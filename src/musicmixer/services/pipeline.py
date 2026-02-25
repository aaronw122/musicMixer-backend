"""Day 2 pipeline orchestrator.

Runs the remix pipeline in a background thread, emitting SSE progress events.
Complete 15-step chain: separation -> analysis -> plan -> processing -> render -> export.
"""

import logging
import queue

from musicmixer.models import SessionState

logger = logging.getLogger(__name__)


def emit_progress(
    event_queue: queue.Queue,
    event: dict,
    session: SessionState | None = None,
) -> None:
    """Non-blocking event push. Drops non-terminal events on full queue.

    Terminal events (complete/error) drain one old event first to guarantee delivery.
    If *session* is provided, also updates ``session.last_event`` so reconnecting
    SSE clients can pick up from where things left off even if no client was
    connected when the event was emitted.
    """
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        if event.get("step") in ("complete", "error"):
            try:
                event_queue.get_nowait()
            except queue.Empty:
                pass
            event_queue.put_nowait(event)
        else:
            logger.warning("Event queue full, dropping: %s", event.get("step"))

    if session is not None:
        session.last_event = event


def run_pipeline(
    session_id: str,
    song_a_path: str,
    song_b_path: str,
    prompt: str,
    event_queue: queue.Queue,
    session: SessionState,
) -> None:
    """Complete Day 2 pipeline: separation, analysis, tempo matching, arrangement, export.

    Pipeline steps:
      1. Separate stems (concurrent for both songs)
      2. Analyze both original songs (BPM, beat grid, duration)
      3. Reconcile BPM between songs
      4. Generate deterministic fallback plan
      5. Determine vocal/instrumental sources from plan
      6. Load and standardize all stems (44.1kHz, stereo, float32)
      7. Trim stems to source time ranges
      8. Compute tempo plan (target BPM, which stems to stretch)
      9. Tempo match via rubberband
     10. Post-stretch beat grid re-detection
     11. Cross-song level matching
     12. Render arrangement into vocal + instrumental buses
     13. Sum buses into final mix
     14. Peak limit (bring peaks under control before normalization)
     15. Peak-constrained LUFS normalization
     16. Fade-in / fade-out
     17. Export to MP3
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import librosa
    import numpy as np

    from musicmixer.config import settings
    from musicmixer.services.analysis import analyze_audio, reconcile_bpm
    from musicmixer.services.interpreter import generate_fallback_plan
    from musicmixer.services.processor import (
        apply_fades,
        auto_level,
        compress_dynamic_range,
        cross_song_level_match,
        compute_tempo_plan,
        export_mp3,
        highpass_filter,
        lufs_normalize_constrained,
        rubberband_process,
        soft_clip,
        trim_audio,
        true_peak,
        validate_stem,
    )
    from musicmixer.services.renderer import render_arrangement
    from musicmixer.services.separation import separate_stems

    song_a_path = Path(song_a_path)
    song_b_path = Path(song_b_path)
    stems_dir = settings.data_dir / "stems" / session_id
    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    # === STEP 1: Separate stems (50% of total time) ===
    emit_progress(event_queue, {
        "step": "separating",
        "detail": "Extracting stems from both songs...",
        "progress": 0.10,
    }, session=session)

    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # Run both separations concurrently
    with ThreadPoolExecutor(max_workers=2) as sep_executor:
        future_a = sep_executor.submit(separate_stems, song_a_path, song_a_stems_dir)
        future_b = sep_executor.submit(separate_stems, song_b_path, song_b_stems_dir)
        song_a_stems = future_a.result(timeout=900)
        song_b_stems = future_b.result(timeout=900)

    emit_progress(event_queue, {
        "step": "separating",
        "detail": "Stems extracted!",
        "progress": 0.50,
    }, session=session)

    # === STEP 2: Analyze both original songs ===
    emit_progress(event_queue, {
        "step": "analyzing",
        "detail": "Detecting tempo and key...",
        "progress": 0.52,
    }, session=session)

    meta_a = analyze_audio(song_a_path)
    meta_b = analyze_audio(song_b_path)
    logger.info(
        "Session %s: Song A BPM=%.1f, Song B BPM=%.1f",
        session_id, meta_a.bpm, meta_b.bpm,
    )

    # === STEP 3: Reconcile BPM between songs ===
    meta_a, meta_b = reconcile_bpm(meta_a, meta_b)
    logger.info(
        "Session %s: Reconciled BPM: A=%.1f, B=%.1f",
        session_id, meta_a.bpm, meta_b.bpm,
    )

    emit_progress(event_queue, {
        "step": "analyzing",
        "detail": f"Song A: {meta_a.bpm:.0f} BPM, Song B: {meta_b.bpm:.0f} BPM",
        "progress": 0.55,
    }, session=session)

    # === STEP 4: Generate deterministic fallback plan ===
    plan = generate_fallback_plan(meta_a, meta_b)
    logger.info(
        "Session %s: Plan -- vocals from %s, tempo from %s",
        session_id, plan.vocal_source, plan.tempo_source,
    )

    # === STEP 5: Determine vocal/instrumental sources ===
    if plan.vocal_source == "song_a":
        vocal_stems_paths = song_a_stems
        inst_stems_paths = song_b_stems
        vocal_meta = meta_a
        inst_meta = meta_b
    else:
        vocal_stems_paths = song_b_stems
        inst_stems_paths = song_a_stems
        vocal_meta = meta_b
        inst_meta = meta_a

    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Standardizing audio...",
        "progress": 0.58,
    }, session=session)

    # === STEP 6: Load and standardize all stems ===
    vocal_audio: dict[str, np.ndarray] = {}
    inst_audio: dict[str, np.ndarray] = {}

    # Load vocal stems (just "vocals" for now)
    for stem_name in ["vocals"]:
        path = vocal_stems_paths.get(stem_name)
        if path is not None:
            audio, _sr = validate_stem(path)
            vocal_audio[stem_name] = audio

    # Load instrumental stems (filter out None paths from 4-stem fallback)
    for stem_name in ["drums", "bass", "guitar", "piano", "other"]:
        path = inst_stems_paths.get(stem_name)
        if path is not None:
            audio, _sr = validate_stem(path)
            inst_audio[stem_name] = audio

    sr = 44100  # All stems standardized to this by validate_stem

    # === STEP 7: Trim stems to source time ranges ===
    for stem_name, audio in vocal_audio.items():
        vocal_audio[stem_name] = trim_audio(
            audio, sr, plan.start_time_vocal, plan.end_time_vocal,
        )
    for stem_name, audio in inst_audio.items():
        inst_audio[stem_name] = trim_audio(
            audio, sr, plan.start_time_instrumental, plan.end_time_instrumental,
        )

    # === STEP 8: Compute tempo plan ===
    target_bpm, stretch_vocals, stretch_instrumentals, tempo_warnings = compute_tempo_plan(
        vocal_meta.bpm, inst_meta.bpm, plan.tempo_source,
    )
    plan.warnings.extend(tempo_warnings)
    logger.info(
        "Session %s: Target BPM=%.1f, stretch_vocals=%s, stretch_inst=%s",
        session_id, target_bpm, stretch_vocals, stretch_instrumentals,
    )

    # === STEP 9: Tempo match via rubberband ===
    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Matching tempo...",
        "progress": 0.62,
    }, session=session)

    total_stems_to_stretch = 0
    if stretch_vocals:
        total_stems_to_stretch += len(vocal_audio)
    if stretch_instrumentals:
        total_stems_to_stretch += len(inst_audio)

    stretched_count = 0

    if stretch_vocals:
        for stem_name in list(vocal_audio.keys()):
            stretched_count += 1
            emit_progress(event_queue, {
                "step": "processing",
                "detail": f"Matching tempo ({stretched_count}/{total_stems_to_stretch} stems)...",
                "progress": 0.62 + (stretched_count / max(total_stems_to_stretch, 1)) * 0.13,
            }, session=session)
            vocal_audio[stem_name] = rubberband_process(
                vocal_audio[stem_name], sr, vocal_meta.bpm, target_bpm,
                is_vocal=(stem_name == "vocals"),
            )

    if stretch_instrumentals:
        for stem_name in list(inst_audio.keys()):
            stretched_count += 1
            emit_progress(event_queue, {
                "step": "processing",
                "detail": f"Matching tempo ({stretched_count}/{total_stems_to_stretch} stems)...",
                "progress": 0.62 + (stretched_count / max(total_stems_to_stretch, 1)) * 0.13,
            }, session=session)
            inst_audio[stem_name] = rubberband_process(
                inst_audio[stem_name], sr, inst_meta.bpm, target_bpm,
            )

    # === STEP 10: Post-stretch beat grid ===
    # Scale the instrumental's original beat grid proportionally as fallback
    beat_scale = inst_meta.bpm / target_bpm if abs(inst_meta.bpm - target_bpm) > 0.001 else 1.0
    # beat_frames are at 22050 Hz analysis rate with hop_length=512 (default).
    # Scale for tempo change. The renderer uses these frames directly with hop_length=512.
    post_stretch_beat_frames = (inst_meta.beat_frames * beat_scale).astype(int)

    # Try re-detecting beats on the summed stretched instrumental (more accurate)
    try:
        if inst_audio:
            inst_arrays = list(inst_audio.values())
            # Sum instrumental stems for beat detection
            inst_sum = inst_arrays[0].copy()
            for arr in inst_arrays[1:]:
                min_len = min(len(inst_sum), len(arr))
                inst_sum = inst_sum[:min_len] + arr[:min_len]

            # Convert to mono for librosa
            mono = np.mean(inst_sum, axis=1) if inst_sum.ndim == 2 else inst_sum
            # Downsample for analysis (librosa beat_track expects 22050)
            mono_22k = librosa.resample(mono, orig_sr=sr, target_sr=22050)
            _, new_beat_frames = librosa.beat.beat_track(
                y=mono_22k, sr=22050, units="frames", start_bpm=target_bpm,
            )
            if len(new_beat_frames) > 10:
                post_stretch_beat_frames = new_beat_frames
                logger.info(
                    "Session %s: Re-detected %d beats post-stretch",
                    session_id, len(new_beat_frames),
                )
            else:
                logger.warning(
                    "Session %s: Post-stretch beat detection found only %d beats, using scaled grid",
                    session_id, len(new_beat_frames),
                )
    except Exception as e:
        logger.warning(
            "Session %s: Post-stretch beat detection failed, using scaled grid: %s",
            session_id, e,
        )

    # === STEP 11: Compress vocal dynamic range ===
    # Compress BEFORE level matching so the LUFS measurement reflects
    # post-compression loudness (otherwise level match is wasted).
    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Normalizing loudness...",
        "progress": 0.80,
    }, session=session)

    if "vocals" in vocal_audio:
        vocal_audio["vocals"] = compress_dynamic_range(
            vocal_audio["vocals"],
            sr,
            threshold_db=-20.0,  # Moderate: only compress loud phrases
            ratio=3.0,           # Standard vocal ratio, preserves natural dynamics
            attack_ms=10.0,      # Preserve plosive transients
            release_ms=80.0,     # Fast release: recover quickly between phrases
            makeup_db=4.0,       # Proportional to reduced compression
            gate_floor_db=-50.0, # Low gate: only ignore true silence
        )
        logger.info("Session %s: Vocal compression applied", session_id)

    # === STEP 11.2: High-pass filter vocals ===
    # Remove low-frequency bleed from stem separation (bass rumble, kick artifacts).
    if "vocals" in vocal_audio:
        vocal_audio["vocals"] = highpass_filter(vocal_audio["vocals"], sr, cutoff_hz=100.0)
        logger.info("Session %s: Vocal HPF applied at 100 Hz", session_id)

    # === STEP 11.5: Cross-song level matching ===
    # Runs AFTER compression so LUFS measurement reflects actual vocal loudness.
    vocal_audio_main = vocal_audio.get("vocals")
    if vocal_audio_main is not None and inst_audio:
        # Sum instrumental stems for LUFS measurement
        inst_arrays = list(inst_audio.values())
        inst_sum_for_lufs = inst_arrays[0].copy()
        for arr in inst_arrays[1:]:
            min_len = min(len(inst_sum_for_lufs), len(arr))
            inst_sum_for_lufs = inst_sum_for_lufs[:min_len] + arr[:min_len]

        vocal_audio["vocals"] = cross_song_level_match(
            vocal_audio_main, inst_sum_for_lufs, sr,
        )

    # === STEP 12: Render arrangement ===
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Building your remix...",
        "progress": 0.85,
    }, session=session)

    vocal_bus, instrumental_bus = render_arrangement(
        sections=plan.sections,
        vocal_stems=vocal_audio,
        instrumental_stems=inst_audio,
        beat_frames=post_stretch_beat_frames,
        sr=sr,
    )

    # === STEP 13: Sum buses into final mix ===
    # Ensure buses are the same length (pad shorter one)
    max_len = max(len(vocal_bus), len(instrumental_bus))
    if len(vocal_bus) < max_len:
        vocal_bus = np.pad(vocal_bus, ((0, max_len - len(vocal_bus)), (0, 0)))
    if len(instrumental_bus) < max_len:
        instrumental_bus = np.pad(instrumental_bus, ((0, max_len - len(instrumental_bus)), (0, 0)))

    mixed = vocal_bus + instrumental_bus

    # Mix bus compression REMOVED -- vocal compression (step 11) + auto-leveler
    # (step 13.7) handle dynamics. A second 3:1 compressor at the same threshold
    # produced ~9:1 effective ratio on vocals, making them flat and lifeless.

    # === STEP 13.7: Slow auto-leveler ===
    # Maintains consistent overall volume over 2-second windows.
    # Gently boosts instrumental-only moments (between vocal phrases)
    # and slightly reduces the loudest peaks. Uses long window so
    # gain changes are imperceptible (no pumping).
    # Detects from the INSTRUMENTAL bus so vocal gain transitions at
    # section boundaries don't trigger pre-emptive cuts via the
    # windowed look-ahead effect (which caused volume dips at ~11s/~22s).
    mixed = auto_level(
        mixed, sr,
        window_sec=3.0,       # 3s compromise: slower than 2s but still responsive to inter-phrase gaps
        max_boost_db=2.0,     # Conservative boost to avoid lifting tail artifacts
        max_cut_db=3.0,       # Up to 3dB cut for loud sections
        target_percentile=50, # Target the median level
        detector_audio=instrumental_bus,  # CRITICAL: drives detection from stable instrumental bus
        active_floor_db=-40.0,  # Treat lower-level tail/noise windows as inactive
    )
    logger.info("Session %s: Auto-leveler applied", session_id)

    # === STEP 14: Peak limit first (bring peaks under control) ===
    ceiling = 10 ** (-1.0 / 20.0)  # -1.0 dBTP ~ 0.891
    peak = true_peak(mixed)
    if peak > ceiling:
        mixed = soft_clip(mixed, ceiling)
        logger.info(
            "Session %s: Pre-normalize peak limiter (peak was %.3f, ceiling %.3f)",
            session_id, peak, ceiling,
        )

    # === STEP 15: Peak-constrained LUFS normalization ===
    # Applies the minimum of (LUFS gain, peak-safe gain) so we never boost
    # past the ceiling. Eliminates the old normalize-then-trim loop that was
    # undoing LUFS gains entirely.
    mixed = lufs_normalize_constrained(mixed, sr, target_lufs=-12.0, ceiling_dbtp=-1.0)
    logger.info("Session %s: Peak-constrained LUFS normalization applied", session_id)

    # === STEP 16: Fades ===
    skip_fade_in = plan.sections[0].transition_in == "fade" if plan.sections else False
    # Renderer no longer applies a terminal fade-out; keep one authoritative
    # final fade stage here in the pipeline for every remix.
    skip_fade_out = False
    mixed = apply_fades(mixed, sr, skip_fade_in=skip_fade_in, skip_fade_out=skip_fade_out)

    # === STEP 17: Export to MP3 ===
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Rendering final mix...",
        "progress": 0.95,
    }, session=session)

    export_mp3(mixed, sr, output_path)

    # === DONE ===
    session.remix_path = str(output_path)
    session.explanation = plan.explanation
    session.status = "complete"

    emit_progress(event_queue, {
        "step": "complete",
        "detail": "Remix ready!",
        "progress": 1.0,
        "explanation": plan.explanation,
        "warnings": plan.warnings,
        "usedFallback": plan.used_fallback,
    }, session=session)

    logger.info("Session %s: Pipeline complete. Output: %s", session_id, output_path)
