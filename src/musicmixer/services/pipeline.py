"""Day 2 pipeline orchestrator.

Runs the remix pipeline in a background thread, emitting SSE progress events.
Complete 15-step chain: separation -> analysis -> plan -> processing -> render -> export.
"""

import logging
import queue
import time

from musicmixer.models import LyricsData, SessionState

logger = logging.getLogger(__name__)

# Maximum wall-clock time (seconds) for any single DSP step on the enhanced
# pipeline path.  If a step exceeds this, its output is discarded and
# processing continues with the pre-step signal state.  This is a post-hoc
# check, not a preemptive kill -- a degenerate step will still block for
# the full duration.  The guard's value is (1) logging which step was slow,
# and (2) not applying potentially corrupted output.
DSP_STEP_TIMEOUT_S = 120.0


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
    song_a_original_filename: str = "",
    song_b_original_filename: str = "",
    source_quality_a: str | None = None,
    source_quality_b: str | None = None,
    force_vocal_source: str | None = None,
) -> None:
    """Complete remix pipeline: separation, analysis, tempo matching, arrangement, export.

    Pipeline steps:
      1. Separate stems (concurrent for both songs)
      2. Analyze both original songs (BPM, beat grid, duration)
      3. Reconcile BPM between songs
      4. Generate mix plan (LLM or deterministic fallback)
     4.5. Taste stage (candidate generation + scoring, if enabled)
      5. Determine vocal/instrumental sources from plan
      6. Load and standardize all stems (44.1kHz, stereo, float32)
      7. Trim stems to source time ranges
     7.5. Detect and exclude near-silent stems
     7.7. Vocal pre-filter bandpass (150Hz-16kHz)
    7.75. Corrective EQ per stem (always on)
      8. Compute tempo plan (target BPM, which stems to stretch)
      9. Tempo match via rubberband
     10. Post-stretch beat grid re-detection
     11. Vocal compression (3:1, -20dB, 3.0dB makeup)
    11.5. Cross-song level matching
    11.8. Pre-limit drum/bass transients
     12. Render arrangement into vocal + instrumental buses
    12.5. Spectral ducking (300-3kHz pocket)
     13. Sum buses into final mix
    13.7. Auto-leveler (4s window, 1.5dB boost, 2.5dB cut)
     14. Static mastering (LUFS normalize + limiter + correction loop + soft clip)
     15. Fade-in / fade-out
     16. Export to MP3 (320kbps, no pre-dither)
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    from pathlib import Path

    import librosa
    import numpy as np
    import pyloudnorm as pyln

    from musicmixer.config import settings
    from musicmixer.services.lyrics import lookup_lyrics_for_song, map_lyrics_to_bars, map_plain_lyrics_to_bars
    from musicmixer.services.analysis import (
        analyze_audio,
        analyze_audio_full,
        analyze_stems,
        compute_relationships,
        detect_key,
        detect_modulation,
        reconcile_bpm,
    )
    from musicmixer.services.interpreter import interpret_prompt, TARGET_REMIX_DURATION_SECONDS
    from musicmixer.services.processor import (
        apply_fades,
        auto_level,
        bandpass_filter,
        compress_dynamic_range,
        cross_song_level_match,
        compute_tempo_plan,
        export_mp3,
        rubberband_process,
        soft_clip,
        trim_audio,
        true_peak,
        true_peak_limit,
        validate_stem,
    )
    from musicmixer.services.ducking import spectral_duck
    from musicmixer.services.eq import apply_corrective_eq
    from musicmixer.services.mastering import master_static
    from musicmixer.services.renderer import render_arrangement
    from musicmixer.services.separation import separate_stems

    song_a_path = Path(song_a_path)
    song_b_path = Path(song_b_path)
    stems_dir = settings.data_dir / "stems" / session_id
    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    logger.info("Session %s: pipeline started (prompt=%r)", session_id, prompt[:80])

    # === STEPS 1+2: Separation + analysis (overlapped) ===
    # Separation and audio analysis run concurrently.  Analysis operates on the
    # original uploaded files (not stems), so it can start immediately alongside
    # separation.  Lyrics lookups also run in the same pool.
    logger.info("Session %s: [1/17] separating stems + analyzing audio...", session_id)
    emit_progress(event_queue, {
        "step": "separating",
        "detail": "Extracting stems from both songs...",
        "progress": 0.10,
    }, session=session)

    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # --- Submit all concurrent work into a single pool ---
    # Workers: 2 separation + 2 analysis + (optionally) 2 lyrics = up to 6
    lyrics_a_data: LyricsData | None = None
    lyrics_b_data: LyricsData | None = None
    lyrics_future_a = None
    lyrics_future_b = None

    pool_workers = 4  # 2 separation + 2 analysis
    if settings.lyrics_lookup_enabled:
        pool_workers = 6

    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        # Separation futures
        sep_future_a = pool.submit(separate_stems, song_a_path, song_a_stems_dir)
        sep_future_b = pool.submit(separate_stems, song_b_path, song_b_stems_dir)

        # Analysis futures (overlapped with separation -- no stem dependency)
        analysis_future_a = pool.submit(analyze_audio_full, song_a_path)
        analysis_future_b = pool.submit(analyze_audio_full, song_b_path)

        # Lyrics futures (optional, also overlapped)
        if settings.lyrics_lookup_enabled:
            try:
                lyrics_future_a = pool.submit(
                    lookup_lyrics_for_song, song_a_path, song_a_original_filename,
                )
                lyrics_future_b = pool.submit(
                    lookup_lyrics_for_song, song_b_path, song_b_original_filename,
                )
                logger.info("Session %s: Lyrics lookup submitted for both songs", session_id)
            except Exception:
                logger.warning("Session %s: Failed to submit lyrics lookups", session_id, exc_info=True)

        # --- Collect analysis results (typically finishes before separation) ---
        meta_a = analysis_future_a.result(timeout=120)
        meta_b = analysis_future_b.result(timeout=120)

        logger.info("Session %s: [2/17] analysis done (A=%.1f BPM, B=%.1f BPM)", session_id, meta_a.bpm, meta_b.bpm)

        # --- Collect separation results ---
        song_a_stems = sep_future_a.result(timeout=900)
        song_b_stems = sep_future_b.result(timeout=900)

        # --- Collect lyrics results ---
        if lyrics_future_a is not None:
            try:
                lyrics_a_data = lyrics_future_a.result(timeout=15)
            except FuturesTimeoutError:
                logger.warning("Session %s: Lyrics lookup timed out for Song A", session_id)
                lyrics_future_a.cancel()
            except Exception:
                logger.warning("Session %s: Lyrics lookup failed for Song A", session_id, exc_info=True)

        if lyrics_future_b is not None:
            try:
                lyrics_b_data = lyrics_future_b.result(timeout=15)
            except FuturesTimeoutError:
                logger.warning("Session %s: Lyrics lookup timed out for Song B", session_id)
                lyrics_future_b.cancel()
            except Exception:
                logger.warning("Session %s: Lyrics lookup failed for Song B", session_id, exc_info=True)

    # Log lyrics results
    for label, data in [("A", lyrics_a_data), ("B", lyrics_b_data)]:
        if data is not None:
            logger.info(
                "Session %s: Song %s lyrics: %d lines, synced=%s, source=%s (%.0fms)",
                session_id, label, len(data.lines), data.is_synced, data.source,
                data.lookup_duration_ms,
            )
        else:
            logger.info("Session %s: Song %s lyrics: not found", session_id, label)

    emit_progress(event_queue, {
        "step": "separating",
        "detail": "Stems extracted!",
        "progress": 0.50,
    }, session=session)

    logger.info("Session %s: [1/17] stems done (%d song_a, %d song_b)", session_id, len(song_a_stems), len(song_b_stems))

    # Emit analyzing progress AFTER separation progress to maintain monotonic
    # step ordering for SSE clients (separating -> analyzing -> processing -> ...)
    emit_progress(event_queue, {
        "step": "analyzing",
        "detail": "Detecting tempo and key...",
        "progress": 0.52,
    }, session=session)

    # Propagate source quality metadata (YouTube inputs carry codec/bitrate info)
    if source_quality_a is not None:
        meta_a.source_quality = source_quality_a
    if source_quality_b is not None:
        meta_b.source_quality = source_quality_b

    logger.info(
        "Session %s: Song A BPM=%.1f key=%s %s, Song B BPM=%.1f key=%s %s",
        session_id, meta_a.bpm, meta_a.key, meta_a.scale,
        meta_b.bpm, meta_b.key, meta_b.scale,
    )

    # === STEP 3: Reconcile BPM between songs ===
    logger.info("Session %s: [3/17] reconciling BPM...", session_id)
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

    # === STEP 3.5: Analyze song structure ===
    # Key detection and modulation are already done by analyze_audio_full above.
    # Only stem-level structure analysis (which depends on separation output)
    # runs here.
    logger.info("Session %s: [3.5/17] analyzing song structure...", session_id)
    emit_progress(event_queue, {
        "step": "analyzing",
        "detail": "Analyzing song structure...",
        "progress": 0.56,
    }, session=session)

    # Log key detection results (already populated by analyze_audio_full)
    for label, meta in [("A", meta_a), ("B", meta_b)]:
        if meta.key is not None:
            logger.info(
                "Session %s: Song %s key=%s %s (confidence=%.2f, modulation=%s)",
                session_id, label, meta.key, meta.scale,
                meta.key_confidence if meta.key_confidence is not None else 0.0,
                meta.has_modulation,
            )

    # Stem-level structure analysis for both songs (depends on stem output)
    for label, meta, s_dir in [
        ("A", meta_a, song_a_stems_dir),
        ("B", meta_b, song_b_stems_dir),
    ]:
        try:
            stem_paths = {name: s_dir / f"{name}.wav" for name in ["vocals", "drums", "bass", "guitar", "piano", "other"]}
            # Filter to stems that actually exist
            stem_paths = {k: v for k, v in stem_paths.items() if v.exists()}
            if stem_paths:
                stem_analysis, song_structure = analyze_stems(
                    stem_paths=stem_paths,
                    beat_frames=meta.beat_frames,
                    bpm=meta.bpm,
                )
                meta.stem_analysis = stem_analysis
                meta.song_structure = song_structure
                logger.info(
                    "Session %s: Song %s structure: %d sections, %d vocal gaps, %d total bars",
                    session_id, label,
                    len(song_structure.sections),
                    len(song_structure.vocal_gaps),
                    song_structure.total_bars,
                )
        except Exception as e:
            logger.warning("Session %s: Structure analysis failed for Song %s: %s", session_id, label, e)

    # Cross-song relationships
    try:
        relationships = compute_relationships(meta_a, meta_b)
        logger.info(
            "Session %s: Cross-song: loudness_diff=%.1fdB, vocal_source=%s, stretch=%.1f%%",
            session_id, relationships.loudness_diff_db,
            relationships.vocal_source, relationships.stretch_pct,
        )
    except Exception as e:
        logger.warning("Session %s: Cross-song relationship analysis failed: %s", session_id, e)

    emit_progress(event_queue, {
        "step": "analyzing",
        "detail": "Song structure analyzed!",
        "progress": 0.57,
    }, session=session)

    # === STEP 3.7: Map lyrics to bars ===
    # Now that beat_frames and bpm are available from analysis, map lyric
    # timestamps to bar numbers so the LLM can cross-reference lyrics with
    # the section map.
    for label, lyrics_data, meta in [
        ("A", lyrics_a_data, meta_a),
        ("B", lyrics_b_data, meta_b),
    ]:
        if lyrics_data is None:
            continue
        try:
            if lyrics_data.is_synced:
                lyrics_data.lines = map_lyrics_to_bars(
                    lyrics_data.lines,
                    beat_frames=meta.beat_frames,
                    bpm=meta.bpm,
                )
            else:
                # Plain lyrics: distribute across vocal-active bars
                vocal_active = None
                total_bars = 0
                if meta.stem_analysis is not None:
                    vocal_active = meta.stem_analysis.vocal_active
                if meta.song_structure is not None:
                    total_bars = meta.song_structure.total_bars
                if total_bars > 0:
                    lyrics_data.lines = map_plain_lyrics_to_bars(
                        lyrics_data.lines,
                        vocal_active=vocal_active,
                        total_bars=total_bars,
                    )
            mapped_count = sum(1 for l in lyrics_data.lines if l.bar_number is not None)
            logger.info(
                "Session %s: Song %s lyrics: %d/%d lines mapped to bars",
                session_id, label, mapped_count, len(lyrics_data.lines),
            )
        except Exception:
            logger.warning(
                "Session %s: Bar mapping failed for Song %s lyrics",
                session_id, label, exc_info=True,
            )

    # === STEP 4: Interpret prompt via LLM (fallback to deterministic plan) ===
    logger.info("Session %s: [4/17] interpreting prompt via LLM...", session_id)
    emit_progress(event_queue, {
        "step": "interpreting",
        "detail": "Planning your remix...",
        "progress": 0.58,
    }, session=session)

    plan = interpret_prompt(prompt, meta_a, meta_b, lyrics_a=lyrics_a_data, lyrics_b=lyrics_b_data)

    if force_vocal_source is not None:
        plan.vocal_source = force_vocal_source
        logger.info("Session %s: Forced vocal_source=%s", session_id, force_vocal_source)

    if plan.used_fallback:
        logger.warning(
            "Session %s: LLM failed, using deterministic fallback",
            session_id,
        )

    logger.info(
        "Session %s: Plan -- vocals from %s, tempo from %s, used_fallback=%s",
        session_id, plan.vocal_source, plan.tempo_source, plan.used_fallback,
    )

    logger.info("Session %s: [4/17] plan ready (vocals=%s, %d sections, fallback=%s)", session_id, plan.vocal_source, len(plan.sections), plan.used_fallback)

    # === STEP 4.5: Taste training stage (candidate generation + scoring) ===
    if settings.ab_taste_model_v1:
        from musicmixer.services.taste_stage import run_taste_stage
        taste_result = run_taste_stage(
            meta_a=meta_a,
            meta_b=meta_b,
            prompt=prompt,
            fallback_plan=plan,  # use LLM/fallback plan as safety net
        )
        if not taste_result.fallback_triggered:
            plan = taste_result.selected_plan
        logger.info(
            "Session %s: Taste stage: %d candidates, %d after filter, method=%s, "
            "latency=%.0fms, fallback=%s",
            session_id,
            taste_result.candidates_generated,
            taste_result.candidates_after_filter,
            taste_result.selection_method,
            taste_result.total_latency_ms,
            taste_result.fallback_triggered,
        )

    # Post-interpret arrangement logging
    if plan.sections:
        _pi_total_beats = plan.sections[-1].end_beat
        from musicmixer.services.tempo import estimate_target_bpm as _est_bpm
        _pi_approx_bpm = _est_bpm(meta_a.bpm, meta_b.bpm, plan.tempo_source)
        _pi_est_duration = _pi_total_beats * 60 / _pi_approx_bpm if _pi_approx_bpm > 0 else 0
        logger.info(
            "Session %s: Arrangement: %d sections, %d beats, est %.0fs at %.0f BPM",
            session_id, len(plan.sections), _pi_total_beats, _pi_est_duration, _pi_approx_bpm,
        )

    # === STEP 5: Determine vocal/instrumental sources ===
    # Fixed convention: Song A always provides vocals, Song B always provides instrumentals.
    vocal_stems_paths = song_a_stems
    inst_stems_paths = song_b_stems
    vocal_meta = meta_a
    inst_meta = meta_b

    # === Source-quality-aware processing: derive per-song lossy flags ===
    is_lossy_source_a = source_quality_a is not None and source_quality_a.startswith("youtube")
    is_lossy_source_b = source_quality_b is not None and source_quality_b.startswith("youtube")

    # Song A = vocal source, Song B = instrumental source (fixed convention)
    is_lossy_vocal_source = is_lossy_source_a
    is_lossy_inst_source = is_lossy_source_b

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

    # === STEP 7.5: Detect and exclude near-silent stems ===
    # Empty-stem guard: stems with integrated LUFS below this threshold are
    # considered inactive and excluded from normalization/summing.
    inactive_lufs_floor = -50.0
    loudness_meter = pyln.Meter(sr)

    def _filter_inactive(stems: dict[str, np.ndarray], stem_group: str) -> dict[str, np.ndarray]:
        active: dict[str, np.ndarray] = {}
        inactive: list[tuple[str, float]] = []
        for stem_name, stem_audio in stems.items():
            try:
                stem_lufs = loudness_meter.integrated_loudness(stem_audio)
            except Exception:
                # If loudness measurement fails, keep the stem to avoid data loss.
                logger.warning(
                    "Session %s: Could not measure loudness for %s stem '%s'; keeping it active",
                    session_id, stem_group, stem_name,
                )
                active[stem_name] = stem_audio
                continue

            if stem_lufs < inactive_lufs_floor:
                inactive.append((stem_name, stem_lufs))
            else:
                active[stem_name] = stem_audio

        if inactive:
            inactive_desc = ", ".join(f"{name} ({lufs:.1f} LUFS)" for name, lufs in inactive)
            logger.info(
                "Session %s: Excluding near-silent %s stems: %s",
                session_id, stem_group, inactive_desc,
            )
            plan.warnings.append(
                f"Ignored near-silent {stem_group} stem(s): {', '.join(name for name, _ in inactive)}"
            )
        return active

    vocal_audio = _filter_inactive(vocal_audio, "vocal")
    inst_audio = _filter_inactive(inst_audio, "instrumental")

    # === STEP 7.7: Vocal pre-filter bandpass ===
    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Pre-filtering vocals...",
        "progress": 0.59,
    }, session=session)
    # Apply 150Hz-16kHz bandpass to vocal stems before tempo stretching.
    # Removes low-frequency bleed (bass rumble, kick artifacts) and
    # high-frequency separation noise, giving rubberband's R3 engine
    # cleaner input for transient detection. 16kHz preserves vocal
    # air/breathiness.
    if "vocals" in vocal_audio:
        vocal_audio["vocals"] = bandpass_filter(
            vocal_audio["vocals"], sr, low_hz=150.0, high_hz=16000.0,
        )
        logger.info(
            "Session %s: Vocal bandpass pre-filter applied (150Hz-16kHz)",
            session_id,
        )

    # === STEP 7.75: Broad preset EQ (before tempo stretch) ===
    # Apply corrective EQ profiles per stem type. Only broad cuts/boosts (Q~1-3)
    # are safe before stretching.
    for stem_type, audio in vocal_audio.items():
        _pre = vocal_audio[stem_type]
        _t0 = time.monotonic()
        _result = apply_corrective_eq(audio, sr, stem_type, apply_preset=True)
        _elapsed = time.monotonic() - _t0
        if _elapsed > DSP_STEP_TIMEOUT_S:
            logger.error(
                "Session %s: DSP step 'preset_eq vocal/%s' exceeded %.0fs timeout (%.1fs), skipping",
                session_id, stem_type, DSP_STEP_TIMEOUT_S, _elapsed,
            )
            vocal_audio[stem_type] = _pre
        else:
            vocal_audio[stem_type] = _result
            logger.info("Session %s: preset_eq vocal/%s took %.2fs", session_id, stem_type, _elapsed)
    for stem_type, audio in inst_audio.items():
        _pre = inst_audio[stem_type]
        _t0 = time.monotonic()
        _result = apply_corrective_eq(audio, sr, stem_type, apply_preset=True)
        _elapsed = time.monotonic() - _t0
        if _elapsed > DSP_STEP_TIMEOUT_S:
            logger.error(
                "Session %s: DSP step 'preset_eq inst/%s' exceeded %.0fs timeout (%.1fs), skipping",
                session_id, stem_type, DSP_STEP_TIMEOUT_S, _elapsed,
            )
            inst_audio[stem_type] = _pre
        else:
            inst_audio[stem_type] = _result
            logger.info("Session %s: preset_eq inst/%s took %.2fs", session_id, stem_type, _elapsed)

    # LUFS checkpoint: after preset EQ
    _eq_meter = pyln.Meter(sr)
    for stem_type, audio in vocal_audio.items():
        _eq_lufs = _eq_meter.integrated_loudness(audio)
        logger.info("Session %s: LUFS after preset EQ (vocal/%s): %.1f", session_id, stem_type, _eq_lufs)
    for stem_type, audio in inst_audio.items():
        _eq_lufs = _eq_meter.integrated_loudness(audio)
        logger.info("Session %s: LUFS after preset EQ (inst/%s): %.1f", session_id, stem_type, _eq_lufs)

    logger.info("Session %s: Preset EQ applied", session_id)

    # === STEP 8: Compute tempo plan ===
    target_bpm, stretch_vocals, stretch_instrumentals, tempo_warnings, stretch_pct = compute_tempo_plan(
        vocal_meta.bpm, inst_meta.bpm, plan.tempo_source,
    )
    plan.warnings.extend(tempo_warnings)
    logger.info(
        "Session %s: Target BPM=%.1f, stretch_vocals=%s, stretch_inst=%s",
        session_id, target_bpm, stretch_vocals, stretch_instrumentals,
    )

    # === STEP 9: Tempo match via rubberband (parallel) ===
    logger.info("Session %s: [9/17] tempo matching via rubberband...", session_id)

    total_stems_to_stretch = (
        (len(vocal_audio) if stretch_vocals else 0)
        + (len(inst_audio) if stretch_instrumentals else 0)
    )

    # Emit batch-level progress BEFORE the pool starts (avoids 5-10s silent gap in SSE stream)
    emit_progress(event_queue, {
        "step": "processing",
        "detail": f"Matching tempo (all {total_stems_to_stretch} stems in parallel)...",
        "progress": 0.62,
    }, session=session)

    with ThreadPoolExecutor(max_workers=6) as rb_executor:
        futures = {}
        if stretch_vocals:
            for stem_name in list(vocal_audio.keys()):
                futures[("vocal", stem_name)] = rb_executor.submit(
                    rubberband_process, vocal_audio[stem_name], sr,
                    vocal_meta.bpm, target_bpm, is_vocal=(stem_name == "vocals"),
                )
        if stretch_instrumentals:
            for stem_name in list(inst_audio.keys()):
                futures[("inst", stem_name)] = rb_executor.submit(
                    rubberband_process, inst_audio[stem_name], sr,
                    inst_meta.bpm, target_bpm,
                )
        for (group, stem_name), future in futures.items():
            result = future.result(timeout=120)
            if group == "vocal":
                vocal_audio[stem_name] = result
            else:
                inst_audio[stem_name] = result

    # Emit completion progress AFTER all futures resolve
    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Tempo matched!",
        "progress": 0.75,
    }, session=session)

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

    vocal_makeup_db = 3.0
    if "vocals" in vocal_audio:
        vocal_audio["vocals"] = compress_dynamic_range(
            vocal_audio["vocals"],
            sr,
            threshold_db=-20.0,  # Moderate: only compress loud phrases
            ratio=3.0,           # Standard vocal ratio, preserves natural dynamics
            attack_ms=10.0,      # Preserve plosive transients
            release_ms=80.0,     # Fast release: recover quickly between phrases
            makeup_db=vocal_makeup_db,
            gate_floor_db=-50.0, # Low gate: only ignore true silence
        )
        logger.info(
            "Session %s: Vocal compression applied (makeup_db=%.1f)",
            session_id, vocal_makeup_db,
        )

    # === STEP 11.2: REMOVED (100Hz HPF) ===
    # The 150Hz-8kHz bandpass pre-filter at step 7.7 subsumes this.
    # Bass bleed is now cleaned before both rubberband and the compressor.

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

    # === STEP 11.8: Pre-limit drum and bass transients ===
    emit_progress(event_queue, {
        "step": "processing",
        "detail": "Taming drum transients...",
        "progress": 0.82,
    }, session=session)
    # Drum transients have 12-15 dB crest factor, consuming all headroom
    # at the mix bus. Pre-limiting reduces crest factor so the LUFS
    # normalizer can actually boost to target. Ceiling/release are tunable
    # parameters (agent can override per-mix in Day 3+).
    if "drums" in inst_audio:
        pre_drum_peak = true_peak(inst_audio["drums"])
        inst_audio["drums"] = true_peak_limit(
            inst_audio["drums"], sr,
            ceiling_dbtp=-6.0,    # Below typical drum peak (~-3 dBTP) to actually engage
            lookahead_ms=3.0,
            release_ms=30.0,
        )
        post_drum_peak = true_peak(inst_audio["drums"])
        logger.info(
            "Session %s: Drum pre-limit: peak %.3f -> %.3f",
            session_id, pre_drum_peak, post_drum_peak,
        )

    if "bass" in inst_audio:
        pre_bass_peak = true_peak(inst_audio["bass"])
        inst_audio["bass"] = true_peak_limit(
            inst_audio["bass"], sr,
            ceiling_dbtp=-4.0,
            lookahead_ms=5.0,
            release_ms=50.0,
        )
        post_bass_peak = true_peak(inst_audio["bass"])
        logger.info(
            "Session %s: Bass pre-limit: peak %.3f -> %.3f",
            session_id, pre_bass_peak, post_bass_peak,
        )

    # === STEP 12: Render arrangement ===
    logger.info("Session %s: [12/17] rendering arrangement...", session_id)
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

    # Post-render duration sanity check
    _pr_actual_duration = vocal_bus.shape[0] / sr
    if plan.sections:
        _pr_estimated_duration = plan.sections[-1].end_beat * 60 / target_bpm if target_bpm > 0 else 0
        logger.info(
            "Session %s: Post-render duration: actual=%.1fs, estimated=%.1fs, target=%ds",
            session_id, _pr_actual_duration, _pr_estimated_duration, TARGET_REMIX_DURATION_SECONDS,
        )
        if _pr_estimated_duration > 0 and abs(_pr_actual_duration - _pr_estimated_duration) / _pr_estimated_duration > 0.15:
            logger.warning(
                "Session %s: Post-render duration mismatch >15%%: actual=%.1fs vs estimated=%.1fs "
                "(delta=%.1f%%). Beat grid or BPM estimation may be inaccurate.",
                session_id, _pr_actual_duration, _pr_estimated_duration,
                (_pr_actual_duration - _pr_estimated_duration) / _pr_estimated_duration * 100,
            )

    # === STEP 12.5: Spectral ducking ===
    # Carve a mid-range pocket (300-3000 Hz) in the instrumental where vocals
    # are active. This is the highest-ROI mixing improvement -- without it,
    # vocals and instruments compete in the 300Hz-5kHz range with zero
    # frequency-aware interaction.
    #
    # CRITICAL: Use a NEW variable (ducked_instrumental). Do NOT mutate
    # instrumental_bus -- the auto-leveler at step 13.7 must continue using
    # the un-ducked instrumental_bus as its detector signal.
    logger.info("Session %s: [12.5/17] applying spectral ducking...", session_id)
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Applying spectral ducking...",
        "progress": 0.87,
    }, session=session)

    ducked_instrumental = spectral_duck(instrumental_bus, vocal_bus, sr)

    # === STEP 13: Sum buses into final mix ===
    # Ensure buses are the same length (pad shorter one).
    # Use ducked_instrumental for the sum, but keep instrumental_bus intact
    # for the auto-leveler detector (pad it to match too).
    max_len = max(len(vocal_bus), len(ducked_instrumental))
    if len(vocal_bus) < max_len:
        vocal_bus = np.pad(vocal_bus, ((0, max_len - len(vocal_bus)), (0, 0)))
    if len(ducked_instrumental) < max_len:
        ducked_instrumental = np.pad(ducked_instrumental, ((0, max_len - len(ducked_instrumental)), (0, 0)))
    if len(instrumental_bus) < max_len:
        instrumental_bus = np.pad(instrumental_bus, ((0, max_len - len(instrumental_bus)), (0, 0)))

    mixed = vocal_bus + ducked_instrumental

    # LUFS checkpoint: after bus sum
    _meter = pyln.Meter(sr)
    _lufs_post_sum = _meter.integrated_loudness(mixed)
    logger.info("Session %s: LUFS after bus sum: %.1f", session_id, _lufs_post_sum)

    # Mix bus compression REMOVED -- vocal compression (step 11) + auto-leveler
    # (step 13.7) handle dynamics. A second 3:1 compressor at the same threshold
    # produced ~9:1 effective ratio on vocals, making them flat and lifeless.

    # === STEP 13.7: Slow auto-leveler ===
    # Maintains consistent overall volume over multi-second windows.
    # Gently boosts instrumental-only moments (between vocal phrases)
    # and slightly reduces the loudest peaks. Uses long window so
    # gain changes are imperceptible (no pumping).
    auto_level_kwargs = dict(window_sec=4.0, max_boost_db=1.5, max_cut_db=2.5)

    # CRITICAL: detector_audio must be set to instrumental_bus to avoid the volume-dip
    # regression (the ~11s/~22s dip bug). Without this, auto_level uses the mixed signal
    # for detection, which causes vocals to trigger gain reduction on themselves.
    auto_level_kwargs["detector_audio"] = instrumental_bus
    auto_level_kwargs["target_percentile"] = 50.0
    auto_level_kwargs["active_floor_db"] = -50.0

    mixed = auto_level(mixed, sr, **auto_level_kwargs)
    logger.info(
        "Session %s: Auto-leveler applied (window_sec=%.1f, max_boost_db=%.1f, max_cut_db=%.1f)",
        session_id,
        auto_level_kwargs["window_sec"],
        auto_level_kwargs["max_boost_db"],
        auto_level_kwargs["max_cut_db"],
    )

    # LUFS checkpoint: after auto-level
    _lufs_post_autolevel = _meter.integrated_loudness(mixed)
    logger.info("Session %s: LUFS after auto-level: %.1f", session_id, _lufs_post_autolevel)

    # === STEPS 14/14.5/15/15.5: Mastering chain (mutual exclusion) ===
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Mastering: peak limiting...",
        "progress": 0.89,
    }, session=session)
    # === STEP 14: Static mastering chain ===
    # Constrained LUFS normalize (-12 LUFS) then limiter (-1.0 dBTP).
    master_kwargs: dict = dict(target_lufs=-12.0, ceiling_dbtp=-1.0)
    if is_lossy_source_a or is_lossy_source_b:
        master_kwargs["lossy_lpf_hz"] = 16000  # gentle LPF at spectral ceiling
    _pre_master = mixed
    _t0 = time.monotonic()
    _master_result = master_static(mixed, sr, **master_kwargs)
    _elapsed = time.monotonic() - _t0
    if _elapsed > DSP_STEP_TIMEOUT_S:
        logger.error(
            "Session %s: DSP step 'master_static' exceeded %.0fs timeout (%.1fs), skipping",
            session_id, DSP_STEP_TIMEOUT_S, _elapsed,
        )
        mixed = _pre_master
    else:
        mixed = _master_result
        logger.info("Session %s: master_static took %.2fs", session_id, _elapsed)

    # LUFS checkpoint: after static mastering
    _lufs_post_master = _meter.integrated_loudness(mixed)
    logger.info("Session %s: LUFS after static mastering: %.1f", session_id, _lufs_post_master)

    # === STEP 14.5: Post-mastering LUFS correction (iterate-and-converge) ===
    # master_static normalizes + limits, but the limiter eats 2-3 dB of
    # integrated loudness. This loop measures the actual LUFS and applies
    # a bounded correction + a light second limiter pass to catch any
    # re-introduced peaks.
    TARGET_MASTER_LUFS = -12.0
    _lufs_meter = pyln.Meter(sr)
    _meas = np.column_stack([mixed, mixed]) if mixed.ndim == 1 else mixed
    post_master_lufs = _lufs_meter.integrated_loudness(_meas)

    if post_master_lufs > -70.0 and post_master_lufs < TARGET_MASTER_LUFS - 1.0:
        correction_db = TARGET_MASTER_LUFS - post_master_lufs
        correction_db = min(correction_db, 3.0)  # safety cap: never boost more than 3 dB

        # Apply correction unconditionally -- do NOT cap by peak headroom.
        # The second limiter pass below will handle any peaks that exceed ceiling.
        mixed = mixed * (10 ** (correction_db / 20.0))
        logger.info(
            "Session %s: post-mastering LUFS correction +%.1f dB (%.1f -> ~%.1f LUFS)",
            session_id, correction_db, post_master_lufs, post_master_lufs + correction_db,
        )

        # Second (lighter) limiter pass to catch re-introduced peaks.
        mixed = true_peak_limit(
            mixed, sr,
            ceiling_dbtp=-1.0,
            lookahead_ms=5.0,
            release_ms=50.0,
        )

        # Log final LUFS for verification
        _meas_final = np.column_stack([mixed, mixed]) if mixed.ndim == 1 else mixed
        final_lufs = _lufs_meter.integrated_loudness(_meas_final)
        logger.info(
            "Session %s: final LUFS after correction + re-limit: %.1f",
            session_id, final_lufs,
        )

    # === STEP 14.6: Safety soft clip ===
    # Catches inter-sample true peaks that can exceed -1.0 dBTP after
    # MP3 encoding. Without this, lossy codecs can reconstruct peaks
    # above the limiter ceiling, causing audible distortion.
    safety_ceiling = 10 ** (-1.0 / 20.0)
    mixed = soft_clip(mixed, safety_ceiling, knee_db=2.0)

    # === STEP 16: Fades ===
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Applying fades...",
        "progress": 0.93,
    }, session=session)
    skip_fade_in = plan.sections[0].transition_in == "fade" if plan.sections else False
    # Renderer no longer applies a terminal fade-out; keep one authoritative
    # final fade stage here in the pipeline for every remix.
    skip_fade_out = False
    mixed = apply_fades(mixed, sr, skip_fade_in=skip_fade_in, skip_fade_out=skip_fade_out)

    # === STEP 17: Export to MP3 ===
    logger.info("Session %s: [17/17] exporting MP3...", session_id)
    emit_progress(event_queue, {
        "step": "rendering",
        "detail": "Rendering final mix...",
        "progress": 0.95,
    }, session=session)

    export_mp3(mixed, sr, output_path, use_s16_dither=False)

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
