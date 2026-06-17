"""Day 2 pipeline orchestrator.

Runs the remix pipeline in a background thread, emitting SSE progress events.
Complete 16-step chain: separation -> analysis -> plan -> processing -> render -> export.
"""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from musicmixer.models import AnalyzedSongs, AudioMetadata, LyricsData, SessionState
from musicmixer.services.pipeline_metrics import PipelineMetrics

logger = logging.getLogger(__name__)

# Stem name constants for the two separation models.
# Song A (vocal source) uses MelBand Roformer → 3 stems.
# Song B (instrumental source) uses BS-RoFormer → 6 stems.
VOCAL_STEM_NAMES = ("lead_vocals", "backing_vocals")
INSTRUMENTAL_STEM_NAMES = ("drums", "bass", "guitar", "piano", "other")

# Stem names that structure analysis (analyze_stems) will load if present on disk.
# This is the domain contract for which separator output shapes are supported:
# the vocal-shape names (single "vocals", or split lead/backing, plus "instrumental")
# and the 6-stem instrumental names. analyze_stems filters this to existing files.
ANALYSIS_STEM_CANDIDATES = (
    # Vocal-shape names
    "vocals",
    "lead_vocals",
    "backing_vocals",
    "instrumental",
    # 6-stem instrumental names
    "drums",
    "bass",
    "guitar",
    "piano",
    "other",
)

# Maximum wall-clock time (seconds) for any single DSP step on the enhanced
# pipeline path.  If a step exceeds this, its output is discarded and
# processing continues with the pre-step signal state.  This is a post-hoc
# check, not a preemptive kill -- a degenerate step will still block for
# the full duration.  The guard's value is (1) logging which step was slow,
# and (2) not applying potentially corrupted output.
DSP_STEP_TIMEOUT_S = 120.0


class CancelledError(Exception):
    """Raised when a session is cancelled by the user."""
    pass


def check_cancelled(session: SessionState | None) -> None:
    """Raise CancelledError if the session has been cancelled."""
    if session is not None and session.cancelled.is_set():
        raise CancelledError(f"Session cancelled by user")


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
        if event.get("step") in ("complete", "error", "cancelled"):
            try:
                event_queue.get_nowait()
            except queue.Empty:
                pass
            event_queue.put_nowait(event)
        else:
            logger.warning("Event queue full, dropping: %s", event.get("step"))

    if session is not None:
        session.last_event = event


def progress_event(step: str, detail: str, progress: float, **extra: Any) -> dict:
    """Build a progress event payload. ``**extra`` carries divergent metadata
    (e.g. ``position``, ``total``, ``wait_seconds``) emitted by some sites."""
    return {"step": step, "detail": detail, "progress": progress, **extra}


def progress_ticker(event_queue, session, start, end, step_name, detail, interval=5):
    """Background thread that slowly advances progress from start toward end."""
    stop = threading.Event()
    def _run():
        current = start
        increment = (end - start) * 0.12
        while current < end - 0.01:
            if stop.wait(timeout=interval):
                return
            current = min(current + increment, end - 0.01)
            increment *= 0.8
            emit_progress(event_queue, progress_event(
                step_name, detail, round(current, 3),
            ), session=session)
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return stop


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _check_remix_cache(
    session_id: str,
    song_a_path,
    song_b_path,
    prompt: str,
    output_path,
    session: SessionState,
    event_queue: queue.Queue,
) -> str | None:
    """Check the remix cache; copy cached result and return cache key if hit.

    Returns the cache key (always computed when caching is enabled).
    If the cache was hit, session/event_queue are updated and the caller
    should return early from the pipeline.  The caller can detect a hit
    by checking ``session.status == "complete"`` after this call.
    """
    from musicmixer.config import settings

    remix_cache_key: str | None = None
    if not settings.remix_cache_enabled:
        return remix_cache_key

    try:
        from musicmixer.services.remix_cache import (
            compute_remix_cache_key,
            get_cached_metadata,
            get_cached_remix,
        )
        import shutil as _shutil

        remix_cache_key = compute_remix_cache_key(song_a_path, song_b_path, prompt)
        cached_path = get_cached_remix(remix_cache_key, settings.remix_cache_dir)

        if cached_path is not None:
            logger.info(
                "Session %s: Remix cache hit (key=%s), skipping pipeline",
                session_id, remix_cache_key[:12],
            )
            _shutil.copy2(cached_path, output_path)

            meta = get_cached_metadata(remix_cache_key, settings.remix_cache_dir) or {}

            session.remix_path = str(output_path)
            session.explanation = meta.get("explanation", "")
            session.used_fallback = meta.get("used_fallback", False)
            session.warnings = meta.get("warnings", [])
            session.key_warning = meta.get("key_warning")
            session.status = "complete"

            complete_event = {
                "step": "complete",
                "detail": "Your remix is ready! 🎧",
                "progress": 1.0,
                "explanation": session.explanation,
                "warnings": session.warnings,
                "usedFallback": session.used_fallback,
            }
            if session.key_warning:
                complete_event["keyWarning"] = session.key_warning

            emit_progress(event_queue, complete_event, session=session)

            logger.info("Session %s: Pipeline complete (cached). Output: %s", session_id, output_path)
    except Exception:
        logger.warning(
            "Session %s: Remix cache check failed, proceeding with full pipeline",
            session_id, exc_info=True,
        )

    return remix_cache_key


def _step_separate_and_analyze(
    session_id: str,
    song_a_path,
    song_b_path,
    stems_dir,
    song_a_original_filename: str,
    song_b_original_filename: str,
    event_queue: queue.Queue,
    session: SessionState,
    cached_meta_a: AudioMetadata | None = None,
    cached_meta_b: AudioMetadata | None = None,
    cached_lyrics_a: LyricsData | None = None,
    cached_lyrics_b: LyricsData | None = None,
) -> tuple:
    """Steps 1+2: Separation + analysis (overlapped).

    When cached metadata is provided for a song, analysis/lyrics/structure-ML
    are skipped for that song (medium cache path).  Separation always runs.

    Returns (song_a_stems, song_b_stems, meta_a, meta_b,
             lyrics_a_data, lyrics_b_data, ml_segments_a, ml_segments_b,
             song_a_stems_dir, song_b_stems_dir).
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

    from musicmixer.config import settings
    from musicmixer.services.lyrics import lookup_lyrics_for_song
    from musicmixer.services.analysis import analyze_audio_full
    from musicmixer.services.separation import separate_stems, separate_vocal_song

    logger.info("Session %s: [1/17] separating stems + analyzing audio...", session_id)
    emit_progress(event_queue, progress_event(
        "separating", "Pulling each instrument out of the mix...", 0.10,
    ), session=session)

    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # --- Medium cache path: log which songs use cached metadata ---
    if cached_meta_a is not None:
        logger.info("Session %s: Using cached metadata for song A (skipping analysis/lyrics/structure-ML)", session_id)
    if cached_meta_b is not None:
        logger.info("Session %s: Using cached metadata for song B (skipping analysis/lyrics/structure-ML)", session_id)

    # --- Submit all concurrent work into a single pool ---
    # Workers: 2 separation + (0-2) analysis + (optionally 0-2) lyrics = up to 8
    lyrics_a_data: LyricsData | None = cached_lyrics_a
    lyrics_b_data: LyricsData | None = cached_lyrics_b
    lyrics_future_a = None
    lyrics_future_b = None

    # --- ML structure detection (SongFormer) ---
    # Runs on the original mix audio (no stems needed), so it can overlap with
    # separation.  Config-aware: "heuristic" skips ML entirely, "auto" falls
    # back on failure, "ml" raises on failure.
    ml_segments_a: list[dict] | None = None
    ml_segments_b: list[dict] | None = None
    structure_ml_enabled = settings.section_detection_backend in ("auto", "ml")

    from musicmixer.services.structure_ml import analyze_structure_ml
    if structure_ml_enabled:
        logger.info(
            "Session %s: ML structure detection enabled (backend=%s)",
            session_id, settings.section_detection_backend,
        )
    else:
        logger.info("Session %s: ML structure detection skipped (backend=heuristic)", session_id)

    # Count pool workers: separation always runs, analysis/lyrics/structure
    # are skipped per-song when cached metadata is available.
    _need_fresh_a = cached_meta_a is None
    _need_fresh_b = cached_meta_b is None
    pool_workers = 2  # 2 separation (always)
    pool_workers += int(_need_fresh_a) + int(_need_fresh_b)  # 0-2 analysis
    if settings.lyrics_lookup_enabled:
        pool_workers += int(_need_fresh_a) + int(_need_fresh_b)  # 0-2 lyrics
    if structure_ml_enabled:
        pool_workers += int(_need_fresh_a) + int(_need_fresh_b)  # 0-2 ML structure

    structure_future_a = None
    structure_future_b = None
    analysis_future_a = None
    analysis_future_b = None

    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        # Separation futures: ALWAYS run (even with cached metadata, stems
        # must be separated for the new role).
        sep_future_a = pool.submit(separate_vocal_song, song_a_path, song_a_stems_dir)
        sep_future_b = pool.submit(separate_stems, song_b_path, song_b_stems_dir)

        # Analysis futures (skip when cached metadata is available)
        if _need_fresh_a:
            analysis_future_a = pool.submit(analyze_audio_full, song_a_path)
        if _need_fresh_b:
            analysis_future_b = pool.submit(analyze_audio_full, song_b_path)

        # ML structure futures (skip when cached metadata is available)
        if structure_ml_enabled:
            if _need_fresh_a:
                structure_future_a = pool.submit(analyze_structure_ml, song_a_path)
            if _need_fresh_b:
                structure_future_b = pool.submit(analyze_structure_ml, song_b_path)

        # Lyrics futures (skip when cached metadata is available)
        if settings.lyrics_lookup_enabled:
            try:
                if _need_fresh_a:
                    lyrics_future_a = pool.submit(
                        lookup_lyrics_for_song, song_a_path, song_a_original_filename,
                    )
                if _need_fresh_b:
                    lyrics_future_b = pool.submit(
                        lookup_lyrics_for_song, song_b_path, song_b_original_filename,
                    )
                if _need_fresh_a or _need_fresh_b:
                    if _need_fresh_a and _need_fresh_b:
                        lyrics_scope = "both songs"
                    elif _need_fresh_a:
                        lyrics_scope = "Song A only"
                    else:
                        lyrics_scope = "Song B only"
                    logger.info("Session %s: Lyrics lookup submitted for %s", session_id, lyrics_scope)
            except Exception:
                logger.warning("Session %s: Failed to submit lyrics lookups", session_id, exc_info=True)

        # --- Collect analysis results (use cached values when available) ---
        meta_a = cached_meta_a if cached_meta_a is not None else analysis_future_a.result(timeout=120)
        meta_b = cached_meta_b if cached_meta_b is not None else analysis_future_b.result(timeout=120)

        logger.info("Session %s: [2/17] analysis done (A=%.1f BPM, B=%.1f BPM)", session_id, meta_a.bpm, meta_b.bpm)

        # --- Collect separation results via as_completed + tickers ---
        sep_futures = {sep_future_a: "a", sep_future_b: "b"}
        ticker = progress_ticker(event_queue, session, 0.12, 0.23, "separating",
                                 "Pulling apart every instrument...", interval=5)
        try:
            completed_sep = 0
            for future in as_completed(sep_futures, timeout=900):
                completed_sep += 1
                ticker.set()
                if completed_sep == 1:
                    emit_progress(event_queue, progress_event(
                        "separating", "Got the first song's stems!", 0.24,
                    ), session=session)
                    ticker = progress_ticker(event_queue, session, 0.25, 0.36, "separating",
                                             "Working on the second song...", interval=5)
                else:
                    emit_progress(event_queue, progress_event(
                        "separating", "Got both songs' stems!", 0.37,
                    ), session=session)
        finally:
            ticker.set()

        # Collect results (already resolved, returns immediately)
        song_a_stems = sep_future_a.result()
        song_b_stems = sep_future_b.result()

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

        # --- Collect ML structure results (120s timeout for Modal cold starts) ---
        _STRUCTURE_ML_TIMEOUT_S = 120
        for label, future, target in [
            ("A", structure_future_a, "ml_segments_a"),
            ("B", structure_future_b, "ml_segments_b"),
        ]:
            if future is None:
                continue
            try:
                segments = future.result(timeout=_STRUCTURE_ML_TIMEOUT_S)
                if target == "ml_segments_a":
                    ml_segments_a = segments
                else:
                    ml_segments_b = segments
                logger.info(
                    "Session %s: ML structure for Song %s: %d segments",
                    session_id, label, len(segments),
                )
            except FuturesTimeoutError:
                msg = f"ML structure detection timed out for Song {label} (>{_STRUCTURE_ML_TIMEOUT_S}s)"
                logger.warning("Session %s: %s", session_id, msg)
                future.cancel()
                if settings.section_detection_backend == "ml":
                    raise RuntimeError(msg)
            except Exception:
                msg = f"ML structure detection failed for Song {label}"
                logger.warning("Session %s: %s", session_id, msg, exc_info=True)
                if settings.section_detection_backend == "ml":
                    raise

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

    emit_progress(event_queue, progress_event(
        "separating", "Got all the pieces!", 0.39,
    ), session=session)

    logger.info("Session %s: [1/17] stems done (%d song_a, %d song_b)", session_id, len(song_a_stems), len(song_b_stems))

    return (
        song_a_stems, song_b_stems, meta_a, meta_b,
        lyrics_a_data, lyrics_b_data, ml_segments_a, ml_segments_b,
        song_a_stems_dir, song_b_stems_dir,
    )


def _step_reconcile_bpm(session_id: str, meta_a, meta_b, event_queue, session):
    """Step 3: Reconcile BPM between songs.

    Returns updated (meta_a, meta_b).
    """
    from musicmixer.services.analysis import reconcile_bpm

    logger.info("Session %s: [3/17] reconciling BPM...", session_id)
    meta_a, meta_b = reconcile_bpm(meta_a, meta_b)
    logger.info(
        "Session %s: Reconciled BPM: A=%.1f, B=%.1f",
        session_id, meta_a.bpm, meta_b.bpm,
    )

    emit_progress(event_queue, progress_event(
        "analyzing",
        f"Song A vibes at {meta_a.bpm:.0f} BPM, Song B grooves at {meta_b.bpm:.0f} BPM",
        0.45,
    ), session=session)

    return meta_a, meta_b


def _run_pulsemap_analysis(
    meta_a, meta_b,
    song_a_path, song_b_path,
    song_a_stems_dir, song_b_stems_dir,
    lyrics_a_data, session_id: str,
) -> None:
    """Run PulseMap analyses (chords sequential, lightweight tasks parallel).

    Chords run one-at-a-time to limit peak memory (~500MB per ensemble).
    Polyphony, drums, and word alignment run in parallel (lightweight).
    Each analysis is config-gated and wrapped in try/except so failures
    don't block the pipeline.  Mutates meta_a / meta_b in place.
    """
    from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    from musicmixer.config import settings

    pulsemap_futures: dict[str, Future] = {}
    # Use lead_vocals as the primary vocal stem for PulseMap analysis.
    # Fall back to "vocals.wav" for backward compatibility with older cached stems.
    vocal_stem_a = song_a_stems_dir / "lead_vocals.wav"
    if not vocal_stem_a.exists():
        vocal_stem_a = song_a_stems_dir / "vocals.wav"
    drum_stem_b = song_b_stems_dir / "drums.wav"

    from musicmixer.services.pulsemap import (
        align_words,
        detect_chords,
        detect_polyphony,
        transcribe_drum_pattern,
    )

    _PULSEMAP_TIMEOUT_S = 120
    task_count = 0

    # --- Sequential chord detection (each loads ~500MB of ensemble models) ---
    if settings.pulsemap_chords_enabled and song_a_path is not None:
        task_count += 1
        try:
            meta_a.chord_progression = detect_chords(song_a_path)
            logger.info(
                "Session %s: Song A chords: %d events, summary=%s",
                session_id,
                len(meta_a.chord_progression.chords),
                meta_a.chord_progression.progression_summary,
            )
        except Exception:
            logger.warning("Session %s: Chord detection failed for Song A", session_id, exc_info=True)

    if settings.pulsemap_chords_enabled and song_b_path is not None:
        task_count += 1
        try:
            meta_b.chord_progression = detect_chords(song_b_path)
            logger.info(
                "Session %s: Song B chords: %d events, summary=%s",
                session_id,
                len(meta_b.chord_progression.chords),
                meta_b.chord_progression.progression_summary,
            )
        except Exception:
            logger.warning("Session %s: Chord detection failed for Song B", session_id, exc_info=True)

    # --- Parallel lightweight tasks (polyphony + drums) ---
    pulsemap_workers = 0
    if settings.pulsemap_polyphony_enabled and vocal_stem_a.exists():
        pulsemap_workers += 1
    if settings.pulsemap_drums_enabled and drum_stem_b.exists():
        pulsemap_workers += 1
    if settings.pulsemap_word_alignment_enabled and vocal_stem_a.exists():
        pulsemap_workers += 1

    if pulsemap_workers == 0 and task_count == 0:
        logger.info("Session %s: PulseMap analysis: all analyses disabled or stems missing", session_id)
        return

    task_count += pulsemap_workers
    logger.info("Session %s: PulseMap analysis: running %d tasks (chords sequential, rest parallel)", session_id, task_count)

    if pulsemap_workers > 0:
        with ThreadPoolExecutor(max_workers=pulsemap_workers) as pool:
            if settings.pulsemap_polyphony_enabled and vocal_stem_a.exists():
                pulsemap_futures["polyphony"] = pool.submit(detect_polyphony, vocal_stem_a)

            if settings.pulsemap_drums_enabled and drum_stem_b.exists():
                pulsemap_futures["drums"] = pool.submit(transcribe_drum_pattern, drum_stem_b)

            if settings.pulsemap_word_alignment_enabled and vocal_stem_a.exists():
                pulsemap_futures["word_align"] = pool.submit(
                    align_words, vocal_stem_a, lyrics_a_data,
                )

            if "polyphony" in pulsemap_futures:
                try:
                    meta_a.polyphony_info = pulsemap_futures["polyphony"].result(timeout=_PULSEMAP_TIMEOUT_S)
                    logger.info(
                        "Session %s: Polyphony: %s (method=%s)",
                        session_id,
                        "polyphonic" if meta_a.polyphony_info.polyphonic else "solo",
                        meta_a.polyphony_info.method,
                    )
                except FuturesTimeoutError:
                    logger.warning("Session %s: Polyphony detection timed out", session_id)
                    pulsemap_futures["polyphony"].cancel()
                except Exception:
                    logger.warning("Session %s: Polyphony detection failed", session_id, exc_info=True)

            if "drums" in pulsemap_futures:
                try:
                    meta_b.drum_pattern = pulsemap_futures["drums"].result(timeout=_PULSEMAP_TIMEOUT_S)
                    logger.info(
                        "Session %s: Drum pattern: %s (kick=%d, snare=%d, hihat=%d)",
                        session_id,
                        meta_b.drum_pattern.style_hint,
                        meta_b.drum_pattern.kick_count,
                        meta_b.drum_pattern.snare_count,
                        meta_b.drum_pattern.hihat_count,
                    )
                except FuturesTimeoutError:
                    logger.warning("Session %s: Drum transcription timed out", session_id)
                    pulsemap_futures["drums"].cancel()
                except Exception:
                    logger.warning("Session %s: Drum transcription failed", session_id, exc_info=True)

            if "word_align" in pulsemap_futures:
                try:
                    meta_a.word_alignment = pulsemap_futures["word_align"].result(timeout=_PULSEMAP_TIMEOUT_S)
                    logger.info(
                        "Session %s: Word alignment: %d words, validated=%s",
                        session_id,
                        len(meta_a.word_alignment.words),
                        meta_a.word_alignment.lrclib_validated,
                    )
                except FuturesTimeoutError:
                    logger.warning("Session %s: Word alignment timed out", session_id)
                    pulsemap_futures["word_align"].cancel()
                except Exception:
                    logger.warning("Session %s: Word alignment failed", session_id, exc_info=True)


def _step_analyze_structure(
    session_id: str,
    meta_a, meta_b,
    song_a_stems_dir, song_b_stems_dir,
    ml_segments_a, ml_segments_b,
    event_queue, session,
    song_a_path=None,
    song_b_path=None,
    lyrics_a_data=None,
    has_cached_meta_a: bool = False,
    has_cached_meta_b: bool = False,
):
    """Step 3.5: Analyze song structure (key, sections, cross-song relationships).

    Mutates meta_a and meta_b in place (adds stem_analysis, song_structure,
    and PulseMap analysis fields: chord_progression, polyphony_info,
    drum_pattern, word_alignment).

    When has_cached_meta_a/has_cached_meta_b are True, stem analysis and
    PulseMap are skipped for that song (cached metadata already has those fields).
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

    from musicmixer.config import settings
    from musicmixer.services.analysis import analyze_stems, compute_relationships

    logger.info("Session %s: [3.5/17] analyzing song structure...", session_id)
    emit_progress(event_queue, progress_event(
        "analyzing", "Mapping out verses, choruses, drops...", 0.50,
    ), session=session)

    # When both songs are cached, skip everything in this step
    if has_cached_meta_a and has_cached_meta_b:
        logger.info("Session %s: [3.5/17] skipping structure + PulseMap (both songs cached)", session_id)
        emit_progress(event_queue, progress_event(
            "analyzing", "Got the blueprint!", 0.57,
        ), session=session)
        return

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
    # Skip for songs with cached metadata (already has stem_analysis, song_structure)
    for label, meta, s_dir, ml_segs, has_cached_meta in [
        ("A", meta_a, song_a_stems_dir, ml_segments_a, has_cached_meta_a),
        ("B", meta_b, song_b_stems_dir, ml_segments_b, has_cached_meta_b),
    ]:
        if has_cached_meta:
            logger.info("Session %s: Song %s structure: skipped (cached)", session_id, label)
            continue
        try:
            stem_paths = {name: s_dir / f"{name}.wav" for name in ANALYSIS_STEM_CANDIDATES}
            # Filter to stems that actually exist
            stem_paths = {k: v for k, v in stem_paths.items() if v.exists()}
            if stem_paths:
                stem_analysis, song_structure = analyze_stems(
                    stem_paths=stem_paths,
                    beat_frames=meta.beat_frames,
                    bpm=meta.bpm,
                    audio_path=song_a_path if label == "A" else song_b_path,
                    ml_segments=ml_segs,
                    downbeat_times=meta.downbeat_times,
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

    # Cross-song relationships (output is logged then discarded — skip when
    # either song is cached to avoid unnecessary compute)
    if not has_cached_meta_a and not has_cached_meta_b:
        try:
            relationships = compute_relationships(meta_a, meta_b)
            logger.info(
                "Session %s: Cross-song: loudness_diff=%.1fdB, vocal_source=%s, stretch=%.1f%%",
                session_id, relationships.loudness_diff_db,
                relationships.vocal_source, relationships.stretch_pct,
            )
        except Exception as e:
            logger.warning("Session %s: Cross-song relationship analysis failed: %s", session_id, e)

    # --- PulseMap analysis (parallel) ---
    # Skip for songs with cached metadata (already has PulseMap fields)
    if not has_cached_meta_a and not has_cached_meta_b:
        _run_pulsemap_analysis(
            meta_a, meta_b,
            song_a_path, song_b_path,
            song_a_stems_dir, song_b_stems_dir,
            lyrics_a_data, session_id,
        )
    elif not has_cached_meta_a or not has_cached_meta_b:
        # One song cached, one not — PulseMap only operates on specific
        # per-song fields, so we can still run it and the cached song's
        # fields won't be overwritten (they're already set).
        _run_pulsemap_analysis(
            meta_a, meta_b,
            song_a_path if not has_cached_meta_a else None,
            song_b_path if not has_cached_meta_b else None,
            song_a_stems_dir, song_b_stems_dir,
            lyrics_a_data, session_id,
        )

    emit_progress(event_queue, progress_event(
        "analyzing", "Got the blueprint!", 0.57,
    ), session=session)


def _step_measure_stem_lufs(
    session_id: str,
    song_a_stems_dir,
    song_b_stems_dir,
) -> tuple[dict[str, float], dict[str, float]]:
    """Step 3.8: Measure per-stem LUFS for LLM interpreter.

    Returns (vocal_stem_lufs, inst_stem_lufs).
    """
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    vocal_stem_lufs: dict[str, float] = {}
    inst_stem_lufs: dict[str, float] = {}

    _lufs_meter_raw = pyln.Meter(44100)
    # Song A = vocal source stems (MelBand Roformer: lead_vocals, backing_vocals;
    # fall back to "vocals" for backward compat with older cached stems)
    for stem_name in ["lead_vocals", "backing_vocals", "vocals"]:
        stem_path = song_a_stems_dir / f"{stem_name}.wav"
        if stem_path.exists():
            try:
                audio_data, _stem_sr = sf.read(stem_path, dtype="float32")
                if audio_data.ndim == 1:
                    audio_data = np.column_stack([audio_data, audio_data])
                lufs_val = _lufs_meter_raw.integrated_loudness(audio_data)
                vocal_stem_lufs[stem_name] = lufs_val
                logger.info(
                    "Session %s: Raw stem LUFS (vocal/%s): %.1f",
                    session_id, stem_name, lufs_val,
                )
            except Exception:
                logger.warning(
                    "Session %s: Failed to measure LUFS for vocal/%s",
                    session_id, stem_name, exc_info=True,
                )

    # Song B = instrumental source stems
    for stem_name in ["drums", "bass", "guitar", "piano", "other"]:
        stem_path = song_b_stems_dir / f"{stem_name}.wav"
        if stem_path.exists():
            try:
                audio_data, _stem_sr = sf.read(stem_path, dtype="float32")
                if audio_data.ndim == 1:
                    audio_data = np.column_stack([audio_data, audio_data])
                lufs_val = _lufs_meter_raw.integrated_loudness(audio_data)
                inst_stem_lufs[stem_name] = lufs_val
                logger.info(
                    "Session %s: Raw stem LUFS (inst/%s): %.1f",
                    session_id, stem_name, lufs_val,
                )
            except Exception:
                logger.warning(
                    "Session %s: Failed to measure LUFS for inst/%s",
                    session_id, stem_name, exc_info=True,
                )

    logger.info(
        "Session %s: Raw stem LUFS measured: %d vocal, %d instrumental",
        session_id, len(vocal_stem_lufs), len(inst_stem_lufs),
    )

    return vocal_stem_lufs, inst_stem_lufs


def _step_interpret_prompt(
    session_id: str,
    prompt: str,
    meta_a, meta_b,
    lyrics_a_data, lyrics_b_data,
    vocal_stem_lufs: dict[str, float],
    inst_stem_lufs: dict[str, float],
    force_vocal_source: str | None,
    event_queue, session,
) -> tuple:
    """Step 4 + 4.5: Interpret prompt via LLM, map gains, optional taste stage.

    Returns (plan, vocal_type).
    """
    from musicmixer.config import settings
    from musicmixer.models import IntentPlan
    from musicmixer.services.gain_mapper import map_intent_to_gains
    from musicmixer.services.interpreter import interpret_prompt

    logger.info("Session %s: [4/17] interpreting prompt via LLM...", session_id)
    emit_progress(event_queue, progress_event(
        "interpreting", "Your AI DJ is reading the prompt...", 0.58,
    ), session=session)

    intent_or_plan = interpret_prompt(
        prompt, meta_a, meta_b,
        lyrics_a=lyrics_a_data,
        lyrics_b=lyrics_b_data,
    )

    # Capture vocal_type before IntentPlan is converted to RemixPlan
    # (RemixPlan doesn't carry vocal_type). Default to "sung" if unavailable.
    vocal_type = "sung"
    if isinstance(intent_or_plan, IntentPlan):
        vocal_type = getattr(intent_or_plan, "vocal_type", "sung") or "sung"

    # If the LLM succeeded, we get an IntentPlan that needs gain mapping.
    # If it fell back, we already have a RemixPlan with concrete gains.
    if isinstance(intent_or_plan, IntentPlan):
        plan = map_intent_to_gains(
            intent_or_plan,
            vocal_stem_lufs=vocal_stem_lufs or None,
            inst_stem_lufs=inst_stem_lufs or None,
        )
    else:
        plan = intent_or_plan

    if force_vocal_source is not None:
        plan.vocal_source = force_vocal_source
        logger.info("Session %s: Forced vocal_source=%s", session_id, force_vocal_source)

    if plan.used_fallback:
        logger.warning(
            "Session %s: using deterministic fallback plan (LLM unavailable or failed)",
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
        total_beats = plan.sections[-1].end_beat
        from musicmixer.services.tempo import estimate_target_bpm as _est_bpm
        approx_bpm = _est_bpm(meta_a.bpm, meta_b.bpm, plan.tempo_source)
        estimated_duration = total_beats * 60 / approx_bpm if approx_bpm > 0 else 0
        logger.info(
            "Session %s: Arrangement: %d sections, %d beats, est %.0fs at %.0f BPM",
            session_id, len(plan.sections), total_beats, estimated_duration, approx_bpm,
        )

    return plan, vocal_type


def _step_load_and_standardize_stems(
    session_id: str,
    song_a_stems: dict,
    song_b_stems: dict,
    source_quality_a: str | None,
    source_quality_b: str | None,
    event_queue, session,
) -> tuple:
    """Steps 5+6: Determine vocal/instrumental sources, load and standardize stems.

    Returns (vocal_audio, inst_audio, vocal_meta_ref, inst_meta_ref,
             is_lossy_vocal_source, is_lossy_inst_source).

    Note: vocal_meta_ref and inst_meta_ref are string tags ("a"/"b") rather than
    the meta objects themselves, since the caller already has them.
    """
    import numpy as np
    from musicmixer.services.processor import validate_stem

    # Fixed convention: Song A always provides vocals, Song B always provides instrumentals.
    vocal_stems_paths = song_a_stems
    inst_stems_paths = song_b_stems

    # === Source-quality-aware processing: derive per-song lossy flags ===
    is_lossy_source_a = source_quality_a is not None and source_quality_a.startswith("youtube")
    is_lossy_source_b = source_quality_b is not None and source_quality_b.startswith("youtube")

    # Song A = vocal source, Song B = instrumental source (fixed convention)
    is_lossy_vocal_source = is_lossy_source_a
    is_lossy_inst_source = is_lossy_source_b

    emit_progress(event_queue, progress_event(
        "processing", "Getting everything on the same page...", 0.62,
    ), session=session)

    # === STEP 6: Load and standardize all stems ===
    vocal_audio: dict[str, np.ndarray] = {}
    inst_audio: dict[str, np.ndarray] = {}

    # Load vocal stems from Song A (MelBand Roformer: lead_vocals, backing_vocals).
    # Also check for legacy "vocals" key for backward compatibility with old cached stems.
    for stem_name in ["lead_vocals", "backing_vocals", "vocals"]:
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

    return vocal_audio, inst_audio, is_lossy_vocal_source, is_lossy_inst_source


def _step_trim_filter_eq(
    session_id: str,
    vocal_audio: dict,
    inst_audio: dict,
    plan,
    sr: int,
    event_queue, session,
) -> tuple[dict, dict]:
    """Steps 7 through 7.75: Trim, silence filter, bandpass, spectral analysis, corrective EQ.

    Returns updated (vocal_audio, inst_audio).
    """
    import numpy as np
    import pyloudnorm as pyln
    from musicmixer.services.processor import bandpass_filter, trim_audio
    from musicmixer.services.eq import apply_corrective_eq
    from musicmixer.services.spectral import (
        compute_adaptive_corrections,
        compute_spectral_profile,
        detect_conflicts,
    )

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
        return active

    vocal_audio = _filter_inactive(vocal_audio, "vocal")
    inst_audio = _filter_inactive(inst_audio, "instrumental")

    check_cancelled(session)

    # === STEP 7.7: Vocal pre-filter bandpass ===
    emit_progress(event_queue, progress_event(
        "processing", "Cleaning up the vocals...", 0.65,
    ), session=session)
    # Apply 150Hz-16kHz bandpass to vocal stems before tempo stretching.
    # Removes low-frequency bleed (bass rumble, kick artifacts) and
    # high-frequency separation noise, giving rubberband's R3 engine
    # cleaner input for transient detection. 16kHz preserves vocal
    # air/breathiness.
    for voc_stem in list(vocal_audio.keys()):
        vocal_audio[voc_stem] = bandpass_filter(
            vocal_audio[voc_stem], sr, low_hz=150.0, high_hz=16000.0,
        )
    if vocal_audio:
        logger.info(
            "Session %s: Vocal bandpass pre-filter applied (150Hz-16kHz) to %s",
            session_id, sorted(vocal_audio.keys()),
        )

    # === STEP 7.72: Adaptive spectral analysis ===
    # Compute spectral profiles, detect cross-stem conflicts, and generate
    # adaptive correction parameters.  Runs AFTER the vocal bandpass filter
    # so adaptive EQ only operates on post-bandpass frequencies.
    vocal_corrections: dict[str, list[tuple[float, float, float]]] = {}
    inst_corrections: dict[str, list[tuple[float, float, float]]] = {}

    try:
        _t0_spectral = time.monotonic()
        vocal_profiles = []
        for stem_type, audio in vocal_audio.items():
            try:
                vocal_profiles.append(compute_spectral_profile(audio, sr, stem_type))
            except Exception:
                logger.warning(
                    "Session %s: Spectral profile failed for vocal/%s, skipping",
                    session_id, stem_type, exc_info=True,
                )

        inst_profiles = []
        for stem_type, audio in inst_audio.items():
            try:
                inst_profiles.append(compute_spectral_profile(audio, sr, stem_type))
            except Exception:
                logger.warning(
                    "Session %s: Spectral profile failed for inst/%s, skipping",
                    session_id, stem_type, exc_info=True,
                )

        if vocal_profiles and inst_profiles:
            conflicts = detect_conflicts(vocal_profiles, inst_profiles)
            vocal_corrections, inst_corrections = compute_adaptive_corrections(
                conflicts, vocal_profiles, inst_profiles,
            )
            _elapsed_spectral = time.monotonic() - _t0_spectral
            logger.info(
                "Session %s: Adaptive EQ analysis: %d conflicts, %d vocal corrections, "
                "%d inst corrections (%.2fs)",
                session_id, len(conflicts),
                sum(len(v) for v in vocal_corrections.values()),
                sum(len(v) for v in inst_corrections.values()),
                _elapsed_spectral,
            )
        else:
            logger.info(
                "Session %s: Adaptive EQ skipped — insufficient profiles "
                "(vocal=%d, inst=%d)",
                session_id, len(vocal_profiles), len(inst_profiles),
            )
    except Exception:
        logger.error(
            "Session %s: Adaptive EQ analysis failed, falling back to preset-only",
            session_id, exc_info=True,
        )
        vocal_corrections = {}
        inst_corrections = {}

    # === STEP 7.75: Corrective EQ: adaptive-only when available, preset fallback otherwise ===
    # Apply corrective EQ per stem type. Only broad cuts/boosts (Q~1-3)
    # are safe before stretching.  When adaptive EQ produced corrections,
    # preset is skipped to avoid overcorrection; preset serves as fallback only.
    for stem_type, audio in vocal_audio.items():
        _pre = vocal_audio[stem_type]
        _adaptive = vocal_corrections.get(stem_type) or None
        _t0 = time.monotonic()
        _result = apply_corrective_eq(audio, sr, stem_type, apply_preset=(_adaptive is None), adaptive_corrections=_adaptive)
        _elapsed = time.monotonic() - _t0
        if _elapsed > DSP_STEP_TIMEOUT_S:
            logger.error(
                "Session %s: DSP step 'eq vocal/%s' exceeded %.0fs timeout (%.1fs), skipping",
                session_id, stem_type, DSP_STEP_TIMEOUT_S, _elapsed,
            )
            vocal_audio[stem_type] = _pre
        else:
            vocal_audio[stem_type] = _result
            logger.info("Session %s: EQ vocal/%s: path=%s, took %.2fs", session_id, stem_type, "adaptive" if _adaptive is not None else "preset-fallback", _elapsed)
    for stem_type, audio in inst_audio.items():
        _pre = inst_audio[stem_type]
        _adaptive = inst_corrections.get(stem_type) or None
        _t0 = time.monotonic()
        _result = apply_corrective_eq(audio, sr, stem_type, apply_preset=(_adaptive is None), adaptive_corrections=_adaptive)
        _elapsed = time.monotonic() - _t0
        if _elapsed > DSP_STEP_TIMEOUT_S:
            logger.error(
                "Session %s: DSP step 'eq inst/%s' exceeded %.0fs timeout (%.1fs), skipping",
                session_id, stem_type, DSP_STEP_TIMEOUT_S, _elapsed,
            )
            inst_audio[stem_type] = _pre
        else:
            inst_audio[stem_type] = _result
            logger.info("Session %s: EQ inst/%s: path=%s, took %.2fs", session_id, stem_type, "adaptive" if _adaptive is not None else "preset-fallback", _elapsed)

    # LUFS checkpoint: after corrective EQ
    _eq_meter = pyln.Meter(sr)
    for stem_type, audio in vocal_audio.items():
        _eq_lufs = _eq_meter.integrated_loudness(audio)
        logger.info("Session %s: LUFS after corrective EQ (vocal/%s): %.1f", session_id, stem_type, _eq_lufs)
    for stem_type, audio in inst_audio.items():
        _eq_lufs = _eq_meter.integrated_loudness(audio)
        logger.info("Session %s: LUFS after corrective EQ (inst/%s): %.1f", session_id, stem_type, _eq_lufs)

    logger.info("Session %s: Corrective EQ applied", session_id)

    return vocal_audio, inst_audio


# Fragment of the auto-handled tempo warning (text owned by processor.py). The
# minor "stretched, slight distortion" note is corrected silently and not shown;
# only the unstretchable-gap warning reaches the listener.
_AUTO_HANDLED_TEMPO_WARNING_FRAGMENT = "minor distortions"


def _add_key_match_note(plan) -> None:
    """Fold a light, positive note into the description after a successful key
    shift, instead of surfacing a key-clash warning the listener can't act on."""
    plan.explanation = (
        (plan.explanation or "").rstrip()
        + " The two tracks were tuned to a shared key for a smoother blend."
    )


def _step_compute_tempo_and_key_plan(
    session_id: str,
    meta_a, meta_b,
    plan,
    vocal_type: str,
    session: SessionState,
) -> tuple:
    """Steps 8 + 8.5: Compute tempo plan and key convergence.

    meta_a is the vocal source, meta_b is the instrumental source
    (fixed convention).

    Returns (target_bpm, need_vocal_rb, need_inst_rb,
             vocal_semitones, inst_semitones).
    """
    from musicmixer.services.processor import compute_tempo_plan
    from musicmixer.services.key_matching import compute_key_plan

    # === STEP 8: Compute tempo plan ===
    target_bpm, stretch_vocals, stretch_instrumentals, tempo_warnings, stretch_pct = compute_tempo_plan(
        meta_a.bpm, meta_b.bpm, plan.tempo_source,
    )
    plan.warnings.extend(
        w for w in tempo_warnings if _AUTO_HANDLED_TEMPO_WARNING_FRAGMENT not in w
    )
    logger.info(
        "Session %s: Target BPM=%.1f, stretch_vocals=%s, stretch_inst=%s",
        session_id, target_bpm, stretch_vocals, stretch_instrumentals,
    )

    # === STEP 8.5: Key convergence ===
    # Compute pitch shifts needed to align the keys of both songs.
    # Must run AFTER analysis (keys detected) and BEFORE rubberband (which
    # applies the pitch shifts alongside tempo stretching).
    rap_vocals = vocal_type == "rap"
    key_plan = compute_key_plan(
        meta_a.key, meta_a.scale, meta_a.key_confidence, meta_a.has_modulation,
        meta_b.key, meta_b.scale, meta_b.key_confidence, meta_b.has_modulation,
        rap_vocals=rap_vocals,
    )
    logger.info(
        "Session %s: [KEY] Song A: %s %s (conf=%.2f, mod=%s) | Song B: %s %s (conf=%.2f, mod=%s)",
        session_id,
        meta_a.key, meta_a.scale, meta_a.key_confidence or 0, meta_a.has_modulation,
        meta_b.key, meta_b.scale, meta_b.key_confidence or 0, meta_b.has_modulation,
    )
    logger.info(
        "Session %s: [KEY] Plan: action=%s, distance=%d, shift_a=%.1f st, shift_b=%.1f st, target=%s %s, reason=%s",
        session_id, key_plan.action, key_plan.distance, key_plan.shift_a, key_plan.shift_b,
        key_plan.target_key, key_plan.target_scale, key_plan.reason,
    )

    # Extract semitone shifts from key plan
    vocal_semitones = key_plan.shift_a if key_plan.action in ("shift", "warning") else 0
    inst_semitones = key_plan.shift_b if key_plan.action in ("shift", "warning") else 0

    # Handle warnings and incompatible cases
    if key_plan.action == "warning":
        session.key_warning = (
            f"Large key difference ({meta_a.key} {meta_a.scale} vs {meta_b.key} {meta_b.scale}) — "
            "the key match is pushing quality limits."
        )
    elif key_plan.action == "incompatible":
        # Zero out shifts — proceed without key matching
        session.key_warning = (
            f"Songs are too far apart in key to match "
            f"({meta_a.key} {meta_a.scale} vs {meta_b.key} {meta_b.scale}) — "
            "remix built without key matching."
        )
        vocal_semitones = 0
        inst_semitones = 0
    elif key_plan.action == "shift":
        _add_key_match_note(plan)

    # Determine which stems need rubberband processing (tempo stretch OR key shift)
    need_vocal_rb = stretch_vocals or vocal_semitones != 0
    need_inst_rb = stretch_instrumentals or inst_semitones != 0

    return target_bpm, need_vocal_rb, need_inst_rb, vocal_semitones, inst_semitones


def _step_tempo_match(
    session_id: str,
    vocal_audio: dict,
    inst_audio: dict,
    vocal_meta, inst_meta,
    target_bpm: float,
    need_vocal_rb: bool,
    need_inst_rb: bool,
    vocal_semitones: float,
    inst_semitones: float,
    sr: int,
    event_queue, session,
) -> tuple[dict, dict]:
    """Step 9: Tempo match via rubberband (parallel).

    Returns updated (vocal_audio, inst_audio).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from musicmixer.services.processor import rubberband_process

    logger.info("Session %s: [9/17] tempo matching via rubberband...", session_id)

    total_stems_to_process = (
        (len(vocal_audio) if need_vocal_rb else 0)
        + (len(inst_audio) if need_inst_rb else 0)
    )

    # Emit batch-level progress BEFORE the pool starts (avoids 5-10s silent gap in SSE stream)
    emit_progress(event_queue, progress_event(
        "processing", f"Syncing tempos ({total_stems_to_process} tracks)...", 0.68,
    ), session=session)

    with ThreadPoolExecutor(max_workers=6) as rb_executor:
        futures = {}
        if need_vocal_rb:
            # Stretch all vocal stems (lead_vocals, backing_vocals, or legacy "vocals").
            for voc_stem_name in list(vocal_audio.keys()):
                futures[("vocal", voc_stem_name)] = rb_executor.submit(
                    rubberband_process, vocal_audio[voc_stem_name], sr,
                    vocal_meta.bpm, target_bpm,
                    semitones=vocal_semitones, is_vocal=True,
                )
        if need_inst_rb:
            for stem_name in list(inst_audio.keys()):
                # Drums are exempt from pitch shifting — they're unpitched,
                # and shifting smears transients.
                # "other" stem from Song B shifts with inst_semitones (keeps
                # backing elements aligned with the instrumental group).
                stem_semitones = 0 if stem_name == "drums" else inst_semitones
                futures[("inst", stem_name)] = rb_executor.submit(
                    rubberband_process, inst_audio[stem_name], sr,
                    inst_meta.bpm, target_bpm,
                    semitones=stem_semitones,
                )
        # Dynamic timeout: 60s base + 2s per second of longest song.
        # Sized for modest hardware (e.g., i5-8500T). Generous on fast
        # machines, but timeouts should never fire under normal operation.
        max_duration = max(vocal_meta.duration_seconds, inst_meta.duration_seconds)
        rb_timeout = 60 + int(max_duration * 2)

        total = len(futures)
        if total < 3:
            # Too few stems for meaningful per-stem events; use ticker fallback
            ticker = progress_ticker(event_queue, session, 0.69, 0.88, "processing",
                                     f"Syncing tempos ({total} tracks)...", interval=5)
            try:
                for (group, stem_name), future in futures.items():
                    result = future.result(timeout=rb_timeout)
                    if group == "vocal":
                        vocal_audio[stem_name] = result
                    else:
                        inst_audio[stem_name] = result
            finally:
                ticker.set()
        else:
            # Invert the futures dict for as_completed lookup
            future_to_key = {v: k for k, v in futures.items()}
            completed_rb = 0
            for future in as_completed(futures.values(), timeout=rb_timeout):
                group, stem_name = future_to_key[future]
                result = future.result()
                if group == "vocal":
                    vocal_audio[stem_name] = result
                else:
                    inst_audio[stem_name] = result
                completed_rb += 1
                sub_progress = 0.69 + (completed_rb / total) * 0.19
                emit_progress(event_queue, progress_event(
                    "processing", f"Synced {completed_rb}/{total} tracks...", round(sub_progress, 3),
                ), session=session)

    # Emit completion progress AFTER all futures resolve
    emit_progress(event_queue, progress_event(
        "processing", "Everything's locked in!", 0.89,
    ), session=session)

    return vocal_audio, inst_audio


def _step_post_stretch_beat_grid(
    session_id: str,
    inst_audio: dict,
    inst_meta,
    target_bpm: float,
    plan,
    sr: int,
):
    """Step 10: Post-stretch beat grid re-detection.

    Returns (post_stretch_beat_frames, beat_grid_source) where
    beat_grid_source is "re-detected" or "scaled_fallback".
    """
    import librosa
    import numpy as np

    beat_grid_source = "scaled_fallback"

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
                # Validate: does the new grid cover enough of the plan's beat range?
                plan_end_beat = plan.sections[-1].end_beat if plan.sections else 0

                if plan_end_beat > 0:
                    # Estimate how many beats the new grid can reliably provide
                    # (allow up to 20% extrapolation beyond detected beats)
                    reliable_beats = len(new_beat_frames)
                    max_reliable_beat = int(reliable_beats * 1.2)

                    if plan_end_beat > max_reliable_beat:
                        logger.warning(
                            "Session %s: Re-detected beat grid (%d beats) cannot reliably "
                            "cover plan range (%d beats). Falling back to scaled grid.",
                            session_id, len(new_beat_frames), plan_end_beat,
                        )
                        # DON'T use new_beat_frames, keep the scaled original grid
                    else:
                        post_stretch_beat_frames = new_beat_frames
                        beat_grid_source = "re-detected"
                        logger.info(
                            "Session %s: Re-detected %d beats post-stretch (plan needs %d)",
                            session_id, len(new_beat_frames), plan_end_beat,
                        )
                else:
                    post_stretch_beat_frames = new_beat_frames
                    beat_grid_source = "re-detected"
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

    return post_stretch_beat_frames, beat_grid_source


def _step_compress_and_level_match(
    session_id: str,
    vocal_audio: dict,
    inst_audio: dict,
    sr: int,
    event_queue, session,
) -> tuple[dict, dict, float]:
    """Steps 11, 11.5, 11.8: Vocal compression, cross-song level matching, pre-limiting.

    Returns updated (vocal_audio, inst_audio, level_match_gain_db).
    """
    import numpy as np
    from musicmixer.services.processor import (
        compress_dynamic_range,
        cross_song_level_match,
        true_peak,
        true_peak_limit,
    )

    # === STEP 11: Compress vocal dynamic range ===
    # Compress BEFORE level matching so the LUFS measurement reflects
    # post-compression loudness (otherwise level match is wasted).
    emit_progress(event_queue, progress_event(
        "processing", "Balancing the volume...", 0.90,
    ), session=session)

    vocal_makeup_db = 3.0
    for voc_stem in list(vocal_audio.keys()):
        vocal_audio[voc_stem] = compress_dynamic_range(
            vocal_audio[voc_stem],
            sr,
            threshold_db=-20.0,  # Moderate: only compress loud phrases
            ratio=3.0,           # Standard vocal ratio, preserves natural dynamics
            attack_ms=10.0,      # Preserve plosive transients
            release_ms=80.0,     # Fast release: recover quickly between phrases
            makeup_db=vocal_makeup_db,
            gate_floor_db=-50.0, # Low gate: only ignore true silence
        )
    if vocal_audio:
        logger.info(
            "Session %s: Vocal compression applied (makeup_db=%.1f) to %s",
            session_id, vocal_makeup_db, sorted(vocal_audio.keys()),
        )

    # === STEP 11.5: Cross-song level matching ===
    # Runs AFTER compression so LUFS measurement reflects actual vocal loudness.
    import pyloudnorm as _pyln_lm
    _lm_meter = _pyln_lm.Meter(sr)
    _pre_lm_lufs = None
    _level_match_gain_db = 0.0

    # Use the primary vocal stem for level matching: lead_vocals preferred, fall back to vocals.
    _primary_vocal_key = None
    for _vk in ("lead_vocals", "vocals"):
        if _vk in vocal_audio:
            _primary_vocal_key = _vk
            break

    if _primary_vocal_key is not None and inst_audio:
        vocal_audio_main = vocal_audio[_primary_vocal_key]
        _pre_lm_lufs = _lm_meter.integrated_loudness(vocal_audio_main)

        # Sum instrumental stems for LUFS measurement
        inst_arrays = list(inst_audio.values())
        inst_sum_for_lufs = inst_arrays[0].copy()
        for arr in inst_arrays[1:]:
            min_len = min(len(inst_sum_for_lufs), len(arr))
            inst_sum_for_lufs = inst_sum_for_lufs[:min_len] + arr[:min_len]

        # Apply cross-song level match to ALL vocal stems using the same gain
        # derived from the primary vocal stem's LUFS vs instrumentals.
        for _vk_apply in list(vocal_audio.keys()):
            vocal_audio[_vk_apply] = cross_song_level_match(
                vocal_audio[_vk_apply], inst_sum_for_lufs, sr,
            )

        _post_lm_lufs = _lm_meter.integrated_loudness(vocal_audio[_primary_vocal_key])
        _level_match_gain_db = _post_lm_lufs - _pre_lm_lufs if _pre_lm_lufs > -70 else 0.0

    # === STEP 11.8: Pre-limit drum and bass transients ===
    emit_progress(event_queue, progress_event(
        "processing", "Smoothing out the drum hits...", 0.92,
    ), session=session)
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

    check_cancelled(session)

    return vocal_audio, inst_audio, _level_match_gain_db


def _step_render_and_duck(
    session_id: str,
    plan,
    vocal_audio: dict,
    inst_audio: dict,
    post_stretch_beat_frames,
    sr: int,
    target_bpm: float,
    event_queue, session,
) -> tuple:
    """Steps 12 + 12.5: Render arrangement into buses and apply spectral ducking.

    Returns (vocal_bus, instrumental_bus, ducked_instrumental).
    """
    from musicmixer.services.renderer import render_arrangement
    from musicmixer.services.ducking import spectral_duck
    from musicmixer.services.interpreter import TARGET_REMIX_DURATION_SECONDS

    logger.info("Session %s: [12/17] rendering arrangement...", session_id)
    emit_progress(event_queue, progress_event(
        "rendering", "Stitching the pieces together...", 0.93,
    ), session=session)

    vocal_bus, instrumental_bus = render_arrangement(
        sections=plan.sections,
        vocal_stems=vocal_audio,
        instrumental_stems=inst_audio,
        beat_frames=post_stretch_beat_frames,
        sr=sr,
        target_bpm=target_bpm,
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
    emit_progress(event_queue, progress_event(
        "rendering", "Making room so nothing clashes...", 0.94,
    ), session=session)

    ducked_instrumental = spectral_duck(instrumental_bus, vocal_bus, sr)

    return vocal_bus, instrumental_bus, ducked_instrumental


def _step_sum_and_auto_level(
    session_id: str,
    vocal_bus,
    instrumental_bus,
    ducked_instrumental,
    sr: int,
):
    """Steps 13 + 13.7: Sum buses into final mix and apply auto-leveler.

    Returns mixed array.
    """
    import numpy as np
    import pyloudnorm as pyln
    from musicmixer.services.processor import auto_level

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
    auto_level_kwargs: dict[str, Any] = dict(window_sec=4.0, max_boost_db=1.5, max_cut_db=2.5)

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

    return mixed


def _step_master(
    session_id: str,
    mixed,
    sr: int,
    is_lossy_vocal_source: bool,
    is_lossy_inst_source: bool,
    event_queue, session,
):
    """Steps 14, 14.5, 14.6: Static mastering, LUFS correction, safety soft clip.

    Returns mastered mixed array.
    """
    import numpy as np
    import pyloudnorm as pyln
    from musicmixer.services.mastering import master_static
    from musicmixer.services.processor import soft_clip, true_peak_limit

    emit_progress(event_queue, progress_event(
        "rendering", "Final polish...", 0.95,
    ), session=session)
    # === STEP 14: Static mastering chain ===
    # Constrained LUFS normalize (-12 LUFS) then limiter (-1.0 dBTP).
    master_kwargs: dict = dict(target_lufs=-12.0, ceiling_dbtp=-1.0)
    if is_lossy_vocal_source or is_lossy_inst_source:
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
    _meter = pyln.Meter(sr)
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

    return mixed


def _step_fades(
    session_id: str,
    mixed,
    sr: int,
    plan,
    event_queue, session,
):
    """Step 15: Fade-in / fade-out.

    Returns mixed array with fades applied.
    """
    from musicmixer.services.processor import apply_fades

    emit_progress(event_queue, progress_event(
        "rendering", "Adding smooth transitions...", 0.97,
    ), session=session)
    skip_fade_in = plan.sections[0].transition_in == "fade" if plan.sections else False
    # Renderer no longer applies a terminal fade-out; keep one authoritative
    # final fade stage here in the pipeline for every remix.
    skip_fade_out = False
    mixed = apply_fades(mixed, sr, skip_fade_in=skip_fade_in, skip_fade_out=skip_fade_out)

    return mixed


def _step_export_and_finalize(
    session_id: str,
    mixed,
    sr: int,
    output_path,
    plan,
    remix_cache_key: str | None,
    session: SessionState,
    event_queue: queue.Queue,
):
    """Step 16: Export MP3, write cache, update session, emit complete event."""
    from musicmixer.config import settings
    from musicmixer.services.processor import export_mp3

    logger.info("Session %s: [16/16] exporting MP3...", session_id)
    emit_progress(event_queue, progress_event(
        "rendering", "Bouncing your remix...", 0.98,
    ), session=session)

    export_mp3(mixed, sr, output_path, use_s16_dither=False)

    # === REMIX CACHE WRITE ===
    if settings.remix_cache_enabled and remix_cache_key is not None:
        try:
            from musicmixer.services.remix_cache import cache_remix, write_url_alias

            cache_remix(remix_cache_key, output_path, settings.remix_cache_dir, metadata={
                "explanation": plan.explanation,
                "warnings": plan.warnings,
                "used_fallback": plan.used_fallback,
                "key_warning": session.key_warning,
            })

            # Write URL-based alias so future requests with the same URLs skip the queue
            url_key = getattr(session, "url_cache_key", None)
            if url_key:
                write_url_alias(url_key, remix_cache_key, settings.remix_cache_dir)
        except Exception:
            logger.warning(
                "Session %s: Remix cache write failed (non-fatal)",
                session_id, exc_info=True,
            )

    # === DONE ===
    session.remix_path = str(output_path)
    session.explanation = plan.explanation

    # Copy plan results to session (needed by /public endpoint)
    session.used_fallback = plan.used_fallback
    session.warnings = plan.warnings

    # Capture phone BEFORE setting status to "complete" — once complete,
    # cleanup considers the session eligible for purging.
    phone = session.notify_phone
    session.notify_phone = None

    session.status = "complete"

    complete_event = {
        "step": "complete",
        "detail": "Your remix is ready! 🎧",
        "progress": 1.0,
        "explanation": plan.explanation,
        "warnings": plan.warnings,
        "usedFallback": plan.used_fallback,
    }
    if session.key_warning:
        complete_event["keyWarning"] = session.key_warning

    emit_progress(event_queue, complete_event, session=session)

    # Send SMS notification if a phone was registered
    if phone:
        try:
            from musicmixer.services.sms import send_remix_ready

            send_remix_ready(phone, session_id)
        except Exception:
            logger.exception("Session %s: SMS notification failed", session_id)

    logger.info("Session %s: Pipeline complete. Output: %s", session_id, output_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def analyze_songs(
    session_id: str,
    song_a_path: str,
    song_b_path: str,
    event_queue: queue.Queue | None = None,
    session: SessionState | None = None,
    song_a_original_filename: str = "",
    song_b_original_filename: str = "",
    source_quality_a: str | None = None,
    source_quality_b: str | None = None,
    metrics: PipelineMetrics | None = None,
    cached_meta_a: AudioMetadata | None = None,
    cached_meta_b: AudioMetadata | None = None,
    cached_lyrics_a: LyricsData | None = None,
    cached_lyrics_b: LyricsData | None = None,
) -> AnalyzedSongs:
    """Analysis phase (steps 1-3.8): separation, audio analysis, structure detection.

    When cached metadata is provided for a song (medium cache path), analysis,
    lyrics lookup, and ML structure detection are skipped for that song.
    Separation and LUFS measurement always run.

    Returns an AnalyzedSongs with everything the remix phase needs.
    """
    assert event_queue is not None
    assert session is not None

    from pathlib import Path
    from musicmixer.config import settings

    _song_a = Path(song_a_path)
    _song_b = Path(song_b_path)
    stems_dir = settings.data_dir / "stems" / session_id

    _t_analysis_start = time.monotonic()
    _step_times: dict[str, float] = {}

    # === STEPS 1+2: Separation + analysis (overlapped) ===
    _t0 = time.monotonic()
    (
        song_a_stems, song_b_stems, meta_a, meta_b,
        lyrics_a_data, lyrics_b_data, ml_segments_a, ml_segments_b,
        song_a_stems_dir, song_b_stems_dir,
    ) = _step_separate_and_analyze(
        session_id, _song_a, _song_b, stems_dir,
        song_a_original_filename, song_b_original_filename,
        event_queue, session,
        cached_meta_a=cached_meta_a,
        cached_meta_b=cached_meta_b,
        cached_lyrics_a=cached_lyrics_a,
        cached_lyrics_b=cached_lyrics_b,
    )
    _step_times["1+2 separate+analyze"] = time.monotonic() - _t0

    check_cancelled(session)

    emit_progress(event_queue, progress_event(
        "analyzing", "Figuring out the BPM and musical key...", 0.40,
    ), session=session)

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

    # --- Structured metrics: input + separation + analysis ---
    if metrics is not None:
        metrics.song_a_title = song_a_original_filename
        metrics.song_b_title = song_b_original_filename
        metrics.stem_backend = settings.stem_backend
        metrics.stem_count = len(song_a_stems)
        metrics.separation_time_a_s = _step_times.get("1+2 separate+analyze", 0.0)
        metrics.log_input()

        metrics.log_separation()

        metrics.bpm_a = meta_a.bpm
        metrics.bpm_b = meta_b.bpm
        metrics.key_a = meta_a.key or ""
        metrics.scale_a = meta_a.scale or ""
        metrics.key_b = meta_b.key or ""
        metrics.scale_b = meta_b.scale or ""
        metrics.key_confidence_a = meta_a.key_confidence or 0.0
        metrics.key_confidence_b = meta_b.key_confidence or 0.0
        metrics.duration_a_s = meta_a.duration_seconds
        metrics.duration_b_s = meta_b.duration_seconds
        metrics.log_analysis()

    # === STEP 3: Reconcile BPM ===
    _t0 = time.monotonic()
    meta_a, meta_b = _step_reconcile_bpm(session_id, meta_a, meta_b, event_queue, session)
    _step_times["3 reconcile_bpm"] = time.monotonic() - _t0

    # === STEP 3.5: Analyze song structure ===
    _t0 = time.monotonic()
    _step_analyze_structure(
        session_id, meta_a, meta_b,
        song_a_stems_dir, song_b_stems_dir,
        ml_segments_a, ml_segments_b,
        event_queue, session,
        song_a_path=_song_a,
        song_b_path=_song_b,
        lyrics_a_data=lyrics_a_data,
        has_cached_meta_a=cached_meta_a is not None,
        has_cached_meta_b=cached_meta_b is not None,
    )
    _step_times["3.5 structure+pulsemap"] = time.monotonic() - _t0

    # === STEP 3.8: Measure per-stem LUFS ===
    _t0 = time.monotonic()
    vocal_stem_lufs, inst_stem_lufs = _step_measure_stem_lufs(
        session_id, song_a_stems_dir, song_b_stems_dir,
    )
    _step_times["3.8 stem_lufs"] = time.monotonic() - _t0

    check_cancelled(session)

    _analysis_total = time.monotonic() - _t_analysis_start
    _timing_lines = " | ".join(f"{k}={v:.1f}s" for k, v in _step_times.items())
    logger.info(
        "Session %s: ANALYSIS TIMING (%.1fs total): %s",
        session_id, _analysis_total, _timing_lines,
    )

    # Attach timing to session for the remix phase to merge
    if session is not None:
        session._analysis_step_times = _step_times  # type: ignore[attr-defined]

    return AnalyzedSongs(
        meta_a=meta_a,
        meta_b=meta_b,
        song_a_stems=song_a_stems,
        song_b_stems=song_b_stems,
        song_a_stems_dir=song_a_stems_dir,
        song_b_stems_dir=song_b_stems_dir,
        lyrics_a=lyrics_a_data,
        lyrics_b=lyrics_b_data,
        vocal_stem_lufs=vocal_stem_lufs,
        inst_stem_lufs=inst_stem_lufs,
    )


def run_remix(
    session_id: str,
    analysis: AnalyzedSongs,
    prompt: str = "",
    event_queue: queue.Queue | None = None,
    session: SessionState | None = None,
    source_quality_a: str | None = None,
    source_quality_b: str | None = None,
    force_vocal_source: str | None = None,
    remix_cache_key: str | None = None,
    metrics: PipelineMetrics | None = None,
) -> None:
    """Remix phase (steps 4-16): LLM planning, DSP processing, export.

    Takes pre-analyzed song data and a prompt, produces a finished MP3.
    """
    assert event_queue is not None
    assert session is not None

    from musicmixer.config import settings

    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    meta_a = analysis.meta_a
    meta_b = analysis.meta_b

    _t_remix_start = time.monotonic()
    _step_times: dict[str, float] = {}

    # Merge analysis timing if available
    if hasattr(session, "_analysis_step_times"):
        _step_times.update(session._analysis_step_times)  # type: ignore[attr-defined]

    # === STEP 4 + 4.5: Interpret prompt + taste stage ===
    _t0 = time.monotonic()
    plan, vocal_type = _step_interpret_prompt(
        session_id, prompt, meta_a, meta_b,
        analysis.lyrics_a, analysis.lyrics_b,
        analysis.vocal_stem_lufs, analysis.inst_stem_lufs,
        force_vocal_source,
        event_queue, session,
    )
    _step_times["4 llm_interpret"] = time.monotonic() - _t0

    # --- Structured metrics: LLM plan ---
    if metrics is not None:
        metrics.start_time_vocal = plan.start_time_vocal
        metrics.start_time_instrumental = plan.start_time_instrumental
        metrics.section_count = len(plan.sections)
        metrics.vocal_type = vocal_type
        metrics.llm_plan_failed = plan.used_fallback

        # Build sections summary (e.g., "intro:0-8, verse:8-40, ...")
        if plan.sections:
            metrics.sections_summary = ", ".join(
                f"{s.label}:{s.start_beat}-{s.end_beat}" for s in plan.sections
            )
            # intro_beats: beats in the first section if it's labeled "intro"
            first = plan.sections[0]
            if first.label.lower() == "intro":
                metrics.intro_beats = first.end_beat - first.start_beat

        metrics.log_llm_plan()

    # === STEPS 5+6: Load and standardize stems ===
    _t0 = time.monotonic()
    vocal_audio, inst_audio, is_lossy_vocal_source, is_lossy_inst_source = (
        _step_load_and_standardize_stems(
            session_id, analysis.song_a_stems, analysis.song_b_stems,
            source_quality_a, source_quality_b,
            event_queue, session,
        )
    )
    _step_times["5+6 load_stems"] = time.monotonic() - _t0

    sr = 44100  # All stems standardized to this by validate_stem

    # === STEPS 7-7.75: Trim, filter, EQ ===
    _t0 = time.monotonic()
    vocal_audio, inst_audio = _step_trim_filter_eq(
        session_id, vocal_audio, inst_audio, plan, sr,
        event_queue, session,
    )
    _step_times["7 trim_filter_eq"] = time.monotonic() - _t0

    # --- Structured metrics: stem health ---
    if metrics is not None:
        metrics.stems_active = sorted(list(vocal_audio.keys()) + list(inst_audio.keys()))
        # Determine inactive stems by comparing against full set
        all_possible = {"vocals", "lead_vocals", "backing_vocals", "drums", "bass", "guitar", "piano", "other"}
        active_set = set(vocal_audio.keys()) | set(inst_audio.keys())
        metrics.stems_inactive = sorted(all_possible - active_set)

        # Measure per-stem LUFS after trim/filter/EQ
        import pyloudnorm as _pyln
        _health_meter = _pyln.Meter(sr)
        for stem_name, audio in vocal_audio.items():
            try:
                metrics.per_stem_lufs[f"vocal/{stem_name}"] = _health_meter.integrated_loudness(audio)
            except Exception:
                pass
        for stem_name, audio in inst_audio.items():
            try:
                metrics.per_stem_lufs[f"inst/{stem_name}"] = _health_meter.integrated_loudness(audio)
            except Exception:
                pass

        # Check for ghost stems before they get filtered
        metrics.check_ghost_stems(inst_audio)
        metrics.log_stem_health()

    # === STEPS 8+8.5: Tempo plan + key convergence ===
    _t0 = time.monotonic()
    target_bpm, need_vocal_rb, need_inst_rb, vocal_semitones, inst_semitones = (
        _step_compute_tempo_and_key_plan(
            session_id, meta_a, meta_b,
            plan, vocal_type, session,
        )
    )
    _step_times["8 tempo_key_plan"] = time.monotonic() - _t0

    # --- Structured metrics: tempo/key ---
    if metrics is not None:
        metrics.target_bpm = target_bpm
        # Compute max stretch % across both songs
        vocal_stretch = abs(meta_a.bpm - target_bpm) / meta_a.bpm * 100 if meta_a.bpm > 0 else 0
        inst_stretch = abs(meta_b.bpm - target_bpm) / meta_b.bpm * 100 if meta_b.bpm > 0 else 0
        metrics.stretch_pct = max(vocal_stretch, inst_stretch)
        metrics.vocal_semitones = vocal_semitones
        metrics.inst_semitones = inst_semitones
        metrics.tempo_key_warnings = list(plan.warnings)
        metrics.log_tempo_key()

    # === STEP 9: Tempo match via rubberband ===
    _t0 = time.monotonic()
    vocal_audio, inst_audio = _step_tempo_match(
        session_id, vocal_audio, inst_audio,
        meta_a, meta_b,
        target_bpm, need_vocal_rb, need_inst_rb,
        vocal_semitones, inst_semitones,
        sr, event_queue, session,
    )
    _step_times["9 tempo_match"] = time.monotonic() - _t0

    # === STEP 10: Post-stretch beat grid ===
    _t0 = time.monotonic()
    post_stretch_beat_frames, beat_grid_source = _step_post_stretch_beat_grid(
        session_id, inst_audio, meta_b, target_bpm, plan, sr,
    )
    _step_times["10 beat_grid"] = time.monotonic() - _t0

    # --- Structured metrics: beat grid ---
    if metrics is not None:
        metrics.beat_grid_source = beat_grid_source
        metrics.post_stretch_beat_count = len(post_stretch_beat_frames)
        metrics.log_beat_grid()

    # === STEPS 11-11.8: Compress, level match, pre-limit ===
    _t0 = time.monotonic()
    vocal_audio, inst_audio, level_match_gain_db = _step_compress_and_level_match(
        session_id, vocal_audio, inst_audio, sr,
        event_queue, session,
    )
    _step_times["11 compress_level"] = time.monotonic() - _t0

    # --- Structured metrics: processing ---
    if metrics is not None:
        metrics.level_match_gain_db = level_match_gain_db
        metrics.log_processing()

    # === STEPS 12+12.5: Render arrangement + spectral ducking ===
    _t0 = time.monotonic()
    vocal_bus, instrumental_bus, ducked_instrumental = _step_render_and_duck(
        session_id, plan, vocal_audio, inst_audio,
        post_stretch_beat_frames, sr, target_bpm,
        event_queue, session,
    )
    _step_times["12 render_duck"] = time.monotonic() - _t0

    # --- Structured metrics: render ---
    if metrics is not None:
        metrics.render_duration_s = vocal_bus.shape[0] / sr
        # Check duration mismatch
        if plan.sections and target_bpm > 0:
            estimated_duration = plan.sections[-1].end_beat * 60 / target_bpm
            metrics.check_duration_mismatch(estimated_duration)

    # === STEPS 13+13.7: Sum buses + auto-leveler ===
    _t0 = time.monotonic()
    mixed = _step_sum_and_auto_level(
        session_id, vocal_bus, instrumental_bus, ducked_instrumental, sr,
    )
    _step_times["13 sum_autolevel"] = time.monotonic() - _t0

    # === STEPS 14-14.6: Mastering chain ===
    _t0 = time.monotonic()
    mixed = _step_master(
        session_id, mixed, sr,
        is_lossy_vocal_source, is_lossy_inst_source,
        event_queue, session,
    )
    _step_times["14 master"] = time.monotonic() - _t0

    # === STEP 15: Fades ===
    _t0 = time.monotonic()
    mixed = _step_fades(session_id, mixed, sr, plan, event_queue, session)
    _step_times["15 fades"] = time.monotonic() - _t0

    check_cancelled(session)

    # === STEP 16: Export + finalize ===
    _t0 = time.monotonic()
    _step_export_and_finalize(
        session_id, mixed, sr, output_path, plan,
        remix_cache_key, session, event_queue,
    )
    _step_times["16 export"] = time.monotonic() - _t0

    # === TIMING SUMMARY ===
    _remix_total = time.monotonic() - _t_remix_start
    _sorted = sorted(_step_times.items(), key=lambda kv: kv[1], reverse=True)
    _timing_lines = " | ".join(f"{k}={v:.1f}s" for k, v in _sorted)
    logger.info(
        "Session %s: PIPELINE TIMING (remix=%.1fs): %s",
        session_id, _remix_total, _timing_lines,
    )

    # --- Structured metrics: output + completion summary ---
    if metrics is not None:
        import pyloudnorm as _pyln_out

        # Final LUFS
        _out_meter = _pyln_out.Meter(sr)
        try:
            metrics.final_lufs = _out_meter.integrated_loudness(mixed)
        except Exception:
            pass

        # Output file size
        if output_path.exists():
            metrics.output_size_mb = output_path.stat().st_size / (1024 * 1024)

        # Collect all per-step times and warnings
        metrics.per_step_times = dict(_step_times)
        metrics.warnings = list(plan.warnings)
        metrics.log_render()
        metrics.log_output()
        metrics.log_completion()


def run_pipeline(
    session_id: str,
    song_a_path: str | Path,
    song_b_path: str | Path,
    prompt: str = "",
    event_queue: queue.Queue | None = None,
    session: SessionState | None = None,
    song_a_original_filename: str = "",
    song_b_original_filename: str = "",
    source_quality_a: str | None = None,
    source_quality_b: str | None = None,
    force_vocal_source: str | None = None,
) -> None:
    """Complete remix pipeline: analyze songs, then remix.

    Convenience wrapper that calls analyze_songs() then run_remix().
    Used by the file upload path. The YouTube path calls them separately.
    """
    assert event_queue is not None
    assert session is not None

    from pathlib import Path
    from musicmixer.config import settings

    _song_a = Path(song_a_path)
    _song_b = Path(song_b_path)
    remix_dir = settings.data_dir / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)
    output_path = remix_dir / "remix.mp3"

    logger.info("Session %s: pipeline started (prompt=%r)", session_id, prompt[:80])

    # === REMIX CACHE CHECK ===
    remix_cache_key = _check_remix_cache(
        session_id, _song_a, _song_b, prompt,
        output_path, session, event_queue,
    )
    if session.status == "complete":
        return

    # Create pipeline metrics tracker for structured logging
    metrics = PipelineMetrics(session_id=session_id)

    # === ANALYSIS PHASE (steps 1-3.8) ===
    analysis = analyze_songs(
        session_id=session_id,
        song_a_path=str(_song_a),
        song_b_path=str(_song_b),
        event_queue=event_queue,
        session=session,
        song_a_original_filename=song_a_original_filename,
        song_b_original_filename=song_b_original_filename,
        source_quality_a=source_quality_a,
        source_quality_b=source_quality_b,
        metrics=metrics,
    )

    # === REMIX PHASE (steps 4-16) ===
    run_remix(
        session_id=session_id,
        analysis=analysis,
        prompt=prompt,
        event_queue=event_queue,
        session=session,
        source_quality_a=source_quality_a,
        source_quality_b=source_quality_b,
        force_vocal_source=force_vocal_source,
        remix_cache_key=remix_cache_key,
        metrics=metrics,
    )
