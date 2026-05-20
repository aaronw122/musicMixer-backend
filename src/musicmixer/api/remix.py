"""Remix API endpoints.

Day 2: Async pipeline with SSE progress events.
- POST /api/remix          -- Accept uploads, start pipeline in background, return session_id
- POST /api/remix/youtube  -- Accept YouTube URLs, download + pipeline in background
- GET  /api/remix/{id}/progress -- SSE stream of pipeline progress events
- GET  /api/remix/{id}/status   -- JSON snapshot of current session state
- GET  /api/remix/{id}/audio    -- Serve the rendered remix MP3

Day 3: Admin/debug stem endpoints.
- GET  /api/remix/{id}/stems                    -- List available stems for both songs
- GET  /api/remix/{id}/stems/{song}/{stem_name} -- Serve raw WAV stem file
"""

import asyncio
import dataclasses
import functools
import json
import logging
import queue
import shutil
import subprocess
import threading
import time
import urllib.parse
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from musicmixer.config import settings
from musicmixer.models import CachedSong, SessionState
from musicmixer.services.cleanup import cleanup_expired_sessions
from musicmixer.api.shelf import ensure_on_shelf

if TYPE_CHECKING:
    from musicmixer.services.youtube import YouTubeAudioResult

logger = logging.getLogger(__name__)
router = APIRouter()

# Average remix duration in seconds, used for queue wait estimates.
# Updated after each completed remix for better accuracy.
_AVG_REMIX_DURATION_S = 600.0  # initial estimate: 10 minutes
_avg_lock = threading.Lock()


def _probe_duration(file_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


def _update_avg_remix_duration(elapsed_seconds: float) -> None:
    """Update the running average remix duration with exponential smoothing."""
    global _AVG_REMIX_DURATION_S
    with _avg_lock:
        # Exponential moving average (alpha=0.3 gives recent runs more weight)
        _AVG_REMIX_DURATION_S = 0.7 * _AVG_REMIX_DURATION_S + 0.3 * elapsed_seconds


def _pre_trim_youtube_download(
    download_result: "YouTubeAudioResult | None",
    max_duration_seconds: int,
) -> None:
    """Trim long YouTube downloads before GPU-heavy processing."""
    if download_result is None:
        return

    if download_result.duration_seconds <= max_duration_seconds:
        return

    from musicmixer.services.processor import pre_trim_for_processing

    pre_trim_for_processing(
        download_result.wav_path,
        max_duration_seconds=max_duration_seconds,
    )
    download_result.duration_seconds = min(
        download_result.duration_seconds,
        max_duration_seconds,
    )


# ---------------------------------------------------------------------------
# Queue work item — what gets enqueued when all slots are busy
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _QueueItem:
    """A queued remix request waiting for a processing slot."""
    session_id: str
    session: SessionState
    run_fn: Callable[[], None]  # callable that runs the pipeline (already bound)
    enqueued_at: float = dataclasses.field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

def _pipeline_wrapper(
    session_id: str,
    song_a_path: Path,
    song_b_path: Path,
    prompt: str,
    session: SessionState,
    processing_lock,
    app_state,
    song_a_original_filename: str = "",
    song_b_original_filename: str = "",
) -> None:
    """Runs the pipeline in a background thread.

    Ownership note: after successful executor.submit(), this wrapper is the sole
    owner of the acquired processing slot and MUST release it on exit.
    """
    from musicmixer.services.pipeline import CancelledError, run_pipeline

    pipeline_start = time.monotonic()
    try:
        session.status = "processing"

        run_pipeline(
            session_id=session_id,
            song_a_path=str(song_a_path),
            song_b_path=str(song_b_path),
            prompt=prompt,
            event_queue=session.events,
            session=session,
            song_a_original_filename=song_a_original_filename,
            song_b_original_filename=song_b_original_filename,
        )
        _update_avg_remix_duration(time.monotonic() - pipeline_start)
    except CancelledError:
        logger.info("Session %s: pipeline cancelled by user", session_id)
        session.status = "cancelled"
        from musicmixer.services.pipeline import emit_progress

        emit_progress(session.events, {
            "step": "cancelled",
            "detail": "Remix cancelled",
            "progress": 0,
        }, session=session)
    except BaseException as exc:
        logger.exception("Session %s: pipeline failed", session_id)
        session.status = "error"
        from musicmixer.services.pipeline import emit_progress

        emit_progress(session.events, {
            "step": "error",
            "detail": str(exc),
            "progress": 0,
        }, session=session)
    finally:
        processing_lock.release()
        _process_next_queued(app_state)


# ---------------------------------------------------------------------------
# YouTube URL validation (SSRF prevention)
# ---------------------------------------------------------------------------

_YOUTUBE_ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
}


def _thumbnail_from_youtube_url(url: str) -> str | None:
    """Derive a YouTube thumbnail URL from a video URL. Returns None on failure."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path_parts = [p for p in parsed.path.split("/") if p]

    video_id = None
    if hostname == "youtu.be" and path_parts:
        video_id = path_parts[0]
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        if qs.get("v"):
            video_id = qs["v"][0]
        elif path_parts and path_parts[0] in {"shorts", "embed"} and len(path_parts) > 1:
            video_id = path_parts[1]

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None


def _validate_youtube_url(url: str) -> str:
    """Validate a YouTube URL for SSRF safety. Returns the validated URL.

    Validation order matters (step 3 before step 5 prevents userinfo bypass).

    Raises HTTPException(422) on invalid URLs.
    """
    # 1. Parse URL
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise HTTPException(422, "Invalid URL format")

    # 2. Reject non-https schemes (allow http, upgrade mentally but don't rewrite)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(422, "Only HTTP/HTTPS YouTube URLs are accepted")

    # 3. Reject URLs containing @ in netloc (prevents userinfo bypass like youtube.com@evil.com)
    if "@" in (parsed.netloc or ""):
        raise HTTPException(422, "Invalid URL — only YouTube links are accepted")

    # 4. Reject IP literals and non-standard ports
    hostname = parsed.hostname
    if hostname is None:
        raise HTTPException(422, "Invalid URL — missing hostname")

    if parsed.port is not None and parsed.port not in (80, 443):
        raise HTTPException(422, "Invalid URL — only YouTube links are accepted")

    # Check for IP literals (v4 and v6)
    # Simple check: if hostname starts with digit or contains ':', it's likely an IP
    if hostname[0].isdigit() or ":" in hostname:
        raise HTTPException(422, "Invalid URL — only YouTube links are accepted")

    # 5. Validate hostname against allowlist
    if hostname not in _YOUTUBE_ALLOWED_HOSTS:
        raise HTTPException(422, "Invalid URL — only YouTube links are accepted")

    return url


# ---------------------------------------------------------------------------
# YouTube remix request model
# ---------------------------------------------------------------------------

def _resolve_shelf_song_id(url: str) -> str | None:
    """Resolve a YouTube URL to a shelf song ID if it matches a shelf record.

    Uses _extract_video_id for matching — does NOT raise on invalid URLs.
    Returns the shelf record's ID if matched, else None.
    """
    from musicmixer.api.shelf import _extract_video_id, _shelf_path, _read_shelf_unlocked, _shelf_lock

    video_id = _extract_video_id(url)
    if not video_id:
        return None

    try:
        with _shelf_lock:
            records = _read_shelf_unlocked(_shelf_path())
    except Exception:
        return None

    for record in records:
        record_video_id = _extract_video_id(record["youtube_url"])
        if record_video_id == video_id:
            return record["id"]

    return None


class YouTubeRemixRequest(BaseModel):
    url_a: str  # YouTube URL for song A
    url_b: str  # YouTube URL for song B
    prompt: str = ""  # Remix prompt (optional — defaults to deterministic plan)


# ---------------------------------------------------------------------------
# YouTube pipeline wrapper (download + existing pipeline)
# ---------------------------------------------------------------------------

def _youtube_pipeline_wrapper(
    session_id: str,
    url_a: str,
    url_b: str,
    prompt: str,
    session: SessionState,
    processing_lock,
    app_state,
    shelf_song_id_a: str | None = None,
    shelf_song_id_b: str | None = None,
    cached_song_a: CachedSong | None = None,
    cached_song_b: CachedSong | None = None,
) -> None:
    """Downloads YouTube audio in parallel, then runs the pipeline.

    Both downloads are independent so we use a 2-thread pool to overlap them.
    A lock-guarded monotonic progress tracker prevents progress values from
    jumping backward when the two callbacks interleave.

    Ownership note: after successful executor.submit(), this wrapper is the sole
    owner of the acquired processing slot and MUST release it on exit.
    """
    pipeline_start = time.monotonic()
    from musicmixer.services.pipeline import (
        CancelledError, analyze_songs, emit_progress, run_remix, run_pipeline,
    )
    from musicmixer.services.pipeline_metrics import PipelineMetrics

    try:
        session.status = "processing"
        from musicmixer.services.youtube import download_youtube_audio

        upload_dir = settings.data_dir / "uploads" / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # --- Check if both songs are fully cached (stems + metadata) ---
        both_fully_cached = (
            cached_song_a is not None and cached_song_a.has_stems
            and cached_song_b is not None and cached_song_b.has_stems
        )

        if both_fully_cached:
            # Skip downloads + analysis — jump straight to remix with cached data
            logger.info("Session %s: Both songs fully cached, skipping downloads + analysis", session_id)
            emit_progress(session.events, {
                "step": "downloading",
                "detail": "Both songs cached — skipping download!",
                "progress": 0.10,
            }, session=session)

            from musicmixer.services.pipeline import check_cancelled, _step_measure_stem_lufs
            from musicmixer.services.song_cache import get_cached_stems
            from musicmixer.models import AnalyzedSongs

            check_cancelled(session)

            assert cached_song_a is not None
            assert cached_song_b is not None

            # Copy cached stems to session directory
            stems_dir = settings.data_dir / "stems" / session_id
            song_a_stems_dir = stems_dir / "song_a"
            song_b_stems_dir = stems_dir / "song_b"
            stems_restored = (
                get_cached_stems(cached_song_a.video_id, song_a_stems_dir)
                and get_cached_stems(cached_song_b.video_id, song_b_stems_dir)
            )

            if stems_restored:
                # Build stem path dicts
                stem_names = ["vocals", "drums", "bass", "guitar", "piano", "other"]
                song_a_stems = {n: song_a_stems_dir / f"{n}.wav" for n in stem_names if (song_a_stems_dir / f"{n}.wav").exists()}
                song_b_stems = {n: song_b_stems_dir / f"{n}.wav" for n in stem_names if (song_b_stems_dir / f"{n}.wav").exists()}

                # Measure LUFS from restored stems
                vocal_stem_lufs, inst_stem_lufs = _step_measure_stem_lufs(
                    session_id, song_a_stems_dir, song_b_stems_dir,
                )

                analysis = AnalyzedSongs(
                    meta_a=cached_song_a.meta,
                    meta_b=cached_song_b.meta,
                    song_a_stems=song_a_stems,
                    song_b_stems=song_b_stems,
                    song_a_stems_dir=song_a_stems_dir,
                    song_b_stems_dir=song_b_stems_dir,
                    lyrics_a=cached_song_a.lyrics,
                    lyrics_b=cached_song_b.lyrics,
                    vocal_stem_lufs=vocal_stem_lufs,
                    inst_stem_lufs=inst_stem_lufs,
                )

                _metrics = PipelineMetrics(session_id=session_id)
                _metrics.song_a_title = cached_song_a.title
                _metrics.song_b_title = cached_song_b.title
                _metrics.song_a_video_id = cached_song_a.video_id
                _metrics.song_b_video_id = cached_song_b.video_id
                _metrics.song_a_cache_hit = True
                _metrics.song_b_cache_hit = True
                _metrics.bpm_a = cached_song_a.meta.bpm
                _metrics.bpm_b = cached_song_b.meta.bpm
                _metrics.key_a = cached_song_a.meta.key or ""
                _metrics.scale_a = cached_song_a.meta.scale or ""
                _metrics.key_b = cached_song_b.meta.key or ""
                _metrics.scale_b = cached_song_b.meta.scale or ""
                _metrics.key_confidence_a = cached_song_a.meta.key_confidence or 0.0
                _metrics.key_confidence_b = cached_song_b.meta.key_confidence or 0.0
                _metrics.duration_a_s = cached_song_a.meta.duration_seconds
                _metrics.duration_b_s = cached_song_b.meta.duration_seconds
                _metrics.log_input()
                _metrics.log_analysis()

                run_remix(
                    session_id=session_id,
                    analysis=analysis,
                    prompt=prompt,
                    event_queue=session.events,
                    session=session,
                    source_quality_a=cached_song_a.meta.source_quality,
                    source_quality_b=cached_song_b.meta.source_quality,
                    metrics=_metrics,
                )
                _update_avg_remix_duration(time.monotonic() - pipeline_start)
                return

            # Stems gone from disk — fall through to normal download path
            logger.warning("Session %s: Cached stems missing, falling back to full pipeline", session_id)

        # --- Monotonic progress tracker ---
        # Both downloads report into the 0.02-0.10 range.  Song A maps to
        # 0.02-0.06 and Song B maps to 0.06-0.10 — but because they run
        # concurrently, raw values can arrive out-of-order.  We only emit
        # when the new value exceeds the high-water mark.
        _progress_lock = threading.Lock()
        _progress_hwm = 0.0  # high-water mark

        def _emit_monotonic(detail: str, progress: float) -> None:
            nonlocal _progress_hwm
            with _progress_lock:
                if progress <= _progress_hwm:
                    return
                _progress_hwm = progress
            emit_progress(session.events, {
                "step": "downloading",
                "detail": detail,
                "progress": round(progress, 3),
            }, session=session)

        # --- Emit initial downloading event ---
        _emit_monotonic("Getting your songs ready...", 0.02)

        # --- Progress callbacks ---
        def _progress_a(fraction: float, status: str) -> None:
            # Map fraction 0-1 to progress 0.02-0.06
            progress = 0.02 + fraction * 0.04
            if fraction >= 1.0:
                _emit_monotonic("Got the first song!", progress)
            else:
                _emit_monotonic("Grabbing your first song...", progress)

        def _progress_b(fraction: float, status: str) -> None:
            # Map fraction 0-1 to progress 0.06-0.10
            progress = 0.06 + fraction * 0.04
            if fraction >= 1.0:
                _emit_monotonic("Got the second song!", progress)
            else:
                _emit_monotonic("Grabbing your second song...", progress)

        # --- Download helpers (run async fn in a fresh event loop per thread) ---
        def _download_a():
            return _run_sync(download_youtube_audio(
                url=url_a,
                output_dir=upload_dir,
                progress_callback=_progress_a,
            ))

        def _download_b():
            return _run_sync(download_youtube_audio(
                url=url_b,
                output_dir=upload_dir,
                progress_callback=_progress_b,
            ))

        # --- Parallel downloads ---
        with ThreadPoolExecutor(max_workers=2) as dl_executor:
            future_a = dl_executor.submit(_download_a)
            future_b = dl_executor.submit(_download_b)

            try:
                result_a = future_a.result(timeout=300)
            except Exception:
                # If Song A fails, cancel Song B and re-raise
                future_b.cancel()
                dl_executor.shutdown(wait=False, cancel_futures=True)
                raise

            try:
                result_b = future_b.result(timeout=300)
            except Exception:
                # Song B failed after Song A succeeded
                dl_executor.shutdown(wait=False, cancel_futures=True)
                raise

        _emit_monotonic("Got both songs!", 0.10)

        max_processing_duration = settings.processing_max_duration_seconds
        _pre_trim_youtube_download(result_a, max_processing_duration)
        _pre_trim_youtube_download(result_b, max_processing_duration)

        # Check cancellation before starting heavy pipeline work
        from musicmixer.services.pipeline import check_cancelled
        check_cancelled(session)

        logger.info(
            "Session %s: YouTube downloads complete. A=%r (%ds, %s %dkbps), B=%r (%ds, %s %dkbps)",
            session_id,
            result_a.title, int(result_a.duration_seconds),
            result_a.source_codec, result_a.source_bitrate,
            result_b.title, int(result_b.duration_seconds),
            result_b.source_codec, result_b.source_bitrate,
        )

        # Build source quality strings for metadata propagation
        source_quality_a = f"youtube-{result_a.source_codec}-{result_a.source_bitrate}kbps"
        source_quality_b = f"youtube-{result_b.source_codec}-{result_b.source_bitrate}kbps"

        logger.info(
            "Session %s: Source quality: A=%s, B=%s",
            session_id, source_quality_a, source_quality_b,
        )

        # --- Analyze + remix (45-100%) ---
        _metrics = PipelineMetrics(session_id=session_id)
        # Populate video IDs and titles for YouTube inputs
        from musicmixer.api.shelf import _extract_video_id as _yt_extract_vid
        _vid_a = _yt_extract_vid(url_a)
        _vid_b = _yt_extract_vid(url_b)
        _metrics.song_a_video_id = _vid_a or ""
        _metrics.song_b_video_id = _vid_b or ""

        analysis = analyze_songs(
            session_id=session_id,
            song_a_path=str(result_a.wav_path),
            song_b_path=str(result_b.wav_path),
            event_queue=session.events,
            session=session,
            song_a_original_filename=result_a.title,
            song_b_original_filename=result_b.title,
            source_quality_a=source_quality_a,
            source_quality_b=source_quality_b,
            shelf_song_id_a=shelf_song_id_a,
            shelf_song_id_b=shelf_song_id_b,
            metrics=_metrics,
        )

        # Cache analysis results to Redis for future cache hits
        from musicmixer.api.shelf import _extract_video_id
        from musicmixer.services.song_cache import cache_song_metadata, cache_song_stems
        from pathlib import Path

        for url, title, meta, lyrics, stems_dir in [
            (url_a, result_a.title, analysis.meta_a, analysis.lyrics_a, analysis.song_a_stems_dir),
            (url_b, result_b.title, analysis.meta_b, analysis.lyrics_b, analysis.song_b_stems_dir),
        ]:
            vid = _extract_video_id(url)
            if vid is not None:
                cache_song_metadata(vid, title, "", meta, lyrics)
                cache_song_stems(vid, Path(stems_dir))

        run_remix(
            session_id=session_id,
            analysis=analysis,
            prompt=prompt,
            event_queue=session.events,
            session=session,
            source_quality_a=source_quality_a,
            source_quality_b=source_quality_b,
            metrics=_metrics,
        )
        _update_avg_remix_duration(time.monotonic() - pipeline_start)

    except CancelledError:
        logger.info("Session %s: YouTube pipeline cancelled by user", session_id)
        session.status = "cancelled"
        emit_progress(session.events, {
            "step": "cancelled",
            "detail": "Remix cancelled",
            "progress": 0,
        }, session=session)
    except BaseException as exc:
        logger.exception("Session %s: YouTube pipeline failed", session_id)
        session.status = "error"
        emit_progress(session.events, {
            "step": "error",
            "detail": str(exc),
            "progress": 0,
        }, session=session)
    finally:
        processing_lock.release()
        _process_next_queued(app_state)


def _run_sync(coro):
    """Run an async coroutine synchronously from a thread (no running event loop)."""
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def _process_next_queued(app_state) -> None:
    """Pull the next valid item from the wait queue and submit it.

    Called from the pipeline finally block after releasing the processing lock.
    Skips items whose SSE clients have disconnected or that have exceeded
    the queue entry TTL.
    """
    from musicmixer.services.pipeline import emit_progress

    queue_entry_ttl_s = settings.queue_entry_ttl_minutes * 60

    while True:
        try:
            item: _QueueItem = app_state.wait_queue.get_nowait()
        except queue.Empty:
            return

        # Broadcast updated positions to remaining queued sessions
        _broadcast_queue_positions(app_state)

        # Check if this queue entry has expired
        if time.monotonic() - item.enqueued_at > queue_entry_ttl_s:
            logger.info(
                "Session %s: queue entry expired (waited %.0fs), skipping",
                item.session_id,
                time.monotonic() - item.enqueued_at,
            )
            item.session.status = "error"
            emit_progress(item.session.events, {
                "step": "error",
                "detail": "Queue wait time exceeded, please try again",
                "progress": 0,
            }, session=item.session)
            continue

        # Check if session was cancelled or abandoned while queued
        if item.session.cancelled.is_set() or item.session.status == "cancelled":
            logger.info("Session %s: cancelled while queued, skipping", item.session_id)
            item.session.status = "cancelled"
            continue

        # Check if the SSE client is still connected by checking session status.
        # If the session was marked as abandoned (client disconnected), skip it.
        if item.session.status == "abandoned":
            logger.info(
                "Session %s: client disconnected while queued, skipping",
                item.session_id,
            )
            continue

        # Acquire slot and start processing
        if not app_state.processing_lock.acquire(blocking=False):
            # Slot was taken (race condition); re-queue the item
            try:
                app_state.wait_queue.put_nowait(item)
            except queue.Full:
                item.session.status = "error"
                emit_progress(item.session.events, {
                    "step": "error",
                    "detail": "Server overloaded, please try again",
                    "progress": 0,
                }, session=item.session)
            return

        # Emit processing_started event
        emit_progress(item.session.events, {
            "step": "processing_started",
            "detail": "Your remix is starting now",
            "progress": 0,
        }, session=item.session)

        try:
            app_state.executor.submit(item.run_fn)
        except Exception:
            app_state.processing_lock.release()
            logger.exception("Session %s: failed to submit queued pipeline", item.session_id)
            item.session.status = "error"
            emit_progress(item.session.events, {
                "step": "error",
                "detail": "Failed to start pipeline",
                "progress": 0,
            }, session=item.session)
            continue
        return


def _get_queue_position(app_state, session_id: str) -> tuple[int, int]:
    """Return (position, total) for a session in the wait queue.

    Position is 1-based. Returns (0, total) if not found in queue.
    Thread-safe: reads queue internals under queue_lock.
    """
    with app_state.queue_lock:
        # Access the underlying deque for read-only position lookup
        items = list(app_state.wait_queue.queue)
        total = len(items)
        for i, item in enumerate(items):
            if item.session_id == session_id:
                return (i + 1, total)
    return (0, total)


def _broadcast_queue_positions(app_state) -> None:
    """Send updated queue_position events to all queued sessions."""
    from musicmixer.services.pipeline import emit_progress

    with app_state.queue_lock:
        items = list(app_state.wait_queue.queue)

    for i, item in enumerate(items):
        position = i + 1
        total = len(items)
        emit_progress(item.session.events, {
            "step": "queue_position",
            "detail": f"Position {position} of {total}",
            "position": position,
            "total": total,
            "progress": 0,
        }, session=item.session)
        emit_progress(item.session.events, {
            "step": "queue_estimate",
            "detail": f"Estimated wait: {int(position * _AVG_REMIX_DURATION_S)}s",
            "wait_seconds": int(position * _AVG_REMIX_DURATION_S),
            "progress": 0,
        }, session=item.session)


def _enqueue_or_start(app_state, session_id: str, session: SessionState, run_fn: Callable[[], None]) -> None:
    """Try to acquire a processing slot; if busy, enqueue the request.

    Raises HTTPException(503) if the queue is full.
    """
    from musicmixer.services.pipeline import emit_progress

    processing_lock = app_state.processing_lock

    if processing_lock.acquire(blocking=False):
        # Slot available — submit immediately
        emit_progress(session.events, {
            "step": "processing_started",
            "detail": "Your remix is starting now",
            "progress": 0,
        }, session=session)
        try:
            app_state.executor.submit(run_fn)
        except Exception:
            processing_lock.release()
            raise
        return

    # All slots busy — enqueue
    item = _QueueItem(
        session_id=session_id,
        session=session,
        run_fn=run_fn,
    )

    try:
        app_state.wait_queue.put_nowait(item)
    except queue.Full:
        raise HTTPException(503, "Server is at capacity, please try again later")

    # Get position and send queue events
    position, total = _get_queue_position(app_state, session_id)
    emit_progress(session.events, {
        "step": "queue_position",
        "detail": f"Position {position} of {total}",
        "position": position,
        "total": total,
        "progress": 0,
    }, session=session)
    emit_progress(session.events, {
        "step": "queue_estimate",
        "detail": f"Estimated wait: {int(position * _AVG_REMIX_DURATION_S)}s",
        "wait_seconds": int(position * _AVG_REMIX_DURATION_S),
        "progress": 0,
    }, session=session)

    logger.info(
        "Session %s: queued at position %d/%d",
        session_id, position, total,
    )


# ---------------------------------------------------------------------------
# POST endpoints
# ---------------------------------------------------------------------------

@router.post("/remix/youtube")
def create_youtube_remix(
    request: Request,
    body: YouTubeRemixRequest,
):
    """Accept two YouTube URLs, download audio, and start remix pipeline.

    Both URLs must be valid YouTube links. Returns session_id immediately.
    If all processing slots are busy, the request is queued.
    Returns 503 if the queue is full.
    """
    if not settings.youtube_enabled:
        raise HTTPException(403, "YouTube input is disabled")

    # Validate both URLs (SSRF prevention) before doing any work
    _validate_youtube_url(body.url_a)
    _validate_youtube_url(body.url_b)

    # Resolve shelf song IDs BEFORE starting the pipeline (server-side reverse lookup)
    shelf_song_id_a = _resolve_shelf_song_id(body.url_a)
    shelf_song_id_b = _resolve_shelf_song_id(body.url_b)
    if shelf_song_id_a:
        logger.info("Resolved shelf song ID for URL A: %s", shelf_song_id_a[:12])
    if shelf_song_id_b:
        logger.info("Resolved shelf song ID for URL B: %s", shelf_song_id_b[:12])

    # Check Redis song cache for each URL
    from musicmixer.api.shelf import _extract_video_id
    from musicmixer.services.song_cache import get_cached_song

    video_id_a = _extract_video_id(body.url_a)
    video_id_b = _extract_video_id(body.url_b)
    cached_a = get_cached_song(video_id_a) if video_id_a else None
    cached_b = get_cached_song(video_id_b) if video_id_b else None
    if cached_a:
        logger.info("Song cache HIT for URL A (video_id=%s, has_stems=%s)", video_id_a, cached_a.has_stems)
    if cached_b:
        logger.info("Song cache HIT for URL B (video_id=%s, has_stems=%s)", video_id_b, cached_b.has_stems)

    # Auto-save both songs to the shelf (idempotent — skips duplicates)
    try:
        ensure_on_shelf(body.url_a)
        ensure_on_shelf(body.url_b)
    except Exception:
        pass  # Shelf save is best-effort; don't block the remix

    # Clean up expired sessions before processing
    cleanup_expired_sessions(request.app.state.sessions, request.app.state.sessions_lock)

    # === PRE-QUEUE CACHE CHECK (URL-based, no file I/O) ===
    # Same URLs + prompt always produce the same remix. Check before queueing
    # so cached remixes are served instantly without consuming a processing slot.
    if settings.remix_cache_enabled:
        try:
            from musicmixer.services.remix_cache import (
                compute_url_cache_key,
                get_cached_remix,
                get_cached_metadata,
            )

            url_key = compute_url_cache_key(body.url_a, body.url_b, body.prompt)
            cached_path = get_cached_remix(url_key, settings.remix_cache_dir)
            if cached_path is not None:
                logger.info("Pre-queue URL cache hit (key=%s), serving instantly", url_key[:12])
                meta = get_cached_metadata(url_key, settings.remix_cache_dir) or {}

                session_id = str(uuid.uuid4())
                session = SessionState()
                session.status = "complete"
                session.thumbnail_url_a = _thumbnail_from_youtube_url(body.url_a)
                session.thumbnail_url_b = _thumbnail_from_youtube_url(body.url_b)
                session.explanation = meta.get("explanation", "")
                session.used_fallback = meta.get("used_fallback", False)
                session.warnings = meta.get("warnings", [])
                session.key_warning = meta.get("key_warning")

                # Copy cached remix to session output dir
                import shutil as _shutil
                remix_dir = settings.data_dir / "remixes" / session_id
                remix_dir.mkdir(parents=True, exist_ok=True)
                output_path = remix_dir / "remix.mp3"
                _shutil.copy2(cached_path, output_path)
                session.remix_path = str(output_path)

                with request.app.state.sessions_lock:
                    request.app.state.sessions[session_id] = session

                return {"session_id": session_id}
        except Exception:
            logger.debug("Pre-queue URL cache check failed, proceeding normally", exc_info=True)

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Create session state with thumbnail URLs derived from YouTube video IDs
    session = SessionState()
    session.thumbnail_url_a = _thumbnail_from_youtube_url(body.url_a)
    session.thumbnail_url_b = _thumbnail_from_youtube_url(body.url_b)
    # Store URL cache key so the pipeline can write an alias after the content-based cache
    if settings.remix_cache_enabled:
        try:
            from musicmixer.services.remix_cache import compute_url_cache_key
            session.url_cache_key = compute_url_cache_key(body.url_a, body.url_b, body.prompt)
        except Exception:
            pass
    with request.app.state.sessions_lock:
        request.app.state.sessions[session_id] = session

    app_state = request.app.state
    processing_lock = app_state.processing_lock

    # Build the run function (binds all args for deferred execution)
    def run_fn():
        _youtube_pipeline_wrapper(
            session_id,
            body.url_a,
            body.url_b,
            body.prompt,
            session,
            processing_lock,
            app_state,
            shelf_song_id_a=shelf_song_id_a,
            shelf_song_id_b=shelf_song_id_b,
            cached_song_a=cached_a,
            cached_song_b=cached_b,
        )

    _enqueue_or_start(app_state, session_id, session, run_fn)

    return {"session_id": session_id}


@router.post("/remix")
def create_remix(
    request: Request,
    song_a: UploadFile = File(...),
    song_b: UploadFile = File(...),
    prompt: str = Form(""),
):
    """Accept two songs, start async pipeline, return session ID immediately.

    If all processing slots are busy, the request is queued.
    Returns 503 if the queue is full.
    """
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    # Validate extensions
    for label, file in [("song_a", song_a), ("song_b", song_b)]:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                422,
                f"Invalid file type for {label}: '{ext}'. "
                f"Allowed: {settings.allowed_extensions}",
            )

    # Clean up expired sessions before processing
    cleanup_expired_sessions(request.app.state.sessions, request.app.state.sessions_lock)

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Save uploaded files
    upload_dir = settings.data_dir / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Capture original filenames before saving (needed for lyrics lookup)
    song_a_original_filename = song_a.filename or ""
    song_b_original_filename = song_b.filename or ""

    song_a_ext = Path(song_a.filename or "song_a.mp3").suffix.lower()
    song_b_ext = Path(song_b.filename or "song_b.mp3").suffix.lower()

    song_a_path = upload_dir / f"song_a{song_a_ext}"
    song_b_path = upload_dir / f"song_b{song_b_ext}"

    for label, file, dest in [
        ("song_a", song_a, song_a_path),
        ("song_b", song_b, song_b_path),
    ]:
        file.file.seek(0)
        chunks = []
        total = 0
        while True:
            chunk = file.file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(
                    413,
                    f"{label} exceeds {settings.max_file_size_mb}MB limit",
                )
            chunks.append(chunk)
        data = b"".join(chunks)
        dest.write_bytes(data)

    logger.info(
        "Session %s: saved uploads (%s, %s)",
        session_id,
        song_a_path.name,
        song_b_path.name,
    )

    # Validate duration via ffprobe
    max_dur = settings.max_upload_duration_seconds
    for label, path in [("song_a", song_a_path), ("song_b", song_b_path)]:
        duration = _probe_duration(path)
        if duration is not None and duration > max_dur:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(
                413,
                f"{label} duration ({int(duration)}s) exceeds "
                f"{max_dur // 60} minute limit",
            )

    # Create session state
    session = SessionState()
    with request.app.state.sessions_lock:
        request.app.state.sessions[session_id] = session

    app_state = request.app.state
    processing_lock = app_state.processing_lock

    # Build the run function (binds all args for deferred execution)
    def run_fn():
        _pipeline_wrapper(
            session_id,
            song_a_path,
            song_b_path,
            prompt,
            session,
            processing_lock,
            app_state,
            song_a_original_filename,
            song_b_original_filename,
        )

    _enqueue_or_start(app_state, session_id, session, run_fn)

    return {"session_id": session_id}


@router.post("/remix/{session_id}/cancel")
async def cancel_remix(session_id: str, request: Request):
    """Cancel a running or queued remix session."""
    _validate_uuid(session_id)

    with request.app.state.sessions_lock:
        session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    if session.status in ("complete", "error", "cancelled"):
        return {"status": session.status, "message": "Session already finished"}

    session.cancelled.set()
    logger.info("Session %s: cancel requested (status was %s)", session_id, session.status)

    if session.status == "queued":
        session.status = "cancelled"
        from musicmixer.services.pipeline import emit_progress
        emit_progress(session.events, {
            "step": "cancelled",
            "detail": "Remix cancelled",
            "progress": 0,
        }, session=session)

    return {"status": "cancelling", "message": "Cancel signal sent"}


# ---------------------------------------------------------------------------
# SMS notification registration
# ---------------------------------------------------------------------------

# E.164: '+' followed by 10-15 digits
_E164_RE = re.compile(r"^\+\d{10,15}$")


class NotifySmsRequest(BaseModel):
    phone: str


@router.post("/remix/{session_id}/notify-sms")
def register_sms_notification(
    session_id: str,
    body: NotifySmsRequest,
    request: Request,
):
    """Register a phone number to receive an SMS when the remix is ready.

    - 202: phone stored, confirmation SMS sent (best-effort)
    - 200: session already complete, ready notification sent directly
    - 409: session in error state
    - 422: invalid phone format
    - 503: SMS feature disabled
    """
    _validate_uuid(session_id)

    if not settings.sms_enabled:
        raise HTTPException(503, "SMS notifications are not available")

    if not _E164_RE.match(body.phone):
        raise HTTPException(422, "Phone must be E.164 format (e.g. +15551234567)")

    with request.app.state.sessions_lock:
        session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    if session.status == "error":
        raise HTTPException(409, "Session failed — cannot register for notification")

    if session.status == "complete":
        # Remix already done — send ready notification directly, no confirmation
        from musicmixer.services.sms import send_remix_ready

        try:
            send_remix_ready(body.phone, session_id)
        except Exception:
            logger.exception(
                "Session %s: failed to send ready SMS", session_id
            )
        return {"status": "sent", "message": "Remix is already ready — notification sent"}

    # Store phone on session (idempotent: overwrites any previous value)
    session.notify_phone = body.phone

    # Send confirmation SMS (best-effort — failure is non-blocking)
    from musicmixer.services.sms import send_confirmation

    try:
        send_confirmation(body.phone)
    except Exception:
        logger.exception(
            "Session %s: failed to send confirmation SMS", session_id
        )

    return JSONResponse(
        status_code=202,
        content={"status": "registered", "message": "We'll text you when your remix is ready"},
    )


@router.get("/remix/{session_id}/progress")
async def get_progress(session_id: str, request: Request):
    """SSE endpoint streaming pipeline progress events."""
    _validate_uuid(session_id)

    with request.app.state.sessions_lock:
        session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    sse_executor = request.app.state.sse_executor

    return StreamingResponse(
        _event_stream(session, sse_executor, request.app.state, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Maximum progress the SSE nudge may reach within each pipeline step.
# Prevents the bar from drifting past the range allocated to the
# current step while the pipeline is idle.
_STEP_PROGRESS_CEILING: dict[str, float] = {
    "downloading": 0.10,
    "separating": 0.39,
    "analyzing": 0.57,
    "interpreting": 0.62,
    "processing": 0.92,
    "rendering": 0.98,
}


async def _event_stream(
    session: SessionState,
    sse_executor,
    app_state=None,
    session_id: str = "",
) -> AsyncGenerator[str, None]:
    """Generate SSE data events from the session's event queue.

    When the queue is idle for >3s during processing, the stream emits
    a small synthetic progress nudge so the client-side bar never
    appears frozen.  Nudges are capped at the current step's ceiling
    so progress can't drift into the next step's range.
    """
    loop = asyncio.get_running_loop()
    start = time.monotonic()
    last_heartbeat = time.monotonic()

    # Track the last emitted values so we can nudge forward during idle
    last_step = ""
    last_detail = ""
    last_progress = 0.0

    # On connect: send current state so reconnecting clients catch up
    if session.last_event:
        yield f"data: {json.dumps(session.last_event)}\n\n"
        last_step = session.last_event.get("step", "")
        last_detail = session.last_event.get("detail", "")
        last_progress = session.last_event.get("progress", 0.0)
        if last_step in ("complete", "error"):
            return

    # Drain stale events that arrived before the client connected
    while not session.events.empty():
        try:
            session.events.get_nowait()
        except queue.Empty:
            break

    while True:
        # 20-minute safety cap
        if time.monotonic() - start > 1200:
            yield 'data: {"step":"error","detail":"Processing timed out","progress":0}\n\n'
            break

        try:
            event = await loop.run_in_executor(
                sse_executor,
                functools.partial(session.events.get, timeout=2),
            )
        except queue.Empty:
            now = time.monotonic()
            # Send heartbeat every 15s while queued
            if session.status == "queued" and now - last_heartbeat >= 15:
                yield 'data: {"step":"heartbeat","detail":"","progress":0}\n\n'
                last_heartbeat = now
            elif session.status == "processing" and 0 < last_progress < 0.98:
                # Nudge progress forward so the bar never freezes.
                # Capped at the current step's ceiling to prevent drift
                # into the next step's allocated range.
                ceiling = _STEP_PROGRESS_CEILING.get(last_step, 0.98)
                if last_progress < ceiling:
                    last_progress = min(last_progress + 0.008, ceiling)
                    nudge = {
                        "step": last_step,
                        "detail": last_detail,
                        "progress": round(last_progress, 3),
                    }
                    yield f"data: {json.dumps(nudge)}\n\n"
                else:
                    # At ceiling — send keepalive to maintain connection
                    yield 'data: {"step":"keepalive","detail":"","progress":-1}\n\n'
            else:
                yield 'data: {"step":"keepalive","detail":"","progress":-1}\n\n'
            continue
        except asyncio.CancelledError:
            # Client disconnected — let the pipeline continue running.
            # Client can reconnect via /progress endpoint to resume updates.
            logger.info("Session %s: SSE client disconnected (pipeline continues)", session_id)
            break

        session.last_event = event
        last_step = event.get("step", last_step)
        last_detail = event.get("detail", last_detail)
        last_progress = max(last_progress, event.get("progress", last_progress))
        yield f"data: {json.dumps(event)}\n\n"

        if event.get("step") in ("complete", "error", "cancelled"):
            break


@router.get("/remix/{session_id}/status")
async def get_status(session_id: str, request: Request):
    """Return JSON snapshot of session state."""
    _validate_uuid(session_id)

    with request.app.state.sessions_lock:
        session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    return {
        "session_id": session_id,
        "status": session.status,
        "remix_path": session.remix_path,
        "explanation": session.explanation,
        "last_event": session.last_event,
    }


@router.get("/remix/{session_id}/audio")
async def get_audio(session_id: str):
    """Serve the rendered remix MP3."""
    _validate_uuid(session_id)

    remix_path = (settings.data_dir / "remixes" / session_id / "remix.mp3").resolve()

    # Belt-and-suspenders: ensure resolved path is still inside data_dir
    if not remix_path.is_relative_to(settings.data_dir.resolve()):
        raise HTTPException(400, "Invalid session ID")

    if not remix_path.exists():
        raise HTTPException(404, "Remix not found")
    return FileResponse(remix_path, media_type="audio/mpeg", filename="remix.mp3")


VALID_SONGS = {"song_a", "song_b"}
VALID_STEMS = {"vocals", "drums", "bass", "guitar", "piano", "other"}


@router.get("/remix/{session_id}/stems")
async def list_stems(session_id: str):
    """List available stems for both songs in a session."""
    _validate_uuid(session_id)

    stems_dir = (settings.data_dir / "stems" / session_id).resolve()

    if not stems_dir.is_relative_to(settings.data_dir.resolve()):
        raise HTTPException(400, "Invalid session ID")

    if not stems_dir.exists():
        raise HTTPException(404, "Stems not found for this session")

    result: dict[str, list[str]] = {}
    for song in sorted(VALID_SONGS):
        song_dir = stems_dir / song
        if not song_dir.is_dir():
            result[song] = []
            continue
        result[song] = sorted(
            f.stem
            for f in song_dir.iterdir()
            if f.suffix.lower() == ".wav" and f.stem in VALID_STEMS
        )

    return result


@router.get("/remix/{session_id}/stems/{song}/{stem_name}")
async def get_stem(session_id: str, song: str, stem_name: str):
    """Serve a raw WAV stem file."""
    _validate_uuid(session_id)

    if song not in VALID_SONGS:
        raise HTTPException(400, f"Invalid song: must be one of {sorted(VALID_SONGS)}")
    if stem_name not in VALID_STEMS:
        raise HTTPException(400, f"Invalid stem: must be one of {sorted(VALID_STEMS)}")

    stem_path = (
        settings.data_dir / "stems" / session_id / song / f"{stem_name}.wav"
    ).resolve()

    if not stem_path.is_relative_to(settings.data_dir.resolve()):
        raise HTTPException(400, "Invalid session ID")

    if not stem_path.exists():
        raise HTTPException(404, f"Stem '{stem_name}' not found for {song}")

    return FileResponse(
        stem_path,
        media_type="audio/wav",
        filename=f"{song}_{stem_name}.wav",
    )


@router.get("/remix/{session_id}/public")
async def get_public_remix(session_id: str, request: Request):
    """Public endpoint for share/listen links.

    Returns remix info for completed sessions, 202 for in-progress,
    404 for not found, and 410 for expired or errored sessions.
    """
    _validate_uuid(session_id)

    with request.app.state.sessions_lock:
        session = request.app.state.sessions.get(session_id)

    # 1. Session not found
    if session is None:
        raise HTTPException(404, "Session not found")

    # 2. Session expired (TTL check)
    ttl_seconds = settings.session_ttl_hours * 3600
    created_at = getattr(session, "created_at", None)
    if created_at is not None and (time.time() - created_at) > ttl_seconds:
        return JSONResponse(status_code=410, content={"status": "expired"})

    # 3. Complete + remix file exists -> 200
    if session.status == "complete":
        remix_path = settings.data_dir / "remixes" / session_id / "remix.mp3"
        if remix_path.exists():
            from datetime import datetime, timezone

            warnings = getattr(session, "warnings", [])
            used_fallback = getattr(session, "used_fallback", False)
            expires_at_ts = created_at + ttl_seconds if created_at else time.time() + ttl_seconds
            expires_at = datetime.fromtimestamp(expires_at_ts, tz=timezone.utc).isoformat()

            return {
                "session_id": session_id,
                "status": "ready",
                "audio_url": f"/api/remix/{session_id}/audio",
                "explanation": session.explanation or "",
                "warnings": warnings,
                "usedFallback": used_fallback,
                "expires_at": expires_at,
                "thumbnail_url_a": getattr(session, "thumbnail_url_a", None),
                "thumbnail_url_b": getattr(session, "thumbnail_url_b", None),
            }

    # 4. Processing or queued -> 202
    if session.status in ("processing", "queued"):
        return JSONResponse(status_code=202, content={"status": "processing"})

    # 5. Error (or any other terminal state) -> 410
    return JSONResponse(status_code=410, content={"status": "error"})


def _validate_uuid(session_id: str) -> None:
    """Validate that session_id is a well-formed UUID. Raises 400 if not."""
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")
