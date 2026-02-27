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
import functools
import json
import logging
import queue
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from musicmixer.config import settings
from musicmixer.models import SessionState

logger = logging.getLogger(__name__)
router = APIRouter()


def _pipeline_wrapper(
    session_id: str,
    song_a_path: Path,
    song_b_path: Path,
    prompt: str,
    session: SessionState,
    processing_lock,
    song_a_original_filename: str = "",
    song_b_original_filename: str = "",
) -> None:
    """Runs the pipeline in a background thread. Releases processing_lock on exit."""
    try:
        session.status = "processing"
        from musicmixer.services.pipeline import run_pipeline

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

class YouTubeRemixRequest(BaseModel):
    url_a: str  # YouTube URL for song A
    url_b: str  # YouTube URL for song B
    prompt: str  # Remix prompt


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
) -> None:
    """Downloads YouTube audio, then runs the pipeline. Releases processing_lock on exit."""
    try:
        session.status = "processing"
        from musicmixer.services.pipeline import emit_progress, run_pipeline
        from musicmixer.services.youtube import download_youtube_audio

        upload_dir = settings.data_dir / "uploads" / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        # --- Download Song A (5-25% progress) ---
        emit_progress(session.events, {
            "step": "downloading",
            "detail": "Downloading song A from YouTube...",
            "progress": 0.05,
        }, session=session)

        def _progress_a(fraction: float, status: str) -> None:
            # Map fraction 0-1 to progress 0.05-0.25
            progress = 0.05 + fraction * 0.20
            emit_progress(session.events, {
                "step": "downloading",
                "detail": f"Downloading song A: {status}",
                "progress": round(progress, 3),
            }, session=session)

        result_a = _run_sync(download_youtube_audio(
            url=url_a,
            output_dir=upload_dir,
            progress_callback=_progress_a,
        ))

        emit_progress(session.events, {
            "step": "downloading",
            "detail": "Song A downloaded!",
            "progress": 0.25,
        }, session=session)

        # --- Download Song B (25-45% progress) ---
        emit_progress(session.events, {
            "step": "downloading",
            "detail": "Downloading song B from YouTube...",
            "progress": 0.25,
        }, session=session)

        def _progress_b(fraction: float, status: str) -> None:
            # Map fraction 0-1 to progress 0.25-0.45
            progress = 0.25 + fraction * 0.20
            emit_progress(session.events, {
                "step": "downloading",
                "detail": f"Downloading song B: {status}",
                "progress": round(progress, 3),
            }, session=session)

        result_b = _run_sync(download_youtube_audio(
            url=url_b,
            output_dir=upload_dir,
            progress_callback=_progress_b,
        ))

        emit_progress(session.events, {
            "step": "downloading",
            "detail": "Both songs downloaded!",
            "progress": 0.45,
        }, session=session)

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

        # --- Run existing pipeline (45-100%) ---
        # The pipeline's own progress events map into the remaining 45-100% range.
        # We pass the WAV paths directly — the pipeline doesn't know they came from YouTube.
        run_pipeline(
            session_id=session_id,
            song_a_path=str(result_a.wav_path),
            song_b_path=str(result_b.wav_path),
            prompt=prompt,
            event_queue=session.events,
            session=session,
            song_a_original_filename=result_a.title,
            song_b_original_filename=result_b.title,
        )

    except BaseException as exc:
        logger.exception("Session %s: YouTube pipeline failed", session_id)
        session.status = "error"
        from musicmixer.services.pipeline import emit_progress

        emit_progress(session.events, {
            "step": "error",
            "detail": str(exc),
            "progress": 0,
        }, session=session)
    finally:
        processing_lock.release()


def _run_sync(coro):
    """Run an async coroutine synchronously from a thread (no running event loop)."""
    import asyncio as _asyncio
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@router.post("/remix/youtube")
def create_youtube_remix(
    request: Request,
    body: YouTubeRemixRequest,
):
    """Accept two YouTube URLs, download audio, and start remix pipeline.

    Both URLs must be valid YouTube links. Returns session_id immediately.
    Only one remix can be processed at a time. Returns 409 if another is in progress.
    """
    if not settings.youtube_enabled:
        raise HTTPException(403, "YouTube input is disabled")

    # Validate both URLs (SSRF prevention) before doing any work
    _validate_youtube_url(body.url_a)
    _validate_youtube_url(body.url_b)

    # Fail-fast: check processing lock BEFORE downloading (don't waste time if busy)
    processing_lock = request.app.state.processing_lock
    if not processing_lock.acquire(blocking=False):
        raise HTTPException(409, "Another remix is being processed")

    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Create session state
        session = SessionState()
        with request.app.state.sessions_lock:
            request.app.state.sessions[session_id] = session

        # Submit YouTube download + pipeline to background executor
        request.app.state.executor.submit(
            _youtube_pipeline_wrapper,
            session_id,
            body.url_a,
            body.url_b,
            body.prompt,
            session,
            processing_lock,
        )

        return {"session_id": session_id}

    except HTTPException:
        raise
    except Exception:
        processing_lock.release()
        raise


@router.post("/remix")
def create_remix(
    request: Request,
    song_a: UploadFile = File(...),
    song_b: UploadFile = File(...),
    prompt: str = Form(""),
):
    """Accept two songs, start async pipeline, return session ID immediately.

    Only one remix can be processed at a time. Returns 409 if another is in progress.
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

    # Acquire processing lock (non-blocking) -- authoritative single-remix gate
    processing_lock = request.app.state.processing_lock
    if not processing_lock.acquire(blocking=False):
        raise HTTPException(409, "Another remix is being processed")

    try:
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
            data = file.file.read()
            if len(data) > max_bytes:
                processing_lock.release()
                raise HTTPException(
                    413, f"{label} exceeds {settings.max_file_size_mb}MB limit"
                )
            dest.write_bytes(data)

        logger.info(
            "Session %s: saved uploads (%s, %s)",
            session_id,
            song_a_path.name,
            song_b_path.name,
        )

        # Create session state
        session = SessionState()
        with request.app.state.sessions_lock:
            request.app.state.sessions[session_id] = session

        # Submit pipeline to background executor
        request.app.state.executor.submit(
            _pipeline_wrapper,
            session_id,
            song_a_path,
            song_b_path,
            prompt,
            session,
            processing_lock,
            song_a_original_filename,
            song_b_original_filename,
        )

        return {"session_id": session_id}

    except HTTPException:
        # Re-raise HTTP exceptions (like 413) without double-releasing the lock
        raise
    except Exception:
        # If anything unexpected fails before submit, release the lock
        processing_lock.release()
        raise


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
        _event_stream(session, sse_executor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _event_stream(
    session: SessionState,
    sse_executor,
) -> AsyncGenerator[str, None]:
    """Generate SSE data events from the session's event queue."""
    loop = asyncio.get_running_loop()
    start = time.monotonic()

    # On connect: send current state so reconnecting clients catch up
    if session.last_event:
        yield f"data: {json.dumps(session.last_event)}\n\n"
        if session.last_event.get("step") in ("complete", "error"):
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
                functools.partial(session.events.get, timeout=5),
            )
        except queue.Empty:
            # Send keepalive as a data event (not SSE comment) so EventSource.onmessage fires
            yield 'data: {"step":"keepalive","detail":"","progress":-1}\n\n'
            continue
        except asyncio.CancelledError:
            break

        session.last_event = event
        yield f"data: {json.dumps(event)}\n\n"

        if event.get("step") in ("complete", "error"):
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


def _validate_uuid(session_id: str) -> None:
    """Validate that session_id is a well-formed UUID. Raises 400 if not."""
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(400, "Invalid session ID")
