import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from musicmixer.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/remix")
def create_remix(
    song_a: UploadFile = File(...),
    song_b: UploadFile = File(...),
    prompt: str = Form(""),
):
    """Accept two songs, separate stems, overlay, return session ID.

    Day 1: Synchronous. Blocks until remix is complete.
    """
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    # Validate extensions
    for label, file in [("song_a", song_a), ("song_b", song_b)]:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in settings.allowed_extensions:
            raise HTTPException(
                422,
                f"Invalid file type for {label}: '{ext}'. "
                f"Allowed: {settings.allowed_extensions}"
            )

    # Generate session ID
    session_id = str(uuid.uuid4())

    # Save uploaded files
    upload_dir = settings.data_dir / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    song_a_ext = Path(song_a.filename or "song_a.mp3").suffix.lower()
    song_b_ext = Path(song_b.filename or "song_b.mp3").suffix.lower()

    song_a_path = upload_dir / f"song_a{song_a_ext}"
    song_b_path = upload_dir / f"song_b{song_b_ext}"

    for label, file, dest in [("song_a", song_a, song_a_path), ("song_b", song_b, song_b_path)]:
        data = file.file.read()
        if len(data) > max_bytes:
            raise HTTPException(413, f"{label} exceeds {settings.max_file_size_mb}MB limit")
        dest.write_bytes(data)

    logger.info(f"Session {session_id}: saved uploads ({song_a_path.name}, {song_b_path.name})")

    # Run pipeline synchronously (Day 1 -- no background processing)
    try:
        from musicmixer.services.pipeline_day1 import run_pipeline_sync
        remix_path = run_pipeline_sync(session_id, song_a_path, song_b_path)
        logger.info(f"Session {session_id}: remix complete at {remix_path}")
    except Exception as e:
        logger.exception(f"Session {session_id}: pipeline failed")
        raise HTTPException(500, f"Remix failed: {str(e)}")

    return {"session_id": session_id}


@router.get("/remix/{session_id}/audio")
async def get_audio(session_id: str):
    """Serve the rendered remix MP3."""
    remix_path = settings.data_dir / "remixes" / session_id / "remix.mp3"
    if not remix_path.exists():
        raise HTTPException(404, "Remix not found")
    return FileResponse(remix_path, media_type="audio/mpeg", filename="remix.mp3")
