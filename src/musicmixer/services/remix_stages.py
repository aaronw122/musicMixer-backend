"""Pure, API-agnostic helpers for the remix endpoints.

These functions perform stage-adjacent input work (thumbnail derivation, upload
validation/write, duration probing) with no FastAPI, SSE, or ``SessionState``
coupling. They raise plain exceptions or return values; ``api/remix.py``
translates results into HTTP responses, preserving the route's exact status
codes and detail strings.
"""

import subprocess
import urllib.parse
from pathlib import Path
from typing import BinaryIO, Iterable


class UploadTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""


def thumbnail_from_youtube_url(url: str) -> str | None:
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


def probe_duration(file_path: Path) -> float | None:
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


def upload_extension(filename: str | None) -> str:
    """Return the lowercased file suffix for an uploaded file name."""
    return Path(filename or "").suffix.lower()


def extension_allowed(filename: str | None, allowed_extensions: Iterable[str]) -> bool:
    """Return whether the upload's extension is in the allowed set."""
    return upload_extension(filename) in allowed_extensions


def write_upload_file(file: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Stream an uploaded file to ``dest`` in 1 MB chunks.

    Raises ``UploadTooLargeError`` if the cumulative size exceeds ``max_bytes``.
    The size check happens mid-stream so an oversized upload is rejected without
    buffering the whole payload.
    """
    file.seek(0)
    chunks = []
    total = 0
    while True:
        chunk = file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError
        chunks.append(chunk)
    data = b"".join(chunks)
    dest.write_bytes(data)
