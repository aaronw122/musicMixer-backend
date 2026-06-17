"""Pure, API-agnostic helpers for the remix endpoints.

These functions perform stage-adjacent input work (thumbnail derivation, upload
validation/write, duration probing) with no FastAPI, SSE, or ``SessionState``
coupling. They raise plain exceptions or return values; ``api/remix.py``
translates results into HTTP responses, preserving the route's exact status
codes and detail strings.
"""

import logging
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

logger = logging.getLogger("musicmixer.api.remix")


class UploadTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""


@dataclass(frozen=True)
class PreQueueCacheHit:
    """A served-from-cache remix resolved before the queue/processing lock.

    Carries the copied remix path and the metadata-derived session fields the
    route applies to its ``SessionState``. The route owns session lifecycle and
    the response; this is just the resolved data.
    """

    remix_path: str
    explanation: str
    used_fallback: bool
    key_warning: str | None
    warnings: list[str] = field(default_factory=list)


def restore_prequeue_cached_remix(
    url_a: str,
    url_b: str,
    prompt: str,
    *,
    session_id: str,
    cache_dir: Path,
    data_dir: Path,
) -> PreQueueCacheHit | None:
    """Resolve a URL-cache hit before queueing, or return ``None`` to fall through.

    Same URLs + prompt always produce the same remix, so a hit can be served
    instantly without consuming a processing slot. On a hit, the cached remix is
    copied into the session's output dir and the metadata-derived fields are
    returned. Any failure returns ``None`` so the route enqueues normally.
    """
    from musicmixer.services.remix_cache import (
        compute_url_cache_key,
        get_cached_remix,
        get_cached_metadata,
    )

    try:
        url_key = compute_url_cache_key(url_a, url_b, prompt)
        cached_path = get_cached_remix(url_key, cache_dir)
        if cached_path is None:
            return None

        logger.info(
            "Pre-queue URL cache hit (key=%s), serving instantly", url_key[:12]
        )
        meta = get_cached_metadata(url_key, cache_dir) or {}

        remix_dir = data_dir / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)
        output_path = remix_dir / "remix.mp3"
        shutil.copy2(cached_path, output_path)

        return PreQueueCacheHit(
            remix_path=str(output_path),
            explanation=meta.get("explanation", ""),
            used_fallback=meta.get("used_fallback", False),
            key_warning=meta.get("key_warning"),
            warnings=meta.get("warnings", []),
        )
    except Exception:
        logger.debug(
            "Pre-queue URL cache check failed, proceeding normally", exc_info=True
        )
        return None


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
