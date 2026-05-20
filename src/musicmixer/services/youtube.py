"""YouTube audio download service.

Downloads audio from YouTube videos as PCM WAV files for the remix pipeline.
Uses yt-dlp as a Python library with single-decode, zero generation loss:
YouTube (Opus/AAC) -> decode -> int16 PCM WAV at native sample rate.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
import yt_dlp

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Hosts allowed for YouTube URLs (SSRF prevention)
_YOUTUBE_HOSTS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "music.youtube.com",
})


class YouTubeDownloadError(Exception):
    """User-facing error from YouTube download operations."""

    pass


@dataclass
class YouTubeAudioResult:
    """Result of downloading audio from a YouTube video."""

    wav_path: Path
    title: str
    duration_seconds: float
    source_codec: str
    source_bitrate: int


def validate_youtube_url(url: str) -> None:
    """Validate a URL for SSRF prevention before passing to yt-dlp.

    Validation order matters — step 3 (reject @) must happen before step 5
    (hostname allowlist) to prevent userinfo bypass attacks like
    ``youtube.com@evil.com``.

    Raises:
        YouTubeDownloadError: If the URL fails any validation check.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # 1. Reject non-https schemes
    if parsed.scheme != "https":
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # 2. Reject URLs containing @ in the netloc (userinfo bypass)
    if parsed.netloc and "@" in parsed.netloc:
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # 3. Reject IP literals and non-standard ports
    hostname = parsed.hostname
    if hostname is None:
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # Check for IP literals (IPv4 or IPv6)
    if _is_ip_literal(hostname):
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # Reject non-standard ports
    port = parsed.port
    if port is not None and port != 443:
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )

    # 4. Validate hostname against allowlist
    if hostname not in _YOUTUBE_HOSTS:
        raise YouTubeDownloadError(
            "Invalid URL — only YouTube links are accepted"
        )


def _is_ip_literal(hostname: str) -> bool:
    """Check if a hostname is an IP literal (IPv4 or IPv6)."""
    # IPv6 brackets
    if hostname.startswith("[") or hostname.startswith("::"):
        return True

    # IPv4: all parts are digits
    parts = hostname.split(".")
    if all(part.isdigit() for part in parts):
        return True

    return False


def _map_ytdlp_error(error: Exception) -> str:
    """Map yt-dlp exceptions to user-friendly error messages.

    Order matters: more specific patterns (e.g. "live stream") must be checked
    before broader patterns (e.g. "is not available") to avoid false matches.
    """
    msg = str(error).lower()

    # Live stream check BEFORE unavailable (a live stream message may contain
    # "not available" and would otherwise be misclassified)
    if "live" in msg and ("stream" in msg or "event" in msg):
        return "Live streams cannot be used — please use a completed video"

    if "video unavailable" in msg or "is not available" in msg:
        return "This video is unavailable or has been removed"

    if "age" in msg and ("restrict" in msg or "gate" in msg or "confirm" in msg):
        return "Age-restricted videos are not supported"

    if (
        "geo" in msg
        or "country" in msg
        or "region" in msg
        or "not available in your" in msg
        or "blocked" in msg
    ):
        return "This video is not available in the server's region"

    if "no audio" in msg or "no suitable" in msg or "requested format" in msg:
        return "No audio track found in this video"

    if "429" in msg or "too many" in msg or "rate" in msg:
        return (
            "YouTube is temporarily limiting downloads. "
            "Try again in a minute"
        )

    if (
        "urlopen" in msg
        or "connection" in msg
        or "network" in msg
        or "timed out" in msg
        or "resolve" in msg
    ):
        return (
            "Failed to download from YouTube. Check the URL and try again"
        )

    # Fallback
    return "Failed to download from YouTube. Check the URL and try again"


async def _download_via_proxy(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> YouTubeAudioResult:
    """Download audio via the yt-proxy service (residential IP)."""
    import asyncio as _asyncio

    proxy_url = settings.youtube_proxy_service_url
    api_key = settings.youtube_proxy_api_key

    if progress_callback:
        progress_callback(0.05, "Connecting...")

    # Background ticker that bumps progress while the HTTP POST blocks
    download_done = _asyncio.Event()

    async def _tick_progress():
        if not progress_callback:
            return
        frac = 0.1
        step = 0.06
        while frac < 0.85:
            await _asyncio.sleep(2.5)
            if download_done.is_set():
                return
            frac = min(frac + step, 0.85)
            step *= 0.9  # decelerate toward cap
            progress_callback(frac, "Downloading...")

    ticker = _asyncio.create_task(_tick_progress())

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{proxy_url}/download",
                json={"url": url, "max_duration_seconds": settings.youtube_max_duration_seconds},
                headers={"X-API-Key": api_key} if api_key else {},
            )
    finally:
        download_done.set()
        await ticker

    if resp.status_code != 200:
        detail = resp.json().get("detail", "Download failed") if resp.headers.get("content-type", "").startswith("application/json") else "Download failed"
        raise YouTubeDownloadError(detail)

    title = resp.headers.get("X-Title", "Unknown")
    duration = float(resp.headers.get("X-Duration", "0"))

    file_id = uuid.uuid4().hex
    audio_path = output_dir / f"{file_id}.mp3"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(resp.content)

    if progress_callback:
        progress_callback(0.95, "Saving...")

    if progress_callback:
        progress_callback(1.0, "Done!")

    logger.info("Downloaded via proxy: title=%s, duration=%.1fs, size=%.1fMB, path=%s", title, duration, len(resp.content) / 1e6, audio_path)

    return YouTubeAudioResult(
        wav_path=audio_path,
        title=title,
        duration_seconds=duration,
        source_codec="unknown",
        source_bitrate=0,
    )


async def download_youtube_audio(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> YouTubeAudioResult:
    """Download audio from a YouTube video as a PCM WAV file.

    Args:
        url: YouTube video URL (must pass SSRF validation).
        output_dir: Directory to write the output WAV file.
        progress_callback: Optional callback ``(progress_fraction, message)``
            for reporting download progress. Throttled to max 1 call/sec or
            5% increments (whichever is less frequent).

    Returns:
        YouTubeAudioResult with path to WAV file and metadata.

    Raises:
        YouTubeDownloadError: On validation failure or download error.
    """
    # --- SSRF validation (must happen before yt-dlp touches the URL) ---
    validate_youtube_url(url)

    # Use remote proxy service if configured (bypasses datacenter IP blocks)
    if settings.youtube_proxy_service_url:
        return await _download_via_proxy(url, output_dir, progress_callback)

    max_duration = settings.youtube_max_duration_seconds

    # Unique filename to avoid wild filenames from video titles
    file_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)

    # We'll track the final output path through the postprocessor hook
    final_path_holder: list[Path] = []

    # --- Progress throttling state ---
    last_progress_time = 0.0
    last_progress_pct = -5.0  # ensure first callback fires

    def _progress_hook(d: dict) -> None:
        nonlocal last_progress_time, last_progress_pct

        if progress_callback is None:
            return

        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)

            if total > 0:
                pct = (downloaded / total) * 100.0
            else:
                pct = 0.0

            # Throttle: max 1/sec or 5% increments
            now = time.monotonic()
            if (now - last_progress_time < 1.0) and (pct - last_progress_pct < 5.0):
                return

            last_progress_time = now
            last_progress_pct = pct

            fraction = min(pct / 100.0, 0.95)  # cap at 95% during download
            progress_callback(fraction, f"Downloading: {pct:.0f}%")

        elif status == "finished":
            progress_callback(0.95, "Download complete, converting to WAV...")

    def _postprocessor_hook(d: dict) -> None:
        status = d.get("status")
        if status == "started":
            if progress_callback:
                progress_callback(0.96, "Converting to WAV...")
        elif status == "finished":
            info = d.get("info_dict", {})
            filepath = info.get("filepath")
            if filepath:
                final_path_holder.append(Path(filepath))
            if progress_callback:
                progress_callback(1.0, "Conversion complete")

    # yt-dlp options
    outtmpl = str(output_dir / f"{file_id}.%(ext)s")
    cookies_path = Path("/app/yt-cookies.txt")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_progress_hook],
        "postprocessor_hooks": [_postprocessor_hook],
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
    }
    if cookies_path.exists():
        ydl_opts["cookiefile"] = str(cookies_path)
    proxy = settings.youtube_proxy
    if proxy:
        ydl_opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to check duration before downloading
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise YouTubeDownloadError(
                    "This video is unavailable or has been removed"
                )

            duration = info.get("duration") or 0
            if duration > max_duration:
                raise YouTubeDownloadError(
                    f"Videos must be under {max_duration // 60} minutes"
                )

            # Check for live streams
            is_live = info.get("is_live", False)
            if is_live:
                raise YouTubeDownloadError(
                    "Live streams cannot be used — "
                    "please use a completed video"
                )

            title = info.get("title", "Unknown")

            # Determine source codec and bitrate from format info
            source_codec = "unknown"
            source_bitrate = 0
            # yt-dlp populates 'acodec' and 'abr' for the selected format
            acodec = info.get("acodec", "unknown")
            if acodec and acodec != "none":
                source_codec = acodec
            abr = info.get("abr")
            if abr:
                source_bitrate = int(abr)

            # Now download
            ydl.download([url])

    except YouTubeDownloadError:
        raise
    except yt_dlp.utils.DownloadError as e:
        user_msg = _map_ytdlp_error(e)
        raise YouTubeDownloadError(user_msg) from e
    except Exception as e:
        user_msg = _map_ytdlp_error(e)
        raise YouTubeDownloadError(user_msg) from e

    # Determine the output WAV path
    if final_path_holder:
        wav_path = final_path_holder[-1]
    else:
        # Fallback: postprocessor hook might not have fired; look for the file
        wav_path = output_dir / f"{file_id}.wav"

    if not wav_path.exists():
        raise YouTubeDownloadError(
            "Failed to download from YouTube. Check the URL and try again"
        )

    logger.info(
        "Downloaded YouTube audio: title=%s, duration=%.1fs, "
        "codec=%s, bitrate=%dkbps, path=%s",
        title,
        duration,
        source_codec,
        source_bitrate,
        wav_path,
    )

    return YouTubeAudioResult(
        wav_path=wav_path,
        title=title,
        duration_seconds=float(duration),
        source_codec=source_codec,
        source_bitrate=source_bitrate,
    )
