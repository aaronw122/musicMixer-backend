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


# Wire-contract values for the SSE `error` event's `error_class` field.
# Consumed by frontend PR #76: `error_class: "transient" | "permanent"`.
#   - transient  → retryable (403 / throttle / timeout / 5xx). Worth a retry.
#   - permanent  → not retryable (video unavailable/private/removed/age-gated).
ERROR_CLASS_TRANSIENT = "transient"
ERROR_CLASS_PERMANENT = "permanent"


class YouTubeDownloadError(Exception):
    """User-facing error from YouTube download operations.

    Carries an ``error_class`` (``"transient"`` | ``"permanent"``) so the SSE
    ``error`` event can tell the frontend whether a retry is worth offering.
    Defaults to ``"permanent"`` — the safe default is to NOT promise a retry for
    a failure we can't prove is transient (mirrors the frontend's own default in
    PR #76: a missing ``error_class`` is treated as permanent).
    """

    def __init__(self, message: str, error_class: str = ERROR_CLASS_PERMANENT) -> None:
        super().__init__(message)
        self.error_class = error_class


def classify_youtube_error(*, status_code: int | None, detail: str) -> str:
    """Classify a download failure as ``transient`` (retryable) or ``permanent``.

    This is the SINGLE choke-point for transient/permanent classification. It is
    net-new logic — NOT a reuse of ``_map_ytdlp_error`` (which returns DISPLAY
    strings, not a retry class). It borrows that function's string-matching
    *patterns* but maps to the wire-contract ``error_class`` instead.

    Best-effort from what the backend actually has today: the proxy collapses
    every yt-dlp error to a fixed ``HTTPException(400, "Failed to download from
    YouTube")`` (``yt-proxy/main.py``), so the only signals available are the
    proxy's HTTP ``status_code`` and its ``detail`` string. We classify from
    those.

    TODO(proxy-forwarded-category): once ``yt-proxy`` forwards a structured
    category (e.g. a raw yt-dlp error string and/or an explicit ``error_class``
    in the JSON body), plug it in HERE — prefer the forwarded category over the
    status/detail heuristics below. This is the seam; do not classify anywhere
    else. (Proxy change is OUT OF SCOPE for this wave.)

    Args:
        status_code: HTTP status from the proxy response, or None when the
            failure was raised before/without an HTTP response (e.g. a network
            error talking to the proxy itself).
        detail: The proxy's ``detail`` string (or any error text available).

    Returns:
        ``ERROR_CLASS_TRANSIENT`` or ``ERROR_CLASS_PERMANENT``.
    """
    msg = (detail or "").lower()

    # --- Permanent signals first (a removed/private/age-gated video will never
    #     succeed on retry, so these must win even if a transient-looking token
    #     also appears in the text). ---
    # Live streams: not a download target (mirrors _map_ytdlp_error ordering —
    # checked before "not available" since a live message may contain it).
    if "live" in msg and ("stream" in msg or "event" in msg):
        return ERROR_CLASS_PERMANENT
    if (
        "video unavailable" in msg
        or "is not available" in msg
        or "has been removed" in msg
        or "private video" in msg
        or "this video is private" in msg
        or "removed by the user" in msg
        or "account associated" in msg  # "account...has been terminated"
        or "terminated" in msg
    ):
        return ERROR_CLASS_PERMANENT
    if "age" in msg and ("restrict" in msg or "gate" in msg or "confirm" in msg):
        return ERROR_CLASS_PERMANENT
    if "no audio" in msg or "no suitable" in msg or "requested format" in msg:
        return ERROR_CLASS_PERMANENT
    # Geo-blocking is permanent from THIS server's vantage point — a retry from
    # the same IP/region won't clear it.
    if (
        "not available in your" in msg
        or "not available in the server" in msg
        or "geo" in msg
        or "country" in msg
        or "region" in msg
    ):
        return ERROR_CLASS_PERMANENT

    # --- Transient signals: throttle / forbidden / timeout / network / 5xx. ---
    # These are exactly the incident class (a 403 that succeeded a beat later).
    if (
        "403" in msg
        or "forbidden" in msg
        or "429" in msg
        or "too many" in msg
        or "throttl" in msg
        or "rate" in msg
        or "timed out" in msg
        or "timeout" in msg
        or "temporarily" in msg
        or "connection" in msg
        or "network" in msg
        or "urlopen" in msg
        or "resolve" in msg
    ):
        return ERROR_CLASS_TRANSIENT

    # --- Fall back to the HTTP status code when the detail text is unhelpful
    #     (it usually is — the proxy sends a generic "Failed to download"). ---
    if status_code is not None:
        # 5xx and 429 are server-side / throttle → retryable.
        if status_code >= 500 or status_code == 429:
            return ERROR_CLASS_TRANSIENT
        # The proxy collapses transient yt-dlp 403s into its own 400. We can't
        # distinguish a transient 403 from a permanent "unavailable" purely from
        # a generic 400 + opaque detail, so we bias the generic-400 case toward
        # TRANSIENT: the observed incident (HTTP 403 on the media stream) is the
        # dominant generic-400 cause, and an over-eager retry is cheaper than
        # wrongly telling the user a recoverable video is gone forever.
        if status_code == 400:
            return ERROR_CLASS_TRANSIENT

    # Truly unknown → permanent (safe default: don't promise a retry we can't
    # justify).
    return ERROR_CLASS_PERMANENT


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
    except httpx.RequestError as exc:
        # Transport-level failure talking to the proxy (timeout, connection
        # reset, DNS) — there's no HTTP response to read. These are inherently
        # retryable, so classify transient (status_code=None).
        logger.warning("Proxy request error (transient): %s", exc)
        raise YouTubeDownloadError(
            "Failed to download from YouTube. Check the URL and try again",
            error_class=ERROR_CLASS_TRANSIENT,
        ) from exc
    finally:
        download_done.set()
        await ticker

    if resp.status_code != 200:
        detail = resp.json().get("detail", "Download failed") if resp.headers.get("content-type", "").startswith("application/json") else "Download failed"
        # Best-effort transient/permanent classification from the only signals
        # the proxy gives us today (status code + detail string). See the TODO
        # seam in classify_youtube_error() for the future proxy-forwarded category.
        error_class = classify_youtube_error(status_code=resp.status_code, detail=detail)
        logger.warning(
            "Proxy download failed: status=%s class=%s detail=%r",
            resp.status_code, error_class, detail,
        )
        raise YouTubeDownloadError(detail, error_class=error_class)

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


def _copy_cached_audio_to_session(
    cached_path: Path,
    output_dir: Path,
    meta: dict,
) -> YouTubeAudioResult:
    """Build a YouTubeAudioResult from a cached audio file, copied into the session.

    The pipeline reads from a session-scoped path, so the cached artifact is
    copied (not referenced) — the cache stays immutable and the session owns its
    copy.
    """
    import shutil as _shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    session_copy = output_dir / f"{uuid.uuid4().hex}{cached_path.suffix}"
    _shutil.copy2(cached_path, session_copy)
    return YouTubeAudioResult(
        wav_path=session_copy,
        title=meta.get("title", "Unknown"),
        duration_seconds=float(meta.get("duration_seconds", 0.0)),
        source_codec=meta.get("source_codec", "unknown"),
        source_bitrate=int(meta.get("source_bitrate", 0)),
    )


async def download_youtube_audio(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    video_id: str | None = None,
) -> YouTubeAudioResult:
    """Download audio from a YouTube video, using the per-video_id audio cache.

    Tiered behavior (Part A): when ``video_id`` is known, check the audio cache
    first — a hit skips YouTube entirely (Tier 2/3 bridge). On a miss, download
    once and persist to the cache so the next request reuses it. A per-video_id
    single-flight lock prevents two concurrent requests for the same video from
    both downloading.

    Args:
        url: YouTube video URL (must pass SSRF validation).
        output_dir: Directory to write the output audio file.
        progress_callback: Optional callback ``(progress_fraction, message)``.
        video_id: YouTube video ID for cache keying. When None, caching is
            skipped (download always runs) — preserves back-compat for callers
            that don't supply it.

    Returns:
        YouTubeAudioResult with path to the audio file and metadata.

    Raises:
        YouTubeDownloadError: On validation failure or download error.
    """
    # --- SSRF validation (must happen before yt-dlp touches the URL) ---
    validate_youtube_url(url)

    # No cache key → always download (back-compat bypass for callers w/o video_id).
    if video_id is None:
        return await _download_youtube_audio_uncached(url, output_dir, progress_callback)

    # Fast path: cache hit before taking the single-flight lock.
    cached = _cached_audio_result(video_id, output_dir, progress_callback)
    if cached is not None:
        return cached

    # Otherwise serialize concurrent same-video requests and download once.
    return await _download_with_single_flight(url, output_dir, progress_callback, video_id)


def _cached_audio_result(
    video_id: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None,
    *,
    after_wait: bool = False,
) -> YouTubeAudioResult | None:
    """Return a session-scoped result from the audio cache, or None on a miss.

    Used for both the pre-lock check and the in-lock re-check; ``after_wait``
    only affects the log message.
    """
    from musicmixer.services.song_cache import get_cached_audio

    hit = get_cached_audio(video_id)
    if hit is None:
        return None
    cached_path, meta = hit
    if after_wait:
        logger.info("Audio cache HIT for video %s after single-flight wait", video_id)
    else:
        logger.info("Audio cache HIT for video %s (skipping YouTube)", video_id)
    if progress_callback:
        progress_callback(1.0, "Already had this one!")
    return _copy_cached_audio_to_session(cached_path, output_dir, meta)


def _cache_download_result(video_id: str, result: YouTubeAudioResult) -> None:
    """Persist a freshly downloaded result to the audio cache (atomic).

    Best-effort: a cache failure must NOT fail the download the user is waiting
    on, so any exception is logged and swallowed.
    """
    from musicmixer.services.song_cache import cache_audio

    try:
        audio_bytes = result.wav_path.read_bytes()
        cache_audio(
            video_id,
            audio_bytes,
            meta={
                "title": result.title,
                "duration_seconds": result.duration_seconds,
                "source_codec": result.source_codec,
                "source_bitrate": result.source_bitrate,
            },
        )
    except Exception:
        logger.warning("Failed to persist audio cache for video %s", video_id, exc_info=True)


async def _download_with_single_flight(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None,
    video_id: str,
) -> YouTubeAudioResult:
    """Single-flight download: serialize concurrent downloads of the same video.

    The second caller waits on the per-video_id lock, then re-checks the cache
    (the load-bearing second check that prevents a double download) and reuses
    the first caller's result. On a real miss, download once and persist.
    """
    from musicmixer.services.song_cache import single_flight

    with single_flight(video_id):
        cached = _cached_audio_result(
            video_id, output_dir, progress_callback, after_wait=True
        )
        if cached is not None:
            return cached

        result = await _download_youtube_audio_uncached(url, output_dir, progress_callback)
        _cache_download_result(video_id, result)
        return result


async def _download_youtube_audio_uncached(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[float, str], None] | None = None,
) -> YouTubeAudioResult:
    """Download audio from YouTube without any cache interaction (the raw path)."""
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
        # Direct (non-proxy) path: we have the RAW yt-dlp error string here, a
        # richer signal than the proxy path gets. Classify from it so this path
        # also emits a correct error_class on the SSE error event.
        user_msg = _map_ytdlp_error(e)
        error_class = classify_youtube_error(status_code=None, detail=str(e))
        raise YouTubeDownloadError(user_msg, error_class=error_class) from e
    except Exception as e:
        user_msg = _map_ytdlp_error(e)
        error_class = classify_youtube_error(status_code=None, detail=str(e))
        raise YouTubeDownloadError(user_msg, error_class=error_class) from e

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
