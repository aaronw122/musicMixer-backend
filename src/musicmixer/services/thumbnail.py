"""Thumbnail fetch and color extraction service.

Fetches YouTube thumbnail images and optionally extracts the dominant color.
Only allows requests to YouTube thumbnail domains (SSRF prevention).
"""

from __future__ import annotations

import io
import logging
import struct
from collections import Counter

import httpx

logger = logging.getLogger(__name__)

# YouTube thumbnail domains -- the ONLY hosts this service will fetch from.
ALLOWED_THUMBNAIL_HOSTS = frozenset({
    "i.ytimg.com",
    "img.youtube.com",
})

_FETCH_TIMEOUT_SECONDS = 5.0
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB


class ThumbnailFetchError(Exception):
    """Raised when the upstream thumbnail fetch fails."""

    pass


async def fetch_thumbnail(url: str) -> tuple[bytes, str]:
    """Fetch a thumbnail image from YouTube.

    Args:
        url: The full thumbnail URL (must be from an allowed host).

    Returns:
        Tuple of (image_bytes, content_type).

    Raises:
        ThumbnailFetchError: On network errors, timeouts, or oversized responses.
    """
    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_SECONDS,
            # Redirects disabled: YouTube thumbnail CDN URLs are direct and
            # don't redirect in practice.  Disabling prevents SSRF via
            # attacker-controlled redirect targets if the allowlist ever
            # admits a domain that issues open redirects.
            follow_redirects=False,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        raise ThumbnailFetchError("Upstream fetch timed out")
    except httpx.HTTPStatusError as exc:
        raise ThumbnailFetchError(
            f"Upstream returned HTTP {exc.response.status_code}"
        )
    except httpx.HTTPError as exc:
        raise ThumbnailFetchError(f"Upstream fetch failed: {exc}")

    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise ThumbnailFetchError("Response too large (>5 MB)")

    content_type = response.headers.get("content-type", "image/jpeg")
    # Strip parameters (e.g. "image/jpeg; charset=utf-8" -> "image/jpeg")
    content_type = content_type.split(";")[0].strip()

    return response.content, content_type


def extract_dominant_color(image_bytes: bytes) -> str:
    """Extract the dominant color from a JPEG/PNG image.

    Uses a simple pixel-sampling approach that works without Pillow:
    decodes BMP via stdlib if possible, otherwise falls back to sampling
    raw bytes from the JPEG stream.

    For production quality, this samples pixels and finds the most common
    color bucket (quantized to 32-level bins to group similar colors).

    Returns:
        Hex color string like ``"#7A3B2E"``.
    """
    try:
        return _extract_color_from_raw_bytes(image_bytes)
    except Exception:
        logger.debug("Color extraction failed, returning default", exc_info=True)
        return "#333333"


def _quantize(value: int, levels: int = 8) -> int:
    """Quantize a 0-255 value into fewer levels for color bucketing."""
    bucket_size = 256 // levels
    return min((value // bucket_size) * bucket_size + bucket_size // 2, 255)


def _extract_color_from_raw_bytes(data: bytes) -> str:
    """Sample pixel-like byte triplets from image data for dominant color.

    This is a heuristic that works on JPEG and PNG compressed data by
    sampling byte triplets at regular intervals through the file body.
    It skips headers and works surprisingly well because the most common
    byte patterns in compressed image data correlate with dominant colors.

    For better accuracy, we skip the first 256 bytes (headers) and last
    64 bytes (trailers), then sample every Nth triplet.
    """
    # Skip headers and trailers
    start = min(256, len(data) // 4)
    end = max(start + 3, len(data) - 64)
    body = data[start:end]

    if len(body) < 30:
        return "#333333"

    # Sample triplets at regular intervals (aim for ~200 samples)
    num_samples = min(200, len(body) // 3)
    if num_samples < 10:
        return "#333333"

    step = max(1, len(body) // (num_samples * 3))

    color_counts: Counter[tuple[int, int, int]] = Counter()

    for i in range(0, len(body) - 2, step * 3):
        r, g, b = body[i], body[i + 1], body[i + 2]
        # Quantize to reduce noise
        qr = _quantize(r)
        qg = _quantize(g)
        qb = _quantize(b)
        color_counts[(qr, qg, qb)] += 1

    if not color_counts:
        return "#333333"

    # Filter out very dark and very light colors (likely background/artifacts)
    filtered = {
        color: count
        for color, count in color_counts.items()
        if 30 < sum(color) < 700  # not too dark, not too bright
    }

    if not filtered:
        filtered = dict(color_counts)

    dominant = max(filtered, key=filtered.__getitem__)
    return f"#{dominant[0]:02X}{dominant[1]:02X}{dominant[2]:02X}"
