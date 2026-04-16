"""Thumbnail proxy endpoint.

Proxies YouTube thumbnail images with CORS headers so the frontend can use
them as WebGL textures and extract dominant colors via canvas.  Only allows
YouTube thumbnail domains (i.ytimg.com, img.youtube.com) to prevent SSRF.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from musicmixer.services.thumbnail import (
    fetch_thumbnail,
    extract_dominant_color,
    ThumbnailFetchError,
    ALLOWED_THUMBNAIL_HOSTS,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_URL_LENGTH = 500


def _validate_thumbnail_url(url: str) -> None:
    """Validate that *url* points to a YouTube thumbnail domain.

    Raises HTTPException on failure.
    """
    if len(url) > _MAX_URL_LENGTH:
        raise HTTPException(status_code=400, detail="URL too long (max 500 characters)")

    try:
        parsed = urlparse(url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL")

    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=403,
            detail="Only YouTube thumbnail URLs are allowed",
        )

    hostname = parsed.hostname
    if hostname is None or hostname not in ALLOWED_THUMBNAIL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="Only YouTube thumbnail URLs are allowed",
        )


@router.get("/thumbnail-proxy")
async def thumbnail_proxy(url: str = Query(..., description="YouTube thumbnail URL to proxy")):
    """Proxy a YouTube thumbnail image with CORS headers.

    Fetches the image from YouTube and returns it with appropriate
    Cache-Control and Access-Control-Allow-Origin headers so the
    frontend can use it in WebGL and canvas operations.
    """
    _validate_thumbnail_url(url)

    try:
        image_bytes, content_type = await fetch_thumbnail(url)
    except ThumbnailFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return Response(
        content=image_bytes,
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/thumbnail-color")
async def thumbnail_color(url: str = Query(..., description="YouTube thumbnail URL")):
    """Extract the dominant color from a YouTube thumbnail.

    Returns ``{"color": "#RRGGBB"}``.
    """
    _validate_thumbnail_url(url)

    try:
        image_bytes, _ = await fetch_thumbnail(url)
    except ThumbnailFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    hex_color = extract_dominant_color(image_bytes)
    return {"color": hex_color}
