"""Record shelf API endpoints."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import urllib.parse
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from musicmixer.config import settings
from musicmixer.services.youtube import YouTubeDownloadError, validate_youtube_url

router = APIRouter()

_shelf_lock = threading.Lock()
_NOEMBED_URL = "https://noembed.com/embed"


class ShelfRecord(BaseModel):
    id: str
    youtube_url: str
    title: str
    artist: str
    thumbnail_url: str
    sleeve_image_url: str
    added_at: str
    is_curated: bool


class ShelfResponse(BaseModel):
    records: list[ShelfRecord]


class AddShelfRecordRequest(BaseModel):
    youtube_url: str


def _shelf_path() -> Path:
    return settings.data_dir / "shelf.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_for_api(url: str) -> None:
    try:
        validate_youtube_url(url)
    except YouTubeDownloadError as exc:
        raise HTTPException(422, str(exc)) from exc


def _extract_video_id(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path_parts = [part for part in parsed.path.split("/") if part]

    if hostname == "youtu.be" and path_parts:
        return path_parts[0]

    query = urllib.parse.parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]

    if path_parts and path_parts[0] in {"shorts", "embed"} and len(path_parts) > 1:
        return path_parts[1]

    return None


def _normalize_youtube_url(url: str) -> str:
    _validate_for_api(url)
    video_id = _extract_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.urlencode(sorted(urllib.parse.parse_qsl(parsed.query)))
    return urllib.parse.urlunparse(
        ("https", parsed.hostname or "", parsed.path.rstrip("/"), "", query, "")
    )


def _extract_artist(title: str) -> str:
    if " - " in title:
        artist = title.split(" - ", 1)[0].strip()
        if artist:
            return artist
    return "Unknown Artist"


def _fetch_noembed_metadata(youtube_url: str) -> dict[str, str]:
    params = urllib.parse.urlencode({"url": youtube_url})

    try:
        response = httpx.get(
            f"{_NOEMBED_URL}?{params}",
            headers={"User-Agent": "musicMixer/0.1"},
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HTTPException(502, "Failed to fetch YouTube metadata") from exc

    title = str(payload.get("title") or "").strip()
    if not title:
        raise HTTPException(422, "Unable to read YouTube video metadata")

    return {
        "title": title,
        "thumbnail_url": str(payload.get("thumbnail_url") or "").strip(),
    }


def _record_from_seed(
    *,
    youtube_url: str,
    title: str,
    thumbnail_url: str,
    added_at: str,
) -> dict[str, Any]:
    record_id = str(uuid.uuid5(uuid.NAMESPACE_URL, youtube_url))
    return {
        "id": record_id,
        "youtube_url": youtube_url,
        "title": title,
        "artist": _extract_artist(title),
        "thumbnail_url": thumbnail_url,
        "sleeve_image_url": f"/api/shelf/sleeve/{record_id}",
        "added_at": added_at,
        "is_curated": True,
    }


def _starter_records() -> list[dict[str, Any]]:
    base_time = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    seeds = [
        (
            "https://www.youtube.com/watch?v=yYl597lyZvo",
            "House of Pain - Jump Around",
            "https://i.ytimg.com/vi/yYl597lyZvo/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=qchPLaiKocI",
            "Kool & The Gang - Get Down On It",
            "https://i.ytimg.com/vi/qchPLaiKocI/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=DUZ7-C0wPF4",
            "MF DOOM - Gazzillion Ear",
            "https://i.ytimg.com/vi/DUZ7-C0wPF4/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=k9Yz0MC4bkQ",
            "Grateful Dead - Althea",
            "https://i.ytimg.com/vi/k9Yz0MC4bkQ/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=QNAVrQ96mpA",
            "Roy Orbison - You Got It",
            "https://i.ytimg.com/vi/QNAVrQ96mpA/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=OPf0YbXqDm0",
            "Mark Ronson - Uptown Funk ft. Bruno Mars",
            "https://i.ytimg.com/vi/OPf0YbXqDm0/hqdefault.jpg",
        ),
        (
            "https://www.youtube.com/watch?v=QDYfEBY9NM4",
            "The Beatles - Let It Be",
            "https://i.ytimg.com/vi/QDYfEBY9NM4/hqdefault.jpg",
        ),
    ]

    return [
        _record_from_seed(
            youtube_url=url,
            title=title,
            thumbnail_url=thumbnail,
            added_at=(base_time - timedelta(minutes=index)).isoformat().replace(
                "+00:00", "Z"
            ),
        )
        for index, (url, title, thumbnail) in enumerate(seeds)
    ]


def _read_shelf_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        records = _starter_records()
        _write_shelf_unlocked(path, records)
        return records

    with path.open("r", encoding="utf-8") as shelf_file:
        payload = json.load(shelf_file)

    records = payload.get("records")
    if not isinstance(records, list):
        raise HTTPException(500, "Shelf data is malformed")

    # Sync with seed list: add missing seeds, remove stale curated records
    seed_urls = {s["youtube_url"] for s in _starter_records()}
    existing_urls = {r["youtube_url"] for r in records}
    missing = [s for s in _starter_records() if s["youtube_url"] not in existing_urls]
    stale = [r for r in records if r.get("is_curated") and r["youtube_url"] not in seed_urls]
    changed = bool(missing) or bool(stale)
    if stale:
        stale_urls = {r["youtube_url"] for r in stale}
        records = [r for r in records if r["youtube_url"] not in stale_urls]
    if missing:
        records.extend(missing)
    if changed:
        _write_shelf_unlocked(path, records)

    return records


def _write_shelf_unlocked(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": records}

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        json.dump(payload, temp_file, indent=2)
        temp_file.write("\n")
        temp_file.flush()

    temp_path.replace(path)


def _load_records() -> list[dict[str, Any]]:
    with _shelf_lock:
        return list(_read_shelf_unlocked(_shelf_path()))


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda record: record["added_at"], reverse=True)


@router.get("/shelf", response_model=ShelfResponse)
def list_shelf() -> ShelfResponse:
    records = _load_records()
    return ShelfResponse(
        records=[ShelfRecord(**record) for record in _sorted_records(records)]
    )


def ensure_on_shelf(youtube_url: str) -> ShelfRecord:
    """Add a YouTube URL to the shelf if it isn't already there.

    Returns the ShelfRecord (existing or newly created). Safe to call
    from any context — silently returns the existing record on duplicates.
    """
    normalized_url = _normalize_youtube_url(youtube_url)

    # Fast path: already on the shelf
    with _shelf_lock:
        records = _read_shelf_unlocked(_shelf_path())
        for record in records:
            if record["youtube_url"] == normalized_url:
                return ShelfRecord(**record)

    metadata = _fetch_noembed_metadata(normalized_url)

    with _shelf_lock:
        path = _shelf_path()
        records = _read_shelf_unlocked(path)
        # Re-check after fetching metadata (another request may have added it)
        for record in records:
            if record["youtube_url"] == normalized_url:
                return ShelfRecord(**record)

        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "youtube_url": normalized_url,
            "title": metadata["title"],
            "artist": _extract_artist(metadata["title"]),
            "thumbnail_url": metadata["thumbnail_url"],
            "sleeve_image_url": f"/api/shelf/sleeve/{record_id}",
            "added_at": _now_iso(),
            "is_curated": False,
        }
        records.append(record)
        _write_shelf_unlocked(path, records)

    return ShelfRecord(**record)


@router.post("/shelf", response_model=ShelfRecord)
def add_shelf_record(body: AddShelfRecordRequest) -> ShelfRecord:
    return ensure_on_shelf(body.youtube_url)


def _sleeve_svg(record: dict[str, Any]) -> str:
    digest = hashlib.sha256(record["youtube_url"].encode("utf-8")).digest()
    colors = [
        f"#{digest[index]:02x}{digest[index + 1]:02x}{digest[index + 2]:02x}"
        for index in (0, 3, 6, 9)
    ]
    variant = digest[12] % 3
    title = escape(str(record["title"]), {'"': "&quot;"})
    artist = escape(str(record["artist"]), {'"': "&quot;"})

    if variant == 0:
        pattern = (
            f'<circle cx="160" cy="160" r="122" fill="none" stroke="{colors[2]}" '
            'stroke-width="18" opacity="0.58"/>'
            f'<circle cx="160" cy="160" r="72" fill="none" stroke="{colors[3]}" '
            'stroke-width="12" opacity="0.72"/>'
            '<circle cx="160" cy="160" r="18" fill="#f7f1df" opacity="0.9"/>'
        )
    elif variant == 1:
        pattern = "".join(
            f'<rect x="{x}" y="-80" width="28" height="480" fill="{colors[(x // 40) % 4]}" '
            'opacity="0.46" transform="rotate(34 160 160)"/>'
            for x in range(-120, 440, 40)
        )
    else:
        pattern = (
            f'<polygon points="18,292 132,48 270,82" fill="{colors[2]}" opacity="0.62"/>'
            f'<polygon points="302,26 206,302 92,228" fill="{colors[3]}" opacity="0.58"/>'
            '<circle cx="236" cy="116" r="50" fill="#f7f1df" opacity="0.48"/>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" viewBox="0 0 320 320" role="img" aria-label="{title}">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="{colors[0]}"/>
      <stop offset="1" stop-color="{colors[1]}"/>
    </linearGradient>
  </defs>
  <rect width="320" height="320" rx="10" fill="url(#bg)"/>
  {pattern}
  <rect x="0" y="218" width="320" height="102" fill="#111111" opacity="0.38"/>
  <text x="24" y="262" fill="#fff8e8" font-family="Arial, sans-serif" font-size="23" font-weight="700">{title}</text>
  <text x="24" y="292" fill="#fff8e8" font-family="Arial, sans-serif" font-size="17" opacity="0.86">{artist}</text>
</svg>
"""


@router.get("/shelf/sleeve/{record_id}")
def get_shelf_sleeve(record_id: str) -> Response:
    records = _load_records()
    record = next((item for item in records if item["id"] == record_id), None)
    if record is None:
        raise HTTPException(404, "Record not found")

    return Response(
        content=_sleeve_svg(record),
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
