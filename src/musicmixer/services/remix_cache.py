"""Remix output caching by composite content hash.

Avoids re-running the full pipeline for identical (song_a, song_b, prompt)
requests. The cache key is SHA-256 of the concatenation of the two song
content hashes and the normalized prompt.

Cache layout:
    data/remix_cache/{sha256_hex}/remix.mp3

Thread-safety: writes use atomic rename (write to temp dir, then
``os.rename`` into place), same pattern as stem_cache.py.
"""

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path

from musicmixer.config import settings
from musicmixer.services.stem_cache import get_cache_key

logger = logging.getLogger(__name__)


def compute_remix_cache_key(song_a_path: Path, song_b_path: Path, prompt: str) -> str:
    """Compute an order-aware cache key for a remix request.

    The key is SHA-256 of ``song_a_hash + ":" + song_b_hash + ":" + normalized_prompt``.
    Swapping song_a and song_b produces a different key (intentional -- the
    pipeline assigns vocals from A and instrumentals from B).
    """
    song_a_hash = get_cache_key(song_a_path)
    song_b_hash = get_cache_key(song_b_path)
    normalized_prompt = prompt.strip().lower()
    composite = f"{song_a_hash}:{song_b_hash}:{normalized_prompt}"
    return hashlib.sha256(composite.encode("utf-8")).hexdigest()


def get_cached_remix(cache_key: str, cache_dir: Path) -> Path | None:
    """Return the path to a cached remix MP3 if a valid cache hit exists.

    Returns ``None`` on cache miss, missing file, or zero-size file.
    """
    cache_entry = cache_dir / cache_key
    remix_path = cache_entry / "remix.mp3"

    try:
        if not remix_path.is_file():
            logger.debug("Remix cache miss for %s (no file)", cache_key[:12])
            return None

        if remix_path.stat().st_size == 0:
            logger.warning(
                "Remix cache entry %s has zero-length remix.mp3, treating as miss",
                cache_key[:12],
            )
            return None
    except FileNotFoundError:
        # Concurrent eviction removed the entry between our checks
        logger.debug(
            "Remix cache entry %s disappeared during read (concurrent eviction), treating as miss",
            cache_key[:12],
        )
        return None

    logger.info("Remix cache hit for %s (%s)", cache_key[:12], remix_path)
    return remix_path


def cache_remix(cache_key: str, remix_mp3_path: Path, cache_dir: Path) -> None:
    """Copy the remix MP3 into the cache under *cache_key*.

    Uses atomic rename: writes to a temp directory first, then renames
    into place. On failure, logs a warning but does not raise -- cache
    write failures must never break the pipeline.

    After caching, runs LRU eviction if the cache exceeds the configured
    max size.
    """
    if not remix_mp3_path.is_file() or remix_mp3_path.stat().st_size == 0:
        logger.warning(
            "Remix MP3 at %s is missing or empty, skipping cache write",
            remix_mp3_path,
        )
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = cache_dir / f".tmp-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(remix_mp3_path, tmp_dir / "remix.mp3")

        target = cache_dir / cache_key
        try:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.rename(tmp_dir, target)
        except OSError:
            if target.exists():
                # Another thread won the race -- clean up our temp dir
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        logger.info("Cached remix for %s (%s)", cache_key[:12], target)
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.warning(
            "Failed to cache remix for %s", cache_key[:12], exc_info=True,
        )
        return

    # Run eviction if needed
    try:
        _evict_lru(cache_dir, settings.remix_cache_max_gb)
    except Exception:
        logger.warning("Remix cache eviction failed", exc_info=True)


def _evict_lru(cache_dir: Path, max_gb: float) -> None:
    """Delete oldest cache entries (by mtime) until total size is within limit.

    Only considers directories directly under *cache_dir* that look like
    SHA-256 hex digests (64 hex chars). Skips temp directories (.tmp-*).
    """
    max_bytes = int(max_gb * 1024 * 1024 * 1024)

    entries = []
    total_size = 0

    for entry in os.scandir(cache_dir):
        if not entry.is_dir() or entry.name.startswith(".tmp-"):
            continue
        entry_size = sum(
            f.stat().st_size
            for f in Path(entry.path).iterdir()
            if f.is_file()
        )
        entry_mtime = entry.stat().st_mtime
        entries.append((entry.path, entry_mtime, entry_size))
        total_size += entry_size

    if total_size <= max_bytes:
        return

    # Sort oldest first (lowest mtime)
    entries.sort(key=lambda e: e[1])

    evicted = 0
    for path, _mtime, size in entries:
        if total_size <= max_bytes:
            break
        shutil.rmtree(path, ignore_errors=True)
        total_size -= size
        evicted += 1

    if evicted:
        logger.info(
            "Evicted %d remix cache entries, cache now ~%.1f GB (limit %.1f GB)",
            evicted,
            total_size / (1024 * 1024 * 1024),
            max_gb,
        )
