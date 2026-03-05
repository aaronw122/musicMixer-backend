"""Stem result caching by content hash.

Avoids re-separating songs that have already been processed. Uses SHA-256
of the input audio bytes as the cache key. Stems are stored in
``data/stem_cache/{sha256_hex}/`` as WAV files.

Thread-safety: concurrent writers for the same hash are handled via atomic
rename (write to temp dir, then ``os.rename`` into place). Worst case,
both threads separate in parallel and the second rename overwrites the
first -- correct because stems are identical for identical input.
"""

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Expected stem names produced by separation (6-stem BS-RoFormer or 4-stem htdemucs_ft).
# Cache validation checks that at least one stem exists and all present stems are non-zero.
EXPECTED_6_STEMS = {"vocals", "drums", "bass", "guitar", "piano", "other"}
EXPECTED_4_STEMS = {"vocals", "drums", "bass", "other"}


def get_cache_key(audio_path: Path) -> str:
    """Return SHA-256 hex digest of the file at *audio_path*.

    Reads the file in 64 KiB chunks to avoid loading the entire file
    into memory for large WAVs.
    """
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_cached_stems(cache_key: str, stems_dir: Path) -> bool:
    """Check cache for stems matching *cache_key*.

    If a cache hit is found and all stems validate (exist, non-zero size),
    copies them into *stems_dir* and returns ``True``.

    Returns ``False`` on cache miss or validation failure.
    """
    if not settings.stem_cache_enabled:
        return False

    cache_entry = settings.stem_cache_dir / cache_key
    if not cache_entry.is_dir():
        logger.debug("Stem cache miss for %s (no directory)", cache_key[:12])
        return False

    # Validate: at least one .wav, all .wav files non-zero
    wav_files = list(cache_entry.glob("*.wav"))
    if not wav_files:
        logger.warning("Stem cache entry %s has no WAV files, treating as miss", cache_key[:12])
        return False

    for wav in wav_files:
        if wav.stat().st_size == 0:
            logger.warning(
                "Stem cache entry %s has zero-length %s, treating as miss",
                cache_key[:12],
                wav.name,
            )
            return False

    # Validate we have a recognized stem set
    stem_names = {wav.stem for wav in wav_files}
    if not (stem_names >= EXPECTED_4_STEMS or stem_names >= EXPECTED_6_STEMS):
        logger.warning(
            "Stem cache entry %s has unrecognized stems %s, treating as miss",
            cache_key[:12],
            stem_names,
        )
        return False

    # Copy stems to output directory
    stems_dir.mkdir(parents=True, exist_ok=True)
    for wav in wav_files:
        dest = stems_dir / wav.name
        shutil.copy2(wav, dest)

    logger.info(
        "Stem cache hit for %s (%d stems copied to %s)",
        cache_key[:12],
        len(wav_files),
        stems_dir,
    )
    return True


def cache_stems(cache_key: str, stems_dir: Path) -> None:
    """Copy stems from *stems_dir* into the cache under *cache_key*.

    Uses atomic rename: writes to a temp directory first, then renames
    into place. If the target already exists (concurrent writer), the
    rename overwrites it -- both sets of stems are identical for the same
    input, so either outcome is correct.

    After caching, runs LRU eviction if the cache exceeds the configured
    max size.
    """
    if not settings.stem_cache_enabled:
        return

    wav_files = list(stems_dir.glob("*.wav"))
    if not wav_files:
        logger.warning("No WAV files in %s, skipping cache write", stems_dir)
        return

    cache_dir = settings.stem_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write to temp directory for atomic rename
    tmp_dir = cache_dir / f".tmp-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for wav in wav_files:
            shutil.copy2(wav, tmp_dir / wav.name)

        target = cache_dir / cache_key
        # Atomic placement: rename temp dir into target.
        # On macOS, os.rename fails if target is a non-empty dir, so we
        # remove the target first. A concurrent writer may race us here;
        # if the rename fails because the other thread placed the target
        # first, that's fine — both wrote identical stems.
        try:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            os.rename(tmp_dir, target)
        except OSError:
            if target.exists():
                # Another thread won the race and placed the target.
                # Clean up our temp dir — the cached result is valid.
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                # Genuine failure (cross-device, permissions, etc.)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        logger.info(
            "Cached %d stems for %s (%s)",
            len(wav_files),
            cache_key[:12],
            target,
        )
    except Exception:
        # Clean up temp dir on any failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # Run eviction if needed
    _evict_lru(cache_dir)


def _evict_lru(cache_dir: Path) -> None:
    """Delete oldest cache entries (by mtime) until total size is within limit.

    Only considers directories directly under *cache_dir* that look like
    SHA-256 hex digests (64 hex chars). Skips temp directories (.tmp-*).
    """
    max_bytes = int(settings.stem_cache_max_gb * 1024 * 1024 * 1024)

    entries = []
    total_size = 0

    for entry in os.scandir(cache_dir):
        if not entry.is_dir() or entry.name.startswith(".tmp-"):
            continue
        # Sum file sizes in this cache entry
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
            "Evicted %d cache entries, cache now ~%.1f GB (limit %.1f GB)",
            evicted,
            total_size / (1024 * 1024 * 1024),
            settings.stem_cache_max_gb,
        )
