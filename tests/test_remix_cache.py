"""Tests for musicmixer.services.remix_cache -- remix output caching."""

import os
import shutil
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_mp3(path: Path, size: int = 1024) -> None:
    """Write a dummy file at *path* (doesn't need to be a real MP3 for cache tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff" * size)


def _make_dummy_song(path: Path, content: bytes | None = None) -> None:
    """Write a dummy song file for cache key computation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else b"song-data-" + path.name.encode())


# ---------------------------------------------------------------------------
# Tests: compute_remix_cache_key
# ---------------------------------------------------------------------------

class TestComputeRemixCacheKey:
    """Composite cache key generation."""

    def test_deterministic(self, tmp_path):
        """Same inputs -> same key."""
        from musicmixer.services.remix_cache import compute_remix_cache_key

        song_a = tmp_path / "a.wav"
        song_b = tmp_path / "b.wav"
        _make_dummy_song(song_a, b"song-a-content")
        _make_dummy_song(song_b, b"song-b-content")

        key1 = compute_remix_cache_key(song_a, song_b, "mix them up")
        key2 = compute_remix_cache_key(song_a, song_b, "mix them up")
        assert key1 == key2

    def test_different_key_when_songs_swapped(self, tmp_path):
        """Swapping song_a and song_b must produce a different key (order matters)."""
        from musicmixer.services.remix_cache import compute_remix_cache_key

        song_a = tmp_path / "a.wav"
        song_b = tmp_path / "b.wav"
        _make_dummy_song(song_a, b"song-a-content")
        _make_dummy_song(song_b, b"song-b-content")

        key_ab = compute_remix_cache_key(song_a, song_b, "remix")
        key_ba = compute_remix_cache_key(song_b, song_a, "remix")
        assert key_ab != key_ba

    def test_different_prompt_different_key(self, tmp_path):
        """Different prompts -> different keys."""
        from musicmixer.services.remix_cache import compute_remix_cache_key

        song_a = tmp_path / "a.wav"
        song_b = tmp_path / "b.wav"
        _make_dummy_song(song_a, b"same-content")
        _make_dummy_song(song_b, b"same-content-b")

        key1 = compute_remix_cache_key(song_a, song_b, "chill vibes")
        key2 = compute_remix_cache_key(song_a, song_b, "heavy bass")
        assert key1 != key2

    def test_normalizes_prompt_whitespace_and_case(self, tmp_path):
        """Prompt is stripped and lowercased before hashing."""
        from musicmixer.services.remix_cache import compute_remix_cache_key

        song_a = tmp_path / "a.wav"
        song_b = tmp_path / "b.wav"
        _make_dummy_song(song_a, b"content-a")
        _make_dummy_song(song_b, b"content-b")

        key1 = compute_remix_cache_key(song_a, song_b, "  Mix It Up  ")
        key2 = compute_remix_cache_key(song_a, song_b, "mix it up")
        assert key1 == key2

    def test_returns_64_char_hex(self, tmp_path):
        """Key should be a 64-char hex string (SHA-256)."""
        from musicmixer.services.remix_cache import compute_remix_cache_key

        song_a = tmp_path / "a.wav"
        song_b = tmp_path / "b.wav"
        _make_dummy_song(song_a, b"data-a")
        _make_dummy_song(song_b, b"data-b")

        key = compute_remix_cache_key(song_a, song_b, "test")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Tests: get_cached_remix
# ---------------------------------------------------------------------------

class TestGetCachedRemix:
    """Cache read logic."""

    def test_returns_none_on_cache_miss(self, tmp_path):
        """Returns None when no cache entry exists."""
        from musicmixer.services.remix_cache import get_cached_remix

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        result = get_cached_remix("deadbeef" * 8, cache_dir)
        assert result is None

    def test_returns_path_on_cache_hit(self, tmp_path):
        """Returns path to cached MP3 on valid cache entry."""
        from musicmixer.services.remix_cache import get_cached_remix

        cache_dir = tmp_path / "cache"
        cache_key = "a" * 64
        _make_dummy_mp3(cache_dir / cache_key / "remix.mp3")

        result = get_cached_remix(cache_key, cache_dir)
        assert result is not None
        assert result.name == "remix.mp3"
        assert result.exists()

    def test_returns_none_if_zero_size(self, tmp_path):
        """Returns None when cached file is zero-size."""
        from musicmixer.services.remix_cache import get_cached_remix

        cache_dir = tmp_path / "cache"
        cache_key = "b" * 64
        entry = cache_dir / cache_key
        entry.mkdir(parents=True)
        (entry / "remix.mp3").write_bytes(b"")

        result = get_cached_remix(cache_key, cache_dir)
        assert result is None

    def test_returns_none_if_directory_empty(self, tmp_path):
        """Returns None when cache entry directory exists but has no remix.mp3."""
        from musicmixer.services.remix_cache import get_cached_remix

        cache_dir = tmp_path / "cache"
        cache_key = "c" * 64
        (cache_dir / cache_key).mkdir(parents=True)

        result = get_cached_remix(cache_key, cache_dir)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: cache_remix
# ---------------------------------------------------------------------------

class TestCacheRemix:
    """Cache write logic with atomic rename."""

    def test_writes_to_correct_location(self, tmp_path):
        """Remix MP3 should be copied into cache_dir/cache_key/remix.mp3."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        remix_path = tmp_path / "remix.mp3"
        _make_dummy_mp3(remix_path)
        cache_key = "a" * 64

        with patch("musicmixer.services.remix_cache.settings") as mock_settings:
            mock_settings.remix_cache_max_gb = 5.0

            cache_remix(cache_key, remix_path, cache_dir)

        cached = cache_dir / cache_key / "remix.mp3"
        assert cached.is_file()
        assert cached.stat().st_size > 0

    def test_no_temp_dirs_left(self, tmp_path):
        """After successful cache write, no .tmp-* dirs should remain."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        remix_path = tmp_path / "remix.mp3"
        _make_dummy_mp3(remix_path)

        with patch("musicmixer.services.remix_cache.settings") as mock_settings:
            mock_settings.remix_cache_max_gb = 5.0

            cache_remix("b" * 64, remix_path, cache_dir)

        tmp_dirs = [d for d in cache_dir.iterdir() if d.name.startswith(".tmp-")]
        assert len(tmp_dirs) == 0

    def test_skips_write_on_missing_file(self, tmp_path):
        """No cache entry should be created when source MP3 doesn't exist."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        missing_path = tmp_path / "does_not_exist.mp3"

        with patch("musicmixer.services.remix_cache.settings") as mock_settings:
            mock_settings.remix_cache_max_gb = 5.0

            cache_remix("c" * 64, missing_path, cache_dir)

        # Cache dir may or may not exist, but no entry for this key
        if cache_dir.exists():
            entries = [d for d in cache_dir.iterdir() if not d.name.startswith(".tmp-")]
            assert len(entries) == 0

    def test_skips_write_on_empty_file(self, tmp_path):
        """No cache entry should be created when source MP3 is zero-length."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        empty_mp3 = tmp_path / "empty.mp3"
        empty_mp3.write_bytes(b"")

        with patch("musicmixer.services.remix_cache.settings") as mock_settings:
            mock_settings.remix_cache_max_gb = 5.0

            cache_remix("d" * 64, empty_mp3, cache_dir)

        if cache_dir.exists():
            entries = [d for d in cache_dir.iterdir() if not d.name.startswith(".tmp-")]
            assert len(entries) == 0

    def test_overwrite_existing_cache_entry(self, tmp_path):
        """Writing to an existing cache key should overwrite cleanly."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        cache_key = "e" * 64
        remix_path = tmp_path / "remix.mp3"
        _make_dummy_mp3(remix_path)

        with patch("musicmixer.services.remix_cache.settings") as mock_settings:
            mock_settings.remix_cache_max_gb = 5.0

            cache_remix(cache_key, remix_path, cache_dir)
            cache_remix(cache_key, remix_path, cache_dir)

        cached = cache_dir / cache_key / "remix.mp3"
        assert cached.is_file()
        assert cached.stat().st_size > 0

    def test_failure_does_not_raise(self, tmp_path):
        """Cache write failure should log but not raise."""
        from musicmixer.services.remix_cache import cache_remix

        cache_dir = tmp_path / "cache"
        remix_path = tmp_path / "remix.mp3"
        _make_dummy_mp3(remix_path)

        with patch("musicmixer.services.remix_cache.settings") as mock_settings, \
             patch("musicmixer.services.remix_cache.shutil.copy2", side_effect=PermissionError("no write")):
            mock_settings.remix_cache_max_gb = 5.0

            # Should not raise
            cache_remix("f" * 64, remix_path, cache_dir)


# ---------------------------------------------------------------------------
# Tests: _evict_lru
# ---------------------------------------------------------------------------

class TestEvictLRU:
    """Cache size management."""

    def test_evicts_oldest_entries(self, tmp_path):
        """When cache exceeds max_gb, oldest entries should be evicted first."""
        from musicmixer.services.remix_cache import _evict_lru

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create 4 cache entries with distinct mtimes
        keys = []
        for i in range(4):
            key = f"{i:064x}"
            keys.append(key)
            _make_dummy_mp3(cache_dir / key / "remix.mp3", size=1024)
            entry_path = cache_dir / key
            mtime = time.time() - (4 - i) * 100
            os.utime(entry_path, (mtime, mtime))

        # Measure total size
        total_size = sum(
            f.stat().st_size
            for entry in cache_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".tmp-")
            for f in entry.iterdir()
            if f.is_file()
        )
        entry_size = total_size // 4

        # Set limit to allow ~2 entries
        limit_gb = (entry_size * 2 + 500) / (1024 * 1024 * 1024)

        _evict_lru(cache_dir, limit_gb)

        remaining = {
            d.name for d in cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".tmp-")
        }

        # Oldest entries (keys[0], keys[1]) should be evicted
        assert keys[0] not in remaining, "Oldest entry should have been evicted"
        assert keys[1] not in remaining, "Second-oldest entry should have been evicted"
        # Newest entries should survive
        assert keys[3] in remaining, "Newest entry should survive"
        assert len(remaining) <= 3

    def test_no_eviction_when_under_limit(self, tmp_path):
        """No entries should be evicted when total size is within limit."""
        from musicmixer.services.remix_cache import _evict_lru

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        for i in range(3):
            _make_dummy_mp3(cache_dir / f"{i:064x}" / "remix.mp3", size=512)

        _evict_lru(cache_dir, max_gb=1.0)  # 1GB >> 1.5KB

        remaining = [
            d for d in cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".tmp-")
        ]
        assert len(remaining) == 3

    def test_skips_temp_dirs(self, tmp_path):
        """Temp directories (.tmp-*) should not be counted or evicted."""
        from musicmixer.services.remix_cache import _evict_lru

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Real entry
        _make_dummy_mp3(cache_dir / ("a" * 64) / "remix.mp3", size=512)
        # Temp dir
        tmp_entry = cache_dir / ".tmp-abc123"
        tmp_entry.mkdir()
        (tmp_entry / "remix.mp3").write_bytes(b"\xff" * 512)

        _evict_lru(cache_dir, max_gb=1.0)

        # Temp dir should still exist (not evicted or counted)
        assert tmp_entry.exists()
