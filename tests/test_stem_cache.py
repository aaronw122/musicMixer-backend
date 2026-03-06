"""Tests for musicmixer.services.stem_cache — content-hash stem caching."""

import io
import os
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STEM_NAMES_6 = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]


def _make_wav_bytes(freq: float = 440.0, duration: float = 0.1, sr: int = 44100) -> bytes:
    """Create minimal float32 WAV bytes (stereo sine wave)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _populate_stems_dir(stems_dir: Path, stem_names: list[str] | None = None) -> None:
    """Write small WAV files for each stem into *stems_dir*."""
    if stem_names is None:
        stem_names = STEM_NAMES_6
    stems_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(stem_names):
        wav_path = stems_dir / f"{name}.wav"
        wav_path.write_bytes(_make_wav_bytes(freq=220.0 + i * 50))


def _make_audio_file(path: Path, freq: float = 440.0) -> None:
    """Write a small audio file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_wav_bytes(freq=freq))


# ---------------------------------------------------------------------------
# Tests: get_cache_key
# ---------------------------------------------------------------------------

class TestGetCacheKey:
    """SHA-256 content hashing."""

    def test_deterministic(self, tmp_path):
        """Same file content -> same hash."""
        from musicmixer.services.stem_cache import get_cache_key

        audio = tmp_path / "song.wav"
        audio.write_bytes(_make_wav_bytes(440.0))

        key1 = get_cache_key(audio)
        key2 = get_cache_key(audio)
        assert key1 == key2

    def test_different_content_different_key(self, tmp_path):
        """Different file content -> different hash."""
        from musicmixer.services.stem_cache import get_cache_key

        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        a.write_bytes(_make_wav_bytes(440.0))
        b.write_bytes(_make_wav_bytes(880.0))

        assert get_cache_key(a) != get_cache_key(b)

    def test_returns_hex_string(self, tmp_path):
        """Key should be a 64-char hex string (SHA-256)."""
        from musicmixer.services.stem_cache import get_cache_key

        audio = tmp_path / "song.wav"
        audio.write_bytes(_make_wav_bytes())

        key = get_cache_key(audio)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ---------------------------------------------------------------------------
# Tests: get_cached_stems
# ---------------------------------------------------------------------------

class TestGetCachedStems:
    """Cache read logic."""

    def test_cache_miss_no_directory(self, tmp_path):
        """Returns False when no cache entry exists."""
        from musicmixer.services.stem_cache import get_cached_stems

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = tmp_path / "cache"

            result = get_cached_stems("deadbeef" * 8, tmp_path / "output")
            assert result is False

    def test_cache_hit_copies_stems(self, tmp_path):
        """Returns True and copies stems on valid cache entry."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "a" * 64
        cache_entry = cache_dir / cache_key
        _populate_stems_dir(cache_entry, STEM_NAMES_6)

        output_dir = tmp_path / "output"

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, output_dir)

        assert result is True
        assert output_dir.exists()
        copied = {f.stem for f in output_dir.glob("*.wav")}
        assert copied == set(STEM_NAMES_6)

    def test_cache_miss_empty_directory(self, tmp_path):
        """Returns False when cache entry directory is empty."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "b" * 64
        (cache_dir / cache_key).mkdir(parents=True)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, tmp_path / "output")
            assert result is False

    def test_cache_miss_zero_length_stem(self, tmp_path):
        """Returns False when any stem file has zero size."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "c" * 64
        cache_entry = cache_dir / cache_key
        _populate_stems_dir(cache_entry, STEM_NAMES_6)

        # Truncate one stem to zero
        (cache_entry / "vocals.wav").write_bytes(b"")

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, tmp_path / "output")
            assert result is False

    def test_cache_miss_unrecognized_stems(self, tmp_path):
        """Returns False when stems don't match expected set."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "d" * 64
        cache_entry = cache_dir / cache_key
        _populate_stems_dir(cache_entry, ["foo", "bar", "baz"])

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, tmp_path / "output")
            assert result is False

    def test_cache_disabled_returns_false(self, tmp_path):
        """Returns False when caching is disabled, even if entry exists."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "e" * 64
        _populate_stems_dir(cache_dir / cache_key, STEM_NAMES_6)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = False
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, tmp_path / "output")
            assert result is False

    def test_cache_hit_4_stems(self, tmp_path):
        """Returns True for valid 4-stem (htdemucs_ft) cache entry."""
        from musicmixer.services.stem_cache import get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "f" * 64
        _populate_stems_dir(cache_dir / cache_key, STEM_NAMES_4)

        output_dir = tmp_path / "output"

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir

            result = get_cached_stems(cache_key, output_dir)

        assert result is True
        copied = {f.stem for f in output_dir.glob("*.wav")}
        assert copied == set(STEM_NAMES_4)


# ---------------------------------------------------------------------------
# Tests: cache_stems
# ---------------------------------------------------------------------------

class TestCacheStems:
    """Cache write logic with atomic rename."""

    def test_caches_stems_to_correct_location(self, tmp_path):
        """Stems should be copied into cache_dir/cache_key/."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        stems_dir = tmp_path / "stems"
        _populate_stems_dir(stems_dir, STEM_NAMES_6)

        cache_key = "a" * 64

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            cache_stems(cache_key, stems_dir)

        cached_entry = cache_dir / cache_key
        assert cached_entry.is_dir()
        cached_stems = {f.stem for f in cached_entry.glob("*.wav")}
        assert cached_stems == set(STEM_NAMES_6)

    def test_no_temp_dirs_left(self, tmp_path):
        """After successful cache write, no .tmp-* dirs should remain."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        stems_dir = tmp_path / "stems"
        _populate_stems_dir(stems_dir, STEM_NAMES_6)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            cache_stems("b" * 64, stems_dir)

        tmp_dirs = [d for d in cache_dir.iterdir() if d.name.startswith(".tmp-")]
        assert len(tmp_dirs) == 0

    def test_cache_disabled_skips_write(self, tmp_path):
        """No cache entry should be created when caching is disabled."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        stems_dir = tmp_path / "stems"
        _populate_stems_dir(stems_dir, STEM_NAMES_6)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = False
            mock_settings.stem_cache_dir = cache_dir

            cache_stems("c" * 64, stems_dir)

        assert not cache_dir.exists()

    def test_no_wav_skips_write(self, tmp_path):
        """If stems_dir has no WAVs, cache write should be skipped."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        stems_dir = tmp_path / "empty_stems"
        stems_dir.mkdir()

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            cache_stems("d" * 64, stems_dir)

        # Cache dir may be created, but no entry for this key
        if cache_dir.exists():
            entries = [d for d in cache_dir.iterdir() if not d.name.startswith(".tmp-")]
            assert len(entries) == 0

    def test_overwrite_existing_cache_entry(self, tmp_path):
        """Writing to an existing cache key should overwrite cleanly."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        cache_key = "e" * 64
        stems_dir = tmp_path / "stems"
        _populate_stems_dir(stems_dir, STEM_NAMES_6)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            # Write once
            cache_stems(cache_key, stems_dir)
            # Write again (simulate concurrent finish)
            cache_stems(cache_key, stems_dir)

        cached_entry = cache_dir / cache_key
        assert cached_entry.is_dir()
        assert len(list(cached_entry.glob("*.wav"))) == 6


# ---------------------------------------------------------------------------
# Tests: LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    """Cache size management."""

    def test_evicts_oldest_entries(self, tmp_path):
        """When cache exceeds max_gb, oldest entries should be evicted first."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"

        # Create 4 cache entries, each ~same size
        stems_dirs = []
        keys = []
        for i in range(4):
            stems_dir = tmp_path / f"stems_{i}"
            _populate_stems_dir(stems_dir, STEM_NAMES_6)
            stems_dirs.append(stems_dir)
            keys.append(f"{i:064x}")

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            for i, (key, sd) in enumerate(zip(keys, stems_dirs)):
                cache_stems(key, sd)
                # Set distinct mtimes: key[0] is oldest, key[3] is newest
                entry_path = cache_dir / key
                mtime = time.time() - (4 - i) * 100
                os.utime(entry_path, (mtime, mtime))

        # Measure total cache size
        total_size = sum(
            f.stat().st_size
            for entry in cache_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".tmp-")
            for f in entry.iterdir()
            if f.is_file()
        )
        entry_size = total_size // 4

        # Set limit to allow ~2 entries -- should evict the 2 oldest
        # when the 5th entry is added (bringing us from 4 to 5)
        limit_gb = (entry_size * 2 + 500) / (1024 * 1024 * 1024)

        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = limit_gb

            # Add a new entry to trigger eviction
            new_stems = tmp_path / "stems_new"
            _populate_stems_dir(new_stems, STEM_NAMES_6)
            new_key = f"{99:064x}"
            cache_stems(new_key, new_stems)

        remaining = {
            d.name for d in cache_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".tmp-")
        }

        # The new entry must survive
        assert new_key in remaining
        # The oldest entries (keys[0], keys[1]) should be evicted
        assert keys[0] not in remaining, "Oldest entry should have been evicted"
        assert keys[1] not in remaining, "Second-oldest entry should have been evicted"
        # Total remaining should be at most 3 (within the budget)
        assert len(remaining) <= 3


# ---------------------------------------------------------------------------
# Tests: concurrent access
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    """Thread-safety of cache read/write."""

    def test_concurrent_writes_same_key(self, tmp_path):
        """Two threads writing the same key should not corrupt the cache."""
        from musicmixer.services.stem_cache import cache_stems

        cache_dir = tmp_path / "cache"
        cache_key = "concurrent" + "0" * 54

        stems_a = tmp_path / "stems_a"
        stems_b = tmp_path / "stems_b"
        _populate_stems_dir(stems_a, STEM_NAMES_6)
        _populate_stems_dir(stems_b, STEM_NAMES_6)

        errors = []

        def write_cache(stems_dir):
            try:
                cache_stems(cache_key, stems_dir)
            except Exception as e:
                errors.append(e)

        # Patch ONCE, outside threads — mock.patch is not thread-safe
        with patch("musicmixer.services.stem_cache.settings") as mock_settings:
            mock_settings.stem_cache_enabled = True
            mock_settings.stem_cache_dir = cache_dir
            mock_settings.stem_cache_max_gb = 10.0

            t1 = threading.Thread(target=write_cache, args=(stems_a,))
            t2 = threading.Thread(target=write_cache, args=(stems_b,))
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

        assert not errors, f"Concurrent write errors: {errors}"

        # Cache entry should exist and be valid
        cached_entry = cache_dir / cache_key
        assert cached_entry.is_dir()
        cached_stems = {f.stem for f in cached_entry.glob("*.wav")}
        assert cached_stems == set(STEM_NAMES_6)
        # All files should be non-zero
        for wav in cached_entry.glob("*.wav"):
            assert wav.stat().st_size > 0

    def test_concurrent_read_and_write(self, tmp_path):
        """Reading while another thread writes should not crash."""
        from musicmixer.services.stem_cache import cache_stems, get_cached_stems

        cache_dir = tmp_path / "cache"
        cache_key = "readwrite" + "0" * 55

        stems_dir = tmp_path / "stems"
        _populate_stems_dir(stems_dir, STEM_NAMES_6)

        errors = []

        def write_fn():
            try:
                with patch("musicmixer.services.stem_cache.settings") as mock_settings:
                    mock_settings.stem_cache_enabled = True
                    mock_settings.stem_cache_dir = cache_dir
                    mock_settings.stem_cache_max_gb = 10.0
                    cache_stems(cache_key, stems_dir)
            except Exception as e:
                errors.append(e)

        def read_fn():
            try:
                with patch("musicmixer.services.stem_cache.settings") as mock_settings:
                    mock_settings.stem_cache_enabled = True
                    mock_settings.stem_cache_dir = cache_dir
                    # Result can be True or False depending on timing
                    get_cached_stems(cache_key, tmp_path / "read_output")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=write_fn)
        t2 = threading.Thread(target=read_fn)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Concurrent read/write errors: {errors}"


# ---------------------------------------------------------------------------
# Tests: integration with separation.py
# ---------------------------------------------------------------------------

class TestSeparationCacheIntegration:
    """Test that separate_stems uses the cache correctly."""

    @patch("musicmixer.services.separation._separate_modal")
    def test_cache_miss_calls_backend(self, mock_modal, tmp_path):
        """On cache miss, separate_stems should call the backend and cache result."""
        from musicmixer.services.separation import separate_stems

        # Set up mock to produce stems
        output_dir = tmp_path / "out"

        def fake_modal(audio_path, out_dir, cb=None):
            _populate_stems_dir(out_dir, STEM_NAMES_6)
            return {name: out_dir / f"{name}.wav" for name in STEM_NAMES_6}

        mock_modal.side_effect = fake_modal

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(_make_wav_bytes())

        cache_dir = tmp_path / "cache"

        with patch("musicmixer.services.separation.settings") as sep_settings, \
             patch("musicmixer.services.stem_cache.settings") as cache_settings:
            sep_settings.stem_backend = "modal"
            sep_settings.stem_cache_enabled = True
            sep_settings.stem_cache_dir = cache_dir

            cache_settings.stem_cache_enabled = True
            cache_settings.stem_cache_dir = cache_dir
            cache_settings.stem_cache_max_gb = 10.0

            result = separate_stems(audio_file, output_dir)

        mock_modal.assert_called_once()
        assert set(result.keys()) == set(STEM_NAMES_6)

        # Stems should also be cached
        cache_entries = [d for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith(".tmp-")]
        assert len(cache_entries) == 1

    @patch("musicmixer.services.separation._separate_modal")
    def test_cache_hit_skips_backend(self, mock_modal, tmp_path):
        """On cache hit, separate_stems should NOT call the backend."""
        from musicmixer.services.separation import separate_stems
        from musicmixer.services.stem_cache import get_cache_key

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(_make_wav_bytes())

        cache_dir = tmp_path / "cache"
        cache_key = get_cache_key(audio_file)

        # Pre-populate cache
        _populate_stems_dir(cache_dir / cache_key, STEM_NAMES_6)

        output_dir = tmp_path / "out"

        with patch("musicmixer.services.separation.settings") as sep_settings, \
             patch("musicmixer.services.stem_cache.settings") as cache_settings:
            sep_settings.stem_backend = "modal"
            sep_settings.stem_cache_enabled = True
            sep_settings.stem_cache_dir = cache_dir

            cache_settings.stem_cache_enabled = True
            cache_settings.stem_cache_dir = cache_dir

            result = separate_stems(audio_file, output_dir)

        mock_modal.assert_not_called()
        assert set(result.keys()) == set(STEM_NAMES_6)
        # Verify actual files exist in output
        for name in STEM_NAMES_6:
            assert (output_dir / f"{name}.wav").exists()

    @patch("musicmixer.services.separation._separate_modal")
    def test_cache_disabled_always_calls_backend(self, mock_modal, tmp_path):
        """When stem_cache_enabled=False, always call backend."""
        from musicmixer.services.separation import separate_stems

        def fake_modal(audio_path, out_dir, cb=None):
            _populate_stems_dir(out_dir, STEM_NAMES_6)
            return {name: out_dir / f"{name}.wav" for name in STEM_NAMES_6}

        mock_modal.side_effect = fake_modal

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(_make_wav_bytes())
        output_dir = tmp_path / "out"

        with patch("musicmixer.services.separation.settings") as sep_settings:
            sep_settings.stem_backend = "modal"
            sep_settings.stem_cache_enabled = False

            result = separate_stems(audio_file, output_dir)

        mock_modal.assert_called_once()
        assert set(result.keys()) == set(STEM_NAMES_6)

    @patch("musicmixer.services.separation._separate_modal")
    def test_cache_write_failure_does_not_crash(self, mock_modal, tmp_path):
        """If cache_stems fails, separation should still return results."""
        from musicmixer.services.separation import separate_stems

        def fake_modal(audio_path, out_dir, cb=None):
            _populate_stems_dir(out_dir, STEM_NAMES_6)
            return {name: out_dir / f"{name}.wav" for name in STEM_NAMES_6}

        mock_modal.side_effect = fake_modal

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(_make_wav_bytes())
        output_dir = tmp_path / "out"

        with patch("musicmixer.services.separation.settings") as sep_settings, \
             patch("musicmixer.services.stem_cache.settings") as cache_settings, \
             patch("musicmixer.services.stem_cache.cache_stems", side_effect=PermissionError("no write")):
            sep_settings.stem_backend = "modal"
            sep_settings.stem_cache_enabled = True
            sep_settings.stem_cache_dir = tmp_path / "cache"

            cache_settings.stem_cache_enabled = True
            cache_settings.stem_cache_dir = tmp_path / "cache"

            # Should not raise despite cache failure
            result = separate_stems(audio_file, output_dir)

        assert set(result.keys()) == set(STEM_NAMES_6)
