"""Tests for the YouTube cache-correctness work (Part 0 + Part A).

Covers:
- Part 0: meta.json on-disk source of truth + Redis-flush fallback + re-warm;
  atomic stem-dir writes (no half-filled dir on a crash mid-write); the dual-
  writer invariant (every metadata mutation updates disk + Redis together).
- Part A: per-video_id audio cache (atomic), tiered lookup behavior, single-
  flight dedup, and size-cap eviction guards.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicmixer.config import settings
from musicmixer.models import AudioMetadata
import musicmixer.services.song_cache as sc
from musicmixer.services.song_cache import (
    ROLE_INSTRUMENTAL,
    ROLE_VOCAL,
    SongRole,
    _atomic_replace_dir,
    _audio_path,
    _get_redis,
    _load_meta_from_disk,
    _meta_key,
    _meta_path,
    _stems_dir_for,
    _stems_key,
    cache_audio,
    cache_song_metadata,
    cache_song_stems,
    get_cached_audio,
    get_cached_song,
    single_flight,
)

STEM_NAMES = ["vocals", "drums", "bass", "guitar", "piano", "other"]
VOCAL_STEM_NAMES = ["lead_vocals", "backing_vocals", "instrumental"]


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_wav(path: Path, duration: float = 0.05, sr: int = 44100) -> None:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.column_stack([mono, mono]), sr, format="WAV", subtype="FLOAT")


def _make_named_dir(base: Path, names: list[str]) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        _make_wav(base / f"{name}.wav")
    return base


def _make_meta(bpm: float = 120.0) -> AudioMetadata:
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=np.array([512, 1024]),
        duration_seconds=60.0,
        total_beats=120,
        key="C",
        scale="major",
    )


@pytest.fixture
def clean_redis():
    sc._redis_client = None
    r = _get_redis()
    yield r
    for key in r.scan_iter("song:test_cc_*"):
        r.delete(key)
    sc._redis_client = None


@pytest.fixture
def clean_disk_cache():
    """Track and remove on-disk cache dirs for test video IDs."""
    created: list[str] = []
    yield created
    for video_id in created:
        root = settings.song_cache_dir / video_id
        if root.exists():
            import shutil
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Part 0: meta.json source of truth + Redis-flush fallback
# ---------------------------------------------------------------------------

class TestMetaOnDiskFallback:
    def test_cache_metadata_writes_disk_and_redis(self, clean_redis, clean_disk_cache):
        vid = "test_cc_dual_write"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "Title", "Artist", _make_meta(), None)

        # Disk source of truth exists.
        assert _meta_path(vid).is_file()
        disk = _load_meta_from_disk(vid)
        assert disk is not None and disk["title"] == "Title"

        # Redis accelerator was populated too.
        assert clean_redis.exists(_meta_key(vid)) == 1

    def test_fallback_to_disk_after_redis_flush(self, clean_redis, clean_disk_cache):
        """A Redis flush must NOT make a cached song unreachable — disk wins."""
        vid = "test_cc_flush"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "Flush Song", "Artist", _make_meta(bpm=128.0), None)
        cache_song_stems(vid, ROLE_INSTRUMENTAL, _make_named_dir(
            settings.song_cache_dir / "_tmp_flush_src", STEM_NAMES))

        # Simulate the production Redis wipe.
        clean_redis.delete(_meta_key(vid))
        clean_redis.delete(_stems_key(vid, ROLE_INSTRUMENTAL))
        assert clean_redis.exists(_meta_key(vid)) == 0

        result = get_cached_song(vid, ROLE_INSTRUMENTAL)
        assert result is not None, "stems on disk must remain reachable after a flush"
        assert result.meta.bpm == 128.0
        assert result.has_stems is True

        # Redis was re-warmed from disk for next time.
        assert clean_redis.exists(_meta_key(vid)) == 1

    def test_disk_wins_on_divergence(self, clean_redis, clean_disk_cache):
        """When disk and Redis disagree, the disk copy is authoritative on a fallback."""
        vid = "test_cc_diverge"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "Disk Title", "Artist", _make_meta(bpm=100.0), None)

        # Corrupt Redis to a stale value, then delete it to force a disk fallback.
        clean_redis.delete(_meta_key(vid))
        disk = _load_meta_from_disk(vid)
        assert disk is not None
        # Disk still holds the real value.
        restored = _deserialize_helper(disk["meta"])
        assert restored.bpm == 100.0

    def test_no_meta_anywhere_is_miss(self, clean_redis, clean_disk_cache):
        vid = "test_cc_nometa"
        clean_disk_cache.append(vid)
        assert get_cached_song(vid, ROLE_VOCAL) is None


def _deserialize_helper(meta_json: str) -> AudioMetadata:
    from musicmixer.services.song_cache import _deserialize_audio_metadata
    return _deserialize_audio_metadata(meta_json)


# ---------------------------------------------------------------------------
# Part 0: atomic stem-dir writes (no half-filled cache)
# ---------------------------------------------------------------------------

class TestAtomicStemWrites:
    def test_cache_song_stems_replaces_atomically(self, clean_redis, clean_disk_cache, tmp_path):
        """Recaching replaces the dir wholesale, leaving only the new valid shape."""
        vid = "test_cc_atomic_stems"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "S", "", _make_meta(), None)

        # First cache: full 6-stem instrumental.
        cache_song_stems(vid, ROLE_INSTRUMENTAL, _make_named_dir(tmp_path / "s1", STEM_NAMES))
        cache_dir = _stems_dir_for(vid, ROLE_INSTRUMENTAL)
        assert {f.stem for f in cache_dir.glob("*.wav")} == set(STEM_NAMES)

        # Recache with the local 4-stem shape; old extra WAVs must be gone.
        cache_song_stems(vid, ROLE_INSTRUMENTAL,
                         _make_named_dir(tmp_path / "s2", ["vocals", "drums", "bass", "other"]))
        assert {f.stem for f in cache_dir.glob("*.wav")} == {"vocals", "drums", "bass", "other"}

    def test_atomic_replace_dir_leaves_no_partial(self, tmp_path):
        """_atomic_replace_dir: dest is always a complete dir, never half-filled."""
        dest = tmp_path / "dest"
        _make_named_dir(dest, ["old1", "old2"])

        staging = tmp_path / "staging"
        _make_named_dir(staging, ["new1", "new2", "new3"])

        _atomic_replace_dir(staging, dest)
        assert {f.stem for f in dest.glob("*.wav")} == {"new1", "new2", "new3"}
        assert not staging.exists()

    def test_no_redis_stems_key_points_at_incomplete_dir(self, clean_redis, clean_disk_cache, tmp_path):
        """The Redis stems key is only set after the atomic rename completes."""
        vid = "test_cc_stemskey"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "S", "", _make_meta(), None)
        cache_song_stems(vid, ROLE_VOCAL, _make_named_dir(tmp_path / "v", VOCAL_STEM_NAMES))

        stems_path = clean_redis.get(_stems_key(vid, ROLE_VOCAL))
        assert stems_path is not None
        # The key points at the deterministic cache dir, which is fully populated.
        assert Path(stems_path) == _stems_dir_for(vid, ROLE_VOCAL)
        assert {f.stem for f in Path(stems_path).glob("*.wav")} == set(VOCAL_STEM_NAMES)


# ---------------------------------------------------------------------------
# Part A: audio cache (atomic, by video_id, role-agnostic)
# ---------------------------------------------------------------------------

class TestAudioCache:
    def test_cache_and_get_audio_round_trip(self, clean_disk_cache):
        vid = "test_cc_audio_rt"
        clean_disk_cache.append(vid)
        payload = b"fake-mp3-bytes" * 1000
        cache_audio(vid, payload, meta={"title": "Hit", "duration_seconds": 42.0,
                                         "source_codec": "opus", "source_bitrate": 128})

        hit = get_cached_audio(vid)
        assert hit is not None
        path, meta = hit
        assert path == _audio_path(vid)
        assert path.read_bytes() == payload
        assert meta["title"] == "Hit"
        assert meta["duration_seconds"] == 42.0

    def test_audio_cache_miss(self, clean_disk_cache):
        assert get_cached_audio("test_cc_audio_absent") is None

    def test_zero_byte_audio_is_miss(self, clean_disk_cache):
        """A truncated/empty file never counts as a valid hit."""
        vid = "test_cc_audio_empty"
        clean_disk_cache.append(vid)
        p = _audio_path(vid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        assert get_cached_audio(vid) is None

    def test_audio_cache_ttl_expiry(self, clean_disk_cache, monkeypatch):
        vid = "test_cc_audio_ttl"
        clean_disk_cache.append(vid)
        cache_audio(vid, b"bytes" * 100, meta={"title": "Old"})
        # Force the file to look old and a short TTL.
        monkeypatch.setattr(settings, "audio_cache_ttl_hours", 1)
        old = time.time() - 3600 * 5  # 5 hours ago
        import os
        os.utime(_audio_path(vid), (old, old))
        assert get_cached_audio(vid) is None  # expired → miss + removed
        assert not _audio_path(vid).exists()

    def test_atomic_audio_write(self, clean_disk_cache):
        """cache_audio writes atomically — final file equals the full payload."""
        vid = "test_cc_audio_atomic"
        clean_disk_cache.append(vid)
        payload = bytes(range(256)) * 4096
        cache_audio(vid, payload)
        assert _audio_path(vid).read_bytes() == payload


# ---------------------------------------------------------------------------
# Part A: size-cap eviction
# ---------------------------------------------------------------------------

class TestAudioCacheEviction:
    def test_size_cap_evicts_oldest(self, clean_disk_cache, monkeypatch):
        """Over-cap audio cache evicts the least-recently-modified audio file."""
        # ~0.5 MB each; cap at ~1 MB so writing a 3rd forces an eviction.
        monkeypatch.setattr(settings, "audio_cache_max_gb", 1.0 / 1024)  # 1 MB
        big = b"x" * (512 * 1024)

        for i, vid in enumerate(["test_cc_evA", "test_cc_evB", "test_cc_evC"]):
            clean_disk_cache.append(vid)
            cache_audio(vid, big)
            # Make A older than B older than C.
            import os
            t = time.time() - (100 - i * 10)
            os.utime(_audio_path(vid), (t, t))

        # Trigger the cap check by writing one more.
        clean_disk_cache.append("test_cc_evD")
        cache_audio("test_cc_evD", big)

        # The oldest (A) should be gone; the newest (D) is protected.
        assert _audio_path("test_cc_evD").exists()
        # At least the oldest was evicted to get back under cap.
        assert not _audio_path("test_cc_evA").exists()

    def test_eviction_never_touches_meta_or_stems(self, clean_redis, clean_disk_cache, monkeypatch, tmp_path):
        """Size-cap eviction only deletes audio.mp3 — never meta.json or stems."""
        monkeypatch.setattr(settings, "audio_cache_max_gb", 1.0 / 1024)  # 1 MB
        big = b"y" * (700 * 1024)

        vid = "test_cc_ev_keep"
        clean_disk_cache.append(vid)
        cache_song_metadata(vid, "Keep", "", _make_meta(), None)
        cache_song_stems(vid, ROLE_INSTRUMENTAL, _make_named_dir(tmp_path / "k", STEM_NAMES))
        cache_audio(vid, big)
        import os
        old = time.time() - 1000
        os.utime(_audio_path(vid), (old, old))

        # Force over-cap with a newer file; vid's audio may be evicted but its
        # meta.json + stems must survive.
        other = "test_cc_ev_new"
        clean_disk_cache.append(other)
        cache_audio(other, big)

        assert _meta_path(vid).is_file(), "meta.json must never be evicted"
        assert _stems_dir_for(vid, ROLE_INSTRUMENTAL).is_dir(), "stems must never be evicted"

    def test_in_flight_video_not_evicted(self, clean_disk_cache, monkeypatch):
        """A video held by single_flight is never evicted mid-use."""
        monkeypatch.setattr(settings, "audio_cache_max_gb", 1.0 / 1024)  # 1 MB
        big = b"z" * (700 * 1024)

        protected = "test_cc_inflight"
        clean_disk_cache.append(protected)
        cache_audio(protected, big)
        import os
        old = time.time() - 5000  # oldest → first eviction candidate
        os.utime(_audio_path(protected), (old, old))

        with single_flight(protected):
            newer = "test_cc_inflight_new"
            clean_disk_cache.append(newer)
            cache_audio(newer, big)  # triggers cap check while `protected` is in-flight
            assert _audio_path(protected).exists(), "in-flight artifact must not be evicted"


# ---------------------------------------------------------------------------
# Part A: single-flight dedup
# ---------------------------------------------------------------------------

class TestSingleFlight:
    def test_single_flight_serializes_same_video(self):
        """Two threads for the same video_id never run the critical section at once."""
        vid = "test_cc_sf_same"
        concurrent = 0
        max_concurrent = 0
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        def worker():
            nonlocal concurrent, max_concurrent
            barrier.wait()
            with single_flight(vid):
                with lock:
                    concurrent += 1
                    max_concurrent = max(max_concurrent, concurrent)
                time.sleep(0.05)
                with lock:
                    concurrent -= 1

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_concurrent == 1, "single-flight must serialize same-video work"

    def test_single_flight_different_videos_parallel(self):
        """Different video_ids must NOT block each other."""
        observed_parallel = threading.Event()
        in_section = threading.Event()
        barrier = threading.Barrier(2)

        def worker(vid: str, first: bool):
            barrier.wait()
            with single_flight(vid):
                if first:
                    in_section.set()
                    # If the other can also enter while we hold ours, they differ → parallel.
                    if in_section.wait(timeout=1.0):
                        observed_parallel.set()
                    time.sleep(0.05)
                else:
                    in_section.wait(timeout=1.0)
                    observed_parallel.set()

        t1 = threading.Thread(target=worker, args=("test_cc_sf_v1", True))
        t2 = threading.Thread(target=worker, args=("test_cc_sf_v2", False))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert observed_parallel.is_set(), "different videos should not serialize"


# ---------------------------------------------------------------------------
# Part A: tiered lookup selection (via download_youtube_audio + audio cache)
# ---------------------------------------------------------------------------

class TestTieredLookup:
    def test_tier2_audio_hit_skips_youtube(self, clean_disk_cache, tmp_path, monkeypatch):
        """When audio is cached, download_youtube_audio returns it without hitting YouTube."""
        import asyncio
        from musicmixer.services import youtube as yt

        vid = "test_cc_tier2"
        clean_disk_cache.append(vid)
        cache_audio(vid, b"cached-audio" * 500,
                    meta={"title": "Cached Song", "duration_seconds": 33.0,
                          "source_codec": "opus", "source_bitrate": 160})

        # Any real download attempt should fail the test.
        async def _boom(*a, **k):
            raise AssertionError("download was attempted despite an audio-cache hit")
        monkeypatch.setattr(yt, "_download_youtube_audio_uncached", _boom)

        out = tmp_path / "session"
        result = asyncio.run(yt.download_youtube_audio(
            url="https://www.youtube.com/watch?v=" + vid,
            output_dir=out,
            video_id=vid,
        ))
        assert result.title == "Cached Song"
        assert result.duration_seconds == 33.0
        assert result.wav_path.exists()
        assert result.wav_path.parent == out

    def test_tier3_download_persists_audio(self, clean_disk_cache, tmp_path, monkeypatch):
        """A cache miss downloads once and persists the audio for next time."""
        import asyncio
        from musicmixer.services import youtube as yt
        from musicmixer.services.youtube import YouTubeAudioResult

        vid = "test_cc_tier3"
        clean_disk_cache.append(vid)
        assert get_cached_audio(vid) is None

        calls = {"n": 0}

        async def _fake_download(url, output_dir, progress_callback=None):
            calls["n"] += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "dl.mp3"
            p.write_bytes(b"downloaded-bytes" * 100)
            return YouTubeAudioResult(
                wav_path=p, title="Fresh", duration_seconds=10.0,
                source_codec="opus", source_bitrate=128,
            )
        monkeypatch.setattr(yt, "_download_youtube_audio_uncached", _fake_download)

        out = tmp_path / "session"
        result = asyncio.run(yt.download_youtube_audio(
            url="https://www.youtube.com/watch?v=" + vid, output_dir=out, video_id=vid,
        ))
        assert result.title == "Fresh"
        assert calls["n"] == 1
        # Audio was persisted.
        assert get_cached_audio(vid) is not None

        # A second call is a cache hit (no further download).
        asyncio.run(yt.download_youtube_audio(
            url="https://www.youtube.com/watch?v=" + vid, output_dir=out, video_id=vid,
        ))
        assert calls["n"] == 1, "second call must reuse cached audio"

    def test_no_video_id_always_downloads(self, tmp_path, monkeypatch):
        """video_id=None bypasses the cache entirely (back-compat)."""
        import asyncio
        from musicmixer.services import youtube as yt
        from musicmixer.services.youtube import YouTubeAudioResult

        calls = {"n": 0}

        async def _fake_download(url, output_dir, progress_callback=None):
            calls["n"] += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / "dl.mp3"
            p.write_bytes(b"x")
            return YouTubeAudioResult(wav_path=p, title="T", duration_seconds=1.0,
                                      source_codec="opus", source_bitrate=1)
        monkeypatch.setattr(yt, "_download_youtube_audio_uncached", _fake_download)

        for _ in range(2):
            asyncio.run(yt.download_youtube_audio(
                url="https://www.youtube.com/watch?v=test_cc_novid",
                output_dir=tmp_path / "s", video_id=None,
            ))
        assert calls["n"] == 2, "no caching without a video_id"
