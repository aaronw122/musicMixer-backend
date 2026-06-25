"""Phase 4 pipeline-integration tests for cache-aware stem separation.

Exercises the wiring added in Phase 4: ``_step_separate_and_analyze`` now routes
both separations through ``get_or_create_cached_stems`` keyed by (video_id, role),
and ``_checkpoint_song_metadata`` no longer publishes stems (the wrapper is the sole
publisher). These tests run against a REAL Redis (``clean_redis``, ``test_``-prefixed
keys) with the separators mocked to count calls, covering plan §12 "pipeline
integration":

- both roles ready -> skip download, separation, and analysis (medium/full cache);
- raw audio cached but stems missing -> elect one owner, separate once;
- two simultaneous sessions sharing one (video, role) -> separate once, other waits;
- same video as vocal AND instrumental -> two independent separations;
- one role ready, other missing -> only the missing role separates;
- legacy disk cache adopted without recomputation;
- Redis flushed then request -> ready state reconstructed from disk (reconcile);
- Redis outage -> completes via local fallback.
"""

import queue
import struct
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import musicmixer.services.song_cache as song_cache
from musicmixer.config import settings
from musicmixer.models import AudioMetadata, CachedSong, SessionState
from musicmixer.services.pipeline import _step_separate_and_analyze
from musicmixer.services.song_cache import (
    ROLE_INSTRUMENTAL,
    ROLE_VOCAL,
    StemCacheCoordinator,
    StemLease,
    _get_redis,
    _stems_dir_for,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_redis():
    song_cache._redis_client = None
    r = _get_redis()
    for key in r.scan_iter("song:test_*"):
        r.delete(key)
    yield r
    for key in r.scan_iter("song:test_*"):
        r.delete(key)
    song_cache._redis_client = None


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    root = tmp_path / "song_cache"
    root.mkdir()
    monkeypatch.setattr(settings, "song_cache_dir", root)
    return root


@pytest.fixture
def fast_coord(monkeypatch):
    monkeypatch.setattr(settings, "stem_lock_lease_seconds", 2)
    monkeypatch.setattr(settings, "stem_lock_renew_interval_seconds", 1)
    monkeypatch.setattr(settings, "stem_wait_poll_seconds", 0)
    monkeypatch.setattr(settings, "stem_wait_timeout_seconds", 30)


@pytest.fixture
def no_aux(monkeypatch):
    """Disable lyrics + ML structure so the step's pool is just separation + analysis."""
    monkeypatch.setattr(settings, "lyrics_lookup_enabled", False)
    monkeypatch.setattr(settings, "section_detection_backend", "heuristic")


_VOCAL_STEMS = ("lead_vocals", "backing_vocals", "instrumental")
_INSTRUMENTAL_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _write_wav(path: Path, *, payload_bytes: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00\x00" * (payload_bytes // 2)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 16000, 2, 16)
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    riff_size = 4 + len(fmt) + len(data_chunk)
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    path.write_bytes(header + fmt + data_chunk)


def _publish_ready(coordinator: StemCacheCoordinator, video_id: str, role, stems) -> None:
    """Separate-and-publish a (video_id, role) into the shared cache as ``ready``."""
    lease = coordinator.acquire(video_id, role)
    assert lease is not None
    cache_dir = _stems_dir_for(video_id, role)
    staging = cache_dir.with_name(f".{role}.staging.{lease.owner_token}")
    staging.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _write_wav(staging / f"{stem}.wav")
    coordinator.mark_ready(lease, staging)
    coordinator.release(lease)


def _write_legacy_disk(video_id: str, role, stems) -> None:
    """Write a valid on-disk role dir with NO Redis state (legacy cache)."""
    cache_dir = _stems_dir_for(video_id, role)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _write_wav(cache_dir / f"{stem}.wav")


class _Separator:
    """Counts calls and writes valid role stems into the given output dir."""

    def __init__(self, stems, block_event=None):
        self.stems = stems
        self.block_event = block_event
        self.calls = 0
        self._lock = threading.Lock()
        self.started = threading.Event()

    def __call__(self, audio_path, output_dir, progress_callback=None):
        with self._lock:
            self.calls += 1
        self.started.set()
        if self.block_event is not None:
            self.block_event.wait(timeout=20)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for stem in self.stems:
            p = output_dir / f"{stem}.wav"
            _write_wav(p)
            result[stem] = p
        return result


def _fake_meta() -> AudioMetadata:
    return AudioMetadata(
        bpm=120.0,
        bpm_confidence=1.0,
        beat_frames=np.array([0, 1, 2]),
        duration_seconds=10.0,
        total_beats=4,
    )


def _run_step(
    session_id,
    stems_dir,
    *,
    video_id_a,
    video_id_b,
    sep_a,
    sep_b,
    session=None,
    cached_meta_a=None,
    cached_meta_b=None,
):
    """Drive _step_separate_and_analyze with mocked separators + analysis."""
    session = session or SessionState()
    event_queue: queue.Queue = queue.Queue(maxsize=200)
    song_a_path = stems_dir.parent / "song_a.wav"
    song_b_path = stems_dir.parent / "song_b.wav"
    _write_wav(song_a_path)
    _write_wav(song_b_path)

    with (
        patch("musicmixer.services.separation.separate_vocal_song", side_effect=sep_a),
        patch("musicmixer.services.separation.separate_stems", side_effect=sep_b),
        patch("musicmixer.services.analysis.analyze_audio_full", side_effect=lambda *_a, **_k: _fake_meta()),
    ):
        return _step_separate_and_analyze(
            session_id,
            song_a_path,
            song_b_path,
            stems_dir,
            "Song A",
            "Song B",
            event_queue,
            session,
            cached_meta_a=cached_meta_a,
            cached_meta_b=cached_meta_b,
            video_id_a=video_id_a,
            video_id_b=video_id_b,
        ), event_queue


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("no_aux")
class TestPipelineStemCache:
    def test_both_roles_ready_skips_separation(self, clean_redis, cache_dir, fast_coord, tmp_path):
        coordinator = StemCacheCoordinator()
        _publish_ready(coordinator, "test_ready_a", ROLE_VOCAL, _VOCAL_STEMS)
        _publish_ready(coordinator, "test_ready_b", ROLE_INSTRUMENTAL, _INSTRUMENTAL_STEMS)

        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_ready_a", video_id_b="test_ready_b",
            sep_a=sep_a, sep_b=sep_b,
            cached_meta_a=_fake_meta(), cached_meta_b=_fake_meta(),
        )

        assert sep_a.calls == 0
        assert sep_b.calls == 0
        song_a_stems = result[0]
        song_b_stems = result[1]
        assert set(song_a_stems) == set(_VOCAL_STEMS)
        assert set(song_b_stems) == set(_INSTRUMENTAL_STEMS)
        # Copied into the session dir.
        assert (stems_dir / "song_a" / "lead_vocals.wav").is_file()
        assert (stems_dir / "song_b" / "drums.wav").is_file()

    def test_stems_missing_elects_owner_and_separates_once(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_miss_a", video_id_b="test_miss_b",
            sep_a=sep_a, sep_b=sep_b,
        )

        assert sep_a.calls == 1
        assert sep_b.calls == 1
        # Published to the shared cache as ready.
        coordinator = StemCacheCoordinator()
        assert coordinator.get_state("test_miss_a", ROLE_VOCAL).status == "ready"
        assert coordinator.get_state("test_miss_b", ROLE_INSTRUMENTAL).status == "ready"
        assert set(result[0]) == set(_VOCAL_STEMS)

    def test_two_sessions_share_role_separate_once(self, clean_redis, cache_dir, fast_coord, tmp_path):
        block = threading.Event()
        sep_owner = _Separator(_VOCAL_STEMS, block_event=block)
        sep_inst = _Separator(_INSTRUMENTAL_STEMS)

        stems_dir1 = tmp_path / "stems" / "s1"
        stems_dir2 = tmp_path / "stems" / "s2"
        results: dict[str, object] = {}

        def _session(name, stems_dir, sep_b):
            res, q = _run_step(
                name, stems_dir,
                video_id_a="test_shared", video_id_b=f"test_inst_{name}",
                sep_a=sep_owner, sep_b=sep_b,
            )
            results[name] = (res, q)

        # First session owns the shared vocal; block it mid-separation.
        t1 = threading.Thread(target=_session, args=("s1", stems_dir1, sep_inst))
        t1.start()
        assert sep_owner.started.wait(timeout=10)

        # Second session requests the SAME vocal -> must wait, not separate.
        sep_inst2 = _Separator(_INSTRUMENTAL_STEMS)
        t2 = threading.Thread(target=_session, args=("s2", stems_dir2, sep_inst2))
        t2.start()
        time.sleep(0.5)

        block.set()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert sep_owner.calls == 1  # vocal separated exactly once across both sessions
        res2, q2 = results["s2"]
        assert set(res2[0]) == set(_VOCAL_STEMS)
        # Waiter saw the "already being separated" SSE message.
        msgs = []
        try:
            while True:
                msgs.append(q2.get_nowait())
        except queue.Empty:
            pass
        assert any("already being separated" in m.get("detail", "") for m in msgs)

    def test_same_video_both_roles_separate_independently(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_same", video_id_b="test_same",
            sep_a=sep_a, sep_b=sep_b,
        )

        # Same video used as both vocal and instrumental: two independent separations.
        assert sep_a.calls == 1
        assert sep_b.calls == 1
        coordinator = StemCacheCoordinator()
        assert coordinator.get_state("test_same", ROLE_VOCAL).status == "ready"
        assert coordinator.get_state("test_same", ROLE_INSTRUMENTAL).status == "ready"
        assert set(result[0]) == set(_VOCAL_STEMS)
        assert set(result[1]) == set(_INSTRUMENTAL_STEMS)

    def test_one_role_ready_other_missing(self, clean_redis, cache_dir, fast_coord, tmp_path):
        coordinator = StemCacheCoordinator()
        _publish_ready(coordinator, "test_partial_a", ROLE_VOCAL, _VOCAL_STEMS)

        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        _run_step(
            "s1", stems_dir,
            video_id_a="test_partial_a", video_id_b="test_partial_b",
            sep_a=sep_a, sep_b=sep_b,
            cached_meta_a=_fake_meta(),
        )

        assert sep_a.calls == 0  # vocal already ready
        assert sep_b.calls == 1  # instrumental missing -> separated

    def test_legacy_disk_cache_adopted_without_recomputation(self, clean_redis, cache_dir, fast_coord, tmp_path):
        # Valid on-disk role dirs with NO Redis state.
        _write_legacy_disk("test_legacy_a", ROLE_VOCAL, _VOCAL_STEMS)
        _write_legacy_disk("test_legacy_b", ROLE_INSTRUMENTAL, _INSTRUMENTAL_STEMS)
        assert clean_redis.keys("song:test_legacy_*") == []

        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_legacy_a", video_id_b="test_legacy_b",
            sep_a=sep_a, sep_b=sep_b,
            cached_meta_a=_fake_meta(), cached_meta_b=_fake_meta(),
        )

        assert sep_a.calls == 0
        assert sep_b.calls == 0
        assert set(result[0]) == set(_VOCAL_STEMS)
        # Adoption promoted disk to ready in Redis.
        coordinator = StemCacheCoordinator()
        assert coordinator.get_state("test_legacy_a", ROLE_VOCAL).status == "ready"

    def test_redis_flush_reconstructs_ready_from_disk(self, clean_redis, cache_dir, fast_coord, tmp_path):
        coordinator = StemCacheCoordinator()
        _publish_ready(coordinator, "test_flush_a", ROLE_VOCAL, _VOCAL_STEMS)
        _publish_ready(coordinator, "test_flush_b", ROLE_INSTRUMENTAL, _INSTRUMENTAL_STEMS)

        # Flush Redis state but keep the disk artifacts.
        for key in clean_redis.scan_iter("song:test_flush_*"):
            clean_redis.delete(key)
        assert coordinator.get_state("test_flush_a", ROLE_VOCAL) is None

        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_flush_a", video_id_b="test_flush_b",
            sep_a=sep_a, sep_b=sep_b,
            cached_meta_a=_fake_meta(), cached_meta_b=_fake_meta(),
        )

        assert sep_a.calls == 0  # reconciled from disk, no recompute
        assert sep_b.calls == 0
        assert coordinator.get_state("test_flush_a", ROLE_VOCAL).status == "ready"

    def test_restore_requires_ready_state_for_both_roles(self, clean_redis, cache_dir, fast_coord, monkeypatch, tmp_path):
        from musicmixer.services import remix_stages
        from musicmixer.services.remix_stages import (
            FullyCachedCallbacks,
            FullyCachedInputs,
            restore_fully_cached_youtube_remix,
        )

        coordinator = StemCacheCoordinator()
        _publish_ready(coordinator, "test_restore_a", ROLE_VOCAL, _VOCAL_STEMS)
        _publish_ready(coordinator, "test_restore_b", ROLE_INSTRUMENTAL, _INSTRUMENTAL_STEMS)

        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(
            remix_stages, "_step_measure_stem_lufs", lambda *_a, **_k: ({}, {}),
            raising=False,
        )

        def _cached(video_id, role):
            return CachedSong(
                video_id=video_id, title="t", artist="a", meta=_fake_meta(),
                lyrics=None, stems_path=str(_stems_dir_for(video_id, role)), has_stems=True,
            )

        inputs = FullyCachedInputs(
            session_id="rs",
            cached_song_a=_cached("test_restore_a", ROLE_VOCAL),
            cached_song_b=_cached("test_restore_b", ROLE_INSTRUMENTAL),
        )
        callbacks = FullyCachedCallbacks(check_cancelled=lambda: None, on_cache_skip=lambda: None)

        result = restore_fully_cached_youtube_remix(inputs, callbacks=callbacks, settings=settings)
        assert result.used_cache is True

    def test_restore_misses_when_role_not_ready(self, clean_redis, cache_dir, fast_coord, monkeypatch, tmp_path):
        from musicmixer.services import remix_stages
        from musicmixer.services.remix_stages import (
            FullyCachedCallbacks,
            FullyCachedInputs,
            restore_fully_cached_youtube_remix,
        )

        coordinator = StemCacheCoordinator()
        _publish_ready(coordinator, "test_nr_a", ROLE_VOCAL, _VOCAL_STEMS)
        # Role B has only a partial (invalid) on-disk dir, never published ready.
        bad_dir = _stems_dir_for("test_nr_b", ROLE_INSTRUMENTAL)
        bad_dir.mkdir(parents=True, exist_ok=True)
        _write_wav(bad_dir / "drums.wav")  # one stem -> invalid for instrumental

        monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
        monkeypatch.setattr(
            remix_stages, "_step_measure_stem_lufs", lambda *_a, **_k: ({}, {}),
            raising=False,
        )

        def _cached(video_id, role):
            return CachedSong(
                video_id=video_id, title="t", artist="a", meta=_fake_meta(),
                lyrics=None, stems_path=None, has_stems=True,
            )

        inputs = FullyCachedInputs(
            session_id="rs2",
            cached_song_a=_cached("test_nr_a", ROLE_VOCAL),
            cached_song_b=_cached("test_nr_b", ROLE_INSTRUMENTAL),
        )
        callbacks = FullyCachedCallbacks(check_cancelled=lambda: None, on_cache_skip=lambda: None)

        result = restore_fully_cached_youtube_remix(inputs, callbacks=callbacks, settings=settings)
        assert result.used_cache is False

    def test_redis_outage_completes_via_fallback(self, clean_redis, cache_dir, monkeypatch, tmp_path):
        import redis as redis_lib

        class _DeadRedis:
            def __getattr__(self, _name):
                def _boom(*_a, **_k):
                    raise redis_lib.ConnectionError("redis down")
                return _boom

        monkeypatch.setattr(song_cache, "_get_redis", lambda: _DeadRedis())

        sep_a = _Separator(_VOCAL_STEMS)
        sep_b = _Separator(_INSTRUMENTAL_STEMS)
        stems_dir = tmp_path / "stems" / "s1"

        result, _ = _run_step(
            "s1", stems_dir,
            video_id_a="test_outage_a", video_id_b="test_outage_b",
            sep_a=sep_a, sep_b=sep_b,
        )

        # Degraded: local separation runs, request completes.
        assert sep_a.calls == 1
        assert sep_b.calls == 1
        assert (stems_dir / "song_a" / "lead_vocals.wav").is_file()
        assert (stems_dir / "song_b" / "drums.wav").is_file()
