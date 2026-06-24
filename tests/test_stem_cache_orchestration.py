"""Concurrency tests for the Phase 3 cache-aware separation wrapper.

Exercises ``get_or_create_cached_stems`` under real contention (threads + a real
Redis) with short lease durations and an instrumented ``separate_fn`` that records
its call count and can block on an event. Covers the plan §12 concurrency items:
single-flight dedup, role/video independence, lease renewal vs. takeover, simulated
owner death, lost-lease publish fencing, waiter cancellation, and the Redis-outage
fallback to local separation.

Runs against the same REAL Redis convention as test_stem_cache_coordination.py:
all keys use ``test_``-prefixed video IDs and are cleaned up after each test.
"""

import struct
import threading
import time
from pathlib import Path

import pytest

import musicmixer.services.song_cache as song_cache
from musicmixer.config import settings
from musicmixer.services.song_cache import (
    ROLE_INSTRUMENTAL,
    ROLE_VOCAL,
    STEM_ERROR_TRANSIENT,
    StemCacheCoordinator,
    StemLease,
    StemSeparationError,
    _get_redis,
    _stem_lock_key,
    _stems_dir_for,
    get_or_create_cached_stems,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_redis():
    """Fresh Redis client; clean up test keys after each test."""
    song_cache._redis_client = None
    r = _get_redis()
    yield r
    for key in r.scan_iter("song:test_*"):
        r.delete(key)
    song_cache._redis_client = None


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point ``song_cache_dir`` at a tmp dir so disk publication is isolated."""
    root = tmp_path / "song_cache"
    root.mkdir()
    monkeypatch.setattr(settings, "song_cache_dir", root)
    return root


@pytest.fixture
def fast_coord(monkeypatch):
    """Short lease/renew/poll so renewal, expiry, and waiting run quickly."""
    monkeypatch.setattr(settings, "stem_lock_lease_seconds", 2)
    monkeypatch.setattr(settings, "stem_lock_renew_interval_seconds", 1)
    monkeypatch.setattr(settings, "stem_wait_poll_seconds", 0)
    monkeypatch.setattr(settings, "stem_wait_timeout_seconds", 30)


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


class _Separator:
    """Instrumented separator: counts calls, writes valid stems, can block.

    Call signature matches the real separators ``(audio_path, output_dir, progress_callback=None)``.
    ``block_event`` (if set) gates each call until it is set, simulating a long
    blocking remote separation.
    """

    def __init__(self, stems=_VOCAL_STEMS, block_event: threading.Event | None = None):
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


def _noop_cancel() -> None:
    return None


# ---------------------------------------------------------------------------
# video_id is None -> uncached path
# ---------------------------------------------------------------------------

class TestUncachedPath:
    def test_none_video_runs_uncached(self, clean_redis, cache_dir, tmp_path):
        sep = _Separator()
        out = tmp_path / "session"
        result = get_or_create_cached_stems(
            video_id=None,
            role=ROLE_VOCAL,
            audio_path=tmp_path / "a.mp3",
            session_output_dir=out,
            separate_fn=sep,
            check_cancelled=_noop_cancel,
        )
        assert sep.calls == 1
        assert set(result) == set(_VOCAL_STEMS)
        # No coordination state should be written for an uncached request.
        assert clean_redis.keys("song:test_*") == []


# ---------------------------------------------------------------------------
# Owner path + ready hit
# ---------------------------------------------------------------------------

class TestOwnerAndReadyHit:
    def test_owner_separates_and_publishes(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep = _Separator()
        out = tmp_path / "session"
        result = get_or_create_cached_stems(
            video_id="test_owner",
            role=ROLE_VOCAL,
            audio_path=tmp_path / "a.mp3",
            session_output_dir=out,
            separate_fn=sep,
            check_cancelled=_noop_cancel,
        )
        assert sep.calls == 1
        assert set(result) == set(_VOCAL_STEMS)
        # Published to the deterministic role dir + state is ready.
        published = _stems_dir_for("test_owner", ROLE_VOCAL)
        assert published.is_dir()
        state = StemCacheCoordinator().get_state("test_owner", ROLE_VOCAL)
        assert state is not None and state.status == "ready"
        # Lock released by the owner's finally.
        assert clean_redis.get(_stem_lock_key("test_owner", ROLE_VOCAL)) is None

    def test_second_request_is_ready_hit_no_separation(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep = _Separator()
        common = dict(
            video_id="test_hit",
            role=ROLE_VOCAL,
            audio_path=tmp_path / "a.mp3",
            separate_fn=sep,
            check_cancelled=_noop_cancel,
        )
        get_or_create_cached_stems(session_output_dir=tmp_path / "s1", **common)
        assert sep.calls == 1
        result = get_or_create_cached_stems(session_output_dir=tmp_path / "s2", **common)
        assert sep.calls == 1  # ready hit, no re-separation
        assert set(result) == set(_VOCAL_STEMS)


# ---------------------------------------------------------------------------
# Single-flight: two threads, one separation
# ---------------------------------------------------------------------------

def _spawn(target, *args):
    results: dict[str, object] = {}

    def run(key):
        try:
            results[key] = target(*args)
        except BaseException as exc:  # noqa: BLE001 - test captures all
            results[key] = exc

    return run, results


class TestSingleFlight:
    def test_two_threads_same_key_separate_once(self, clean_redis, cache_dir, fast_coord, tmp_path):
        gate = threading.Event()
        sep = _Separator(block_event=gate)
        results: dict[str, object] = {}

        def worker(key):
            try:
                results[key] = get_or_create_cached_stems(
                    video_id="test_sf",
                    role=ROLE_VOCAL,
                    audio_path=tmp_path / "a.mp3",
                    session_output_dir=tmp_path / f"sess_{key}",
                    separate_fn=sep,
                    check_cancelled=_noop_cancel,
                )
            except BaseException as exc:  # noqa: BLE001
                results[key] = exc

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        sep.started.wait(timeout=5)  # ensure the owner is mid-separation
        t2.start()
        time.sleep(0.5)  # let the waiter enter its poll loop
        gate.set()  # release the blocked separation
        t1.join(timeout=20)
        t2.join(timeout=20)

        assert sep.calls == 1, "separator must run exactly once for the shared key"
        for key in ("a", "b"):
            assert isinstance(results[key], dict), results[key]
            assert set(results[key]) == set(_VOCAL_STEMS)

    def test_different_roles_same_video_separate_independently(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep_v = _Separator(stems=_VOCAL_STEMS)
        sep_i = _Separator(stems=_INSTRUMENTAL_STEMS)
        get_or_create_cached_stems(
            video_id="test_roles", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "v", separate_fn=sep_v, check_cancelled=_noop_cancel,
        )
        get_or_create_cached_stems(
            video_id="test_roles", role=ROLE_INSTRUMENTAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "i", separate_fn=sep_i, check_cancelled=_noop_cancel,
        )
        assert sep_v.calls == 1 and sep_i.calls == 1

    def test_different_videos_separate_independently(self, clean_redis, cache_dir, fast_coord, tmp_path):
        sep1 = _Separator()
        sep2 = _Separator()
        get_or_create_cached_stems(
            video_id="test_v1", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "1", separate_fn=sep1, check_cancelled=_noop_cancel,
        )
        get_or_create_cached_stems(
            video_id="test_v2", role=ROLE_VOCAL, audio_path=tmp_path / "b.mp3",
            session_output_dir=tmp_path / "2", separate_fn=sep2, check_cancelled=_noop_cancel,
        )
        assert sep1.calls == 1 and sep2.calls == 1


# ---------------------------------------------------------------------------
# Lease renewal prevents takeover during a long (blocked) separation
# ---------------------------------------------------------------------------

class TestLeaseRenewal:
    def test_renewal_holds_lease_through_long_separation(self, clean_redis, cache_dir, fast_coord, tmp_path):
        gate = threading.Event()
        sep = _Separator(block_event=gate)
        result_box: dict[str, object] = {}

        def owner():
            try:
                result_box["r"] = get_or_create_cached_stems(
                    video_id="test_renew", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                    session_output_dir=tmp_path / "owner", separate_fn=sep,
                    check_cancelled=_noop_cancel,
                )
            except BaseException as exc:  # noqa: BLE001
                result_box["r"] = exc

        t = threading.Thread(target=owner)
        t.start()
        sep.started.wait(timeout=5)

        # Block for well past the 2s lease; renewal (1s interval) must keep it alive.
        lease_secs = settings.stem_lock_lease_seconds
        time.sleep(lease_secs * 2 + 0.5)
        coord = StemCacheCoordinator()
        # A would-be taker cannot acquire because the lock is still held (renewed).
        assert coord.acquire("test_renew", ROLE_VOCAL) is None

        gate.set()
        t.join(timeout=20)
        assert isinstance(result_box["r"], dict)
        assert sep.calls == 1


# ---------------------------------------------------------------------------
# Simulated owner death allows takeover after lease expiry
# ---------------------------------------------------------------------------

class TestOwnerDeathTakeover:
    def test_takeover_after_lease_expiry(self, clean_redis, cache_dir, fast_coord, tmp_path):
        # Simulate a dead owner: acquire a lease and never renew/release it.
        coord = StemCacheCoordinator()
        dead = coord.acquire("test_dead", ROLE_VOCAL)
        assert dead is not None

        sep = _Separator()
        # New caller competes; it must wait, observe the stale lease, then take over.
        result = get_or_create_cached_stems(
            video_id="test_dead", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "taker", separate_fn=sep,
            check_cancelled=_noop_cancel,
        )
        assert sep.calls == 1
        assert set(result) == set(_VOCAL_STEMS)
        state = coord.get_state("test_dead", ROLE_VOCAL)
        assert state is not None and state.status == "ready"


# ---------------------------------------------------------------------------
# Owner that lost its lease cannot publish over the new owner
# ---------------------------------------------------------------------------

class TestLostLeaseCannotPublish:
    def test_owner_aborts_when_lease_lost_before_publish(self, clean_redis, cache_dir, fast_coord, tmp_path):
        coord = StemCacheCoordinator()
        gate = threading.Event()
        sep = _Separator(block_event=gate)
        box: dict[str, object] = {}

        def owner():
            try:
                box["r"] = get_or_create_cached_stems(
                    video_id="test_lost", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                    session_output_dir=tmp_path / "owner", separate_fn=sep,
                    check_cancelled=_noop_cancel,
                )
            except BaseException as exc:  # noqa: BLE001
                box["r"] = exc

        # Stop renewals from keeping the lease alive: force every renew to fail so
        # the owner detects a lost lease while blocked in separation.
        import musicmixer.services.song_cache as sc
        orig_renew = sc.StemCacheCoordinator.renew
        sc.StemCacheCoordinator.renew = lambda self, lease: False  # type: ignore[assignment]
        try:
            t = threading.Thread(target=owner)
            t.start()
            sep.started.wait(timeout=5)
            # Let the lease expire and renewal-failure be observed.
            time.sleep(settings.stem_lock_lease_seconds + settings.stem_lock_renew_interval_seconds + 1)
            # A different worker takes over the now-free lock.
            taker = coord.acquire("test_lost", ROLE_VOCAL)
            assert taker is not None
            gate.set()  # release the original owner's separation
            t.join(timeout=20)
        finally:
            sc.StemCacheCoordinator.renew = orig_renew  # type: ignore[assignment]

        # The original owner must NOT have published — it raised instead.
        assert isinstance(box["r"], StemSeparationError), box["r"]
        # The new owner's lock is intact (the loser did not release/overwrite it).
        assert clean_redis.get(_stem_lock_key("test_lost", ROLE_VOCAL)) == taker.owner_token


# ---------------------------------------------------------------------------
# Cancellation of a waiter does not cancel the owner
# ---------------------------------------------------------------------------

class TestWaiterCancellation:
    def test_cancelled_waiter_stops_without_disturbing_owner(self, clean_redis, cache_dir, fast_coord, tmp_path):
        coord = StemCacheCoordinator()
        # An owner is mid-flight: hold a real lease (processing) that we keep alive.
        owner_lease = coord.acquire("test_cancel", ROLE_VOCAL)
        assert owner_lease is not None

        class _Cancel(Exception):
            pass

        cancel_flag = threading.Event()

        def check_cancelled():
            if cancel_flag.is_set():
                raise _Cancel()

        box: dict[str, object] = {}

        def waiter():
            try:
                box["r"] = get_or_create_cached_stems(
                    video_id="test_cancel", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                    session_output_dir=tmp_path / "waiter", separate_fn=_Separator(),
                    check_cancelled=check_cancelled,
                )
            except BaseException as exc:  # noqa: BLE001
                box["r"] = exc

        # Keep the owner's lease alive in the background so the waiter keeps polling.
        stop_renew = threading.Event()

        def keep_alive():
            while not stop_renew.wait(0.5):
                coord.renew(owner_lease)

        ka = threading.Thread(target=keep_alive, daemon=True)
        ka.start()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(1.0)  # let the waiter poll a few times
        cancel_flag.set()
        t.join(timeout=10)
        stop_renew.set()

        assert isinstance(box["r"], _Cancel), box["r"]
        # The owner's lock and processing state are untouched by the cancelled waiter.
        assert clean_redis.get(_stem_lock_key("test_cancel", ROLE_VOCAL)) == owner_lease.owner_token
        state = coord.get_state("test_cancel", ROLE_VOCAL)
        assert state is not None and state.status == "processing"


# ---------------------------------------------------------------------------
# Redis-outage fallback runs local separation
# ---------------------------------------------------------------------------

class TestRedisOutageFallback:
    def test_outage_falls_back_to_local_separation(self, clean_redis, cache_dir, monkeypatch, tmp_path):
        import redis as redis_lib

        class _DeadRedis:
            def __getattr__(self, _name):
                def _raise(*a, **k):
                    raise redis_lib.ConnectionError("simulated outage")
                return _raise

        # Patch the accessor so every coordination call hits a connection error.
        monkeypatch.setattr(song_cache, "_get_redis", lambda: _DeadRedis())

        sep = _Separator()
        out = tmp_path / "session"
        result = get_or_create_cached_stems(
            video_id="test_outage", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=out, separate_fn=sep, check_cancelled=_noop_cancel,
        )
        # Degraded path: local separation ran into the session dir.
        assert sep.calls == 1
        assert set(result) == set(_VOCAL_STEMS)
        assert (out / "lead_vocals.wav").is_file()
