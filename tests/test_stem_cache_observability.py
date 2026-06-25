"""Observability tests for the Phase 5 stem-cache outcome instrumentation.

Asserts that ``get_or_create_cached_stems`` (and the coordinator primitives it
drives) emit the correct §9.6 outcome label — plus the expected timing/category
fields — on each path: ready_hit, owner_created, waited_then_hit, stale_takeover,
legacy_adopted, failed_backoff, invalidated.

Runs against the same REAL Redis convention as test_stem_cache_orchestration.py:
``test_``-prefixed video IDs, cleaned up after each test. Outcomes are read from
the ``extra=`` structured fields on captured log records (the project's
pipeline_metrics logging convention), not by parsing message strings.
"""

import logging
import struct
import threading
import time
from pathlib import Path

import pytest

import musicmixer.services.song_cache as song_cache
from musicmixer.config import settings
from musicmixer.services.song_cache import (
    ROLE_VOCAL,
    STEM_ERROR_TRANSIENT,
    StemCacheCoordinator,
    StemSeparationError,
    _get_redis,
    _stems_dir_for,
    get_or_create_cached_stems,
)


# ---------------------------------------------------------------------------
# Fixtures (mirror test_stem_cache_orchestration.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_redis():
    song_cache._redis_client = None
    r = _get_redis()
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
def capture_outcomes(caplog):
    """Capture INFO-level records emitted by song_cache and expose outcomes."""
    caplog.set_level(logging.INFO, logger="musicmixer.services.song_cache")
    return caplog


_VOCAL_STEMS = ("lead_vocals", "backing_vocals", "instrumental")


def _write_wav(path: Path, *, payload_bytes: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00\x00" * (payload_bytes // 2)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 16000, 2, 16)
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    riff_size = 4 + len(fmt) + len(data_chunk)
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    path.write_bytes(header + fmt + data_chunk)


class _Separator:
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


def _outcome_records(caplog):
    """All captured records carrying a stem-cache outcome, in order."""
    return [r for r in caplog.records if getattr(r, "stem_cache_outcome", None)]


def _outcomes(caplog) -> list[str]:
    return [r.stem_cache_outcome for r in _outcome_records(caplog)]


def _last_outcome(caplog, outcome: str):
    for r in reversed(_outcome_records(caplog)):
        if r.stem_cache_outcome == outcome:
            return r
    raise AssertionError(
        f"outcome {outcome!r} not emitted; saw {_outcomes(caplog)}"
    )


def _write_role_dir(cache_dir: Path, video_id: str, role: str, stems=_VOCAL_STEMS) -> Path:
    d = cache_dir / video_id / role
    for stem in stems:
        _write_wav(d / f"{stem}.wav")
    return d


# ---------------------------------------------------------------------------
# owner_created
# ---------------------------------------------------------------------------

class TestOwnerCreated:
    def test_owner_emits_owner_created_with_timing(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        sep = _Separator()
        get_or_create_cached_stems(
            video_id="test_obs_owner", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "s", separate_fn=sep, check_cancelled=_noop_cancel,
        )
        rec = _last_outcome(capture_outcomes, "owner_created")
        assert rec.video_id == "test_obs_owner"
        assert rec.role == ROLE_VOCAL
        assert isinstance(rec.separation_duration_s, float)
        assert hasattr(rec, "lease_renew_count")
        assert rec.separator_version == settings.stem_separator_version
        # An owner that just separated must not also report a ready_hit.
        assert "ready_hit" not in _outcomes(capture_outcomes)


# ---------------------------------------------------------------------------
# ready_hit
# ---------------------------------------------------------------------------

class TestReadyHit:
    def test_second_request_emits_ready_hit(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        sep = _Separator()
        common = dict(
            video_id="test_obs_hit", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            separate_fn=sep, check_cancelled=_noop_cancel,
        )
        get_or_create_cached_stems(session_output_dir=tmp_path / "s1", **common)
        capture_outcomes.clear()
        get_or_create_cached_stems(session_output_dir=tmp_path / "s2", **common)
        assert sep.calls == 1  # no re-separation
        rec = _last_outcome(capture_outcomes, "ready_hit")
        assert rec.video_id == "test_obs_hit"
        assert rec.file_count == len(_VOCAL_STEMS)
        assert "owner_created" not in _outcomes(capture_outcomes)


# ---------------------------------------------------------------------------
# waited_then_hit
# ---------------------------------------------------------------------------

class TestWaitedThenHit:
    def test_waiter_emits_waited_then_hit_with_wait_duration(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        gate = threading.Event()
        sep = _Separator(block_event=gate)
        results: dict[str, object] = {}

        def worker(key):
            try:
                results[key] = get_or_create_cached_stems(
                    video_id="test_obs_wait", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                    session_output_dir=tmp_path / f"sess_{key}", separate_fn=sep,
                    check_cancelled=_noop_cancel,
                )
            except BaseException as exc:  # noqa: BLE001
                results[key] = exc

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        sep.started.wait(timeout=5)
        t2.start()
        time.sleep(0.5)
        gate.set()
        t1.join(timeout=20)
        t2.join(timeout=20)

        assert sep.calls == 1
        outcomes = _outcomes(capture_outcomes)
        assert "owner_created" in outcomes
        rec = _last_outcome(capture_outcomes, "waited_then_hit")
        assert rec.video_id == "test_obs_wait"
        assert isinstance(rec.wait_duration_s, float)
        assert rec.file_count == len(_VOCAL_STEMS)


# ---------------------------------------------------------------------------
# stale_takeover
# ---------------------------------------------------------------------------

class TestStaleTakeover:
    def test_takeover_after_lease_expiry_emits_stale_takeover(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        # Simulate a dead owner: hold a lease and never renew/release it.
        coord = StemCacheCoordinator()
        dead = coord.acquire("test_obs_takeover", ROLE_VOCAL)
        assert dead is not None

        sep = _Separator()
        get_or_create_cached_stems(
            video_id="test_obs_takeover", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "taker", separate_fn=sep, check_cancelled=_noop_cancel,
        )
        assert sep.calls == 1
        rec = _last_outcome(capture_outcomes, "stale_takeover")
        assert rec.video_id == "test_obs_takeover"
        assert isinstance(rec.separation_duration_s, float)
        # A takeover must not be double-counted as a plain owner_created.
        assert "owner_created" not in _outcomes(capture_outcomes)


# ---------------------------------------------------------------------------
# legacy_adopted
# ---------------------------------------------------------------------------

class TestLegacyAdopted:
    def test_on_disk_dir_emits_legacy_adopted(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        # A valid on-disk role dir with no state key -> reconcile_disk adopts it.
        _write_role_dir(cache_dir, "test_obs_legacy", ROLE_VOCAL)
        assert clean_redis.exists("song:test_obs_legacy:vocal:stem_state") == 0

        sep = _Separator()
        result = get_or_create_cached_stems(
            video_id="test_obs_legacy", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "s", separate_fn=sep, check_cancelled=_noop_cancel,
        )
        assert sep.calls == 0  # adopted, no separation
        assert set(result) == set(_VOCAL_STEMS)
        rec = _last_outcome(capture_outcomes, "legacy_adopted")
        assert rec.video_id == "test_obs_legacy"
        assert rec.file_count == len(_VOCAL_STEMS)
        assert rec.separator_version == settings.stem_separator_version
        # Adoption is reported once as legacy_adopted, not also as ready_hit.
        assert "ready_hit" not in _outcomes(capture_outcomes)


# ---------------------------------------------------------------------------
# failed_backoff
# ---------------------------------------------------------------------------

class _FailingSeparator:
    def __init__(self, exc: BaseException):
        self.exc = exc
        self.calls = 0
        self.started = threading.Event()

    def __call__(self, audio_path, output_dir, progress_callback=None):
        self.calls += 1
        self.started.set()
        raise self.exc


class TestFailedBackoff:
    def test_owner_failure_emits_failed_backoff_with_error_code(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        sep = _FailingSeparator(RuntimeError("modal exploded"))
        with pytest.raises(StemSeparationError):
            get_or_create_cached_stems(
                video_id="test_obs_fail", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                session_output_dir=tmp_path / "s", separate_fn=sep, check_cancelled=_noop_cancel,
            )
        rec = _last_outcome(capture_outcomes, "failed_backoff")
        assert rec.video_id == "test_obs_fail"
        assert rec.error_code == STEM_ERROR_TRANSIENT
        assert rec.separator_version == settings.stem_separator_version

    def test_request_within_retry_window_emits_failed_backoff(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        # First request fails and schedules a retry window.
        sep = _FailingSeparator(RuntimeError("boom"))
        with pytest.raises(StemSeparationError):
            get_or_create_cached_stems(
                video_id="test_obs_backoff", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                session_output_dir=tmp_path / "s1", separate_fn=sep, check_cancelled=_noop_cancel,
            )
        state = StemCacheCoordinator().get_state("test_obs_backoff", ROLE_VOCAL)
        assert state is not None and state.status == "failed"
        assert state.retry_after is not None, "expected a scheduled retry window"

        capture_outcomes.clear()
        # Second request, still inside retry_after -> fast-fail with failed_backoff.
        sep2 = _Separator()
        with pytest.raises(StemSeparationError):
            get_or_create_cached_stems(
                video_id="test_obs_backoff", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
                session_output_dir=tmp_path / "s2", separate_fn=sep2, check_cancelled=_noop_cancel,
            )
        assert sep2.calls == 0  # blocked by backoff, no separation
        rec = _last_outcome(capture_outcomes, "failed_backoff")
        assert rec.video_id == "test_obs_backoff"
        assert rec.error_code == STEM_ERROR_TRANSIENT


# ---------------------------------------------------------------------------
# invalidated
# ---------------------------------------------------------------------------

class TestInvalidated:
    def test_invalidate_ready_emits_invalidated(self, clean_redis, cache_dir, fast_coord, tmp_path, capture_outcomes):
        sep = _Separator()
        get_or_create_cached_stems(
            video_id="test_obs_inval", role=ROLE_VOCAL, audio_path=tmp_path / "a.mp3",
            session_output_dir=tmp_path / "s", separate_fn=sep, check_cancelled=_noop_cancel,
        )
        state = StemCacheCoordinator().get_state("test_obs_inval", ROLE_VOCAL)
        assert state is not None and state.status == "ready"

        capture_outcomes.clear()
        StemCacheCoordinator().invalidate_ready("test_obs_inval", ROLE_VOCAL)
        rec = _last_outcome(capture_outcomes, "invalidated")
        assert rec.video_id == "test_obs_inval"
        assert rec.role == ROLE_VOCAL
        # The ready record is gone after invalidation.
        assert StemCacheCoordinator().get_state("test_obs_inval", ROLE_VOCAL) is None

    def test_invalidate_noop_on_missing_does_not_emit(self, clean_redis, cache_dir, fast_coord, capture_outcomes):
        StemCacheCoordinator().invalidate_ready("test_obs_missing", ROLE_VOCAL)
        assert "invalidated" not in _outcomes(capture_outcomes)
