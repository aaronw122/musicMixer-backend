"""Tests for the distributed stem-cache coordination primitives.

Covers the Phase 1 Redis lease + state machine: lock election/contention,
token-checked renew/release, stale-lock takeover, and the mark_failed transition
with attempt-driven exponential backoff and failed-state TTL.

These run against a REAL Redis (same convention as test_song_cache.py): the
``clean_redis`` fixture cleans up ``song:test_*`` keys after each test, so all
tests key off ``test_``-prefixed video IDs.
"""

import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from musicmixer.config import settings
from musicmixer.services.song_cache import (
    ROLE_INSTRUMENTAL,
    ROLE_VOCAL,
    STEM_ERROR_INVALID_INPUT,
    STEM_ERROR_TRANSIENT,
    StemCacheCoordinator,
    StemCacheState,
    StemLease,
    _get_redis,
    _retry_delay_seconds,
    _stem_lock_key,
    _stems_key,
    _stem_state_key,
    sweep_orphaned_staging_dirs,
)


@pytest.fixture
def clean_redis():
    """Fresh Redis client; clean up test keys after each test."""
    import musicmixer.services.song_cache as _mod
    _mod._redis_client = None
    r = _get_redis()
    yield r
    for key in r.scan_iter("song:test_*"):
        r.delete(key)
    _mod._redis_client = None


@pytest.fixture
def short_lease(monkeypatch):
    """Use a sub-second lease so expiry/takeover tests run fast."""
    monkeypatch.setattr(settings, "stem_lock_lease_seconds", 1)
    return settings.stem_lock_lease_seconds


def _coordinator() -> StemCacheCoordinator:
    return StemCacheCoordinator()


# ---------------------------------------------------------------------------
# Filesystem helpers for Phase 2 (manifest / publication / reconciliation)
# ---------------------------------------------------------------------------

# Known valid stem shapes per role (mirror _VALID_STEM_SETS_BY_ROLE).
_VOCAL_STEMS = ("lead_vocals", "backing_vocals", "instrumental")
_INSTRUMENTAL_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")


def _write_wav(path: Path, *, payload_bytes: int = 4096) -> None:
    """Write a minimal but structurally-valid RIFF/WAVE file (mono 16-bit PCM)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\x00\x00" * (payload_bytes // 2)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 8000, 16000, 2, 16)
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    riff_size = 4 + len(fmt) + len(data_chunk)
    header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")
    path.write_bytes(header + fmt + data_chunk)


def _make_role_dir(base: Path, stems: tuple[str, ...]) -> Path:
    """Create a directory of valid WAVs named after ``stems``."""
    base.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        _write_wav(base / f"{stem}.wav")
    return base


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Point ``song_cache_dir`` at a tmp dir so disk-publication tests are isolated."""
    root = tmp_path / "song_cache"
    root.mkdir()
    monkeypatch.setattr(settings, "song_cache_dir", root)
    return root


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestStateSerialization:
    def test_round_trip_minimal(self):
        state = StemCacheState(status="processing", owner_token="abc", attempt=2)
        restored = StemCacheState.from_hash(state.to_hash())
        assert restored == state

    def test_round_trip_with_manifest_and_error(self):
        state = StemCacheState(
            status="failed",
            attempt=3,
            failed_at="2026-06-24T00:00:00+00:00",
            retry_after="2026-06-24T00:01:00+00:00",
            manifest=("lead_vocals.wav", "instrumental.wav"),
            error_code=STEM_ERROR_TRANSIENT,
            separator_version="v1",
        )
        restored = StemCacheState.from_hash(state.to_hash())
        assert restored == state

    def test_hash_has_only_string_values(self):
        state = StemCacheState(status="processing", attempt=5, manifest=("a.wav",))
        for value in state.to_hash().values():
            assert isinstance(value, str)


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------

class TestKeyBuilders:
    def test_state_and_lock_keys(self):
        assert _stem_state_key("test_vid", ROLE_VOCAL) == "song:test_vid:vocal:stem_state"
        assert _stem_lock_key("test_vid", ROLE_INSTRUMENTAL) == "song:test_vid:instrumental:stem_lock"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError):
            _stem_state_key("test_vid", "bogus")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            _stem_lock_key("test_vid", "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Acquire / contention
# ---------------------------------------------------------------------------

class TestAcquire:
    def test_missing_state_elects_one_owner(self, clean_redis):
        coord = _coordinator()
        assert coord.get_state("test_acq", ROLE_VOCAL) is None
        lease = coord.acquire("test_acq", ROLE_VOCAL)
        assert isinstance(lease, StemLease)
        assert lease.owner_token

        state = coord.get_state("test_acq", ROLE_VOCAL)
        assert state is not None
        assert state.status == "processing"
        assert state.owner_token == lease.owner_token
        assert state.separator_version == settings.stem_separator_version

    def test_second_concurrent_acquire_gets_contention(self, clean_redis):
        coord = _coordinator()
        first = coord.acquire("test_acq2", ROLE_VOCAL)
        assert first is not None
        second = coord.acquire("test_acq2", ROLE_VOCAL)
        assert second is None  # contention, not a second lease

    def test_different_roles_acquire_independently(self, clean_redis):
        coord = _coordinator()
        a = coord.acquire("test_acq3", ROLE_VOCAL)
        b = coord.acquire("test_acq3", ROLE_INSTRUMENTAL)
        assert a is not None and b is not None
        assert a.owner_token != b.owner_token


# ---------------------------------------------------------------------------
# Renew
# ---------------------------------------------------------------------------

class TestRenew:
    def test_renew_succeeds_for_owner(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_renew", ROLE_VOCAL)
        assert lease is not None
        assert coord.renew(lease) is True

    def test_renew_fails_for_non_owner_token(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_renew2", ROLE_VOCAL)
        assert lease is not None
        impostor = StemLease(video_id="test_renew2", role=ROLE_VOCAL, owner_token="not-the-owner")
        assert coord.renew(impostor) is False
        # The real owner can still renew — impostor did not disturb the lock.
        assert coord.renew(lease) is True

    def test_renew_fails_after_different_token_holds_lock(self, clean_redis, short_lease):
        import time

        coord = _coordinator()
        first = coord.acquire("test_renew3", ROLE_VOCAL)
        assert first is not None
        time.sleep(short_lease + 0.5)  # lease expires
        second = coord.acquire("test_renew3", ROLE_VOCAL)  # takeover
        assert second is not None
        assert second.owner_token != first.owner_token
        # The old owner can no longer renew — a different token holds the lock.
        assert coord.renew(first) is False
        assert coord.renew(second) is True


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_succeeds_for_owner(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_rel", ROLE_VOCAL)
        assert lease is not None
        assert coord.release(lease) is True
        # Lock gone — a fresh acquire now succeeds.
        assert coord.acquire("test_rel", ROLE_VOCAL) is not None

    def test_non_owner_release_does_not_delete_lock(self, clean_redis):
        coord = _coordinator()
        owner = coord.acquire("test_rel2", ROLE_VOCAL)
        assert owner is not None
        impostor = StemLease(video_id="test_rel2", role=ROLE_VOCAL, owner_token="other")
        assert coord.release(impostor) is False
        # Owner's lock survived: a fresh acquire still hits contention.
        assert coord.acquire("test_rel2", ROLE_VOCAL) is None
        assert clean_redis.get(_stem_lock_key("test_rel2", ROLE_VOCAL)) == owner.owner_token


# ---------------------------------------------------------------------------
# Takeover semantics
# ---------------------------------------------------------------------------

class TestTakeover:
    def test_active_lock_prevents_acquire(self, clean_redis):
        coord = _coordinator()
        first = coord.acquire("test_take", ROLE_VOCAL)
        assert first is not None
        assert coord.acquire("test_take", ROLE_VOCAL) is None

    def test_stale_lock_permits_fresh_acquire(self, clean_redis, short_lease):
        import time

        coord = _coordinator()
        first = coord.acquire("test_take2", ROLE_VOCAL)
        assert first is not None
        time.sleep(short_lease + 0.5)  # lease expires -> lock auto-removed
        second = coord.acquire("test_take2", ROLE_VOCAL)
        assert second is not None
        assert second.owner_token != first.owner_token


# ---------------------------------------------------------------------------
# mark_failed: backoff schedule + TTL
# ---------------------------------------------------------------------------

class TestMarkFailed:
    def test_records_bounded_error_code_and_fields(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_fail", ROLE_VOCAL)
        assert lease is not None
        before = datetime.now(timezone.utc)
        state = coord.mark_failed(lease, STEM_ERROR_TRANSIENT)

        assert state.status == "failed"
        assert state.error_code == STEM_ERROR_TRANSIENT
        assert state.attempt == 1
        assert state.failed_at is not None
        assert _parse(state.failed_at) >= before
        assert state.retry_after is not None
        # Lock released as part of failing.
        assert clean_redis.get(_stem_lock_key("test_fail", ROLE_VOCAL)) is None

    def test_only_lock_owner_release_happens(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_fail_owner", ROLE_VOCAL)
        assert lease is not None
        coord.mark_failed(lease, STEM_ERROR_TRANSIENT)
        # Failed state persisted with no owner_token.
        state = coord.get_state("test_fail_owner", ROLE_VOCAL)
        assert state is not None
        assert state.owner_token is None

    def test_transient_retry_after_follows_jittered_exponential_schedule(self, clean_redis):
        coord = _coordinator()
        base = settings.stem_retry_transient_base_seconds
        cap = settings.stem_retry_backoff_cap_seconds

        for attempt in range(1, settings.stem_retry_max_attempts):
            lease = coord.acquire("test_backoff", ROLE_VOCAL)
            assert lease is not None
            state = coord.mark_failed(lease, STEM_ERROR_TRANSIENT)
            assert state.attempt == attempt
            assert state.retry_after is not None

            capped = min(base * 2 ** (attempt - 1), cap)
            delay = (_parse(state.retry_after) - _parse(state.failed_at)).total_seconds()
            # Equal jitter: delay lands in [capped/2, capped].
            assert capped / 2 - 1.0 <= delay <= capped + 1.0

    def test_invalid_input_uses_fixed_window(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_invalid", ROLE_VOCAL)
        assert lease is not None
        state = coord.mark_failed(lease, STEM_ERROR_INVALID_INPUT)
        assert state.retry_after is not None
        delay = (_parse(state.retry_after) - _parse(state.failed_at)).total_seconds()
        assert abs(delay - settings.stem_retry_invalid_input_seconds) < 1.0

    def test_attempt_increments_across_failures(self, clean_redis):
        coord = _coordinator()
        for expected_attempt in (1, 2, 3):
            lease = coord.acquire("test_increment", ROLE_VOCAL)
            assert lease is not None
            state = coord.mark_failed(lease, STEM_ERROR_TRANSIENT)
            assert state.attempt == expected_attempt

    def test_no_retry_after_max_attempts(self, clean_redis):
        coord = _coordinator()
        state = None
        for _ in range(settings.stem_retry_max_attempts):
            lease = coord.acquire("test_maxattempt", ROLE_VOCAL)
            assert lease is not None
            state = coord.mark_failed(lease, STEM_ERROR_TRANSIENT)

        assert state is not None
        assert state.attempt == settings.stem_retry_max_attempts
        assert state.retry_after is None  # held failed; no further retry scheduled

    def test_failed_state_ttl_is_applied(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_ttl", ROLE_VOCAL)
        assert lease is not None
        coord.mark_failed(lease, STEM_ERROR_TRANSIENT)

        ttl = clean_redis.ttl(_stem_state_key("test_ttl", ROLE_VOCAL))
        assert ttl > 0
        assert ttl <= settings.stem_failed_ttl_seconds

    def test_invalid_error_code_rejected(self, clean_redis):
        coord = _coordinator()
        lease = coord.acquire("test_badcode", ROLE_VOCAL)
        assert lease is not None
        with pytest.raises(ValueError):
            coord.mark_failed(lease, "boom")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _retry_delay_seconds unit
# ---------------------------------------------------------------------------

class TestRetryDelay:
    def test_transient_jittered_within_bounds(self):
        base = settings.stem_retry_transient_base_seconds
        cap = settings.stem_retry_backoff_cap_seconds
        for attempt, uncapped in [(1, base), (2, base * 2), (100, cap)]:
            capped = min(uncapped, cap)
            for _ in range(100):
                delay = _retry_delay_seconds(STEM_ERROR_TRANSIENT, attempt)
                # Equal jitter keeps half the backoff as a floor and never exceeds it.
                assert capped / 2 <= delay <= capped

    def test_jitter_actually_varies(self):
        samples = {_retry_delay_seconds(STEM_ERROR_TRANSIENT, 3) for _ in range(50)}
        assert len(samples) > 1

    def test_invalid_input_fixed(self):
        assert (
            _retry_delay_seconds(STEM_ERROR_INVALID_INPUT, 1)
            == settings.stem_retry_invalid_input_seconds
        )


# ---------------------------------------------------------------------------
# WAV integrity + manifest validation
# ---------------------------------------------------------------------------

class TestWavIntegrity:
    def test_valid_wav_accepted(self, tmp_path):
        from musicmixer.services.song_cache import _is_intact_wav

        p = tmp_path / "lead_vocals.wav"
        _write_wav(p)
        assert _is_intact_wav(p) is True

    def test_zero_byte_wav_rejected(self, tmp_path):
        from musicmixer.services.song_cache import _is_intact_wav

        p = tmp_path / "lead_vocals.wav"
        p.write_bytes(b"")
        assert _is_intact_wav(p) is False

    def test_non_riff_header_rejected(self, tmp_path):
        from musicmixer.services.song_cache import _is_intact_wav

        p = tmp_path / "lead_vocals.wav"
        p.write_bytes(b"NOTAWAVE" + b"\x00" * 4096)
        assert _is_intact_wav(p) is False

    def test_truncated_below_floor_rejected(self, tmp_path):
        from musicmixer.services.song_cache import _is_intact_wav

        p = tmp_path / "lead_vocals.wav"
        p.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # valid header, too small
        assert _is_intact_wav(p) is False

    def test_validate_role_dir_builds_manifest(self, tmp_path):
        from musicmixer.services.song_cache import _validate_role_dir

        d = _make_role_dir(tmp_path / "vocal", _VOCAL_STEMS)
        manifest = _validate_role_dir(ROLE_VOCAL, d)
        assert manifest == tuple(sorted(f"{s}.wav" for s in _VOCAL_STEMS))

    def test_validate_role_dir_rejects_wrong_role(self, tmp_path):
        from musicmixer.services.song_cache import _validate_role_dir

        # Instrumental stem set under the vocal role must not validate.
        d = _make_role_dir(tmp_path / "x", _INSTRUMENTAL_STEMS)
        assert _validate_role_dir(ROLE_VOCAL, d) is None

    def test_validate_role_dir_rejects_partial(self, tmp_path):
        from musicmixer.services.song_cache import _validate_role_dir

        d = _make_role_dir(tmp_path / "x", ("lead_vocals",))  # incomplete
        assert _validate_role_dir(ROLE_VOCAL, d) is None

    def test_validate_role_dir_rejects_truncated_member(self, tmp_path):
        from musicmixer.services.song_cache import _validate_role_dir

        d = _make_role_dir(tmp_path / "vocal", _VOCAL_STEMS)
        (d / "instrumental.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")  # truncate one
        assert _validate_role_dir(ROLE_VOCAL, d) is None


# ---------------------------------------------------------------------------
# Lazy disk adoption (reconcile_disk)
# ---------------------------------------------------------------------------

class TestReconcileDisk:
    def test_valid_vocal_cache_adopted_as_ready(self, clean_redis, cache_dir):
        coord = _coordinator()
        _make_role_dir(cache_dir / "test_recon_v" / "vocal", _VOCAL_STEMS)

        state = coord.reconcile_disk("test_recon_v", ROLE_VOCAL)
        assert state is not None
        assert state.status == "ready"
        assert state.separator_version == settings.stem_separator_version
        assert state.manifest == tuple(sorted(f"{s}.wav" for s in _VOCAL_STEMS))
        assert state.completed_at is not None
        # State persisted and legacy pointer written.
        persisted = coord.get_state("test_recon_v", ROLE_VOCAL)
        assert persisted is not None and persisted.status == "ready"
        assert clean_redis.get(_stems_key("test_recon_v", ROLE_VOCAL)) == str(
            cache_dir / "test_recon_v" / "vocal"
        )

    def test_valid_instrumental_cache_adopted_as_ready(self, clean_redis, cache_dir):
        coord = _coordinator()
        _make_role_dir(cache_dir / "test_recon_i" / "instrumental", _INSTRUMENTAL_STEMS)
        state = coord.reconcile_disk("test_recon_i", ROLE_INSTRUMENTAL)
        assert state is not None and state.status == "ready"

    def test_partial_stems_not_adopted(self, clean_redis, cache_dir):
        coord = _coordinator()
        _make_role_dir(cache_dir / "test_recon_p" / "vocal", ("lead_vocals",))
        assert coord.reconcile_disk("test_recon_p", ROLE_VOCAL) is None
        assert coord.get_state("test_recon_p", ROLE_VOCAL) is None

    def test_wrong_role_stems_not_adopted(self, clean_redis, cache_dir):
        coord = _coordinator()
        # instrumental shape on disk under the vocal path
        _make_role_dir(cache_dir / "test_recon_w" / "vocal", _INSTRUMENTAL_STEMS)
        assert coord.reconcile_disk("test_recon_w", ROLE_VOCAL) is None

    def test_zero_byte_wav_not_adopted(self, clean_redis, cache_dir):
        coord = _coordinator()
        d = _make_role_dir(cache_dir / "test_recon_z" / "vocal", _VOCAL_STEMS)
        (d / "instrumental.wav").write_bytes(b"")
        assert coord.reconcile_disk("test_recon_z", ROLE_VOCAL) is None

    def test_does_not_overwrite_existing_state(self, clean_redis, cache_dir):
        coord = _coordinator()
        _make_role_dir(cache_dir / "test_recon_e" / "vocal", _VOCAL_STEMS)
        lease = coord.acquire("test_recon_e", ROLE_VOCAL)  # state = processing
        assert lease is not None
        assert coord.reconcile_disk("test_recon_e", ROLE_VOCAL) is None
        assert coord.get_state("test_recon_e", ROLE_VOCAL).status == "processing"

    def test_invalid_dir_does_not_promote_legacy_pointer(self, clean_redis, cache_dir):
        coord = _coordinator()
        # Stale legacy pointer set, but the on-disk dir is invalid (partial).
        _make_role_dir(cache_dir / "test_recon_lp" / "vocal", ("lead_vocals",))
        clean_redis.set(
            _stems_key("test_recon_lp", ROLE_VOCAL),
            str(cache_dir / "test_recon_lp" / "vocal"),
        )
        assert coord.reconcile_disk("test_recon_lp", ROLE_VOCAL) is None
        assert coord.get_state("test_recon_lp", ROLE_VOCAL) is None


# ---------------------------------------------------------------------------
# Fenced publication (mark_ready)
# ---------------------------------------------------------------------------

class TestMarkReady:
    def test_owner_publishes_ready_with_manifest_and_version(self, clean_redis, cache_dir):
        coord = _coordinator()
        lease = coord.acquire("test_pub", ROLE_VOCAL)
        assert lease is not None
        staging = _make_role_dir(
            cache_dir / "test_pub" / ".vocal.staging.work", _VOCAL_STEMS
        )
        before = datetime.now(timezone.utc)
        state = coord.mark_ready(lease, staging)

        assert state.status == "ready"
        assert state.manifest == tuple(sorted(f"{s}.wav" for s in _VOCAL_STEMS))
        assert state.separator_version == settings.stem_separator_version
        assert state.completed_at is not None
        assert _parse(state.completed_at) >= before
        # Files atomically moved into the published role dir.
        published = cache_dir / "test_pub" / "vocal"
        assert published.is_dir()
        assert {p.name for p in published.glob("*.wav")} == set(state.manifest)
        # Staging consumed by the move.
        assert not staging.exists()
        # Legacy :stems pointer written.
        assert clean_redis.get(_stems_key("test_pub", ROLE_VOCAL)) == str(published)
        # Lock is NOT released by mark_ready (caller's finally owns release).
        assert clean_redis.get(_stem_lock_key("test_pub", ROLE_VOCAL)) == lease.owner_token

    def test_non_owner_cannot_publish(self, clean_redis, cache_dir):
        coord = _coordinator()
        owner = coord.acquire("test_pub_fence", ROLE_VOCAL)
        assert owner is not None
        staging = _make_role_dir(
            cache_dir / "test_pub_fence" / ".vocal.staging.x", _VOCAL_STEMS
        )
        impostor = StemLease(
            video_id="test_pub_fence", role=ROLE_VOCAL, owner_token="not-the-owner"
        )
        with pytest.raises(PermissionError):
            coord.mark_ready(impostor, staging)
        # No publication happened.
        assert not (cache_dir / "test_pub_fence" / "vocal").exists()
        assert coord.get_state("test_pub_fence", ROLE_VOCAL).status == "processing"

    def test_lost_lease_cannot_publish(self, clean_redis, cache_dir, short_lease):
        import time

        coord = _coordinator()
        lease = coord.acquire("test_pub_lost", ROLE_VOCAL)
        assert lease is not None
        time.sleep(short_lease + 0.5)  # lease expires
        taker = coord.acquire("test_pub_lost", ROLE_VOCAL)  # someone else takes over
        assert taker is not None and taker.owner_token != lease.owner_token

        staging = _make_role_dir(
            cache_dir / "test_pub_lost" / ".vocal.staging.x", _VOCAL_STEMS
        )
        with pytest.raises(PermissionError):
            coord.mark_ready(lease, staging)  # old owner must not publish

    def test_invalid_staging_rejected(self, clean_redis, cache_dir):
        coord = _coordinator()
        lease = coord.acquire("test_pub_bad", ROLE_VOCAL)
        assert lease is not None
        staging = _make_role_dir(
            cache_dir / "test_pub_bad" / ".vocal.staging.x", ("lead_vocals",)
        )
        with pytest.raises(ValueError):
            coord.mark_ready(lease, staging)
        assert not (cache_dir / "test_pub_bad" / "vocal").exists()


# ---------------------------------------------------------------------------
# invalidate_ready
# ---------------------------------------------------------------------------

class TestInvalidateReady:
    def _publish(self, coord, cache_dir, video_id):
        lease = coord.acquire(video_id, ROLE_VOCAL)
        staging = _make_role_dir(
            cache_dir / video_id / ".vocal.staging.x", _VOCAL_STEMS
        )
        coord.mark_ready(lease, staging)
        coord.release(lease)
        return lease

    def test_version_mismatch_invalidates_ready(self, clean_redis, cache_dir, monkeypatch):
        coord = _coordinator()
        self._publish(coord, cache_dir, "test_inv_ver")
        ready = coord.get_state("test_inv_ver", ROLE_VOCAL)
        assert ready is not None and ready.status == "ready"

        # Bump the configured version: the ready record is now incompatible.
        monkeypatch.setattr(settings, "stem_separator_version", "v2")
        assert ready.separator_version != settings.stem_separator_version

        coord.invalidate_ready("test_inv_ver", ROLE_VOCAL)
        assert coord.get_state("test_inv_ver", ROLE_VOCAL) is None
        # A new lease can now be competed for.
        assert coord.acquire("test_inv_ver", ROLE_VOCAL) is not None

    def test_missing_file_invalidates_ready(self, clean_redis, cache_dir):
        from musicmixer.services.song_cache import _manifest_files_intact, _stems_dir_for

        coord = _coordinator()
        self._publish(coord, cache_dir, "test_inv_miss")
        ready = coord.get_state("test_inv_miss", ROLE_VOCAL)
        published = _stems_dir_for("test_inv_miss", ROLE_VOCAL)
        (published / "instrumental.wav").unlink()  # a published file disappears

        assert _manifest_files_intact(published, ready.manifest) is False
        coord.invalidate_ready("test_inv_miss", ROLE_VOCAL)
        assert coord.get_state("test_inv_miss", ROLE_VOCAL) is None
        assert clean_redis.get(_stems_key("test_inv_miss", ROLE_VOCAL)) is None

    def test_invalidate_noop_on_processing(self, clean_redis, cache_dir):
        coord = _coordinator()
        lease = coord.acquire("test_inv_proc", ROLE_VOCAL)
        assert lease is not None
        coord.invalidate_ready("test_inv_proc", ROLE_VOCAL)  # must not touch processing
        assert coord.get_state("test_inv_proc", ROLE_VOCAL).status == "processing"


# ---------------------------------------------------------------------------
# Crash recovery: disk dir promoted by next caller without recompute
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    def test_disk_dir_promoted_after_simulated_crash(self, clean_redis, cache_dir):
        # Owner crashed AFTER the atomic replace but BEFORE writing ready: the
        # role dir exists on disk, the lock has expired, no state key remains.
        _make_role_dir(cache_dir / "test_crash" / "vocal", _VOCAL_STEMS)
        coord = _coordinator()
        assert coord.get_state("test_crash", ROLE_VOCAL) is None

        # Next caller reconciles disk -> ready without recomputing separation.
        state = coord.reconcile_disk("test_crash", ROLE_VOCAL)
        assert state is not None and state.status == "ready"
        assert state.manifest == tuple(sorted(f"{s}.wav" for s in _VOCAL_STEMS))

    def test_stale_staging_never_counts_as_ready(self, clean_redis, cache_dir):
        # Only a staging sibling exists (crash before atomic replace). The
        # deterministic role dir is absent, so reconcile must NOT adopt anything.
        _make_role_dir(cache_dir / "test_stale" / ".vocal.staging.dead", _VOCAL_STEMS)
        coord = _coordinator()
        assert coord.reconcile_disk("test_stale", ROLE_VOCAL) is None
        assert coord.get_state("test_stale", ROLE_VOCAL) is None


# ---------------------------------------------------------------------------
# Orphaned-staging sweeper
# ---------------------------------------------------------------------------

class TestSweeper:
    def test_removes_staging_without_live_lock(self, clean_redis, cache_dir):
        staging = _make_role_dir(
            cache_dir / "test_sweep" / ".vocal.staging.deadtoken", _VOCAL_STEMS
        )
        removed = sweep_orphaned_staging_dirs()
        assert removed == 1
        assert not staging.exists()

    def test_keeps_staging_for_live_lock(self, clean_redis, cache_dir):
        coord = _coordinator()
        lease = coord.acquire("test_sweep_live", ROLE_VOCAL)
        assert lease is not None
        staging = _make_role_dir(
            cache_dir / "test_sweep_live" / f".vocal.staging.{lease.owner_token}",
            _VOCAL_STEMS,
        )
        removed = sweep_orphaned_staging_dirs(coord)
        assert removed == 0
        assert staging.exists()

    def test_removes_old_swap_aside_dirs(self, clean_redis, cache_dir):
        old = _make_role_dir(
            cache_dir / "test_sweep_old" / ".vocal.old.abc123", _VOCAL_STEMS
        )
        removed = sweep_orphaned_staging_dirs()
        assert removed == 1
        assert not old.exists()

    def test_leaves_published_dirs_untouched(self, clean_redis, cache_dir):
        published = _make_role_dir(cache_dir / "test_sweep_keep" / "vocal", _VOCAL_STEMS)
        sweep_orphaned_staging_dirs()
        assert published.is_dir()
