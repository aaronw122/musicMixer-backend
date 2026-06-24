"""Tests for the distributed stem-cache coordination primitives.

Covers the Phase 1 Redis lease + state machine: lock election/contention,
token-checked renew/release, stale-lock takeover, and the mark_failed transition
with attempt-driven exponential backoff and failed-state TTL.

These run against a REAL Redis (same convention as test_song_cache.py): the
``clean_redis`` fixture cleans up ``song:test_*`` keys after each test, so all
tests key off ``test_``-prefixed video IDs.
"""

from datetime import datetime, timezone

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
    _stem_state_key,
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

    def test_transient_retry_after_follows_exponential_schedule(self, clean_redis):
        coord = _coordinator()
        base = settings.stem_retry_transient_base_seconds
        cap = settings.stem_retry_backoff_cap_seconds

        for attempt in range(1, settings.stem_retry_max_attempts):
            lease = coord.acquire("test_backoff", ROLE_VOCAL)
            assert lease is not None
            state = coord.mark_failed(lease, STEM_ERROR_TRANSIENT)
            assert state.attempt == attempt
            assert state.retry_after is not None

            expected = min(base * 2 ** (attempt - 1), cap)
            delay = (_parse(state.retry_after) - _parse(state.failed_at)).total_seconds()
            assert abs(delay - expected) < 1.0

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
    def test_transient_exponential_then_capped(self):
        base = settings.stem_retry_transient_base_seconds
        cap = settings.stem_retry_backoff_cap_seconds
        assert _retry_delay_seconds(STEM_ERROR_TRANSIENT, 1) == base
        assert _retry_delay_seconds(STEM_ERROR_TRANSIENT, 2) == base * 2
        assert _retry_delay_seconds(STEM_ERROR_TRANSIENT, 100) == cap

    def test_invalid_input_fixed(self):
        assert (
            _retry_delay_seconds(STEM_ERROR_INVALID_INPUT, 1)
            == settings.stem_retry_invalid_input_seconds
        )
