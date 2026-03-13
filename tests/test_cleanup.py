"""Tests for TTL-based session cleanup service."""

import logging
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from musicmixer.models import SessionState
from musicmixer.services.cleanup import cleanup_expired_sessions


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Create a temporary data directory with session subdirs."""
    for subdir in ("uploads", "stems", "remixes"):
        (tmp_path / subdir).mkdir()
    monkeypatch.setattr("musicmixer.services.cleanup.settings.data_dir", tmp_path)
    return tmp_path


@pytest.fixture()
def default_settings(monkeypatch):
    """Set default config values for cleanup tests."""
    monkeypatch.setattr(
        "musicmixer.services.cleanup.settings.session_ttl_hours", 3
    )
    monkeypatch.setattr(
        "musicmixer.services.cleanup.settings.processing_timeout_minutes", 20
    )
    monkeypatch.setattr(
        "musicmixer.services.cleanup.settings.queue_entry_ttl_minutes", 150
    )


def _make_session(status="complete", age_seconds=0):
    """Create a SessionState with a given status and age."""
    session = SessionState()
    session.status = status
    session.created_at = time.time() - age_seconds
    return session


def _make_session_dir(data_dir, session_id, subdir="uploads", age_seconds=0):
    """Create a session directory and backdate its mtime."""
    d = data_dir / subdir / session_id
    d.mkdir(parents=True, exist_ok=True)
    # Write a file so directory is non-empty
    (d / "test.txt").write_text("data")
    if age_seconds > 0:
        old_time = time.time() - age_seconds
        os.utime(d, (old_time, old_time))
    return d


class TestInMemoryCleanup:
    """In-memory session purge tests."""

    def test_removes_expired_complete_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        sessions = {
            "expired": _make_session("complete", age_seconds=4 * 3600),
            "fresh": _make_session("complete", age_seconds=1 * 3600),
        }

        cleaned = cleanup_expired_sessions(sessions, lock)

        assert "expired" not in sessions
        assert "fresh" in sessions
        assert cleaned >= 1

    def test_removes_expired_error_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        sessions = {
            "err": _make_session("error", age_seconds=4 * 3600),
        }

        cleanup_expired_sessions(sessions, lock)

        assert "err" not in sessions

    def test_keeps_unexpired_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        sessions = {
            "new": _make_session("complete", age_seconds=60),
        }

        cleaned = cleanup_expired_sessions(sessions, lock)

        assert "new" in sessions
        assert cleaned == 0

    def test_skips_young_processing_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        sessions = {
            "active": _make_session("processing", age_seconds=10 * 60),
        }

        cleanup_expired_sessions(sessions, lock)

        assert "active" in sessions

    def test_purges_stuck_processing_sessions(
        self, data_dir, default_settings, caplog
    ):
        lock = threading.Lock()
        sessions = {
            "stuck": _make_session("processing", age_seconds=25 * 60),
        }

        with caplog.at_level(logging.WARNING):
            cleanup_expired_sessions(sessions, lock)

        assert "stuck" not in sessions
        assert "Purging stuck processing session stuck" in caplog.text

    def test_skips_young_queued_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        sessions = {
            "waiting": _make_session("queued", age_seconds=60 * 60),
        }

        cleanup_expired_sessions(sessions, lock)

        assert "waiting" in sessions

    def test_purges_abandoned_queued_sessions(
        self, data_dir, default_settings, caplog
    ):
        lock = threading.Lock()
        sessions = {
            "abandoned": _make_session("queued", age_seconds=160 * 60),
        }

        with caplog.at_level(logging.WARNING):
            cleanup_expired_sessions(sessions, lock)

        assert "abandoned" not in sessions
        assert "Purging abandoned queued session abandoned" in caplog.text


class TestFileCleanup:
    """File directory cleanup tests."""

    def test_deletes_dirs_for_expired_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        session_id = "sess-expired"
        sessions = {
            session_id: _make_session("complete", age_seconds=4 * 3600),
        }
        d = _make_session_dir(data_dir, session_id, "uploads")

        cleanup_expired_sessions(sessions, lock)

        assert not d.exists()

    def test_keeps_dirs_for_live_sessions(self, data_dir, default_settings):
        lock = threading.Lock()
        session_id = "sess-live"
        sessions = {
            session_id: _make_session("complete", age_seconds=1 * 3600),
        }
        d = _make_session_dir(data_dir, session_id, "uploads")

        cleanup_expired_sessions(sessions, lock)

        assert d.exists()

    def test_orphaned_dir_deleted_by_mtime(self, data_dir, default_settings):
        """Orphaned dir (no in-memory session) deleted when mtime > TTL."""
        lock = threading.Lock()
        sessions = {}
        d = _make_session_dir(
            data_dir, "orphan-old", "stems", age_seconds=4 * 3600
        )

        cleaned = cleanup_expired_sessions(sessions, lock)

        assert not d.exists()
        assert cleaned >= 1

    def test_orphaned_dir_kept_when_young(self, data_dir, default_settings):
        """Orphaned dir kept when mtime is within TTL."""
        lock = threading.Lock()
        sessions = {}
        d = _make_session_dir(data_dir, "orphan-new", "stems", age_seconds=60)

        cleanup_expired_sessions(sessions, lock)

        assert d.exists()

    def test_cleans_across_all_subdirs(self, data_dir, default_settings):
        """Expired session dirs cleaned from uploads, stems, and remixes."""
        lock = threading.Lock()
        session_id = "sess-multi"
        sessions = {
            session_id: _make_session("complete", age_seconds=4 * 3600),
        }
        dirs = []
        for subdir in ("uploads", "stems", "remixes"):
            dirs.append(_make_session_dir(data_dir, session_id, subdir))

        cleanup_expired_sessions(sessions, lock)

        for d in dirs:
            assert not d.exists()


class TestOSErrorHandling:
    """Cleanup doesn't crash on missing or inaccessible directories."""

    def test_missing_subdir_doesnt_crash(self, tmp_path, default_settings, monkeypatch):
        """Cleanup works when a data subdir doesn't exist."""
        monkeypatch.setattr(
            "musicmixer.services.cleanup.settings.data_dir", tmp_path
        )
        # Don't create any subdirs — they're missing
        lock = threading.Lock()
        sessions = {}

        cleaned = cleanup_expired_sessions(sessions, lock)
        assert cleaned == 0

    def test_rmtree_oserror_handled_gracefully(
        self, data_dir, default_settings, caplog
    ):
        """OSError during rmtree is caught and logged."""
        lock = threading.Lock()
        sessions = {}
        d = _make_session_dir(
            data_dir, "fail-delete", "uploads", age_seconds=4 * 3600
        )

        with patch("musicmixer.services.cleanup.shutil.rmtree", side_effect=OSError("permission denied")):
            with caplog.at_level(logging.WARNING):
                # Should not raise
                cleanup_expired_sessions(sessions, lock)

        assert "Failed to delete session dir" in caplog.text

    def test_stat_oserror_handled_gracefully(
        self, data_dir, default_settings
    ):
        """OSError during stat() for orphaned dir is caught."""
        lock = threading.Lock()
        sessions = {}
        d = _make_session_dir(data_dir, "stat-fail", "uploads")

        original_stat = Path.stat

        def stat_that_fails_on_target(self_path, *args, **kwargs):
            if self_path.name == "stat-fail":
                raise OSError("gone")
            return original_stat(self_path, *args, **kwargs)

        with patch.object(Path, "stat", stat_that_fails_on_target):
            # Should not raise
            cleanup_expired_sessions(sessions, lock)


class TestLockBehavior:
    """Verify cleanup correctly uses the sessions lock."""

    def test_lock_acquired_and_released(self, data_dir, default_settings):
        """After cleanup, lock should be available (not held)."""
        lock = threading.Lock()
        sessions = {
            "old": _make_session("complete", age_seconds=4 * 3600),
        }

        cleanup_expired_sessions(sessions, lock)

        # Lock should be acquirable (not stuck)
        assert lock.acquire(timeout=1)
        lock.release()

    def test_concurrent_cleanup_is_safe(self, data_dir, default_settings):
        """Two concurrent cleanup calls don't corrupt state."""
        lock = threading.Lock()
        sessions = {
            f"sess-{i}": _make_session("complete", age_seconds=4 * 3600)
            for i in range(10)
        }

        errors = []

        def run_cleanup():
            try:
                cleanup_expired_sessions(sessions, lock)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cleanup) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        # All sessions should be cleaned (may be cleaned by different threads)
        assert len(sessions) == 0
