"""TTL-based session cleanup.

Purges expired in-memory sessions and their file directories. Also cleans
orphaned directories (from server crashes) using mtime as a fallback clock.

Called periodically by a background asyncio task and once on startup.
"""

import logging
import shutil
import threading
import time
from pathlib import Path

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Subdirectories under data_dir that contain per-session folders
_SESSION_SUBDIRS = ("uploads", "stems", "remixes")


def cleanup_expired_sessions(
    sessions: dict,
    sessions_lock: threading.Lock,
) -> int:
    """Purge expired in-memory sessions and their file directories.

    Two responsibilities:
    1. Remove in-memory sessions past their TTL (respecting status-based guards).
    2. Delete file directories for purged sessions AND orphaned directories
       whose mtime exceeds the TTL.

    Returns the total number of cleaned items (sessions + orphaned dirs).
    """
    ttl_seconds = settings.session_ttl_hours * 3600
    processing_timeout_seconds = settings.processing_timeout_minutes * 60
    queue_timeout_seconds = settings.queue_entry_ttl_minutes * 60
    now = time.time()
    cleaned = 0

    # --- Step 1: Identify expired in-memory sessions ---
    expired_ids: set[str] = set()

    with sessions_lock:
        for session_id, session in list(sessions.items()):
            age_seconds = now - session.created_at

            if session.status == "processing":
                if age_seconds < processing_timeout_seconds:
                    continue  # Pipeline actively working
                age_minutes = age_seconds / 60
                logger.warning(
                    "Purging stuck processing session %s "
                    "(age: %.0fm, timeout: %dm)",
                    session_id,
                    age_minutes,
                    settings.processing_timeout_minutes,
                )
                expired_ids.add(session_id)

            elif session.status == "queued":
                if age_seconds < queue_timeout_seconds:
                    continue  # Still waiting for a worker
                age_minutes = age_seconds / 60
                logger.warning(
                    "Purging abandoned queued session %s "
                    "(age: %.0fm, timeout: %dm)",
                    session_id,
                    age_minutes,
                    settings.queue_entry_ttl_minutes,
                )
                expired_ids.add(session_id)

            else:
                # "complete", "error", "cancelled" — standard TTL check
                if age_seconds > ttl_seconds:
                    expired_ids.add(session_id)

        # Remove expired sessions from dict while holding the lock
        for session_id in expired_ids:
            del sessions[session_id]
            cleaned += 1
            logger.info("Removed expired in-memory session: %s", session_id)

    # --- Step 2: Clean file directories ---
    # Build set of ALL known session IDs (those that survived cleanup)
    with sessions_lock:
        live_session_ids = set(sessions.keys())

    for subdir_name in _SESSION_SUBDIRS:
        subdir = settings.data_dir / subdir_name
        if not subdir.is_dir():
            continue

        for session_dir in subdir.iterdir():
            try:
                if not session_dir.is_dir():
                    continue
            except OSError:
                continue

            dir_session_id = session_dir.name

            # Already marked for deletion from in-memory purge
            if dir_session_id in expired_ids:
                _delete_dir(session_dir)
                continue

            # Live session — skip (it's still active or within TTL)
            if dir_session_id in live_session_ids:
                continue

            # Orphaned directory — no matching in-memory session.
            # Fall back to mtime for TTL check.
            try:
                mtime = session_dir.stat().st_mtime
            except OSError:
                continue

            if (now - mtime) > ttl_seconds:
                _delete_dir(session_dir)
                cleaned += 1
                logger.info(
                    "Cleaned up orphaned session dir: %s (age: %.1fh)",
                    session_dir,
                    (now - mtime) / 3600,
                )

    return cleaned


def _delete_dir(path: Path) -> None:
    """Delete a directory tree, catching OSError gracefully."""
    try:
        shutil.rmtree(path)
        logger.info("Deleted session directory: %s", path)
    except OSError:
        logger.warning(
            "Failed to delete session dir: %s",
            path,
            exc_info=True,
        )
