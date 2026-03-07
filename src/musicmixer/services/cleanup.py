"""TTL-based session cleanup.

Scans data directories for session subdirectories older than the configured
TTL and deletes them. Called at the start of each new remix request to keep
disk usage bounded without requiring an external cron job.
"""

import logging
import shutil
import time
from pathlib import Path

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Subdirectories under data_dir that contain per-session folders
_SESSION_SUBDIRS = ("uploads", "stems", "remixes")


def cleanup_expired_sessions() -> int:
    """Delete session directories older than session_ttl_hours.

    Returns the number of session directories deleted.
    """
    ttl_seconds = settings.session_ttl_hours * 3600
    cutoff = time.time() - ttl_seconds
    deleted_count = 0

    for subdir_name in _SESSION_SUBDIRS:
        subdir = settings.data_dir / subdir_name
        if not subdir.is_dir():
            continue

        for session_dir in subdir.iterdir():
            if not session_dir.is_dir():
                continue

            try:
                mtime = session_dir.stat().st_mtime
            except OSError:
                continue

            if mtime < cutoff:
                try:
                    shutil.rmtree(session_dir)
                    deleted_count += 1
                    logger.info(
                        "Cleaned up expired session dir: %s (age: %.1fh)",
                        session_dir,
                        (time.time() - mtime) / 3600,
                    )
                except OSError:
                    logger.warning(
                        "Failed to delete expired session dir: %s",
                        session_dir,
                        exc_info=True,
                    )

    if deleted_count > 0:
        logger.info("Session cleanup: deleted %d expired directories", deleted_count)

    return deleted_count
