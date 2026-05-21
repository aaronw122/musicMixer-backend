"""Structured logging configuration for musicMixer backend.

Sets up JSON-formatted logging to both stdout and a rotating log file.
Call ``setup_logging()`` once at startup (before any log calls) to configure
the root logger.  All modules that use ``logging.getLogger(__name__)`` will
automatically inherit this configuration.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pythonjsonlogger.json import JsonFormatter

from musicmixer.config import settings

_LOG_DIR = settings.data_dir / "logs"
_LOG_FILE = _LOG_DIR / "musicmixer.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5

# Fields included in every JSON log record
_JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(message)s"


def setup_logging() -> None:
    """Configure the root logger with JSON formatting, stdout, and file handlers."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Build the JSON formatter (python-json-logger v4 API)
    formatter = JsonFormatter(
        fmt=_JSON_FORMAT,
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )

    # --- Root logger ---
    root = logging.getLogger()
    root.setLevel(log_level)

    # Remove any handlers that basicConfig or other libraries may have added
    root.handlers.clear()

    # Stdout handler (Docker/container runtimes capture stdout)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Rotating file handler
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpcore", "httpx", "watchfiles", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
