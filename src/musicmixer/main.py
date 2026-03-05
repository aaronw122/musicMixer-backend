import asyncio
import json
import logging
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from musicmixer.api import health, remix
from musicmixer.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
_WORKER_ENV_VARS = ("UVICORN_WORKERS", "WEB_CONCURRENCY")


def _detect_multi_worker_mode() -> dict[str, int]:
    """Return worker env vars that indicate multi-worker mode (>1)."""
    multi_worker_values: dict[str, int] = {}
    for env_var in _WORKER_ENV_VARS:
        raw_value = os.getenv(env_var)
        if raw_value is None:
            continue
        try:
            worker_count = int(raw_value)
        except ValueError:
            logger.warning("Ignoring non-integer %s=%r", env_var, raw_value)
            continue

        if worker_count > 1:
            multi_worker_values[env_var] = worker_count
    return multi_worker_values


def _enforce_distributed_limiter_startup_guard() -> None:
    """Disallow multi-worker mode without an explicit distributed limiter."""
    multi_worker_values = _detect_multi_worker_mode()
    if settings.distributed_limiter_enabled or not multi_worker_values:
        return

    detected_workers = ", ".join(
        f"{env_var}={count}" for env_var, count in sorted(multi_worker_values.items())
    )
    raise RuntimeError(
        "Multi-worker mode detected via "
        f"{detected_workers}, but distributed limiter is disabled. "
        "Set DISTRIBUTED_LIMITER_ENABLED=true or run with a single worker."
    )


async def _cleanup_expired_remixes() -> int:
    """Delete expired remix artifacts (remixes, uploads, stems).

    Returns the number of sessions cleaned up.
    """
    remixes_dir = settings.data_dir / "remixes"
    if not remixes_dir.exists():
        return 0

    now = datetime.now(timezone.utc)
    cleaned = 0

    for session_dir in remixes_dir.iterdir():
        if not session_dir.is_dir():
            continue

        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = json.loads(manifest_path.read_text())
            expires_at = datetime.fromisoformat(manifest["expires_at"])
            if now < expires_at:
                continue
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Cleanup: skipping malformed manifest in %s", session_dir)
            continue

        session_id = session_dir.name
        for subdir in ("remixes", "uploads", "stems"):
            target = settings.data_dir / subdir / session_id
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

        cleaned += 1
        logger.info("Cleanup: removed expired session %s", session_id)

    return cleaned


async def _cleanup_loop(stop_event: asyncio.Event) -> None:
    """Periodically clean up expired remix artifacts."""
    cleanup_interval = 300  # 5 minutes
    while not stop_event.is_set():
        try:
            cleaned = await _cleanup_expired_remixes()
            if cleaned > 0:
                logger.info("Cleanup: removed %d expired session(s)", cleaned)
        except Exception:
            logger.exception("Cleanup: unexpected error during cleanup sweep")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cleanup_interval)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Query-param log redaction
# ---------------------------------------------------------------------------

_LISTEN_PARAM_RE = re.compile(r"([?&])listen=[^&]*")


def redact_listen_param(url: str) -> str:
    """Replace `listen=<value>` query-param values with `listen=REDACTED`."""
    return _LISTEN_PARAM_RE.sub(r"\1listen=REDACTED", url)


class _ListenRedactFilter(logging.Filter):
    """Redact `listen` query-param values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "msg") and isinstance(record.msg, str):
            record.msg = redact_listen_param(record.msg)
        if hasattr(record, "args") and record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact_listen_param(a) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact_listen_param(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _enforce_distributed_limiter_startup_guard()

    # Install log redaction filter for listen query-param
    for handler in logging.root.handlers:
        handler.addFilter(_ListenRedactFilter())

    # Create data directories
    for subdir in ("uploads", "stems", "remixes"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    mix_capacity = settings.max_concurrent_mixes

    # Thread pool for pipeline execution (bounded by configured mix capacity)
    app.state.executor = ThreadPoolExecutor(max_workers=mix_capacity)
    # Thread pool for SSE blocking reads (up to 4 concurrent SSE readers)
    app.state.sse_executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="sse-reader"
    )
    # In-memory session storage
    app.state.sessions = {}
    # Lock protecting the sessions dict for thread safety
    app.state.sessions_lock = threading.Lock()
    # Shared global capacity gate across ALL remix creation endpoints
    app.state.processing_lock = threading.BoundedSemaphore(value=mix_capacity)

    # Start periodic cleanup of expired remixes
    cleanup_stop = asyncio.Event()
    cleanup_task = asyncio.create_task(_cleanup_loop(cleanup_stop))

    logger.info("musicMixer backend started (max_concurrent_mixes=%d)", mix_capacity)
    yield

    logger.info("musicMixer backend shutting down")
    cleanup_stop.set()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    app.state.executor.shutdown(wait=False)
    app.state.sse_executor.shutdown(wait=False)


app = FastAPI(title="musicMixer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(remix.router, prefix="/api")

# Serve static HTML page -- must come LAST (after all API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


def dev():
    import uvicorn
    uvicorn.run(
        "musicmixer.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_excludes=["data", "data/**", "*.pyc", "__pycache__", "notes"],
    )
