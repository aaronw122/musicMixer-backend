import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    _enforce_distributed_limiter_startup_guard()

    # Create data directories
    for subdir in ("uploads", "stems", "remixes"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Thread pool for pipeline execution (max 1 concurrent remix)
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    # Thread pool for SSE blocking reads (up to 4 concurrent SSE readers)
    app.state.sse_executor = ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="sse-reader"
    )
    # In-memory session storage
    app.state.sessions = {}
    # Lock protecting the sessions dict for thread safety
    app.state.sessions_lock = threading.Lock()
    # Lock ensuring only one pipeline runs at a time (authoritative gate)
    app.state.processing_lock = threading.Lock()

    logger.info("musicMixer backend started")
    yield

    logger.info("musicMixer backend shutting down")
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
