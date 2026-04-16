import asyncio
import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from musicmixer.api import health, remix, thumbnail
from musicmixer.config import settings
from musicmixer.services.cleanup import cleanup_expired_sessions

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
    # Wait queue for requests when all processing slots are busy
    app.state.wait_queue = queue.Queue(maxsize=settings.max_queue_depth)
    # Lock protecting wait queue operations for accurate position reporting
    app.state.queue_lock = threading.Lock()

    logger.info("musicMixer backend started (max_concurrent_mixes=%d)", mix_capacity)

    # Run cleanup once on startup to clear stale data from previous runs
    await asyncio.to_thread(
        cleanup_expired_sessions,
        app.state.sessions,
        app.state.sessions_lock,
    )

    # Background task: run cleanup every 30 minutes
    async def _periodic_cleanup() -> None:
        while True:
            await asyncio.sleep(30 * 60)
            try:
                await asyncio.to_thread(
                    cleanup_expired_sessions,
                    app.state.sessions,
                    app.state.sessions_lock,
                )
            except Exception:
                logger.exception("Periodic cleanup failed")

    cleanup_task = asyncio.create_task(_periodic_cleanup())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

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
app.include_router(thumbnail.router, prefix="/api")


@app.get("/about")
async def about_page() -> FileResponse:
    return FileResponse("static/about.html")


@app.get("/terms")
async def terms_page() -> FileResponse:
    return FileResponse("static/terms.html")


@app.get("/privacy")
async def privacy_page() -> FileResponse:
    return FileResponse("static/privacy.html")


# Serve static files -- must come LAST (after all API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


def dev():
    import uvicorn
    # Force deterministic fallback — no LLM API calls during dev
    settings.anthropic_api_key = ""
    uvicorn.run(
        "musicmixer.main:app",
        host=settings.host,
        port=settings.port,
    )


def serve():
    import uvicorn
    uvicorn.run(
        "musicmixer.main:app",
        host=settings.host,
        port=settings.port,
    )
