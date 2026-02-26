import logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
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
