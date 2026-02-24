from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from musicmixer.config import settings
from musicmixer.api import health, remix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create data directories
    for subdir in ("uploads", "stems", "remixes"):
        (settings.data_dir / subdir).mkdir(parents=True, exist_ok=True)
    logger.info("musicMixer backend started")
    yield
    logger.info("musicMixer backend shutting down")


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
