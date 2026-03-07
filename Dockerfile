# -- Base stage: system deps + Python deps (layer-cached) --
FROM python:3.11-slim AS base

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System dependencies required by audio processing pipeline
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        rubberband-cli \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps (layer-cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# -- Runtime stage: copy source + frontend build --
FROM base AS runtime

WORKDIR /app

# Copy application source
COPY src/ ./src/

# Copy frontend build (staged by deploy.sh)
COPY static-build/ /app/static/

EXPOSE 8880

CMD ["uv", "run", "uvicorn", "musicmixer.main:app", "--host", "0.0.0.0", "--port", "8880"]
