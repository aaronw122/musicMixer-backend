#!/usr/bin/env bash
set -euo pipefail

CONTEXT="home"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    echo "Cleaning up staged frontend build..."
    rm -rf ./static-build
}
trap cleanup EXIT

# Build frontend and stage output inside backend dir
# (Docker cannot COPY from outside the build context)
echo "Building frontend..."
cd ../frontend && bun run build && cd ../backend

echo "Staging frontend build..."
cp -r ../frontend/dist ./static-build

# Copy secrets to homeserver
echo "Deploying secrets..."
scp .env.production homeserver:~/musicmixer/.env

# Deploy via remote Docker context
echo "Deploying to homeserver..."
docker --context "$CONTEXT" compose up -d --build --force-recreate --remove-orphans

echo "Deploy complete."
