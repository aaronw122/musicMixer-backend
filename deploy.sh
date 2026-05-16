#!/usr/bin/env bash
set -euo pipefail

# Deploy target: "hetzner" (default) or "home" (legacy homeserver)
TARGET="${1:-hetzner}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$TARGET" in
  home)
    CONTEXT="home"
    SSH_HOST="homeserver"
    REMOTE_APP_DIR="~/musicmixer"
    REMOTE_MODAL_PATH="/home/lobo/.modal.toml"
    ;;
  hetzner)
    CONTEXT="hetzner"
    SSH_HOST="hetzner"
    REMOTE_APP_DIR="~/musicmixer"
    REMOTE_MODAL_PATH="/root/.modal.toml"
    ;;
  *)
    echo "Usage: ./deploy.sh [hetzner|home]"
    exit 1
    ;;
esac

cleanup() {
    echo "Cleaning up staged frontend build..."
    rm -rf "$SCRIPT_DIR/static-build"
}
trap cleanup EXIT

# Build frontend and stage output inside backend dir
# (Docker cannot COPY from outside the build context)
echo "Building frontend..."
cd ../frontend && bun run build && cd ../backend

echo "Staging frontend build..."
cp -r ../frontend/dist ./static-build

# Copy secrets to target server
echo "Deploying secrets to $TARGET..."
scp .env.production "$SSH_HOST:$REMOTE_APP_DIR/.env"
scp ~/.modal.toml "$SSH_HOST:$REMOTE_MODAL_PATH"

# Deploy via remote Docker context
echo "Deploying to $TARGET..."
docker --context "$CONTEXT" compose up -d --build --force-recreate --remove-orphans

echo "Deploy to $TARGET complete."
