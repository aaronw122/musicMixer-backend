#!/usr/bin/env bash
# Run this on a fresh Hetzner Ubuntu 24.04 server to prepare it for musicMixer.
# Usage: ssh root@<hetzner-ip> 'bash -s' < scripts/hetzner-setup.sh
set -euo pipefail

echo "=== Updating system ==="
apt-get update && apt-get upgrade -y

echo "=== Installing Docker ==="
apt-get install -y docker.io docker-compose-plugin
systemctl enable --now docker

echo "=== Creating app directory ==="
mkdir -p ~/musicmixer

echo "=== Setting up firewall (allow SSH + HTTP + HTTPS) ==="
apt-get install -y ufw
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8880/tcp
ufw --force enable

echo "=== Done ==="
echo "Server is ready. Next steps:"
echo "  1. Add Docker context on your local machine:"
echo "     docker context create hetzner --docker 'host=ssh://root@<hetzner-ip>'"
echo "  2. Add SSH config entry for 'hetzner' in ~/.ssh/config"
echo "  3. Run: ./deploy.sh"
