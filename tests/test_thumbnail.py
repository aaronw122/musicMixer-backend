"""Tests for the thumbnail proxy and color extraction endpoints."""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from musicmixer.main import app
from musicmixer.services.thumbnail import (
    ThumbnailFetchError,
    extract_dominant_color,
    _quantize,
    ALLOWED_THUMBNAIL_HOSTS,
)


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]
        mock_settings.max_concurrent_mixes = 1
        mock_settings.max_queue_depth = 10
        mock_settings.session_ttl_hours = 3
        mock_settings.queue_entry_ttl_minutes = 15
        mock_settings.max_upload_duration_seconds = 900
        mock_settings.distributed_limiter_enabled = False

        # Create required directories
        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.main.settings", mock_settings), \
             patch("musicmixer.api.remix.cleanup_expired_sessions"):
            with TestClient(app) as c:
                yield c


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------


class TestThumbnailProxyValidation:
    def test_missing_url_returns_422(self, client):
        """FastAPI returns 422 when required query param is missing."""
        resp = client.get("/api/thumbnail-proxy")
        assert resp.status_code == 422

    def test_non_youtube_domain_returns_403(self, client):
        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://evil.com/image.jpg"},
        )
        assert resp.status_code == 403
        assert "YouTube" in resp.json()["detail"]

    def test_youtube_watch_url_returns_403(self, client):
        """youtube.com (not a thumbnail host) should be rejected."""
        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://www.youtube.com/watch?v=abc"},
        )
        assert resp.status_code == 403

    def test_url_too_long_returns_400(self, client):
        long_url = "https://i.ytimg.com/vi/" + "a" * 500
        resp = client.get("/api/thumbnail-proxy", params={"url": long_url})
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"]

    def test_ftp_scheme_returns_403(self, client):
        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "ftp://i.ytimg.com/vi/abc/default.jpg"},
        )
        assert resp.status_code == 403

    def test_no_scheme_returns_403(self, client):
        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "i.ytimg.com/vi/abc/default.jpg"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Successful proxy
# ---------------------------------------------------------------------------


class TestThumbnailProxySuccess:
    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    def test_proxies_valid_thumbnail(self, mock_fetch, client):
        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG-ish bytes
        mock_fetch.return_value = (fake_image, "image/jpeg")

        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://i.ytimg.com/vi/abc123/maxresdefault.jpg"},
        )

        assert resp.status_code == 200
        assert resp.content == fake_image
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.headers["cache-control"] == "public, max-age=86400"
        assert resp.headers["access-control-allow-origin"] == "*"

    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    def test_proxies_img_youtube_domain(self, mock_fetch, client):
        fake_image = b"\x89PNG" + b"\x00" * 100
        mock_fetch.return_value = (fake_image, "image/png")

        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://img.youtube.com/vi/xyz/0.jpg"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"


# ---------------------------------------------------------------------------
# Upstream errors
# ---------------------------------------------------------------------------


class TestThumbnailProxyErrors:
    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    def test_upstream_failure_returns_502(self, mock_fetch, client):
        mock_fetch.side_effect = ThumbnailFetchError("Upstream returned HTTP 404")

        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://i.ytimg.com/vi/missing/default.jpg"},
        )

        assert resp.status_code == 502
        assert "404" in resp.json()["detail"]

    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    def test_timeout_returns_502(self, mock_fetch, client):
        mock_fetch.side_effect = ThumbnailFetchError("Upstream fetch timed out")

        resp = client.get(
            "/api/thumbnail-proxy",
            params={"url": "https://i.ytimg.com/vi/slow/default.jpg"},
        )

        assert resp.status_code == 502
        assert "timed out" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Color extraction endpoint
# ---------------------------------------------------------------------------


class TestThumbnailColorEndpoint:
    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    @patch("musicmixer.api.thumbnail.extract_dominant_color")
    def test_returns_hex_color(self, mock_color, mock_fetch, client):
        mock_fetch.return_value = (b"\xff" * 100, "image/jpeg")
        mock_color.return_value = "#7A3B2E"

        resp = client.get(
            "/api/thumbnail-color",
            params={"url": "https://i.ytimg.com/vi/abc123/maxresdefault.jpg"},
        )

        assert resp.status_code == 200
        assert resp.json() == {"color": "#7A3B2E"}

    def test_color_endpoint_rejects_non_youtube(self, client):
        resp = client.get(
            "/api/thumbnail-color",
            params={"url": "https://evil.com/image.jpg"},
        )
        assert resp.status_code == 403

    @patch("musicmixer.api.thumbnail.fetch_thumbnail", new_callable=AsyncMock)
    def test_color_endpoint_upstream_error(self, mock_fetch, client):
        mock_fetch.side_effect = ThumbnailFetchError("Upstream returned HTTP 500")

        resp = client.get(
            "/api/thumbnail-color",
            params={"url": "https://i.ytimg.com/vi/err/default.jpg"},
        )

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Service-layer unit tests
# ---------------------------------------------------------------------------


class TestAllowedHosts:
    def test_expected_hosts(self):
        assert "i.ytimg.com" in ALLOWED_THUMBNAIL_HOSTS
        assert "img.youtube.com" in ALLOWED_THUMBNAIL_HOSTS
        # Must NOT contain general YouTube domains
        assert "youtube.com" not in ALLOWED_THUMBNAIL_HOSTS
        assert "www.youtube.com" not in ALLOWED_THUMBNAIL_HOSTS


class TestQuantize:
    def test_zero(self):
        assert _quantize(0) == 16

    def test_midrange(self):
        result = _quantize(128)
        assert 0 <= result <= 255

    def test_max(self):
        result = _quantize(255)
        assert 200 <= result <= 255  # high bucket, exact value depends on levels


class TestExtractDominantColor:
    def test_returns_hex_string(self):
        """Should always return a valid hex color string."""
        # Random-ish bytes that look like image data
        data = bytes(range(256)) * 10
        result = extract_dominant_color(data)
        assert result.startswith("#")
        assert len(result) == 7

    def test_short_data_returns_default(self):
        """Very short data should return the default color."""
        result = extract_dominant_color(b"\x00\x01\x02")
        assert result == "#333333"

    def test_empty_data_returns_default(self):
        result = extract_dominant_color(b"")
        assert result == "#333333"
