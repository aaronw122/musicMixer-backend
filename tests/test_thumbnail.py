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
    @pytest.mark.parametrize("url,expected_status", [
        ("https://evil.com/image.jpg", 403),
        ("https://www.youtube.com/watch?v=abc", 403),  # not a thumbnail host
    ])
    def test_rejects_non_thumbnail_domains(self, client, url, expected_status):
        resp = client.get("/api/thumbnail-proxy", params={"url": url})
        assert resp.status_code == expected_status

    @pytest.mark.parametrize("url", [
        "ftp://i.ytimg.com/vi/abc/default.jpg",
        "i.ytimg.com/vi/abc/default.jpg",  # no scheme
    ])
    def test_rejects_non_http_schemes(self, client, url):
        resp = client.get("/api/thumbnail-proxy", params={"url": url})
        assert resp.status_code == 403

    def test_url_too_long_returns_400(self, client):
        long_url = "https://i.ytimg.com/vi/" + "a" * 500
        resp = client.get("/api/thumbnail-proxy", params={"url": long_url})
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"]


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
    @pytest.mark.parametrize("value,min_expected,max_expected", [
        (0, 0, 32),
        (128, 96, 160),
        (255, 200, 255),
    ])
    def test_quantize_boundaries(self, value, min_expected, max_expected):
        result = _quantize(value)
        assert min_expected <= result <= max_expected


class TestExtractDominantColor:
    def test_returns_hex_string(self):
        """Should always return a valid hex color string."""
        # Random-ish bytes that look like image data
        data = bytes(range(256)) * 10
        result = extract_dominant_color(data)
        assert result.startswith("#")
        assert len(result) == 7

    @pytest.mark.parametrize("data", [b"", b"\x00\x01\x02"])
    def test_insufficient_data_returns_default(self, data):
        assert extract_dominant_color(data) == "#333333"
