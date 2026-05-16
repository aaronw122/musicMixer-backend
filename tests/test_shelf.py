"""Tests for the record shelf API."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from musicmixer.main import app


@pytest.fixture
def client(tmp_path):
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

        with patch("musicmixer.main.settings", mock_settings), \
             patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.api.shelf.settings", mock_settings), \
             patch("musicmixer.api.remix.cleanup_expired_sessions"):
            with TestClient(app) as test_client:
                yield test_client


def test_get_shelf_seeds_starter_library(client, tmp_path):
    response = client.get("/api/shelf")

    assert response.status_code == 200
    records = response.json()["records"]
    assert len(records) > 0
    assert all(record["is_curated"] for record in records)
    assert (tmp_path / "shelf.json").exists()

    payload = json.loads((tmp_path / "shelf.json").read_text())
    assert len(payload["records"]) == len(records)


def test_post_shelf_adds_record_and_persists(client, tmp_path):
    with patch(
        "musicmixer.api.shelf._fetch_noembed_metadata",
        return_value={
            "title": "The Meters - Cissy Strut",
            "thumbnail_url": "https://example.test/thumb.jpg",
        },
    ):
        response = client.post(
            "/api/shelf",
            json={"youtube_url": "https://youtu.be/abc123"},
        )

    assert response.status_code == 200
    record = response.json()
    assert record["youtube_url"] == "https://www.youtube.com/watch?v=abc123"
    assert record["artist"] == "The Meters"
    assert record["sleeve_image_url"] == f"/api/shelf/sleeve/{record['id']}"
    assert record["is_curated"] is False

    payload = json.loads((tmp_path / "shelf.json").read_text())
    assert any(item["id"] == record["id"] for item in payload["records"])


def test_post_shelf_returns_existing_on_duplicate_url(client):
    with patch(
        "musicmixer.api.shelf._fetch_noembed_metadata",
        return_value={
            "title": "Artist - Song",
            "thumbnail_url": "https://example.test/thumb.jpg",
        },
    ):
        first = client.post(
            "/api/shelf",
            json={"youtube_url": "https://youtu.be/duplicate"},
        )
        second = client.post(
            "/api/shelf",
            json={"youtube_url": "https://www.youtube.com/watch?v=duplicate&t=42"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


def test_post_shelf_rejects_invalid_youtube_url(client):
    response = client.post(
        "/api/shelf",
        json={"youtube_url": "https://example.com/watch?v=abc"},
    )

    assert response.status_code == 422


def test_sleeve_endpoint_returns_deterministic_svg_with_cache_headers(client):
    shelf = client.get("/api/shelf").json()["records"]
    record_id = shelf[0]["id"]

    first = client.get(f"/api/shelf/sleeve/{record_id}")
    second = client.get(f"/api/shelf/sleeve/{record_id}")

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("image/svg+xml")
    assert first.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert first.text == second.text
    assert "<svg" in first.text


def test_sleeve_endpoint_404s_for_unknown_record(client):
    response = client.get("/api/shelf/sleeve/not-a-record")

    assert response.status_code == 404
