"""Tests for the remix API endpoints."""
import sys
import types
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

from musicmixer.main import app


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]

        # Create required directories
        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        # Also patch settings in the remix module since it imports at module level
        with patch("musicmixer.api.remix.settings", mock_settings):
            with TestClient(app) as c:
                yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestCreateRemix:
    def test_rejects_invalid_extension(self, client):
        """Should return 422 for unsupported file types."""
        response = client.post(
            "/api/remix",
            files={
                "song_a": ("song.txt", b"not audio", "text/plain"),
                "song_b": ("song.mp3", b"fake mp3", "audio/mpeg"),
            },
            data={"prompt": "test"},
        )
        assert response.status_code == 422

    def test_rejects_both_invalid_extensions(self, client):
        """Should return 422 when both files have invalid extensions."""
        response = client.post(
            "/api/remix",
            files={
                "song_a": ("song.ogg", b"not audio", "audio/ogg"),
                "song_b": ("song.flac", b"not audio", "audio/flac"),
            },
            data={"prompt": "test"},
        )
        assert response.status_code == 422

    def test_accepts_mp3_files_with_mocked_pipeline(self, client, tmp_path):
        """Should accept .mp3 files and run pipeline successfully when mocked."""
        # Create a fake pipeline_day1 module
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")

        def fake_pipeline(session_id, song_a_path, song_b_path):
            remix_dir = tmp_path / "remixes" / session_id
            remix_dir.mkdir(parents=True, exist_ok=True)
            remix_path = remix_dir / "remix.mp3"
            remix_path.write_bytes(b"fake mp3")
            return remix_path

        fake_module.run_pipeline_sync = fake_pipeline
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]

    def test_successful_remix_returns_session_id(self, client, tmp_path):
        """Should return session_id on successful remix."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")

        def fake_pipeline(session_id, song_a_path, song_b_path):
            remix_dir = tmp_path / "remixes" / session_id
            remix_dir.mkdir(parents=True, exist_ok=True)
            remix_path = remix_dir / "remix.mp3"
            remix_path.write_bytes(b"fake mp3")
            return remix_path

        fake_module.run_pipeline_sync = fake_pipeline
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

            # Verify the session_id is a valid UUID format
            import uuid
            uuid.UUID(data["session_id"])  # Raises ValueError if invalid
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]

    def test_uploads_saved_to_disk(self, client, tmp_path):
        """Should save uploaded files to the upload directory."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")

        def fake_pipeline(session_id, song_a_path, song_b_path):
            remix_dir = tmp_path / "remixes" / session_id
            remix_dir.mkdir(parents=True, exist_ok=True)
            remix_path = remix_dir / "remix.mp3"
            remix_path.write_bytes(b"fake mp3")
            return remix_path

        fake_module.run_pipeline_sync = fake_pipeline
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"song a content", "audio/mpeg"),
                    "song_b": ("song_b.wav", b"song b content", "audio/x-wav"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Check files were saved
            upload_dir = tmp_path / "uploads" / session_id
            assert upload_dir.exists()
            assert (upload_dir / "song_a.mp3").exists()
            assert (upload_dir / "song_b.wav").exists()
            assert (upload_dir / "song_a.mp3").read_bytes() == b"song a content"
            assert (upload_dir / "song_b.wav").read_bytes() == b"song b content"
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]

    def test_pipeline_failure_returns_500(self, client, tmp_path):
        """Should return 500 when pipeline raises an exception."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")

        def failing_pipeline(session_id, song_a_path, song_b_path):
            raise RuntimeError("Pipeline exploded")

        fake_module.run_pipeline_sync = failing_pipeline
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 500
            assert "Pipeline exploded" in response.json()["detail"]
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]

    def test_accepts_wav_files(self, client, tmp_path):
        """Should accept .wav files."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")

        def fake_pipeline(session_id, song_a_path, song_b_path):
            remix_dir = tmp_path / "remixes" / session_id
            remix_dir.mkdir(parents=True, exist_ok=True)
            remix_path = remix_dir / "remix.mp3"
            remix_path.write_bytes(b"fake mp3")
            return remix_path

        fake_module.run_pipeline_sync = fake_pipeline
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.wav", b"fake wav data", "audio/x-wav"),
                    "song_b": ("song_b.wav", b"fake wav data", "audio/x-wav"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]


    def test_rejects_oversized_upload(self, client, tmp_path):
        """Should return 413 when a file exceeds max_file_size_mb."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")
        fake_module.run_pipeline_sync = lambda *a: None
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            # mock_settings.max_file_size_mb is 50, so limit is 50 * 1024 * 1024 bytes
            # Send a file that exceeds 50MB (we'll set max_file_size_mb=0 via a nested patch)
            with patch("musicmixer.api.remix.settings") as inner_settings:
                inner_settings.max_file_size_mb = 0  # 0 MB limit
                inner_settings.allowed_extensions = {".mp3", ".wav"}
                inner_settings.data_dir = tmp_path

                response = client.post(
                    "/api/remix",
                    files={
                        "song_a": ("song_a.mp3", b"some data", "audio/mpeg"),
                        "song_b": ("song_b.mp3", b"some data", "audio/mpeg"),
                    },
                    data={"prompt": "test"},
                )
            assert response.status_code == 413
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]


class TestGetAudio:
    def test_returns_404_for_missing_remix(self, client):
        response = client.get("/api/remix/nonexistent-id/audio")
        assert response.status_code == 404

    def test_returns_mp3_for_existing_remix(self, client, tmp_path):
        """Should serve MP3 file for valid session."""
        session_id = "test-session"
        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3 content")

        response = client.get(f"/api/remix/{session_id}/audio")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"fake mp3 content"
