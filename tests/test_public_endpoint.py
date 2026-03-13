"""Tests for the GET /api/remix/{session_id}/public endpoint."""

import time
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from musicmixer.main import app
from musicmixer.models import SessionState


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory and isolated sessions."""
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


def _inject_session(client, session_id: str, session: SessionState):
    """Inject a session into the app's in-memory sessions dict."""
    with client.app.state.sessions_lock:
        client.app.state.sessions[session_id] = session


class TestPublicEndpointNotFound:
    def test_session_not_found_returns_404(self, client):
        session_id = str(uuid.uuid4())
        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 404

    def test_invalid_session_id_returns_400(self, client):
        response = client.get("/api/remix/not-a-uuid/public")
        assert response.status_code == 400


class TestPublicEndpointExpired:
    def test_expired_session_returns_410(self, client):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        # Set created_at to 4 hours ago (TTL is 3 hours)
        session.created_at = time.time() - (4 * 3600)
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410
        assert response.json()["status"] == "expired"


class TestPublicEndpointComplete:
    def test_complete_session_returns_200(self, client, tmp_path):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        session.explanation = "A great remix"
        session.created_at = time.time() - 60  # 1 minute ago
        _inject_session(client, session_id, session)

        # Create the remix file
        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3 data")

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "ready"
        assert data["audio_url"] == f"/api/remix/{session_id}/audio"
        assert data["explanation"] == "A great remix"
        assert isinstance(data["warnings"], list)
        assert data["usedFallback"] is False
        assert "expires_at" in data

    def test_response_has_correct_schema(self, client, tmp_path):
        """Verify response matches PublicRemixResponse type exactly."""
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        session.explanation = "Test explanation"
        session.created_at = time.time() - 120
        _inject_session(client, session_id, session)

        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3")

        response = client.get(f"/api/remix/{session_id}/public")
        data = response.json()

        # All required keys present
        required_keys = {"session_id", "status", "audio_url", "explanation",
                         "warnings", "usedFallback", "expires_at"}
        assert set(data.keys()) == required_keys

    def test_expires_at_is_valid_iso8601(self, client, tmp_path):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        session.explanation = "Test"
        session.created_at = time.time() - 300
        _inject_session(client, session_id, session)

        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3")

        response = client.get(f"/api/remix/{session_id}/public")
        data = response.json()

        # Should parse without error
        parsed = datetime.fromisoformat(data["expires_at"])
        assert parsed.tzinfo is not None  # Must be timezone-aware

        # expires_at should be in the future (session is only 5 min old, TTL is 3h)
        assert parsed > datetime.now(timezone.utc)

    def test_complete_but_no_file_returns_410(self, client):
        """Complete status but missing remix file should return 410 (error)."""
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        session.created_at = time.time() - 60
        _inject_session(client, session_id, session)

        # Don't create the remix file
        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410
        assert response.json()["status"] == "error"


class TestPublicEndpointProcessing:
    def test_processing_session_returns_202(self, client):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        session.created_at = time.time()
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 202
        assert response.json()["status"] == "processing"

    def test_queued_session_returns_202(self, client):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "queued"
        session.created_at = time.time()
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 202
        assert response.json()["status"] == "processing"


class TestPublicEndpointError:
    def test_error_session_returns_410(self, client):
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "error"
        session.created_at = time.time()
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410
        assert response.json()["status"] == "error"

    def test_cancelled_session_returns_410(self, client):
        """Cancelled sessions should behave like errors (terminal state)."""
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "cancelled"
        session.created_at = time.time()
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410
        assert response.json()["status"] == "error"


class TestPublicEndpointExpiryPrecedence:
    def test_expired_processing_returns_410_expired(self, client):
        """Expiry check should run before status check."""
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        session.created_at = time.time() - (4 * 3600)
        _inject_session(client, session_id, session)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410
        assert response.json()["status"] == "expired"
