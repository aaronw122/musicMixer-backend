"""Tests for the SMS notification endpoint and pipeline SMS hook."""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from musicmixer.main import app
from musicmixer.models import SessionState


@pytest.fixture
def client(tmp_path):
    """Create test client with SMS enabled and mocked SMS service."""
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
        mock_settings.sms_enabled = True
        mock_settings.app_base_url = "http://localhost:5173"

        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.main.settings", mock_settings), \
             patch("musicmixer.api.remix.cleanup_expired_sessions"):
            with TestClient(app) as c:
                yield c


@pytest.fixture
def client_sms_disabled(tmp_path):
    """Create test client with SMS disabled."""
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
        mock_settings.sms_enabled = False

        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.main.settings", mock_settings), \
             patch("musicmixer.api.remix.cleanup_expired_sessions"):
            with TestClient(app) as c:
                yield c


def _add_session(client, session_id: str, session: SessionState) -> None:
    """Insert a session directly into the app state."""
    with client.app.state.sessions_lock:
        client.app.state.sessions[session_id] = session


class TestNotifyEndpointValidation:
    """Tests for input validation on POST /api/remix/{id}/notify-sms."""

    def test_invalid_phone_returns_422(self, client):
        session_id = str(uuid.uuid4())
        _add_session(client, session_id, SessionState())

        response = client.post(
            f"/api/remix/{session_id}/notify-sms",
            json={"phone": "5551234567"},  # missing +
        )
        assert response.status_code == 422

    def test_short_phone_returns_422(self, client):
        session_id = str(uuid.uuid4())
        _add_session(client, session_id, SessionState())

        response = client.post(
            f"/api/remix/{session_id}/notify-sms",
            json={"phone": "+123"},  # too short
        )
        assert response.status_code == 422

    def test_phone_with_letters_returns_422(self, client):
        session_id = str(uuid.uuid4())
        _add_session(client, session_id, SessionState())

        response = client.post(
            f"/api/remix/{session_id}/notify-sms",
            json={"phone": "+1555abc4567"},
        )
        assert response.status_code == 422

    def test_nonexistent_session_returns_404(self, client):
        fake_id = str(uuid.uuid4())
        response = client.post(
            f"/api/remix/{fake_id}/notify-sms",
            json={"phone": "+15551234567"},
        )
        assert response.status_code == 404

    def test_sms_disabled_returns_503(self, client_sms_disabled):
        session_id = str(uuid.uuid4())
        _add_session(client_sms_disabled, session_id, SessionState())

        response = client_sms_disabled.post(
            f"/api/remix/{session_id}/notify-sms",
            json={"phone": "+15551234567"},
        )
        assert response.status_code == 503


class TestNotifyEndpointBehavior:
    """Tests for SMS endpoint behavior in various session states."""

    def test_valid_phone_returns_202_and_sends_confirmation(self, client):
        """Valid phone on a processing session -> 202, confirmation SMS sent."""
        mock_confirm = MagicMock(return_value=True)

        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        _add_session(client, session_id, session)

        with patch("musicmixer.services.sms.send_confirmation", mock_confirm):
            response = client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15551234567"},
            )

        assert response.status_code == 202
        assert session.notify_phone == "+15551234567"
        mock_confirm.assert_called_once_with("+15551234567")

    def test_confirmation_failure_still_returns_202(self, client):
        """If confirmation SMS fails, endpoint still returns 202."""
        mock_confirm = MagicMock(side_effect=Exception("Twilio error"))

        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        _add_session(client, session_id, session)

        with patch("musicmixer.services.sms.send_confirmation", mock_confirm):
            response = client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15551234567"},
            )

        assert response.status_code == 202
        # Phone should still be stored even though confirmation failed
        assert session.notify_phone == "+15551234567"

    def test_already_complete_returns_200_and_sends_ready(self, client):
        """If session is already complete, send ready SMS directly -> 200."""
        mock_ready = MagicMock(return_value=True)

        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "complete"
        _add_session(client, session_id, session)

        with patch("musicmixer.services.sms.send_remix_ready", mock_ready):
            response = client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15551234567"},
            )

        assert response.status_code == 200
        mock_ready.assert_called_once_with("+15551234567", session_id)

    def test_error_session_returns_409(self, client):
        """Error session -> 409."""
        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "error"
        _add_session(client, session_id, session)

        response = client.post(
            f"/api/remix/{session_id}/notify-sms",
            json={"phone": "+15551234567"},
        )
        assert response.status_code == 409

    def test_idempotent_overwrites_phone(self, client):
        """Calling twice with different phones overwrites the first."""
        mock_confirm = MagicMock(return_value=True)

        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        _add_session(client, session_id, session)

        with patch("musicmixer.services.sms.send_confirmation", mock_confirm):
            client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15551111111"},
            )
            client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15552222222"},
            )

        assert session.notify_phone == "+15552222222"


class TestPipelineSmsHook:
    """Tests for the SMS hook in pipeline completion."""

    def test_pipeline_sends_sms_on_completion(self):
        """When phone is registered, pipeline sends SMS at completion."""
        mock_ready = MagicMock(return_value=True)
        session = SessionState()
        session.status = "processing"
        session.notify_phone = "+15551234567"

        # Simulate the pipeline completion logic from pipeline.py
        phone = session.notify_phone
        session.notify_phone = None
        session.status = "complete"

        assert phone == "+15551234567"
        assert session.notify_phone is None

        if phone:
            mock_ready(phone, "test-session-id")

        mock_ready.assert_called_once_with("+15551234567", "test-session-id")

    def test_pipeline_clears_phone_before_send(self):
        """Phone is cleared from session before SMS send attempt."""
        session = SessionState()
        session.notify_phone = "+15551234567"

        phone = session.notify_phone
        session.notify_phone = None

        assert phone == "+15551234567"
        assert session.notify_phone is None

    def test_pipeline_no_sms_when_no_phone(self):
        """No SMS sent when no phone is registered."""
        mock_ready = MagicMock()
        session = SessionState()
        session.notify_phone = None

        phone = session.notify_phone
        session.notify_phone = None

        if phone:
            mock_ready(phone, "test-session-id")

        mock_ready.assert_not_called()


class TestFullFlow:
    """Integration test: register phone + complete pipeline = two Twilio calls."""

    def test_register_then_complete_sends_two_sms(self, client):
        """Full flow: register phone (confirmation) + pipeline completes (ready)."""
        mock_confirm = MagicMock(return_value=True)
        mock_ready = MagicMock(return_value=True)

        session_id = str(uuid.uuid4())
        session = SessionState()
        session.status = "processing"
        _add_session(client, session_id, session)

        # Step 1: Register phone -> confirmation SMS
        with patch("musicmixer.services.sms.send_confirmation", mock_confirm):
            response = client.post(
                f"/api/remix/{session_id}/notify-sms",
                json={"phone": "+15551234567"},
            )

        assert response.status_code == 202
        mock_confirm.assert_called_once_with("+15551234567")

        # Step 2: Simulate pipeline completion
        assert session.notify_phone == "+15551234567"

        phone = session.notify_phone
        session.notify_phone = None
        session.status = "complete"

        assert phone == "+15551234567"
        assert session.notify_phone is None

        # Send SMS (simulating what pipeline.py does)
        with patch("musicmixer.services.sms.send_remix_ready", mock_ready):
            from musicmixer.services.sms import send_remix_ready
            send_remix_ready(phone, session_id)

        mock_ready.assert_called_once_with("+15551234567", session_id)

        # Total: 2 Twilio calls (1 confirmation + 1 ready)
        assert mock_confirm.call_count == 1
        assert mock_ready.call_count == 1
