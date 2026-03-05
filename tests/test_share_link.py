"""Tests for share-link feature: public endpoint, TTL, cleanup, headers, log redaction."""

import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from musicmixer.main import app, redact_listen_param, _ListenRedactFilter


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]
        mock_settings.remix_ttl_seconds = 10800
        mock_settings.max_concurrent_mixes = 1
        mock_settings.distributed_limiter_enabled = False
        mock_settings.youtube_enabled = False

        # Create required directories
        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        with patch("musicmixer.api.remix.settings", mock_settings):
            with TestClient(app) as c:
                yield c


def _create_valid_session(tmp_path: Path, ttl_seconds: int = 10800) -> str:
    """Create a valid remix session with manifest and audio on disk."""
    session_id = str(uuid.uuid4())
    remix_dir = tmp_path / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    manifest = {
        "session_id": session_id,
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "explanation": "Test remix explanation",
        "warnings": ["test warning"],
        "used_fallback": False,
        "audio_filename": "remix.mp3",
    }
    (remix_dir / "manifest.json").write_text(json.dumps(manifest))
    (remix_dir / "remix.mp3").write_bytes(b"fake mp3 content")

    return session_id


def _create_expired_session(tmp_path: Path) -> str:
    """Create an expired remix session (expiry in the past)."""
    session_id = str(uuid.uuid4())
    remix_dir = tmp_path / "remixes" / session_id
    remix_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    expires_at = now - timedelta(seconds=60)  # Expired 1 minute ago

    manifest = {
        "session_id": session_id,
        "created_at": (now - timedelta(hours=4)).isoformat(),
        "expires_at": expires_at.isoformat(),
        "explanation": "Expired remix",
        "warnings": [],
        "used_fallback": True,
        "audio_filename": "remix.mp3",
    }
    (remix_dir / "manifest.json").write_text(json.dumps(manifest))
    (remix_dir / "remix.mp3").write_bytes(b"fake mp3 content")

    return session_id


class TestPublicEndpoint:
    """Tests for GET /api/remix/{session_id}/public."""

    def test_200_valid_remix(self, client, tmp_path):
        """Should return 200 with correct fields for a valid remix."""
        session_id = _create_valid_session(tmp_path)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "ready"
        assert data["audio_url"] == f"/api/remix/{session_id}/audio"
        assert data["explanation"] == "Test remix explanation"
        assert data["warnings"] == ["test warning"]
        assert "expires_at" in data

    def test_404_unknown_session(self, client):
        """Should return 404 for an unknown session ID."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/remix/{fake_id}/public")
        assert response.status_code == 404

    def test_404_missing_audio(self, client, tmp_path):
        """Should return 404 when manifest exists but audio is missing."""
        session_id = _create_valid_session(tmp_path)
        # Remove the audio file
        (tmp_path / "remixes" / session_id / "remix.mp3").unlink()

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 404

    def test_410_expired_remix(self, client, tmp_path):
        """Should return 410 for an expired remix while manifest still exists."""
        session_id = _create_expired_session(tmp_path)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 410

    def test_400_invalid_uuid(self, client):
        """Should return 400 for malformed session IDs."""
        response = client.get("/api/remix/not-a-uuid/public")
        assert response.status_code == 400

    def test_contract_uses_camelcase_fallback(self, client, tmp_path):
        """Response must expose `usedFallback` (camelCase), not `used_fallback`."""
        session_id = _create_valid_session(tmp_path)

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200
        data = response.json()
        assert "usedFallback" in data
        assert "used_fallback" not in data
        assert data["usedFallback"] is False

    def test_contract_used_fallback_true(self, client, tmp_path):
        """usedFallback should be true when manifest has used_fallback=true."""
        session_id = str(uuid.uuid4())
        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=3)

        manifest = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "explanation": "Fallback remix",
            "warnings": [],
            "used_fallback": True,
            "audio_filename": "remix.mp3",
        }
        (remix_dir / "manifest.json").write_text(json.dumps(manifest))
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3")

        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200
        assert response.json()["usedFallback"] is True


class TestPublicEndpointCacheHeaders:
    """Cache header tests for /public endpoint."""

    def test_cache_control_no_store(self, client, tmp_path):
        """/public must return Cache-Control: no-store."""
        session_id = _create_valid_session(tmp_path)
        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"

    def test_referrer_policy(self, client, tmp_path):
        """/public must return Referrer-Policy: no-referrer."""
        session_id = _create_valid_session(tmp_path)
        response = client.get(f"/api/remix/{session_id}/public")
        assert response.status_code == 200
        assert response.headers["referrer-policy"] == "no-referrer"


class TestAudioEndpointTTL:
    """Tests for TTL enforcement on GET /api/remix/{session_id}/audio."""

    def test_410_expired_audio(self, client, tmp_path):
        """Should return 410 for expired remix audio while manifest exists."""
        session_id = _create_expired_session(tmp_path)
        response = client.get(f"/api/remix/{session_id}/audio")
        assert response.status_code == 410

    def test_404_after_cleanup(self, client, tmp_path):
        """Should return 404 after cleanup removes expired artifacts."""
        fake_id = str(uuid.uuid4())
        # No files on disk at all -- simulates post-cleanup state
        response = client.get(f"/api/remix/{fake_id}/audio")
        assert response.status_code == 404

    def test_cache_control_max_age_bounded(self, client, tmp_path):
        """Audio max-age must not exceed remaining TTL."""
        # Create session expiring in 600 seconds
        session_id = _create_valid_session(tmp_path, ttl_seconds=600)

        response = client.get(f"/api/remix/{session_id}/audio")
        assert response.status_code == 200

        cache_control = response.headers["cache-control"]
        assert "private" in cache_control
        assert "must-revalidate" in cache_control

        # Extract max-age value
        import re
        match = re.search(r"max-age=(\d+)", cache_control)
        assert match is not None
        max_age = int(match.group(1))
        # Should be roughly 600 seconds (allow some test execution time)
        assert max_age <= 600
        assert max_age >= 590

    def test_referrer_policy_on_audio(self, client, tmp_path):
        """/audio must return Referrer-Policy: no-referrer."""
        session_id = _create_valid_session(tmp_path)
        response = client.get(f"/api/remix/{session_id}/audio")
        assert response.status_code == 200
        assert response.headers["referrer-policy"] == "no-referrer"


class TestCleanup:
    """Tests for the periodic cleanup of expired remixes."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired_sessions(self, tmp_path):
        """Cleanup should remove expired session directories."""
        with patch("musicmixer.config.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            with patch("musicmixer.main.settings", mock_settings):
                from musicmixer.main import _cleanup_expired_remixes

                # Create expired session with all three directories
                session_id = str(uuid.uuid4())
                for subdir in ("remixes", "uploads", "stems"):
                    d = tmp_path / subdir / session_id
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "test.txt").write_text("data")

                # Write expired manifest
                now = datetime.now(timezone.utc)
                manifest = {
                    "session_id": session_id,
                    "created_at": (now - timedelta(hours=4)).isoformat(),
                    "expires_at": (now - timedelta(hours=1)).isoformat(),
                    "explanation": "expired",
                    "warnings": [],
                    "used_fallback": False,
                    "audio_filename": "remix.mp3",
                }
                (tmp_path / "remixes" / session_id / "manifest.json").write_text(
                    json.dumps(manifest)
                )

                cleaned = await _cleanup_expired_remixes()
                assert cleaned == 1

                # All three directories should be removed
                for subdir in ("remixes", "uploads", "stems"):
                    assert not (tmp_path / subdir / session_id).exists()

    @pytest.mark.asyncio
    async def test_cleanup_preserves_valid_sessions(self, tmp_path):
        """Cleanup should not remove sessions that haven't expired."""
        with patch("musicmixer.config.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            with patch("musicmixer.main.settings", mock_settings):
                from musicmixer.main import _cleanup_expired_remixes

                session_id = str(uuid.uuid4())
                remix_dir = tmp_path / "remixes" / session_id
                remix_dir.mkdir(parents=True, exist_ok=True)

                now = datetime.now(timezone.utc)
                manifest = {
                    "session_id": session_id,
                    "created_at": now.isoformat(),
                    "expires_at": (now + timedelta(hours=2)).isoformat(),
                    "explanation": "still valid",
                    "warnings": [],
                    "used_fallback": False,
                    "audio_filename": "remix.mp3",
                }
                (remix_dir / "manifest.json").write_text(json.dumps(manifest))
                (remix_dir / "remix.mp3").write_bytes(b"audio")

                cleaned = await _cleanup_expired_remixes()
                assert cleaned == 0
                assert remix_dir.exists()


class TestLogRedaction:
    """Tests for listen query-param log redaction."""

    def test_redact_listen_param_in_url(self):
        """Should redact listen param values in URLs."""
        url = "/?listen=abc123-def456&other=ok"
        result = redact_listen_param(url)
        assert "abc123-def456" not in result
        assert "listen=REDACTED" in result
        assert "other=ok" in result

    def test_redact_listen_only_param(self):
        """Should redact when listen is the only param."""
        url = "/?listen=secret-session-id"
        result = redact_listen_param(url)
        assert "secret-session-id" not in result
        assert "listen=REDACTED" in result

    def test_no_redaction_without_listen(self):
        """Should not modify URLs without listen param."""
        url = "/?foo=bar&baz=qux"
        result = redact_listen_param(url)
        assert result == url

    def test_filter_redacts_log_message(self):
        """The logging filter should redact listen params in log messages."""
        import logging

        filt = _ListenRedactFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Request: /?listen=secret-uuid&x=1",
            args=None, exc_info=None,
        )
        filt.filter(record)
        assert "secret-uuid" not in record.msg
        assert "listen=REDACTED" in record.msg

    def test_filter_redacts_log_args(self):
        """The logging filter should redact listen params in format args."""
        import logging

        filt = _ListenRedactFilter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Serving %s", args=("/?listen=my-session-id",),
            exc_info=None,
        )
        filt.filter(record)
        assert "my-session-id" not in record.args[0]
        assert "listen=REDACTED" in record.args[0]


class TestManifestWriting:
    """Tests for the pipeline manifest writing function."""

    def test_write_manifest_creates_valid_json(self, tmp_path):
        """_write_remix_manifest should create a valid manifest.json."""
        with patch("musicmixer.config.settings") as mock_settings:
            mock_settings.remix_ttl_seconds = 10800

            from musicmixer.services.pipeline import _write_remix_manifest

            session_id = str(uuid.uuid4())
            remix_dir = tmp_path / "remixes" / session_id
            remix_dir.mkdir(parents=True, exist_ok=True)

            _write_remix_manifest(
                session_id=session_id,
                remix_dir=remix_dir,
                explanation="Test explanation",
                warnings=["warn1"],
                used_fallback=True,
            )

            manifest_path = remix_dir / "manifest.json"
            assert manifest_path.exists()

            manifest = json.loads(manifest_path.read_text())
            assert manifest["session_id"] == session_id
            assert manifest["explanation"] == "Test explanation"
            assert manifest["warnings"] == ["warn1"]
            assert manifest["used_fallback"] is True
            assert manifest["audio_filename"] == "remix.mp3"
            assert "created_at" in manifest
            assert "expires_at" in manifest

            # Verify expires_at is roughly 3 hours from now
            created = datetime.fromisoformat(manifest["created_at"])
            expires = datetime.fromisoformat(manifest["expires_at"])
            delta = (expires - created).total_seconds()
            assert 10790 < delta < 10810
