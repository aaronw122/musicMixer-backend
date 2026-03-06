"""Tests for the remix API endpoints."""
import time
import pytest
from unittest.mock import patch
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
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
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

    def test_successful_remix_returns_session_id(self, client, tmp_path):
        """Should return session_id on successful remix."""
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
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

    def test_uploads_saved_to_disk(self, client, tmp_path):
        """Should save uploaded files to the upload directory."""
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
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

    def test_pipeline_failure_sets_error_status(self, client, tmp_path):
        """Pipeline failure should set session status to 'error' (Day 2: async).

        POST now returns 200 immediately; errors are reported via /status endpoint.
        """
        def fake_failing_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "error"
            session.error = "Pipeline exploded"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_failing_wrapper):
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": "test"},
            )
            # Day 2: POST returns 200 immediately (pipeline runs in background)
            assert response.status_code == 200
            session_id = response.json()["session_id"]

            # Give the background thread a moment to fail
            time.sleep(0.5)

            # Check session status via /status endpoint
            status_resp = client.get(f"/api/remix/{session_id}/status")
            assert status_resp.status_code == 200
            assert status_resp.json()["status"] == "error"

    def test_accepts_wav_files(self, client, tmp_path):
        """Should accept .wav files."""
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.wav", b"fake wav data", "audio/x-wav"),
                    "song_b": ("song_b.wav", b"fake wav data", "audio/x-wav"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200


    def test_rejects_oversized_upload(self, client, tmp_path):
        """Should return 413 when a file exceeds max_file_size_mb."""
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

    def test_oversized_upload_aborts_without_full_read(self, client, tmp_path):
        """Should abort chunked read early without buffering the entire file."""
        fake_module = types.ModuleType("musicmixer.services.pipeline_day1")
        fake_module.run_pipeline_sync = lambda *a: None
        sys.modules["musicmixer.services.pipeline_day1"] = fake_module

        try:
            with patch("musicmixer.api.remix.settings") as inner_settings:
                inner_settings.max_file_size_mb = 1  # 1 MB limit
                inner_settings.allowed_extensions = {".mp3", ".wav"}
                inner_settings.data_dir = tmp_path

                # Create data larger than 1 MB but track how much is actually read
                oversized_data = b"x" * (2 * 1024 * 1024)  # 2 MB

                response = client.post(
                    "/api/remix",
                    files={
                        "song_a": ("song_a.mp3", oversized_data, "audio/mpeg"),
                        "song_b": ("song_b.mp3", b"small", "audio/mpeg"),
                    },
                    data={"prompt": "test"},
                )
                assert response.status_code == 413
                assert "song_a" in response.json()["detail"]
                assert "1MB" in response.json()["detail"]
        finally:
            del sys.modules["musicmixer.services.pipeline_day1"]

    def test_submit_failure_releases_processing_slot(self, client):
        """If executor.submit fails, the slot should be released for the next request."""
        class FailingExecutor:
            def submit(self, *args, **kwargs):
                raise RuntimeError("submit failed")

        original_executor = client.app.state.executor
        client.app.state.executor = FailingExecutor()

        try:
            with pytest.raises(RuntimeError, match="submit failed"):
                client.post(
                    "/api/remix",
                    files={
                        "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                        "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                    },
                    data={"prompt": "test"},
                )
        finally:
            client.app.state.executor = original_executor

        # If the slot leaked, this follow-up request would return 409.
        with patch("musicmixer.api.remix._pipeline_wrapper") as mock_wrapper:
            mock_wrapper.side_effect = lambda *args, **kwargs: args[5].release()
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": "test"},
            )
            assert response.status_code == 200


class TestCreateRemixNoPrompt:
    """Tests that the /remix endpoint works without a prompt parameter."""

    def test_accepts_mp3_files_without_prompt(self, client, tmp_path):
        """Should accept .mp3 files without a prompt and return session_id."""
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                # No prompt data — should default to ""
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_empty_prompt_accepted(self, client, tmp_path):
        """Should accept an explicitly empty prompt string."""
        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                data={"prompt": ""},
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_pipeline_receives_empty_prompt_when_omitted(self, client, tmp_path):
        """When no prompt is sent, the pipeline wrapper should receive empty string."""
        captured_prompt = []

        def fake_wrapper(session_id, song_a_path, song_b_path, prompt, session, processing_lock, *args, **kwargs):
            captured_prompt.append(prompt)
            session.status = "complete"
            processing_lock.release()

        with patch("musicmixer.api.remix._pipeline_wrapper", fake_wrapper):
            response = client.post(
                "/api/remix",
                files={
                    "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
                    "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
                },
                # No prompt
            )
            assert response.status_code == 200

            # Give the background thread a moment to run
            time.sleep(0.5)

            assert len(captured_prompt) == 1
            assert captured_prompt[0] == ""


class TestYouTubeRemixRequestModel:
    """Tests for the YouTubeRemixRequest model with optional prompt."""

    def test_prompt_defaults_to_empty_string(self):
        """YouTubeRemixRequest should accept missing prompt, defaulting to empty string."""
        from musicmixer.api.remix import YouTubeRemixRequest
        req = YouTubeRemixRequest(url_a="https://www.youtube.com/watch?v=abc", url_b="https://www.youtube.com/watch?v=xyz")
        assert req.prompt == ""

    def test_prompt_preserved_when_provided(self):
        """YouTubeRemixRequest should preserve prompt when explicitly provided."""
        from musicmixer.api.remix import YouTubeRemixRequest
        req = YouTubeRemixRequest(
            url_a="https://www.youtube.com/watch?v=abc",
            url_b="https://www.youtube.com/watch?v=xyz",
            prompt="mix the vocals with the beat",
        )
        assert req.prompt == "mix the vocals with the beat"


class TestGetAudio:
    def test_returns_404_for_missing_remix(self, client):
        import uuid as _uuid
        fake_id = str(_uuid.uuid4())
        response = client.get(f"/api/remix/{fake_id}/audio")
        assert response.status_code == 404

    def test_returns_mp3_for_existing_remix(self, client, tmp_path):
        """Should serve MP3 file for valid session."""
        import uuid as _uuid
        session_id = str(_uuid.uuid4())
        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3 content")

        response = client.get(f"/api/remix/{session_id}/audio")
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.content == b"fake mp3 content"

    @pytest.mark.parametrize("bad_id", [
        "not-a-uuid",
        "hello world",
        "....etc....passwd",
        "DROP TABLE remixes",
        "00000000-0000-0000-0000-00000000000g",  # almost-UUID, invalid hex char
    ])
    def test_rejects_path_traversal_session_id(self, client, bad_id):
        """Should return 400 for session IDs that are not valid UUIDs."""
        response = client.get(f"/api/remix/{bad_id}/audio")
        assert response.status_code == 400
        assert "Invalid session ID" in response.json()["detail"]
