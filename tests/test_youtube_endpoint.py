"""Tests for the YouTube remix API endpoint (POST /api/remix/youtube)."""

import json
import queue
import sys
import threading
import time
import types
import uuid

import pytest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from musicmixer.main import app


@dataclass
class FakeYouTubeAudioResult:
    """Fake result matching the YouTubeAudioResult interface."""
    wav_path: Path
    title: str
    duration_seconds: float
    source_codec: str
    source_bitrate: int


@pytest.fixture(autouse=True)
def _stub_youtube_module():
    """Ensure musicmixer.services.youtube exists as a stub module for patching.

    The real youtube.py is built by another agent in parallel. Tests mock it,
    but unittest.mock.patch needs the module to exist before it can patch attributes.
    """
    mod_name = "musicmixer.services.youtube"
    already_existed = mod_name in sys.modules
    if not already_existed:
        stub = types.ModuleType(mod_name)
        stub.download_youtube_audio = None  # placeholder for patch
        stub.YouTubeAudioResult = FakeYouTubeAudioResult
        sys.modules[mod_name] = stub
    yield
    if not already_existed and mod_name in sys.modules:
        del sys.modules[mod_name]


@pytest.fixture
def client(tmp_path):
    """Create test client with temp data directory."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]
        mock_settings.youtube_enabled = True
        mock_settings.youtube_max_duration_seconds = 900
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


VALID_YT_URL_A = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VALID_YT_URL_B = "https://www.youtube.com/watch?v=9bZkp7q19f0"


class TestYouTubeURLValidation:
    """Test the SSRF prevention URL validation."""

    def test_rejects_non_youtube_host(self, client):
        """Non-YouTube hosts should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "https://evil.com/watch?v=abc123",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422
        assert "only YouTube links" in response.json()["detail"]

    def test_rejects_ftp_scheme(self, client):
        """Non-HTTP(S) schemes should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "ftp://youtube.com/watch?v=abc123",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_rejects_userinfo_bypass(self, client):
        """URLs with @ in netloc (userinfo bypass) should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "https://youtube.com@evil.com/watch?v=abc123",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_rejects_ip_literal(self, client):
        """IP address URLs should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "https://192.168.1.1/watch?v=abc123",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_rejects_non_standard_port(self, client):
        """Non-standard ports should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "https://www.youtube.com:8080/watch?v=abc123",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_accepts_youtu_be_shortlink(self, client):
        """youtu.be shortlinks should be accepted."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            mock_wrapper.side_effect = lambda *a, **kw: a[5].release()

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": "https://youtu.be/dQw4w9WgXcQ",
                    "url_b": "https://youtu.be/9bZkp7q19f0",
                    "prompt": "test remix",
                },
            )
            assert response.status_code == 200

    def test_accepts_music_youtube_url(self, client):
        """music.youtube.com URLs should be accepted."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            mock_wrapper.side_effect = lambda *a, **kw: a[5].release()

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
                    "url_b": "https://music.youtube.com/watch?v=9bZkp7q19f0",
                    "prompt": "test remix",
                },
            )
            assert response.status_code == 200

    def test_rejects_url_b_invalid(self, client):
        """Both URLs must be valid -- test url_b validation."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": VALID_YT_URL_A,
                "url_b": "https://notyoutube.com/watch?v=abc",
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_rejects_javascript_scheme(self, client):
        """javascript: scheme should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "javascript:alert(1)",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422

    def test_rejects_empty_url(self, client):
        """Empty URLs should be rejected."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "url_a": "",
                "url_b": VALID_YT_URL_B,
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422


class TestYouTubeRemixEndpoint:
    """Test the POST /api/remix/youtube endpoint behavior."""

    def test_successful_request_returns_session_id(self, client):
        """Successful request should return a valid session_id immediately."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            # Simulate the wrapper releasing the lock
            def fake_wrapper(session_id, url_a, url_b, prompt, session, lock, app_state=None, **kwargs):
                lock.release()
                if app_state:
                    from musicmixer.api.remix import _process_next_queued
                    _process_next_queued(app_state)

            mock_wrapper.side_effect = fake_wrapper

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "Hendrix guitar with MF Doom rapping",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            # Verify it's a valid UUID
            uuid.UUID(data["session_id"])

    def test_queues_when_processing_lock_held(self, client):
        """Should queue the request (return 200) if another remix is being processed."""
        # Acquire the lock to simulate an in-progress remix
        processing_lock = client.app.state.processing_lock
        processing_lock.acquire()

        try:
            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "test remix",
                },
            )
            # Request should be queued, not rejected
            assert response.status_code == 200
            assert "session_id" in response.json()
        finally:
            processing_lock.release()

    def test_503_when_queue_full(self, client):
        """Should return 503 when processing slot is held and queue is full."""
        processing_lock = client.app.state.processing_lock
        processing_lock.acquire()

        # Replace with a queue of capacity 1 and pre-fill it
        import queue as _queue
        from musicmixer.api.remix import _QueueItem
        from musicmixer.models import SessionState

        old_queue = client.app.state.wait_queue
        tiny_queue = _queue.Queue(maxsize=1)
        dummy_item = _QueueItem(
            session_id="dummy", session=SessionState(), run_fn=lambda: None,
        )
        tiny_queue.put(dummy_item)
        client.app.state.wait_queue = tiny_queue

        try:
            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "test remix",
                },
            )
            assert response.status_code == 503
            assert "capacity" in response.json()["detail"]
        finally:
            processing_lock.release()
            client.app.state.wait_queue = old_queue

    def test_session_created_in_app_state(self, client):
        """Session should be stored in app.state.sessions."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            def _fake(*a):
                a[5].release()  # processing_lock
                from musicmixer.api.remix import _process_next_queued
                _process_next_queued(a[6])  # app_state
            mock_wrapper.side_effect = _fake

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "test remix",
                },
            )

            session_id = response.json()["session_id"]
            assert session_id in client.app.state.sessions

    def test_youtube_disabled_returns_403(self, client):
        """Should return 403 when youtube_enabled is False."""
        with patch("musicmixer.api.remix.settings") as mock_settings:
            mock_settings.youtube_enabled = False

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "test remix",
                },
            )
            assert response.status_code == 403
            assert "disabled" in response.json()["detail"]

    def test_missing_prompt_field_accepted(self, client):
        """Missing prompt field should be accepted (defaults to empty string)."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            mock_wrapper.side_effect = lambda *a, **kw: a[5].release()

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    # missing prompt — should default to ""
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_empty_prompt_accepted(self, client):
        """Explicitly empty prompt should be accepted."""
        with patch("musicmixer.api.remix._youtube_pipeline_wrapper") as mock_wrapper:
            mock_wrapper.side_effect = lambda *a, **kw: a[5].release()

            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data

    def test_missing_url_fields(self, client):
        """Missing URL fields should return 422."""
        response = client.post(
            "/api/remix/youtube",
            json={
                "prompt": "test remix",
            },
        )
        assert response.status_code == 422


class TestYouTubePipelineWrapper:
    """Test the _youtube_pipeline_wrapper function directly."""

    def test_downloads_both_songs_and_runs_pipeline(self, tmp_path):
        """Wrapper should download both songs and call run_pipeline."""
        from musicmixer.api.remix import _youtube_pipeline_wrapper
        from musicmixer.models import SessionState

        session = SessionState()
        lock = threading.Lock()
        lock.acquire()

        wav_a = tmp_path / "song_a.wav"
        wav_b = tmp_path / "song_b.wav"
        wav_a.write_bytes(b"fake wav a")
        wav_b.write_bytes(b"fake wav b")

        result_a = FakeYouTubeAudioResult(
            wav_path=wav_a, title="Song A Title",
            duration_seconds=180.0, source_codec="opus", source_bitrate=128,
        )
        result_b = FakeYouTubeAudioResult(
            wav_path=wav_b, title="Song B Title",
            duration_seconds=200.0, source_codec="aac", source_bitrate=128,
        )

        download_call_count = 0

        async def fake_download(url, output_dir, progress_callback=None):
            nonlocal download_call_count
            download_call_count += 1
            if "dQw4w9WgXcQ" in url:
                return result_a
            return result_b

        mock_app_state = MagicMock()
        mock_app_state.wait_queue = queue.Queue(maxsize=10)
        mock_app_state.queue_lock = threading.Lock()
        mock_app_state.processing_lock = lock
        mock_app_state.executor = MagicMock()

        with patch("musicmixer.api.remix.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_settings.queue_entry_ttl_minutes = 15

            with patch("musicmixer.services.pipeline.run_pipeline") as mock_pipeline:
                with patch("musicmixer.services.youtube.download_youtube_audio", new=fake_download):
                    _youtube_pipeline_wrapper(
                        session_id="test-session",
                        url_a=VALID_YT_URL_A,
                        url_b=VALID_YT_URL_B,
                        prompt="test prompt",
                        session=session,
                        processing_lock=lock,
                        app_state=mock_app_state,
                    )

        # Both songs downloaded
        assert download_call_count == 2

        # Pipeline was called with correct paths and filenames
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args
        assert call_kwargs.kwargs.get("song_a_original_filename") == "Song A Title" or \
               call_kwargs[1].get("song_a_original_filename") == "Song A Title"

        # Lock was released
        assert not lock.locked()

    def test_emits_downloading_progress_events(self, tmp_path):
        """Wrapper should emit 'downloading' step progress events.

        Downloads run in parallel, so event ordering between A and B is
        non-deterministic.  We verify: initial event at 0.05, at least one
        progress callback fires, and the final "Both songs downloaded!" event
        is at 0.45.  The monotonic tracker suppresses out-of-order values.
        """
        from musicmixer.api.remix import _youtube_pipeline_wrapper
        from musicmixer.models import SessionState

        session = SessionState()
        lock = threading.Lock()
        lock.acquire()

        wav_path = tmp_path / "song.wav"
        wav_path.write_bytes(b"fake wav")

        result = FakeYouTubeAudioResult(
            wav_path=wav_path, title="Test",
            duration_seconds=180.0, source_codec="opus", source_bitrate=128,
        )

        async def fake_download(url, output_dir, progress_callback=None):
            if progress_callback:
                progress_callback(0.5, "50%")
            return result

        mock_app_state = MagicMock()
        mock_app_state.wait_queue = queue.Queue(maxsize=10)
        mock_app_state.queue_lock = threading.Lock()
        mock_app_state.processing_lock = lock
        mock_app_state.executor = MagicMock()

        with patch("musicmixer.api.remix.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_settings.queue_entry_ttl_minutes = 15

            with patch("musicmixer.services.pipeline.run_pipeline"):
                with patch("musicmixer.services.youtube.download_youtube_audio", new=fake_download):
                    _youtube_pipeline_wrapper(
                        session_id="test-session",
                        url_a=VALID_YT_URL_A,
                        url_b=VALID_YT_URL_B,
                        prompt="test",
                        session=session,
                        processing_lock=lock,
                        app_state=mock_app_state,
                    )

        # Collect all events from the queue
        events = []
        while not session.events.empty():
            events.append(session.events.get_nowait())

        # Should have downloading events (initial + callbacks + final)
        download_events = [e for e in events if e.get("step") == "downloading"]
        assert len(download_events) >= 2  # At least: initial + "Both songs downloaded!"

        # First event should be the initial download announcement at 0.05
        assert download_events[0]["detail"] == "Downloading songs from YouTube..."
        assert download_events[0]["progress"] == 0.05

        # Last downloading event should be "Both songs downloaded!" at 0.45
        assert download_events[-1]["detail"] == "Both songs downloaded!"
        assert download_events[-1]["progress"] == 0.45

        # Progress values should be monotonically increasing
        progress_values = [e["progress"] for e in download_events]
        for i in range(1, len(progress_values)):
            assert progress_values[i] > progress_values[i - 1], (
                f"Progress went backward: {progress_values}"
            )

    def test_error_releases_lock_and_sets_error_status(self, tmp_path):
        """If download fails, lock should be released and status set to error."""
        from musicmixer.api.remix import _youtube_pipeline_wrapper
        from musicmixer.models import SessionState

        session = SessionState()
        lock = threading.Lock()
        lock.acquire()

        async def failing_download(url, output_dir, progress_callback=None):
            raise RuntimeError("Download failed: video unavailable")

        mock_app_state = MagicMock()
        mock_app_state.wait_queue = queue.Queue(maxsize=10)
        mock_app_state.queue_lock = threading.Lock()
        mock_app_state.processing_lock = lock
        mock_app_state.executor = MagicMock()

        with patch("musicmixer.api.remix.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            mock_settings.queue_entry_ttl_minutes = 15

            with patch("musicmixer.services.youtube.download_youtube_audio", new=failing_download):
                _youtube_pipeline_wrapper(
                    session_id="test-session",
                    url_a=VALID_YT_URL_A,
                    url_b=VALID_YT_URL_B,
                    prompt="test",
                    session=session,
                    processing_lock=lock,
                    app_state=mock_app_state,
                )

        # Lock must be released
        assert not lock.locked()

        # Session status must be error
        assert session.status == "error"

        # Should have an error event in the queue
        events = []
        while not session.events.empty():
            events.append(session.events.get_nowait())
        error_events = [e for e in events if e.get("step") == "error"]
        assert len(error_events) >= 1
        assert "Download failed" in error_events[-1]["detail"]


class TestYouTubeProgressFlow:
    """Test that download progress events flow through the existing SSE mechanism."""

    def test_progress_events_visible_via_status(self, client, tmp_path):
        """After wrapper runs, last_event should reflect download progress."""
        completed = threading.Event()

        def fake_wrapper(session_id, url_a, url_b, prompt, session, lock, app_state=None, **kwargs):
            from musicmixer.services.pipeline import emit_progress
            emit_progress(session.events, {
                "step": "downloading",
                "detail": "Downloading song A from YouTube...",
                "progress": 0.05,
            }, session=session)
            emit_progress(session.events, {
                "step": "downloading",
                "detail": "Both songs downloaded!",
                "progress": 0.45,
            }, session=session)
            emit_progress(session.events, {
                "step": "complete",
                "detail": "Remix ready!",
                "progress": 1.0,
            }, session=session)
            session.status = "complete"
            lock.release()
            if app_state:
                from musicmixer.api.remix import _process_next_queued
                _process_next_queued(app_state)
            completed.set()

        with patch("musicmixer.api.remix._youtube_pipeline_wrapper", side_effect=fake_wrapper):
            response = client.post(
                "/api/remix/youtube",
                json={
                    "url_a": VALID_YT_URL_A,
                    "url_b": VALID_YT_URL_B,
                    "prompt": "test remix",
                },
            )
            session_id = response.json()["session_id"]

        # Wait for background thread to finish
        completed.wait(timeout=5)
        time.sleep(0.1)

        # Check status endpoint reflects the session state
        status_resp = client.get(f"/api/remix/{session_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["status"] == "complete"
        # last_event should be the complete event
        assert status_data["last_event"]["step"] == "complete"
