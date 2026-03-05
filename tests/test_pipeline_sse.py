"""Tests for Day 2 async pipeline orchestrator + SSE progress events."""

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from musicmixer.main import app
from musicmixer.models import SessionState
from musicmixer.services.pipeline import emit_progress


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    """Create a test client with temp data directory and app state initialized."""
    with patch("musicmixer.config.settings") as mock_settings:
        mock_settings.data_dir = tmp_path
        mock_settings.allowed_extensions = {".mp3", ".wav"}
        mock_settings.max_file_size_mb = 50
        mock_settings.cors_origins = ["http://localhost:5173"]
        mock_settings.host = "0.0.0.0"
        mock_settings.port = 8000
        mock_settings.max_concurrent_mixes = 1
        mock_settings.distributed_limiter_enabled = False

        # Create required directories
        (tmp_path / "uploads").mkdir()
        (tmp_path / "stems").mkdir()
        (tmp_path / "remixes").mkdir()

        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.main.settings", mock_settings):
            with TestClient(app) as c:
                yield c


@pytest.fixture
def mock_pipeline_fast(tmp_path):
    """Mock pipeline that completes instantly, emitting progress events."""
    def _mock_run_pipeline(session_id, song_a_path, song_b_path, prompt, event_queue, session, **kwargs):
        from musicmixer.services.pipeline import emit_progress

        emit_progress(event_queue, {
            "step": "separating",
            "detail": "Extracting stems...",
            "progress": 0.10,
        }, session=session)

        # Create fake remix file
        remix_dir = tmp_path / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)
        (remix_dir / "remix.mp3").write_bytes(b"fake mp3 data")

        session.remix_path = str(remix_dir / "remix.mp3")
        session.explanation = "Test remix"
        session.status = "complete"

        emit_progress(event_queue, {
            "step": "complete",
            "detail": "Remix ready!",
            "progress": 1.0,
            "explanation": "Test remix",
            "warnings": [],
            "usedFallback": True,
        }, session=session)

    return _mock_run_pipeline


@pytest.fixture
def mock_pipeline_slow():
    """Mock pipeline that blocks on an event, so we can control timing."""
    go_event = threading.Event()
    done_event = threading.Event()

    def _mock_run_pipeline(session_id, song_a_path, song_b_path, prompt, event_queue, session, **kwargs):
        from musicmixer.services.pipeline import emit_progress

        session.status = "processing"
        emit_progress(event_queue, {
            "step": "separating",
            "detail": "Extracting stems...",
            "progress": 0.10,
        }, session=session)

        # Wait until test signals to proceed
        go_event.wait(timeout=30)

        session.status = "complete"
        emit_progress(event_queue, {
            "step": "complete",
            "detail": "Done!",
            "progress": 1.0,
        }, session=session)
        done_event.set()

    return _mock_run_pipeline, go_event, done_event


def _post_remix(client):
    """Helper to POST a remix with fake files."""
    return client.post(
        "/api/remix",
        files={
            "song_a": ("song_a.mp3", b"fake mp3 data", "audio/mpeg"),
            "song_b": ("song_b.mp3", b"fake mp3 data", "audio/mpeg"),
        },
        data={"prompt": "test remix"},
    )


def _post_youtube(client):
    """Helper to POST a YouTube remix."""
    return client.post(
        "/api/remix/youtube",
        json={
            "url_a": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "url_b": "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "prompt": "test remix",
        },
    )


# ---------------------------------------------------------------------------
# Tests: POST /api/remix returns session_id immediately
# ---------------------------------------------------------------------------


class TestPostRemixAsync:
    def test_returns_session_id_immediately(self, client, mock_pipeline_fast):
        """POST should return session_id without waiting for pipeline to finish."""
        with patch("musicmixer.api.remix._pipeline_wrapper") as mock_wrapper:
            # Replace wrapper to avoid actually running pipeline
            mock_wrapper.side_effect = lambda *args: args[4].events.put_nowait(
                {"step": "complete", "detail": "done", "progress": 1.0}
            ) or args[5].release()

            response = _post_remix(client)

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

        # Verify it's a valid UUID
        import uuid
        uuid.UUID(data["session_id"])

    def test_returns_409_when_processing(self, client, mock_pipeline_slow):
        """A second POST while pipeline is running should return 409."""
        slow_fn, go_event, done_event = mock_pipeline_slow

        with patch("musicmixer.services.pipeline.run_pipeline", side_effect=slow_fn):
            # First request -- should succeed
            resp1 = _post_remix(client)
            assert resp1.status_code == 200

            # Give the executor a moment to start the pipeline
            time.sleep(0.3)

            # Second request -- should get 409
            resp2 = _post_remix(client)
            assert resp2.status_code == 409
            assert "Another remix" in resp2.json()["detail"]

            # Release the slow pipeline so cleanup happens
            go_event.set()
            done_event.wait(timeout=5)

    def test_mixed_endpoints_share_global_capacity_gate(self, client):
        """Two mixed endpoint requests may be accepted at capacity=2; third gets 409."""
        go_event = threading.Event()
        done_event = threading.Event()
        done_count = 0
        done_lock = threading.Lock()

        def _mark_done():
            nonlocal done_count
            with done_lock:
                done_count += 1
                if done_count == 2:
                    done_event.set()

        def _slow_pipeline_wrapper(
            session_id,
            song_a_path,
            song_b_path,
            prompt,
            session,
            processing_lock,
            *_args,
            **_kwargs,
        ):
            try:
                session.status = "processing"
                go_event.wait(timeout=5)
            finally:
                processing_lock.release()
                _mark_done()

        def _slow_youtube_wrapper(
            session_id,
            url_a,
            url_b,
            prompt,
            session,
            processing_lock,
        ):
            try:
                session.status = "processing"
                go_event.wait(timeout=5)
            finally:
                processing_lock.release()
                _mark_done()

        old_executor = client.app.state.executor
        old_executor.shutdown(wait=False, cancel_futures=True)
        client.app.state.executor = ThreadPoolExecutor(max_workers=2)
        client.app.state.processing_lock = threading.BoundedSemaphore(value=2)

        with patch(
            "musicmixer.api.remix._pipeline_wrapper",
            side_effect=_slow_pipeline_wrapper,
        ), patch(
            "musicmixer.api.remix._youtube_pipeline_wrapper",
            side_effect=_slow_youtube_wrapper,
        ):
            try:
                resp1 = _post_remix(client)
                assert resp1.status_code == 200

                resp2 = _post_youtube(client)
                assert resp2.status_code == 200

                resp3 = _post_remix(client)
                assert resp3.status_code == 409
                assert "Another remix" in resp3.json()["detail"]
            finally:
                go_event.set()
                done_event.wait(timeout=5)


# ---------------------------------------------------------------------------
# Tests: GET /api/remix/{id}/status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_returns_session_status(self, client, mock_pipeline_fast):
        """GET /status should return current session state."""
        with patch("musicmixer.services.pipeline.run_pipeline", side_effect=mock_pipeline_fast):
            resp = _post_remix(client)
            session_id = resp.json()["session_id"]

            # Give pipeline a moment to complete
            time.sleep(0.5)

            status_resp = client.get(f"/api/remix/{session_id}/status")
            assert status_resp.status_code == 200

            data = status_resp.json()
            assert data["session_id"] == session_id
            assert data["status"] == "complete"
            assert data["explanation"] == "Test remix"

    def test_returns_404_for_unknown_session(self, client):
        """GET /status should return 404 for non-existent session."""
        import uuid
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/remix/{fake_id}/status")
        assert resp.status_code == 404

    def test_returns_400_for_invalid_uuid(self, client):
        """GET /status should return 400 for malformed session ID."""
        resp = client.get("/api/remix/not-a-uuid/status")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Tests: GET /api/remix/{id}/progress (SSE)
# ---------------------------------------------------------------------------


class TestGetProgress:
    def test_sse_stream_returns_events(self, client, mock_pipeline_fast):
        """SSE endpoint should stream progress events and close on complete."""
        with patch("musicmixer.services.pipeline.run_pipeline", side_effect=mock_pipeline_fast):
            resp = _post_remix(client)
            session_id = resp.json()["session_id"]

            # Give pipeline a moment to complete
            time.sleep(0.5)

            # Request SSE stream
            sse_resp = client.get(
                f"/api/remix/{session_id}/progress",
                headers={"Accept": "text/event-stream"},
            )
            assert sse_resp.status_code == 200
            assert "text/event-stream" in sse_resp.headers["content-type"]

            # Parse SSE events from response body
            body = sse_resp.text
            events = _parse_sse_events(body)

            # Should have at least the complete event (last_event replay on reconnect)
            assert len(events) >= 1

            # The last real event should be complete
            terminal_events = [e for e in events if e.get("step") in ("complete", "error")]
            assert len(terminal_events) >= 1
            assert terminal_events[-1]["step"] == "complete"

    def test_sse_returns_404_for_unknown_session(self, client):
        """SSE endpoint should return 404 for non-existent session."""
        import uuid
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/remix/{fake_id}/progress")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: emit_progress
# ---------------------------------------------------------------------------


class TestEmitProgress:
    def test_puts_event_on_queue(self):
        """emit_progress should add event to queue."""
        q = queue.Queue(maxsize=10)
        event = {"step": "separating", "detail": "test", "progress": 0.5}
        emit_progress(q, event)
        assert q.get_nowait() == event

    def test_drops_non_terminal_on_full_queue(self):
        """Non-terminal events should be silently dropped when queue is full."""
        q = queue.Queue(maxsize=2)
        q.put({"step": "s1"})
        q.put({"step": "s2"})

        # Queue is full -- non-terminal event should be dropped
        emit_progress(q, {"step": "separating", "detail": "dropped", "progress": 0.3})

        # Queue should still have exactly 2 items (the originals)
        assert q.qsize() == 2
        assert q.get_nowait()["step"] == "s1"
        assert q.get_nowait()["step"] == "s2"

    def test_keeps_terminal_complete_on_full_queue(self):
        """Terminal 'complete' events should drain one old event to make room."""
        q = queue.Queue(maxsize=2)
        q.put({"step": "s1"})
        q.put({"step": "s2"})

        # Queue is full -- terminal event should drain one and succeed
        emit_progress(q, {"step": "complete", "detail": "done", "progress": 1.0})

        # Should now have s2 + complete
        assert q.qsize() == 2
        e1 = q.get_nowait()
        e2 = q.get_nowait()
        assert e1["step"] == "s2"
        assert e2["step"] == "complete"

    def test_keeps_terminal_error_on_full_queue(self):
        """Terminal 'error' events should drain one old event to make room."""
        q = queue.Queue(maxsize=2)
        q.put({"step": "s1"})
        q.put({"step": "s2"})

        emit_progress(q, {"step": "error", "detail": "failed", "progress": 0})

        assert q.qsize() == 2
        e1 = q.get_nowait()
        e2 = q.get_nowait()
        assert e1["step"] == "s2"
        assert e2["step"] == "error"

    def test_empty_queue_terminal_event(self):
        """Terminal event on empty queue should just succeed normally."""
        q = queue.Queue(maxsize=2)
        emit_progress(q, {"step": "complete", "detail": "done", "progress": 1.0})
        assert q.qsize() == 1
        assert q.get_nowait()["step"] == "complete"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE text body into a list of JSON event dicts."""
    events = []
    for line in body.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                events.append(data)
            except json.JSONDecodeError:
                pass
    return events
