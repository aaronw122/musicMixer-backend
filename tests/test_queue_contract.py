"""Contract tests for queue lifecycle terminal events (Slice 0 lock).

These pin down event payloads that the orchestration refactor must preserve
byte-for-byte, and which were previously uncovered:

- queued cancellation still emits the `cancelled` event (via the /cancel route
  AND when `_process_next_queued` skips an already-cancelled queued item)
- queue entry TTL expiry emits the `error` "Queue wait time exceeded" event
- a lost-slot race that overflows the queue emits the `error` "Server overloaded"
  event
- the queue-full path raises HTTPException(503) with the existing detail string

The pipeline wrapper and download internals are never exercised here — these
tests drive the queue helpers directly so the payloads are isolated.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fastapi import HTTPException

from musicmixer.api.remix import (
    _QueueItem,
    _enqueue_or_start,
    _process_next_queued,
    cancel_remix,
)
from musicmixer.models import SessionState


def _drain(session: SessionState) -> list[dict]:
    events = []
    while not session.events.empty():
        events.append(session.events.get_nowait())
    return events


def _make_app_state(processing_lock, *, queue_maxsize=10):
    """Minimal app_state stand-in for the queue helpers."""
    return SimpleNamespace(
        wait_queue=queue.Queue(maxsize=queue_maxsize),
        queue_lock=threading.Lock(),
        processing_lock=processing_lock,
        executor=MagicMock(),
    )


# ---------------------------------------------------------------------------
# Queued cancellation — same `cancelled` event
# ---------------------------------------------------------------------------


class TestQueuedCancellation:
    @pytest.mark.asyncio
    async def test_cancel_route_emits_cancelled_event_for_queued_session(self):
        """POST /cancel on a queued session emits the canonical cancelled event."""
        session = SessionState(status="queued")
        session_id = "11111111-1111-1111-1111-111111111111"
        request = MagicMock()
        request.app.state.sessions_lock = threading.Lock()
        request.app.state.sessions = {session_id: session}

        result = await cancel_remix(session_id, request)

        assert result == {"status": "cancelling", "message": "Cancel signal sent"}
        assert session.cancelled.is_set()
        assert session.status == "cancelled"

        events = _drain(session)
        cancelled = [e for e in events if e.get("step") == "cancelled"]
        assert cancelled == [
            {"step": "cancelled", "detail": "Remix cancelled", "progress": 0}
        ]

    def test_process_next_queued_skips_cancelled_item(self):
        """A queued item cancelled before its turn is skipped and marked cancelled.

        It must NOT acquire the processing slot or get submitted to the executor.
        """
        processing_lock = MagicMock()
        app_state = _make_app_state(processing_lock)

        cancelled_session = SessionState(status="queued")
        cancelled_session.cancelled.set()
        item = _QueueItem(
            session_id="cancelled-one",
            session=cancelled_session,
            run_fn=MagicMock(),
        )
        app_state.wait_queue.put(item)

        with patch("musicmixer.api.remix._broadcast_queue_positions"):
            _process_next_queued(app_state)

        assert cancelled_session.status == "cancelled"
        processing_lock.acquire.assert_not_called()
        app_state.executor.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Queue TTL expiry — "Queue wait time exceeded" error event
# ---------------------------------------------------------------------------


class TestQueueTimeout:
    def test_expired_entry_emits_queue_wait_exceeded_error(self):
        processing_lock = MagicMock()
        app_state = _make_app_state(processing_lock)

        session = SessionState(status="queued")
        item = _QueueItem(
            session_id="expired-one",
            session=session,
            run_fn=MagicMock(),
        )
        # Make the entry look older than the TTL window.
        item.enqueued_at = -10_000.0
        app_state.wait_queue.put(item)

        mock_settings = SimpleNamespace(queue_entry_ttl_minutes=15)
        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.api.remix._broadcast_queue_positions"):
            _process_next_queued(app_state)

        assert session.status == "error"
        processing_lock.acquire.assert_not_called()
        app_state.executor.submit.assert_not_called()

        events = _drain(session)
        assert {
            "step": "error",
            "detail": "Queue wait time exceeded, please try again",
            "progress": 0,
        } in events


# ---------------------------------------------------------------------------
# Lost-slot race overflow — "Server overloaded" error event
# ---------------------------------------------------------------------------


class TestServerOverloaded:
    def test_requeue_into_full_queue_emits_server_overloaded(self):
        """If the slot is lost AND the queue is full on re-queue, emit overloaded.

        The item is pulled off the queue, the slot acquire fails (lost race), and
        the re-queue overflows. The overflow is a genuine race in production; here
        we force it deterministically by making put_nowait raise queue.Full only on
        the re-queue attempt.
        """
        processing_lock = MagicMock()
        processing_lock.acquire.return_value = False

        app_state = _make_app_state(processing_lock)

        session = SessionState(status="queued")
        item = _QueueItem(
            session_id="overloaded-one",
            session=session,
            run_fn=MagicMock(),
        )
        app_state.wait_queue.put(item)

        # After the item is pulled, the re-queue put_nowait must overflow.
        def _full_on_requeue(value):
            raise queue.Full

        mock_settings = SimpleNamespace(queue_entry_ttl_minutes=15)
        with patch("musicmixer.api.remix.settings", mock_settings), \
             patch("musicmixer.api.remix._broadcast_queue_positions"), \
             patch.object(app_state.wait_queue, "put_nowait", side_effect=_full_on_requeue):
            _process_next_queued(app_state)

        assert session.status == "error"
        app_state.executor.submit.assert_not_called()

        events = _drain(session)
        assert {
            "step": "error",
            "detail": "Server overloaded, please try again",
            "progress": 0,
        } in events


# ---------------------------------------------------------------------------
# Queue full on enqueue — 503 with the existing detail string
# ---------------------------------------------------------------------------


class TestQueueFull503:
    def test_enqueue_raises_503_when_queue_full(self):
        # Slot busy so enqueue is attempted; queue full so it raises.
        processing_lock = MagicMock()
        processing_lock.acquire.return_value = False
        app_state = _make_app_state(processing_lock, queue_maxsize=1)

        # Pre-fill the queue to capacity.
        app_state.wait_queue.put(
            _QueueItem(session_id="filler", session=SessionState(), run_fn=MagicMock())
        )

        session = SessionState(status="queued")
        with pytest.raises(HTTPException) as exc_info:
            _enqueue_or_start(app_state, "new-one", session, MagicMock())

        assert exc_info.value.status_code == 503
        assert exc_info.value.detail == "Server is at capacity, please try again later"

    def test_enqueue_emits_queue_position_and_estimate_payloads(self):
        """When enqueued, the position + estimate event payloads stay stable."""
        processing_lock = MagicMock()
        processing_lock.acquire.return_value = False
        app_state = _make_app_state(processing_lock, queue_maxsize=10)

        session = SessionState(status="queued")
        with patch("musicmixer.api.remix._AVG_REMIX_DURATION_S", 600.0):
            _enqueue_or_start(app_state, "queued-one", session, MagicMock())

        events = _drain(session)
        position_events = [e for e in events if e.get("step") == "queue_position"]
        estimate_events = [e for e in events if e.get("step") == "queue_estimate"]

        assert position_events == [
            {
                "step": "queue_position",
                "detail": "Position 1 of 1",
                "position": 1,
                "total": 1,
                "progress": 0,
            }
        ]
        assert estimate_events == [
            {
                "step": "queue_estimate",
                "detail": "Estimated wait: 600s",
                "wait_seconds": 600,
                "progress": 0,
            }
        ]
