"""Day 2 pipeline orchestrator.

Runs the remix pipeline in a background thread, emitting SSE progress events.
Steps will be wired in by Step 7; for now wraps the Day 1 pipeline with progress events.
"""

import logging
import queue

from musicmixer.models import SessionState

logger = logging.getLogger(__name__)


def emit_progress(
    event_queue: queue.Queue,
    event: dict,
    session: SessionState | None = None,
) -> None:
    """Non-blocking event push. Drops non-terminal events on full queue.

    Terminal events (complete/error) drain one old event first to guarantee delivery.
    If *session* is provided, also updates ``session.last_event`` so reconnecting
    SSE clients can pick up from where things left off even if no client was
    connected when the event was emitted.
    """
    try:
        event_queue.put_nowait(event)
    except queue.Full:
        if event.get("step") in ("complete", "error"):
            try:
                event_queue.get_nowait()
            except queue.Empty:
                pass
            event_queue.put_nowait(event)
        else:
            logger.warning("Event queue full, dropping: %s", event.get("step"))

    if session is not None:
        session.last_event = event


def run_pipeline(
    session_id: str,
    song_a_path: str,
    song_b_path: str,
    prompt: str,
    event_queue: queue.Queue,
    session: SessionState,
) -> None:
    """Day 2 pipeline skeleton. Steps will be filled in by the wiring agent (Step 7).

    For now: runs the Day 1 separation + overlay logic, emitting progress events.

    Pipeline steps (to be wired in Step 7):
      1. Separate stems (existing Day 1 Modal/local code)
      2. Analyze both songs (BPM, beat_frames, duration)
      3. Reconcile BPM between songs
      4. Generate deterministic fallback plan
      5. Standardize stems (44.1kHz, stereo, float32)
      6. Trim stems to source time ranges
      7. Tempo match vocals via rubberband
      8. Re-run beat detection on stretched audio
      9. Cross-song level match
     10. Render arrangement
     11. Sum buses into final mix
     12. LUFS normalize
     13. Peak limiter
     14. Fade-in/fade-out
     15. Export to MP3
    """
    from pathlib import Path

    emit_progress(event_queue, {
        "step": "separating",
        "detail": "Extracting stems from both songs...",
        "progress": 0.10,
    }, session=session)

    # For now, delegate to Day 1 pipeline as a placeholder
    from musicmixer.services.pipeline_day1 import run_pipeline_sync

    remix_path = run_pipeline_sync(session_id, Path(song_a_path), Path(song_b_path))

    emit_progress(event_queue, {
        "step": "mixing",
        "detail": "Building the remix...",
        "progress": 0.80,
    }, session=session)

    session.remix_path = str(remix_path)
    session.explanation = "Remix created using Day 1 pipeline (Day 2 wiring pending)."
    session.status = "complete"

    emit_progress(event_queue, {
        "step": "complete",
        "detail": "Remix ready!",
        "progress": 1.0,
        "explanation": session.explanation,
        "warnings": [],
        "usedFallback": True,
    }, session=session)
