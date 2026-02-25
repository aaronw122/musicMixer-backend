"""Tests for Day 2 pipeline wiring (Step 7).

Tests the complete run_pipeline() chain with mocked stem separation
but real analysis, processing, rendering, and export.
"""

import queue
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import SessionState


def make_test_wav(path: Path, duration: float = 10.0, sr: int = 44100, freq: float = 440.0) -> Path:
    """Create a synthetic stereo WAV file with a sine tone.

    Uses different frequencies to produce slightly different BPM detections,
    which exercises more of the pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Add a click track at ~120 BPM (every 0.5 seconds) to help BPM detection
    click_interval = int(sr * 0.5)  # 120 BPM
    click = np.zeros_like(t)
    for i in range(0, len(t), click_interval):
        end = min(i + int(sr * 0.01), len(t))
        click[i:end] = 0.8
    signal = np.sin(2 * np.pi * freq * t) * 0.3 + click * 0.3
    audio = np.column_stack([signal, signal]).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return path


def make_mock_stems(stems_dir: Path, stem_names: list[str], duration: float = 10.0) -> dict[str, Path]:
    """Create a set of mock stem WAV files and return the path dict."""
    stems_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    freqs = {"vocals": 220.0, "drums": 100.0, "bass": 80.0, "guitar": 330.0, "piano": 440.0, "other": 550.0}
    for name in stem_names:
        path = stems_dir / f"{name}.wav"
        make_test_wav(path, duration=duration, freq=freqs.get(name, 440.0))
        result[name] = path
    return result


@pytest.fixture
def pipeline_tmp(tmp_path):
    """Set up a temporary directory structure for pipeline tests."""
    # Create song files (full-length songs that analysis will run on)
    song_a_path = tmp_path / "uploads" / "song_a.wav"
    song_b_path = tmp_path / "uploads" / "song_b.wav"
    make_test_wav(song_a_path, duration=15.0, freq=440.0)
    make_test_wav(song_b_path, duration=15.0, freq=330.0)

    # Pre-create mock stem directories
    song_a_stems = make_mock_stems(
        tmp_path / "stems" / "test-session" / "song_a",
        ["vocals", "drums", "bass", "other"],
        duration=15.0,
    )
    song_b_stems = make_mock_stems(
        tmp_path / "stems" / "test-session" / "song_b",
        ["vocals", "drums", "bass", "other"],
        duration=15.0,
    )

    return {
        "tmp_path": tmp_path,
        "song_a_path": song_a_path,
        "song_b_path": song_b_path,
        "song_a_stems": song_a_stems,
        "song_b_stems": song_b_stems,
    }


def _run_pipeline_with_mock_separation(pipeline_tmp, session=None):
    """Helper to run the pipeline with mocked separation returning pre-made stems."""
    from musicmixer.services.pipeline import run_pipeline

    tmp_path = pipeline_tmp["tmp_path"]
    event_queue = queue.Queue(maxsize=100)
    if session is None:
        session = SessionState()

    def mock_separate(audio_path, output_dir, progress_callback=None):
        """Return the pre-made stems for the matching song."""
        if "song_a" in str(audio_path):
            return pipeline_tmp["song_a_stems"]
        else:
            return pipeline_tmp["song_b_stems"]

    with (
        patch("musicmixer.config.settings") as mock_settings,
        patch("musicmixer.services.separation.separate_stems", side_effect=mock_separate),
    ):
        mock_settings.data_dir = tmp_path

        run_pipeline(
            session_id="test-session",
            song_a_path=str(pipeline_tmp["song_a_path"]),
            song_b_path=str(pipeline_tmp["song_b_path"]),
            prompt="test remix",
            event_queue=event_queue,
            session=session,
        )

    return event_queue, session


class TestFullPipelineWithMocks:
    """Run the full pipeline with mocked separation, verify output MP3 exists."""

    def test_full_pipeline_produces_mp3(self, pipeline_tmp):
        """Pipeline should produce a non-empty MP3 file at the expected path."""
        event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)

        output_path = Path(session.remix_path)
        assert output_path.exists(), f"Expected remix MP3 at {output_path}"
        assert output_path.stat().st_size > 0, "Remix MP3 should not be empty"
        assert output_path.suffix == ".mp3"


class TestPipelineEmitsProgressEvents:
    """Verify the event queue contains events with expected steps in order."""

    def test_progress_events_in_order(self, pipeline_tmp):
        """Pipeline should emit progress events in the correct sequence."""
        event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)

        # Drain all events
        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        # Extract step names in order
        steps = [e["step"] for e in events]

        # Verify expected steps appear in sequence
        expected_steps = ["separating", "analyzing", "processing", "rendering", "complete"]
        seen = []
        for step in steps:
            if step in expected_steps and (not seen or seen[-1] != step):
                seen.append(step)

        assert seen == expected_steps, (
            f"Expected steps {expected_steps} in order, got unique sequence: {seen}. "
            f"Full steps list: {steps}"
        )

    def test_progress_values_are_monotonically_increasing(self, pipeline_tmp):
        """Progress values should generally increase over time."""
        event_queue, _session = _run_pipeline_with_mock_separation(pipeline_tmp)

        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        progress_values = [e["progress"] for e in events]
        # The last event should be 1.0 (complete)
        assert progress_values[-1] == 1.0

        # Check that progress never goes backward by more than a trivial amount
        # (small regressions can happen between steps due to weighted percentages)
        for i in range(1, len(progress_values)):
            assert progress_values[i] >= progress_values[i - 1] - 0.01, (
                f"Progress went backward: {progress_values[i-1]} -> {progress_values[i]} "
                f"at event {i}: {events[i]}"
            )

    def test_complete_event_has_required_fields(self, pipeline_tmp):
        """The final 'complete' event should contain explanation, warnings, usedFallback."""
        event_queue, _session = _run_pipeline_with_mock_separation(pipeline_tmp)

        events = []
        while not event_queue.empty():
            events.append(event_queue.get_nowait())

        complete_event = events[-1]
        assert complete_event["step"] == "complete"
        assert "explanation" in complete_event
        assert "warnings" in complete_event
        assert "usedFallback" in complete_event
        assert complete_event["progress"] == 1.0


class TestPipelineSetsSessionState:
    """Verify session state is updated correctly on completion."""

    def test_session_status_is_complete(self, pipeline_tmp):
        """session.status should be 'complete' after successful pipeline run."""
        _event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)
        assert session.status == "complete"

    def test_session_remix_path_is_set(self, pipeline_tmp):
        """session.remix_path should point to the output MP3."""
        _event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)
        assert session.remix_path is not None
        assert Path(session.remix_path).exists()
        assert session.remix_path.endswith("remix.mp3")

    def test_session_explanation_is_set(self, pipeline_tmp):
        """session.explanation should be a non-empty string."""
        _event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)
        assert session.explanation is not None
        assert len(session.explanation) > 0

    def test_session_last_event_is_complete(self, pipeline_tmp):
        """session.last_event should be the 'complete' event."""
        _event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)
        assert session.last_event is not None
        assert session.last_event["step"] == "complete"


class TestPipelineHandlesSeparationError:
    """Verify that separation errors propagate correctly."""

    def test_separation_error_propagates(self, pipeline_tmp):
        """When separation raises, the error should propagate to the caller.

        The pipeline wrapper in remix.py catches exceptions and emits error events.
        The pipeline itself should let the exception propagate.
        """
        from musicmixer.services.pipeline import run_pipeline

        tmp_path = pipeline_tmp["tmp_path"]
        event_queue = queue.Queue(maxsize=100)
        session = SessionState()

        def mock_separate_raises(audio_path, output_dir, progress_callback=None):
            raise RuntimeError("GPU unavailable: Modal service down")

        with (
            patch("musicmixer.config.settings") as mock_settings,
            patch("musicmixer.services.separation.separate_stems", side_effect=mock_separate_raises),
        ):
            mock_settings.data_dir = tmp_path

            with pytest.raises(RuntimeError, match="GPU unavailable"):
                run_pipeline(
                    session_id="test-session",
                    song_a_path=str(pipeline_tmp["song_a_path"]),
                    song_b_path=str(pipeline_tmp["song_b_path"]),
                    prompt="test remix",
                    event_queue=event_queue,
                    session=session,
                )
