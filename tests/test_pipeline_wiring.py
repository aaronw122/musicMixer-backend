"""Tests for Day 2 pipeline wiring (Step 7) + loudness fix stages.

Tests the complete run_pipeline() chain with mocked stem separation
but real analysis, processing, rendering, and export.

Also includes integration tests for the loudness fix pipeline stages:
pre-limiting, look-ahead limiter, LUFS normalization, safety soft clip.
"""

import queue
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

from musicmixer.models import SessionState
from musicmixer.services.processor import (
    lufs_normalize_constrained,
    soft_clip,
    true_peak,
    true_peak_limit,
)


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


def _run_pipeline_with_mock_separation(pipeline_tmp, session=None, settings_overrides=None, **pipeline_kwargs):
    """Helper to run the pipeline with mocked separation returning pre-made stems.

    Args:
        pipeline_tmp: Fixture dict with paths and stems.
        session: Optional pre-created SessionState.
        settings_overrides: Dict of setting name -> value to override on mock_settings.
        **pipeline_kwargs: Extra keyword args passed to run_pipeline (e.g. source_quality_a).
    """
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

        # Set sensible defaults for all settings the pipeline reads.
        # MagicMock returns MagicMock objects for unset attributes, which
        # causes boolean comparisons to behave unpredictably.
        mock_settings.stem_backend = "modal"
        mock_settings.lyrics_lookup_enabled = False
        mock_settings.ab_taste_model_v1 = False
        mock_settings.anthropic_api_key = ""  # Prevent LLM API calls in tests

        # Apply settings overrides (AB flags, etc.)
        if settings_overrides:
            for key, value in settings_overrides.items():
                setattr(mock_settings, key, value)

        # Patch interpreter's module-level settings reference to the same mock
        # (it imports settings at module level, so musicmixer.config.settings
        # patch alone doesn't reach it)
        with patch("musicmixer.services.interpreter.settings", mock_settings):
            run_pipeline(
                session_id="test-session",
                song_a_path=str(pipeline_tmp["song_a_path"]),
                song_b_path=str(pipeline_tmp["song_b_path"]),
                prompt="test remix",
                event_queue=event_queue,
                session=session,
                **pipeline_kwargs,
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


# ---------------------------------------------------------------------------
# Loudness fix integration tests
# ---------------------------------------------------------------------------

SR = 44100


def _make_drum_like_signal(duration: float = 5.0, sr: int = SR) -> np.ndarray:
    """Create a signal with drum-like transients (high crest factor).

    Uses a sine-based sustained level with periodic loud bursts,
    resembling a real drum stem with kick/snare hits.
    Deterministic seed for reproducibility.
    """
    rng = np.random.RandomState(42)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Sustained content at moderate level (hi-hat / room)
    sustained = np.sin(2 * np.pi * 200 * t).astype(np.float32) * 0.15

    # Add transient bursts every ~0.25 seconds (like a fast beat)
    beat_interval = int(0.25 * sr)
    transients = np.zeros(n, dtype=np.float32)
    for i in range(0, n, beat_interval):
        burst_len = int(0.008 * sr)
        end = min(i + burst_len, n)
        transients[i:end] = 0.85

    signal = sustained + transients
    # Make stereo
    return np.column_stack([signal, signal]).astype(np.float32)


def _make_music_like_signal(duration: float = 5.0, sr: int = SR) -> np.ndarray:
    """Create a more realistic music-like signal for testing.

    Combines multiple sine tones at moderate level — simulates
    a mixed signal closer to real audio for pipeline testing.
    """
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Multiple harmonics at decent level
    signal = (
        np.sin(2 * np.pi * 110 * t) * 0.20 +
        np.sin(2 * np.pi * 220 * t) * 0.15 +
        np.sin(2 * np.pi * 440 * t) * 0.10 +
        np.sin(2 * np.pi * 880 * t) * 0.05
    ).astype(np.float32)

    # Add some beats
    beat_interval = int(0.5 * sr)
    for i in range(0, n, beat_interval):
        burst_len = int(0.01 * sr)
        end = min(i + burst_len, n)
        signal[i:end] += 0.4

    return np.column_stack([signal, signal]).astype(np.float32)


class TestLoudnessFixPipeline:
    """Test that the new pipeline stages work correctly."""

    def test_pre_limiting_reduces_peak(self):
        """Pre-limiting drums should reduce peak level."""
        drums = _make_drum_like_signal()

        peak_before = true_peak(drums)

        limited = true_peak_limit(
            drums, SR, ceiling_dbtp=-3.0, lookahead_ms=3.0, release_ms=30.0,
        )

        peak_after = true_peak(limited)

        # Peak should be reduced
        assert peak_after < peak_before, (
            f"Expected peak reduction: before={peak_before:.4f}, after={peak_after:.4f}"
        )

    def test_pre_limiting_preserves_loudness(self):
        """Pre-limiting should not drastically change LUFS."""
        drums = _make_drum_like_signal()
        meter = pyln.Meter(SR)
        lufs_before = meter.integrated_loudness(drums)

        limited = true_peak_limit(
            drums, SR, ceiling_dbtp=-3.0, lookahead_ms=3.0, release_ms=30.0,
        )
        lufs_after = meter.integrated_loudness(limited)

        # LUFS change should be bounded — limiting peaks doesn't destroy
        # average loudness for signals with reasonable crest factor
        assert abs(lufs_after - lufs_before) < 6.0, (
            f"LUFS changed too much: {lufs_before:.1f} -> {lufs_after:.1f}"
        )

    def test_mix_bus_limiter_controls_peaks(self):
        """The mix-bus look-ahead limiter should bring sample peaks under ceiling."""
        signal = _make_music_like_signal()

        limited = true_peak_limit(
            signal, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=50.0,
        )

        # Sample peak (not true peak) should be under ceiling
        sample_peak = float(np.max(np.abs(limited)))
        ceiling_linear = 10 ** (-1.0 / 20.0)
        assert sample_peak <= ceiling_linear + 0.01, (
            f"Sample peak {sample_peak:.4f} exceeds ceiling {ceiling_linear:.4f}"
        )

    def test_safety_clip_catches_residual_peaks(self):
        """Safety soft clip should bring true peak under ceiling."""
        signal = _make_drum_like_signal()

        # Run through limiter first
        limited = true_peak_limit(
            signal, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=50.0,
        )

        # Safety clip
        ceiling_linear = 10 ** (-1.0 / 20.0)
        clipped = soft_clip(limited, ceiling_linear, knee_db=2.0)

        # After safety clip, true peak must be under ceiling
        peak = true_peak(clipped)
        assert peak <= ceiling_linear + 0.001, (
            f"True peak {peak:.4f} exceeds ceiling {ceiling_linear:.4f} after safety clip"
        )

    def test_safety_clip_is_transparent_on_limited_signal(self):
        """Safety soft clip after normalization should not significantly change LUFS."""
        signal = _make_music_like_signal()

        # Limit + normalize
        limited = true_peak_limit(
            signal, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=50.0,
        )
        normalized = lufs_normalize_constrained(
            limited, SR, target_lufs=-12.0, ceiling_dbtp=-1.0,
        )

        meter = pyln.Meter(SR)
        lufs_before_clip = meter.integrated_loudness(normalized)

        # Safety clip with 2.0 dB knee
        ceiling_linear = 10 ** (-1.0 / 20.0)
        clipped = soft_clip(normalized, ceiling_linear, knee_db=2.0)

        lufs_after_clip = meter.integrated_loudness(clipped)

        # LUFS should barely change (safety clip should be mostly transparent)
        assert abs(lufs_after_clip - lufs_before_clip) < 1.0, (
            f"Safety clip changed LUFS too much: {lufs_before_clip:.1f} -> {lufs_after_clip:.1f}"
        )

    def test_full_chain_peak_under_ceiling(self):
        """After the full new chain, true peak should be within ceiling."""
        signal = _make_drum_like_signal()

        # Full chain: pre-limit -> mix-bus limit -> normalize -> safety clip
        step1 = true_peak_limit(
            signal, SR, ceiling_dbtp=-3.0, lookahead_ms=3.0, release_ms=30.0,
        )
        step2 = true_peak_limit(
            step1, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=50.0,
        )
        step3 = lufs_normalize_constrained(
            step2, SR, target_lufs=-12.0, ceiling_dbtp=-1.0,
        )
        ceiling_linear = 10 ** (-1.0 / 20.0)
        step4 = soft_clip(step3, ceiling_linear, knee_db=2.0)

        peak = true_peak(step4)
        assert peak <= ceiling_linear + 0.001, (
            f"True peak {peak:.4f} exceeds ceiling {ceiling_linear:.4f}"
        )

    def test_full_chain_lufs_is_reasonable(self):
        """After the full chain, output LUFS should not be degenerate."""
        signal = _make_music_like_signal()
        meter = pyln.Meter(SR)

        # Full chain
        step1 = true_peak_limit(
            signal, SR, ceiling_dbtp=-3.0, lookahead_ms=3.0, release_ms=30.0,
        )
        step2 = true_peak_limit(
            step1, SR, ceiling_dbtp=-1.0, lookahead_ms=5.0, release_ms=50.0,
        )
        step3 = lufs_normalize_constrained(
            step2, SR, target_lufs=-12.0, ceiling_dbtp=-1.0,
        )
        ceiling_linear = 10 ** (-1.0 / 20.0)
        step4 = soft_clip(step3, ceiling_linear, knee_db=2.0)

        output_lufs = meter.integrated_loudness(step4)

        # Output should not be absurdly quiet or loud
        assert output_lufs > -24.0, (
            f"Output LUFS {output_lufs:.1f} is unreasonably quiet"
        )
        assert output_lufs < -6.0, (
            f"Output LUFS {output_lufs:.1f} is unreasonably loud"
        )


# ---------------------------------------------------------------------------
# Sound quality enhancement wiring tests
# ---------------------------------------------------------------------------


class TestPipelineAutoLeveler:
    """Verify auto-leveler uses hardcoded parameters after flag cleanup."""

    @pytest.fixture(autouse=True)
    def _patch_auto_level(self):
        """Patch auto_level to capture its kwargs without running the full pipeline."""
        self.captured_kwargs = {}

        def _capture_auto_level(audio, sr, **kwargs):
            self.captured_kwargs = kwargs
            return audio  # pass through

        with patch("musicmixer.services.processor.auto_level", side_effect=_capture_auto_level):
            yield

    def test_hardcoded_params(self, pipeline_tmp):
        """Auto-leveler should use hardcoded tuned params: 4s/1.5/2.5."""
        self.captured_kwargs = {}
        _run_pipeline_with_mock_separation(pipeline_tmp)
        kw = self.captured_kwargs
        assert kw["max_boost_db"] == 1.5
        assert kw["max_cut_db"] == 2.5
        assert kw["window_sec"] == 4.0

    def test_detector_audio_is_always_set(self, pipeline_tmp):
        """detector_audio must always be the instrumental bus (not None)."""
        self.captured_kwargs = {}
        _run_pipeline_with_mock_separation(pipeline_tmp)
        assert "detector_audio" in self.captured_kwargs
        assert self.captured_kwargs["detector_audio"] is not None

    def test_active_floor_is_minus_50(self, pipeline_tmp):
        """active_floor_db must be -50.0 (lowered to prevent volume drops)."""
        self.captured_kwargs = {}
        _run_pipeline_with_mock_separation(pipeline_tmp)
        assert self.captured_kwargs["active_floor_db"] == -50.0
        assert self.captured_kwargs["target_percentile"] == 50.0


class TestPipelineOutputQuality:
    """Verify the pipeline (with all enhancements baked in) produces valid output.

    After the flag cleanup, EQ and static mastering are always on -- there are
    no flag combinations to test. These tests verify the single code path
    produces correct output (MP3 exists, LUFS reasonable, peaks within ceiling).
    """

    def test_output_lufs_within_range(self, pipeline_tmp):
        """Output LUFS should be within reasonable range."""
        import soundfile as sf_mod

        event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)

        output_path = Path(session.remix_path)
        assert output_path.exists()

        try:
            audio, sr = sf_mod.read(str(output_path), dtype="float32")
        except Exception:
            pytest.skip("soundfile cannot read MP3 (libsndfile version)")

        meter = pyln.Meter(sr)
        output_lufs = meter.integrated_loudness(audio)

        # LUFS should be within a reasonable range of the -12 target
        assert output_lufs > -24.0, f"Output LUFS {output_lufs:.1f} is unreasonably quiet"
        assert output_lufs < -6.0, f"Output LUFS {output_lufs:.1f} is unreasonably loud"

    def test_output_peak_within_ceiling(self, pipeline_tmp):
        """Output true peak should not grossly exceed -1.0 dBTP ceiling."""
        import soundfile as sf_mod

        event_queue, session = _run_pipeline_with_mock_separation(pipeline_tmp)

        output_path = Path(session.remix_path)
        assert output_path.exists()

        try:
            audio, sr = sf_mod.read(str(output_path), dtype="float32")
        except Exception:
            pytest.skip("soundfile cannot read MP3 (libsndfile version)")

        peak = true_peak(audio)
        # True peak should be under ceiling with some tolerance
        # (MP3 encoding can introduce small peak overshoots)
        ceiling_linear = 10 ** (-1.0 / 20.0)
        assert peak <= ceiling_linear + 0.05, (
            f"True peak {peak:.4f} grossly exceeds ceiling {ceiling_linear:.4f}"
        )


# ---------------------------------------------------------------------------
# Fix 9: Lossy-source processing path tests
# ---------------------------------------------------------------------------


class TestLossySourceWiring:
    """Verify that source_quality flags flow through the pipeline correctly.

    Tests that lossy YouTube source metadata produces the correct downstream
    kwargs (lossy_lpf_hz for mastering).
    """

    def test_lossy_source_passes_lossy_lpf_to_master_static(self, pipeline_tmp):
        """Either source being lossy should pass lossy_lpf_hz=16000 to master_static."""
        captured_kwargs = {}

        original_master = None

        def _capture_master(audio, sr, **kwargs):
            captured_kwargs.update(kwargs)
            return original_master(audio, sr, **kwargs)

        from musicmixer.services import mastering as mastering_mod
        original_master = mastering_mod.master_static

        with patch("musicmixer.services.mastering.master_static", side_effect=_capture_master):
            _run_pipeline_with_mock_separation(
                pipeline_tmp,
                source_quality_a="youtube-opus-128kbps",
            )

        assert captured_kwargs.get("lossy_lpf_hz") == 16000, (
            f"Expected lossy_lpf_hz=16000 passed to master_static, "
            f"got: {captured_kwargs}"
        )
