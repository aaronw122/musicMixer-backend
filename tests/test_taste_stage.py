"""Tests for the taste training pipeline stage.

Tests cover:
- TasteStageResult construction
- Pipeline flag gating (ab_taste_model_v1)
- Timeout wrapper fallback on slow execution
- Circuit breaker: disable after threshold, reset on success, re-enable after cooldown
- Fallback on import errors / exceptions
- TasteStageLog Pydantic model validation
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_logging import TasteStageLog
from musicmixer.services.taste_stage import (
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CIRCUIT_BREAKER_THRESHOLD,
    TasteStageResult,
    _get_circuit_breaker_state,
    _reset_circuit_breaker,
    _set_circuit_open_since,
    _record_fallback,
    run_taste_stage,
)


def _make_audio_metadata(**overrides) -> AudioMetadata:
    """Create a minimal AudioMetadata for testing."""
    defaults = dict(
        bpm=120.0,
        bpm_confidence=0.9,
        beat_frames=np.arange(0, 100, 10),
        duration_seconds=30.0,
        total_beats=64,
    )
    defaults.update(overrides)
    return AudioMetadata(**defaults)


def _make_remix_plan(**overrides) -> RemixPlan:
    """Create a minimal RemixPlan for testing."""
    defaults = dict(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=30.0,
        start_time_instrumental=0.0,
        end_time_instrumental=30.0,
        sections=[
            Section(
                label="main",
                start_beat=0,
                end_beat=64,
                stem_gains={"vocals": 1.0, "drums": 0.8},
                transition_in="cut",
                transition_beats=0,
            ),
        ],
        tempo_source="song_a",
        explanation="Test plan",
    )
    defaults.update(overrides)
    return RemixPlan(**defaults)


@pytest.fixture(autouse=True)
def _reset_cb():
    """Reset circuit breaker state before each test."""
    _reset_circuit_breaker()
    yield
    _reset_circuit_breaker()


class TestTasteStageResult:
    """Test that TasteStageResult can be constructed correctly."""

    def test_basic_construction(self):
        plan = _make_remix_plan()
        result = TasteStageResult(
            selected_plan=plan,
            candidates_generated=10,
            candidates_after_filter=8,
            selection_method="heuristic",
            generation_latency_ms=50.0,
            scoring_latency_ms=30.0,
            total_latency_ms=80.0,
            fallback_triggered=False,
        )
        assert result.candidates_generated == 10
        assert result.candidates_after_filter == 8
        assert result.selection_method == "heuristic"
        assert result.fallback_triggered is False
        assert result.fallback_reason is None

    def test_fallback_result(self):
        plan = _make_remix_plan()
        result = TasteStageResult(
            selected_plan=plan,
            candidates_generated=0,
            candidates_after_filter=0,
            selection_method="fallback",
            generation_latency_ms=0.0,
            scoring_latency_ms=0.0,
            total_latency_ms=5.0,
            fallback_triggered=True,
            fallback_reason="timeout",
        )
        assert result.fallback_triggered is True
        assert result.fallback_reason == "timeout"


class TestFlagGating:
    """When ab_taste_model_v1 is False, taste stage must not be called."""

    def test_taste_stage_not_called_when_flag_off(self, tmp_path):
        """Pipeline should not import or call taste_stage when flag is off."""
        import queue
        from pathlib import Path

        import soundfile as sf

        from musicmixer.models import SessionState

        # Create minimal test WAV files
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        click_interval = int(sr * 0.5)
        click = np.zeros_like(t)
        for i in range(0, len(t), click_interval):
            end = min(i + int(sr * 0.01), len(t))
            click[i:end] = 0.8
        signal = np.sin(2 * np.pi * 440 * t) * 0.3 + click * 0.3
        audio = np.column_stack([signal, signal]).astype(np.float32)

        song_a = tmp_path / "uploads" / "song_a.wav"
        song_b = tmp_path / "uploads" / "song_b.wav"
        song_a.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(song_a), audio, sr, subtype="FLOAT")
        sf.write(str(song_b), audio, sr, subtype="FLOAT")

        # Create mock stems
        stems_dir_a = tmp_path / "stems" / "test" / "song_a"
        stems_dir_b = tmp_path / "stems" / "test" / "song_b"
        stems_dir_a.mkdir(parents=True, exist_ok=True)
        stems_dir_b.mkdir(parents=True, exist_ok=True)
        stem_names = ["vocals", "drums", "bass", "other"]
        stems_a = {}
        stems_b = {}
        for name in stem_names:
            p = stems_dir_a / f"{name}.wav"
            sf.write(str(p), audio, sr, subtype="FLOAT")
            stems_a[name] = p
            p = stems_dir_b / f"{name}.wav"
            sf.write(str(p), audio, sr, subtype="FLOAT")
            stems_b[name] = p

        def mock_separate(audio_path, output_dir, progress_callback=None):
            if "song_a" in str(audio_path):
                return stems_a
            return stems_b

        event_queue = queue.Queue(maxsize=100)
        session = SessionState()

        with (
            patch("musicmixer.config.settings") as mock_settings,
            patch("musicmixer.services.separation.separate_stems", side_effect=mock_separate),
            patch("musicmixer.services.taste_stage.run_taste_stage") as mock_taste,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.stem_backend = "modal"
            mock_settings.lyrics_lookup_enabled = False
            mock_settings.ab_taste_model_v1 = False  # Flag OFF

            # Patch interpreter's module-level settings reference
            with patch("musicmixer.services.interpreter.settings", mock_settings):
                from musicmixer.services.pipeline import run_pipeline
                run_pipeline(
                    session_id="test",
                    song_a_path=str(song_a),
                    song_b_path=str(song_b),
                    prompt="test",
                    event_queue=event_queue,
                    session=session,
                )

                # taste_stage should never have been called
                mock_taste.assert_not_called()


class TestTimeoutFallback:
    """Timeout wrapper should fall back gracefully on slow execution."""

    def test_timeout_returns_fallback(self):
        """When the inner pipeline takes too long, fallback is returned."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()

        def slow_generate(*args, **kwargs):
            time.sleep(2.0)  # Way over the 400ms limit
            return []

        with patch(
            "musicmixer.services.taste_stage._run_taste_pipeline",
            side_effect=slow_generate,
        ):
            # Override the internal function to be slow
            pass

        # Instead, mock the imports inside _run_taste_pipeline to be slow
        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": MagicMock(
                generate_candidates=MagicMock(side_effect=lambda *a, **kw: time.sleep(2.0) or [])
            ),
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert result.fallback_reason == "timeout"
        assert result.selected_plan is fallback

    def test_error_returns_fallback(self):
        """When the inner pipeline raises, fallback is returned."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            side_effect=RuntimeError("boom"),
        )

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert result.selected_plan is fallback


class TestCircuitBreaker:
    """Circuit breaker disables taste stage after consecutive fallbacks."""

    def test_disables_after_threshold(self):
        """After CIRCUIT_BREAKER_THRESHOLD consecutive fallbacks, circuit opens."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            side_effect=RuntimeError("fail"),
        )

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
        }):
            # Trigger CIRCUIT_BREAKER_THRESHOLD fallbacks
            for i in range(CIRCUIT_BREAKER_THRESHOLD):
                result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
                assert result.fallback_triggered is True

            # Next call should be short-circuited by the breaker
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
            assert result.fallback_triggered is True
            assert result.fallback_reason == "circuit breaker open"

    def test_resets_on_success(self):
        """A successful run resets the consecutive fallback counter."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()
        selected = _make_remix_plan(explanation="Selected by model")

        mock_planner = MagicMock()
        mock_constraints = MagicMock()
        mock_model = MagicMock()

        # First: trigger some (but not threshold) fallbacks
        mock_planner.generate_candidates = MagicMock(
            side_effect=RuntimeError("fail"),
        )
        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
        }):
            for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
                run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        # Now: succeed (reset counter)
        mock_planner.generate_candidates = MagicMock(return_value=[selected])
        mock_constraints.validate_candidate = MagicMock(return_value=(True, None))
        mock_model.select_best = MagicMock(return_value=(selected, []))

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": mock_constraints,
            "musicmixer.services.taste_model": mock_model,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
            assert result.fallback_triggered is False
            assert result.selection_method == "heuristic"

        # After reset, more failures should not immediately open breaker
        mock_planner.generate_candidates = MagicMock(
            side_effect=RuntimeError("fail again"),
        )
        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
        }):
            # Should need CIRCUIT_BREAKER_THRESHOLD more failures to open
            for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
                result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
                assert result.fallback_triggered is True
                assert result.fallback_reason != "circuit breaker open"

    def test_reenables_after_cooldown(self):
        """Circuit breaker re-enables after cooldown period."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()
        selected = _make_remix_plan(explanation="After cooldown")

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            side_effect=RuntimeError("fail"),
        )

        # Open the circuit breaker
        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
        }):
            for _ in range(CIRCUIT_BREAKER_THRESHOLD):
                run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        # Verify breaker is open
        result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
        assert result.fallback_reason == "circuit breaker open"

        # Simulate cooldown elapsed by backdating the open timestamp
        _set_circuit_open_since(
            time.monotonic() - CIRCUIT_BREAKER_COOLDOWN_SECONDS - 1,
        )

        # Now set up a successful path
        mock_planner.generate_candidates = MagicMock(return_value=[selected])
        mock_constraints = MagicMock()
        mock_constraints.validate_candidate = MagicMock(return_value=(True, None))
        mock_model = MagicMock()
        mock_model.select_best = MagicMock(return_value=(selected, []))

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": mock_constraints,
            "musicmixer.services.taste_model": mock_model,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)
            assert result.fallback_triggered is False
            assert result.selection_method == "heuristic"

    def test_record_fallback_thread_safe_under_concurrency(self):
        """Concurrent fallback recording preserves an exact counter."""
        workers = 16
        increments_per_worker = 1000
        expected = workers * increments_per_worker

        original_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            def _worker():
                for _ in range(increments_per_worker):
                    _record_fallback()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_worker) for _ in range(workers)]
                for future in futures:
                    future.result()
        finally:
            sys.setswitchinterval(original_switch_interval)

        consecutive_fallbacks, circuit_open_since = _get_circuit_breaker_state()
        assert consecutive_fallbacks == expected
        assert circuit_open_since is not None


class TestSuccessfulRun:
    """Test successful taste stage execution with mocked dependencies."""

    def test_successful_run_returns_selected_plan(self):
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()
        candidate_1 = _make_remix_plan(explanation="Candidate 1")
        candidate_2 = _make_remix_plan(explanation="Candidate 2")

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            return_value=[candidate_1, candidate_2],
        )
        mock_constraints = MagicMock()
        mock_constraints.validate_candidate = MagicMock(return_value=(True, None))
        mock_model = MagicMock()
        mock_model.select_best = MagicMock(return_value=(candidate_1, []))

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": mock_constraints,
            "musicmixer.services.taste_model": mock_model,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is False
        assert result.selected_plan is candidate_1
        assert result.candidates_generated == 2
        assert result.candidates_after_filter == 2
        assert result.selection_method == "heuristic"
        assert result.total_latency_ms > 0

    def test_all_candidates_filtered_out_returns_fallback(self):
        """When all candidates fail validation, fallback is returned."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata(bpm=130.0)
        fallback = _make_remix_plan()
        candidate = _make_remix_plan(explanation="Bad candidate")

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(return_value=[candidate])
        mock_constraints = MagicMock()
        mock_constraints.validate_candidate = MagicMock(
            return_value=(False, "section_too_short"),
        )

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": mock_constraints,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert result.fallback_reason == "all candidates failed constraint validation"
        assert result.selected_plan is fallback

    def test_no_fallback_plan_raises(self):
        """Running without a fallback plan should raise ValueError."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata()
        with pytest.raises(ValueError, match="fallback_plan is required"):
            run_taste_stage(meta_a, meta_b, "test", fallback_plan=None)


class TestImportFallbacks:
    """Test fallback behavior when dependency modules are not importable."""

    def test_candidate_planner_import_error_fallback(self):
        """If candidate_planner is not importable, returns fallback."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata()
        fallback = _make_remix_plan()

        # Remove the module from sys.modules if present, and prevent import
        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": None,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert "not available" in (result.fallback_reason or "")

    def test_taste_constraints_import_error_fallback(self):
        """If taste_constraints is not importable, returns fallback."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata()
        fallback = _make_remix_plan()

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            return_value=[_make_remix_plan()],
        )

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": None,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert "not available" in (result.fallback_reason or "")

    def test_taste_model_import_error_fallback(self):
        """If taste_model is not importable, returns fallback."""
        meta_a = _make_audio_metadata()
        meta_b = _make_audio_metadata()
        fallback = _make_remix_plan()

        mock_planner = MagicMock()
        mock_planner.generate_candidates = MagicMock(
            return_value=[_make_remix_plan()],
        )
        mock_constraints = MagicMock()
        mock_constraints.validate_candidate = MagicMock(return_value=(True, None))

        with patch.dict("sys.modules", {
            "musicmixer.services.candidate_planner": mock_planner,
            "musicmixer.services.taste_constraints": mock_constraints,
            "musicmixer.services.taste_model": None,
        }):
            result = run_taste_stage(meta_a, meta_b, "test", fallback_plan=fallback)

        assert result.fallback_triggered is True
        assert "not available" in (result.fallback_reason or "")


class TestTasteStageLog:
    """TasteStageLog Pydantic model validation."""

    def test_valid_construction(self):
        log = TasteStageLog(
            request_id="abc-123",
            prompt="Hendrix guitar with MF Doom vocals",
            feature_version="v1.0.0",
            model_version="catboost-v0.1",
            flag_config={
                "ab_taste_model_v1": True,
            },
            candidates_generated=10,
            candidates_after_filter=8,
            selected_candidate_index=2,
            selection_method="model",
            generation_latency_ms=95.0,
            scoring_latency_ms=45.0,
            total_latency_ms=150.0,
            fallback_triggered=False,
        )
        assert log.request_id == "abc-123"
        assert log.candidates_generated == 10
        assert log.selection_method == "model"
        assert log.fallback_triggered is False

    def test_defaults(self):
        log = TasteStageLog(request_id="xyz", prompt="test")
        assert log.feature_version is None
        assert log.model_version is None
        assert log.flag_config == {}
        assert log.candidates_generated == 0
        assert log.selection_method == "fallback"
        assert log.fallback_triggered is False
        assert log.fallback_reason is None

    def test_serialization_roundtrip(self):
        log = TasteStageLog(
            request_id="test-123",
            prompt="remix me",
            fallback_triggered=True,
            fallback_reason="timeout",
        )
        data = log.model_dump()
        restored = TasteStageLog(**data)
        assert restored.request_id == log.request_id
        assert restored.fallback_reason == "timeout"

    def test_json_roundtrip(self):
        log = TasteStageLog(
            request_id="test-456",
            prompt="test",
            flag_config={"ab_taste_model_v1": True},
        )
        json_str = log.model_dump_json()
        restored = TasteStageLog.model_validate_json(json_str)
        assert restored.flag_config == {"ab_taste_model_v1": True}
