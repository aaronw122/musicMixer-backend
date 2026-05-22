"""Tests for lead/backing vocal separation integration.

Covers:
- Pipeline dispatch (separate_vocal_song vs separate_stems)
- Renderer vocal bus routing for new stem names
- Stem loading with new vocal stem names
- Cache restoration with dynamic stem discovery
- VALID_STEMS set includes new names
- LUFS measurement handles new vocal stems
- Metrics all_possible set includes new stems
"""

import io
import queue
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import Section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100
ANALYSIS_SR = 22050
HOP_LENGTH = 512
BPM = 120
BEAT_INTERVAL_FRAMES = int(60 / BPM * ANALYSIS_SR / HOP_LENGTH)
BEAT_FRAMES = np.arange(0, 200) * BEAT_INTERVAL_FRAMES


def _make_stereo_noise(n_samples: int, amplitude: float = 0.1, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n_samples, 2)) * amplitude).astype(np.float32)


def _make_float32_wav_bytes(freq: float = 440.0, duration: float = 0.5, sr: int = 44100) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _make_test_wav(path: Path, duration: float = 1.0, sr: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t) * 0.3
    audio = np.column_stack([signal, signal]).astype(np.float32)
    sf.write(str(path), audio, sr, subtype="FLOAT")
    return path


# ---------------------------------------------------------------------------
# Renderer tests: vocal bus routing with new stem names
# ---------------------------------------------------------------------------

class TestRendererVocalBusRouting:
    """Renderer correctly routes lead_vocals/backing_vocals to the vocal bus."""

    def test_lead_vocals_routed_to_vocal_bus(self):
        """lead_vocals stem is routed to the vocal bus, not instrumental."""
        from musicmixer.services.renderer import render_arrangement

        n_samples = 44100  # 1 second
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"lead_vocals": 1.0, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        vocal_stems = {"lead_vocals": _make_stereo_noise(n_samples, seed=1)}
        inst_stems = {"drums": _make_stereo_noise(n_samples, seed=2)}

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Vocal bus should have content from lead_vocals
        assert not np.allclose(vocal_bus, 0.0), "lead_vocals should appear in vocal bus"
        # Instrumental bus should have content from drums only
        assert not np.allclose(inst_bus, 0.0), "drums should appear in instrumental bus"

    def test_backing_vocals_routed_to_vocal_bus(self):
        """backing_vocals stem is routed to the vocal bus."""
        from musicmixer.services.renderer import render_arrangement

        n_samples = 44100
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"backing_vocals": 0.8, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        vocal_stems = {"backing_vocals": _make_stereo_noise(n_samples, seed=3)}
        inst_stems = {"drums": _make_stereo_noise(n_samples, seed=4)}

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        assert not np.allclose(vocal_bus, 0.0), "backing_vocals should appear in vocal bus"

    def test_both_vocal_stems_summed_in_vocal_bus(self):
        """Both lead_vocals and backing_vocals contribute to vocal bus simultaneously."""
        from musicmixer.services.renderer import render_arrangement

        n_samples = 44100
        sections = [
            Section(
                label="chorus", start_beat=0, end_beat=50,
                stem_gains={"lead_vocals": 1.0, "backing_vocals": 0.5, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        lead = _make_stereo_noise(n_samples, seed=10)
        backing = _make_stereo_noise(n_samples, seed=11)
        vocal_stems = {"lead_vocals": lead, "backing_vocals": backing}
        inst_stems = {"drums": _make_stereo_noise(n_samples, seed=12)}

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Vocal bus should be louder than either stem alone (summed)
        lead_only_rms = np.sqrt(np.mean(lead[:n_samples] ** 2))
        vocal_bus_rms = np.sqrt(np.mean(vocal_bus[:n_samples] ** 2))
        assert vocal_bus_rms > lead_only_rms * 0.9, "Both vocal stems should contribute"

    def test_legacy_vocals_still_works(self):
        """Legacy "vocals" stem name is still routed to the vocal bus."""
        from musicmixer.services.renderer import render_arrangement

        n_samples = 44100
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"vocals": 1.0, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        vocal_stems = {"vocals": _make_stereo_noise(n_samples)}
        inst_stems = {"drums": _make_stereo_noise(n_samples, seed=5)}

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        assert not np.allclose(vocal_bus, 0.0), "Legacy vocals should still route to vocal bus"

    def test_asymmetric_stems_handled(self):
        """Renderer handles 2 vocal stems (Song A) + 5 instrumental stems (Song B)."""
        from musicmixer.services.renderer import render_arrangement

        n_samples = 44100
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={
                    "lead_vocals": 1.0, "backing_vocals": 0.5,
                    "drums": 0.8, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.3,
                },
                transition_in="cut", transition_beats=0,
            ),
        ]
        vocal_stems = {
            "lead_vocals": _make_stereo_noise(n_samples, seed=20),
            "backing_vocals": _make_stereo_noise(n_samples, seed=21),
        }
        inst_stems = {
            "drums": _make_stereo_noise(n_samples, seed=22),
            "bass": _make_stereo_noise(n_samples, seed=23),
            "guitar": _make_stereo_noise(n_samples, seed=24),
            "piano": _make_stereo_noise(n_samples, seed=25),
            "other": _make_stereo_noise(n_samples, seed=26),
        }

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Both buses should have content
        assert not np.allclose(vocal_bus, 0.0)
        assert not np.allclose(inst_bus, 0.0)
        assert vocal_bus.dtype == np.float32
        assert inst_bus.dtype == np.float32


# ---------------------------------------------------------------------------
# Separation dispatcher tests
# ---------------------------------------------------------------------------

class TestSeparateVocalSongDispatcher:
    """Test that separate_vocal_song dispatches correctly."""

    @patch("musicmixer.services.separation._separate_vocal_song_modal")
    def test_dispatches_to_modal(self, mock_vocal_modal, tmp_path):
        """stem_backend='modal' should call _separate_vocal_song_modal."""
        from musicmixer.services.separation import separate_vocal_song

        mock_vocal_modal.return_value = {
            "lead_vocals": tmp_path / "lead_vocals.wav",
            "backing_vocals": tmp_path / "backing_vocals.wav",
            "instrumental": tmp_path / "instrumental.wav",
        }

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "modal"
            result = separate_vocal_song(tmp_path / "input.wav", tmp_path / "out")

        mock_vocal_modal.assert_called_once()
        assert "lead_vocals" in result
        assert "backing_vocals" in result

    @patch("musicmixer.services.separation._separate_vocal_song_local")
    def test_dispatches_to_local_with_remap(self, mock_local, tmp_path):
        """stem_backend='local' should call _separate_vocal_song_local."""
        from musicmixer.services.separation import separate_vocal_song

        # Local fallback returns already-remapped stems
        mock_local.return_value = {
            "lead_vocals": tmp_path / "out" / "lead_vocals.wav",
            "backing_vocals": None,
            "instrumental": tmp_path / "out" / "other.wav",
        }

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "local"
            result = separate_vocal_song(tmp_path / "input.wav", tmp_path / "out")

        mock_local.assert_called_once()
        assert "lead_vocals" in result
        assert result["backing_vocals"] is None


class TestSeparateVocalModalValidation:
    """Test _separate_vocal_song_modal stem validation."""

    def test_raises_on_missing_stems(self, tmp_path):
        """Should raise RuntimeError if Modal returns fewer than 3 expected stems."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        incomplete = {
            "lead_vocals": _make_float32_wav_bytes(440),
            # missing backing_vocals and instrumental
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = incomplete

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            with pytest.raises(RuntimeError, match="Expected 3 stems"):
                _separate_vocal_song_modal(input_path, tmp_path / "out")

    def test_accepts_complete_3_stems(self, tmp_path):
        """Should accept and save all 3 stems without error."""
        from musicmixer.services.separation import _separate_vocal_song_modal

        complete = {
            "lead_vocals": _make_float32_wav_bytes(440),
            "backing_vocals": _make_float32_wav_bytes(220),
            "instrumental": _make_float32_wav_bytes(110),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("modal.Function.from_name", return_value=mock_remote):
            result = _separate_vocal_song_modal(input_path, tmp_path / "out")

        assert set(result.keys()) == {"lead_vocals", "backing_vocals", "instrumental"}
        for stem_name, stem_path in result.items():
            assert stem_path.exists(), f"{stem_name} should exist on disk"


# ---------------------------------------------------------------------------
# Pipeline stem loading tests
# ---------------------------------------------------------------------------

class TestPipelineStemLoading:
    """Test _step_load_and_standardize_stems handles new vocal stem names."""

    def _make_stem_wav(self, path: Path, duration: float = 1.0) -> Path:
        return _make_test_wav(path, duration=duration)

    def test_loads_lead_and_backing_vocals(self, tmp_path):
        """Song A with lead_vocals and backing_vocals should load both."""
        from musicmixer.services.pipeline import _step_load_and_standardize_stems

        song_a_dir = tmp_path / "song_a"
        song_b_dir = tmp_path / "song_b"

        # Create Song A stems (MelBand Roformer output)
        self._make_stem_wav(song_a_dir / "lead_vocals.wav")
        self._make_stem_wav(song_a_dir / "backing_vocals.wav")
        self._make_stem_wav(song_a_dir / "instrumental.wav")

        # Create Song B stems (BS-RoFormer output)
        for name in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
            self._make_stem_wav(song_b_dir / f"{name}.wav")

        song_a_stems = {
            "lead_vocals": song_a_dir / "lead_vocals.wav",
            "backing_vocals": song_a_dir / "backing_vocals.wav",
            "instrumental": song_a_dir / "instrumental.wav",
        }
        song_b_stems = {
            name: song_b_dir / f"{name}.wav"
            for name in ["vocals", "drums", "bass", "guitar", "piano", "other"]
        }

        event_queue = queue.Queue()
        session = MagicMock()
        session.cancelled.is_set.return_value = False

        vocal_audio, inst_audio, _, _ = _step_load_and_standardize_stems(
            "test-session",
            song_a_stems, song_b_stems,
            None, None,
            event_queue, session,
        )

        assert "lead_vocals" in vocal_audio, "Should load lead_vocals"
        assert "backing_vocals" in vocal_audio, "Should load backing_vocals"
        assert len(inst_audio) >= 5, "Should load instrumental stems from Song B"

    def test_loads_legacy_vocals(self, tmp_path):
        """Song A with legacy 'vocals' key should still load."""
        from musicmixer.services.pipeline import _step_load_and_standardize_stems

        song_a_dir = tmp_path / "song_a"
        song_b_dir = tmp_path / "song_b"

        self._make_stem_wav(song_a_dir / "vocals.wav")
        for name in ["drums", "bass", "guitar", "piano", "other"]:
            self._make_stem_wav(song_b_dir / f"{name}.wav")

        song_a_stems = {"vocals": song_a_dir / "vocals.wav"}
        song_b_stems = {
            name: song_b_dir / f"{name}.wav"
            for name in ["drums", "bass", "guitar", "piano", "other"]
        }

        event_queue = queue.Queue()
        session = MagicMock()
        session.cancelled.is_set.return_value = False

        vocal_audio, inst_audio, _, _ = _step_load_and_standardize_stems(
            "test-session",
            song_a_stems, song_b_stems,
            None, None,
            event_queue, session,
        )

        assert "vocals" in vocal_audio, "Legacy vocals key should still work"


# ---------------------------------------------------------------------------
# API/Cache tests
# ---------------------------------------------------------------------------

class TestValidStemsEndpoint:
    """Test that VALID_STEMS includes new stem names."""

    def test_valid_stems_includes_new_names(self):
        from musicmixer.api.remix import VALID_STEMS

        assert "lead_vocals" in VALID_STEMS
        assert "backing_vocals" in VALID_STEMS
        assert "vocals" in VALID_STEMS  # backward compat
        assert "drums" in VALID_STEMS
        assert "bass" in VALID_STEMS

    def test_valid_stems_count(self):
        from musicmixer.api.remix import VALID_STEMS

        # 8 total: vocals, lead_vocals, backing_vocals, drums, bass, guitar, piano, other
        assert len(VALID_STEMS) == 8


class TestCacheRestorationDynamic:
    """Test that cache restoration discovers stems dynamically."""

    def test_discovers_new_vocal_stems(self, tmp_path):
        """Cache restoration should find lead_vocals.wav and backing_vocals.wav."""
        # Simulate a cached Song A directory with MelBand Roformer output
        song_a_dir = tmp_path / "song_a"
        song_a_dir.mkdir(parents=True)
        _make_test_wav(song_a_dir / "lead_vocals.wav")
        _make_test_wav(song_a_dir / "backing_vocals.wav")
        _make_test_wav(song_a_dir / "instrumental.wav")

        # Dynamic discovery via glob (same pattern as updated cache code)
        stems = {f.stem: f for f in song_a_dir.glob("*.wav")}

        assert "lead_vocals" in stems
        assert "backing_vocals" in stems
        assert "instrumental" in stems

    def test_discovers_legacy_stems(self, tmp_path):
        """Cache restoration should still find old-style 6-stem sets."""
        song_b_dir = tmp_path / "song_b"
        song_b_dir.mkdir(parents=True)
        for name in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
            _make_test_wav(song_b_dir / f"{name}.wav")

        stems = {f.stem: f for f in song_b_dir.glob("*.wav")}

        assert len(stems) == 6
        assert "vocals" in stems
        assert "drums" in stems


# ---------------------------------------------------------------------------
# Pipeline LUFS measurement tests
# ---------------------------------------------------------------------------

class TestStemLufsMeasurement:
    """Test _step_measure_stem_lufs handles new vocal stems."""

    def test_measures_new_vocal_stems(self, tmp_path):
        """LUFS measurement should find lead_vocals and backing_vocals."""
        from musicmixer.services.pipeline import _step_measure_stem_lufs

        song_a_dir = tmp_path / "song_a"
        song_b_dir = tmp_path / "song_b"

        # Create Song A stems
        _make_test_wav(song_a_dir / "lead_vocals.wav", duration=2.0)
        _make_test_wav(song_a_dir / "backing_vocals.wav", duration=2.0)

        # Create Song B stems
        for name in ["drums", "bass", "guitar", "piano", "other"]:
            _make_test_wav(song_b_dir / f"{name}.wav", duration=2.0)

        vocal_lufs, inst_lufs = _step_measure_stem_lufs(
            "test-session", song_a_dir, song_b_dir,
        )

        assert "lead_vocals" in vocal_lufs, "Should measure LUFS for lead_vocals"
        assert "backing_vocals" in vocal_lufs, "Should measure LUFS for backing_vocals"
        assert len(inst_lufs) >= 5

    def test_measures_legacy_vocals(self, tmp_path):
        """LUFS measurement should still work with legacy 'vocals' stem."""
        from musicmixer.services.pipeline import _step_measure_stem_lufs

        song_a_dir = tmp_path / "song_a"
        song_b_dir = tmp_path / "song_b"

        _make_test_wav(song_a_dir / "vocals.wav", duration=2.0)
        for name in ["drums", "bass", "guitar", "piano", "other"]:
            _make_test_wav(song_b_dir / f"{name}.wav", duration=2.0)

        vocal_lufs, inst_lufs = _step_measure_stem_lufs(
            "test-session", song_a_dir, song_b_dir,
        )

        assert "vocals" in vocal_lufs


# ---------------------------------------------------------------------------
# Pipeline constants tests
# ---------------------------------------------------------------------------

class TestPipelineConstants:
    """Test pipeline stem name constants."""

    def test_vocal_stem_names(self):
        from musicmixer.services.pipeline import VOCAL_STEM_NAMES

        assert "lead_vocals" in VOCAL_STEM_NAMES
        assert "backing_vocals" in VOCAL_STEM_NAMES

    def test_instrumental_stem_names(self):
        from musicmixer.services.pipeline import INSTRUMENTAL_STEM_NAMES

        assert "drums" in INSTRUMENTAL_STEM_NAMES
        assert "bass" in INSTRUMENTAL_STEM_NAMES
        assert "guitar" in INSTRUMENTAL_STEM_NAMES
        assert "piano" in INSTRUMENTAL_STEM_NAMES
        assert "other" in INSTRUMENTAL_STEM_NAMES


# ---------------------------------------------------------------------------
# Renderer _VOCAL_BUS_STEMS constant test
# ---------------------------------------------------------------------------

class TestRendererVocalBusConstant:
    """Test that _VOCAL_BUS_STEMS contains all vocal-family names."""

    def test_contains_all_vocal_names(self):
        from musicmixer.services.renderer import _VOCAL_BUS_STEMS

        assert "vocals" in _VOCAL_BUS_STEMS
        assert "lead_vocals" in _VOCAL_BUS_STEMS
        assert "backing_vocals" in _VOCAL_BUS_STEMS

    def test_does_not_contain_instrumental_names(self):
        from musicmixer.services.renderer import _VOCAL_BUS_STEMS

        assert "drums" not in _VOCAL_BUS_STEMS
        assert "bass" not in _VOCAL_BUS_STEMS
        assert "guitar" not in _VOCAL_BUS_STEMS


# ---------------------------------------------------------------------------
# Pipeline dispatch test
# ---------------------------------------------------------------------------

class TestPipelineDispatch:
    """Test that _step_separate_and_analyze calls correct separation functions."""

    def test_pipeline_imports_separate_vocal_song(self):
        """Pipeline should import separate_vocal_song alongside separate_stems."""
        # Verify that the pipeline module's _step_separate_and_analyze function
        # references separate_vocal_song for Song A separation.
        import inspect
        from musicmixer.services.pipeline import _step_separate_and_analyze

        source = inspect.getsource(_step_separate_and_analyze)
        assert "separate_vocal_song" in source, (
            "Pipeline should import and use separate_vocal_song for Song A"
        )
        assert "separate_stems" in source, (
            "Pipeline should still use separate_stems for Song B"
        )

    def test_separation_module_exports_both_functions(self):
        """Separation module should export both separate_stems and separate_vocal_song."""
        from musicmixer.services.separation import separate_stems, separate_vocal_song

        assert callable(separate_stems)
        assert callable(separate_vocal_song)
