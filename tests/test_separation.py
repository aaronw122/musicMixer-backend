"""Tests for musicmixer.services.separation - dispatcher logic."""

import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import soundfile as sf
import pytest


def _make_float32_wav_bytes(freq: float = 440.0, duration: float = 0.5, sr: int = 44100) -> bytes:
    """Create float32 WAV bytes for a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class TestSeparateStemsDispatcher:
    """Test that separate_stems dispatches to correct backend."""

    @patch("musicmixer.services.separation._separate_modal")
    def test_dispatches_to_modal(self, mock_modal, tmp_path):
        """stem_backend='modal' should call _separate_modal."""
        from musicmixer.services.separation import separate_stems

        mock_modal.return_value = {"vocals": tmp_path / "vocals.wav"}

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "modal"
            result = separate_stems(tmp_path / "input.wav", tmp_path / "out")

        mock_modal.assert_called_once()
        assert "vocals" in result

    @patch("musicmixer.services.separation._separate_local")
    def test_dispatches_to_local(self, mock_local, tmp_path):
        """stem_backend='local' should call _separate_local."""
        from musicmixer.services.separation import separate_stems

        mock_local.return_value = {"vocals": tmp_path / "vocals.wav"}

        with patch("musicmixer.services.separation.settings") as mock_settings:
            mock_settings.stem_backend = "local"
            result = separate_stems(tmp_path / "input.wav", tmp_path / "out")

        mock_local.assert_called_once()
        assert "vocals" in result


class TestSeparateModalValidation:
    """Test _separate_modal stem validation."""

    def test_raises_on_missing_stems(self, tmp_path):
        """Should raise RuntimeError if Modal returns fewer than 6 stems."""
        from musicmixer.services.separation import _separate_modal

        # Mock separate_stems_remote to return only 4 stems
        incomplete_stems = {
            "vocals": _make_float32_wav_bytes(440),
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "other": _make_float32_wav_bytes(330),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = incomplete_stems

        # Create a real input file
        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        with patch("musicmixer.services.separation_modal.separate_stems_remote", mock_remote):
            with pytest.raises(RuntimeError, match="Expected 6 stems"):
                _separate_modal(input_path, tmp_path / "out")

    def test_accepts_complete_6_stems(self, tmp_path):
        """Should accept and save all 6 stems without error."""
        from musicmixer.services.separation import _separate_modal

        complete_stems = {
            "vocals": _make_float32_wav_bytes(440),
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "guitar": _make_float32_wav_bytes(330),
            "piano": _make_float32_wav_bytes(550),
            "other": _make_float32_wav_bytes(660),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        output_dir = tmp_path / "out"

        with patch("musicmixer.services.separation_modal.separate_stems_remote", mock_remote):
            result = _separate_modal(input_path, output_dir)

        assert set(result.keys()) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
        for stem_name, stem_path in result.items():
            assert stem_path.exists()
            assert stem_path.stat().st_size > 0

    def test_float32_validation_warns_on_non_float(self, tmp_path, caplog):
        """Should log a warning if stems are not float32."""
        from musicmixer.services.separation import _separate_modal
        import logging

        # Create a PCM16 WAV (not float32)
        sr = 44100
        t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
        pcm16_bytes = buf.getvalue()

        complete_stems = {
            "vocals": pcm16_bytes,  # PCM_16, not FLOAT
            "drums": _make_float32_wav_bytes(220),
            "bass": _make_float32_wav_bytes(110),
            "guitar": _make_float32_wav_bytes(330),
            "piano": _make_float32_wav_bytes(550),
            "other": _make_float32_wav_bytes(660),
        }

        mock_remote = MagicMock()
        mock_remote.remote.return_value = complete_stems

        input_path = tmp_path / "input.wav"
        input_path.write_bytes(_make_float32_wav_bytes())

        output_dir = tmp_path / "out"

        with patch("musicmixer.services.separation_modal.separate_stems_remote", mock_remote):
            with caplog.at_level(logging.WARNING, logger="musicmixer.services.separation"):
                result = _separate_modal(input_path, output_dir)

        # Should still succeed but log a warning
        assert "vocals" in result
        assert any("PCM_16" in record.message for record in caplog.records)
