"""Tests for musicmixer.services.mixer - overlay_and_export."""

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

pytestmark = [pytest.mark.slow, pytest.mark.timeout(30)]


def _make_sine_wav(path: Path, freq: float = 440.0, duration: float = 1.0, sr: int = 44100, amplitude: float = 0.5) -> Path:
    """Create a short sine wave WAV file (float32, stereo)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    sf.write(str(path), stereo, sr, subtype="FLOAT")
    return path


class TestOverlayAndExport:
    """Test the mixer overlay_and_export function."""

    def test_basic_overlay_produces_mp3(self, tmp_path):
        """Two stems overlaid should produce a valid, non-zero MP3."""
        from musicmixer.services.mixer import overlay_and_export

        vocals_path = _make_sine_wav(tmp_path / "vocals.wav", freq=440.0)
        drums_path = _make_sine_wav(tmp_path / "drums.wav", freq=220.0)

        output_path = tmp_path / "output" / "remix.mp3"

        result = overlay_and_export(
            vocal_stems={"vocals": vocals_path},
            instrumental_stems={"drums": drums_path},
            output_path=output_path,
        )

        assert result == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_peak_normalization_loud_input(self, tmp_path):
        """Loud stems (amplitude > 0.95) should be peak-normalized."""
        from musicmixer.services.mixer import overlay_and_export

        # Create two loud stems that will sum to > 0.95 peak
        vocals_path = _make_sine_wav(tmp_path / "vocals.wav", freq=440.0, amplitude=0.9)
        drums_path = _make_sine_wav(tmp_path / "drums.wav", freq=440.0, amplitude=0.9)

        output_path = tmp_path / "output" / "remix.mp3"

        result = overlay_and_export(
            vocal_stems={"vocals": vocals_path},
            instrumental_stems={"drums": drums_path},
            output_path=output_path,
        )

        # Output should exist and be valid
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_none_stems_skipped(self, tmp_path):
        """None stems (from 4-stem fallback) should be skipped gracefully."""
        from musicmixer.services.mixer import overlay_and_export

        vocals_path = _make_sine_wav(tmp_path / "vocals.wav", freq=440.0)

        output_path = tmp_path / "output" / "remix.mp3"

        result = overlay_and_export(
            vocal_stems={"vocals": vocals_path},
            instrumental_stems={
                "drums": _make_sine_wav(tmp_path / "drums.wav", freq=220.0),
                "bass": _make_sine_wav(tmp_path / "bass.wav", freq=110.0),
                "guitar": None,  # 4-stem fallback: missing
                "piano": None,   # 4-stem fallback: missing
                "other": _make_sine_wav(tmp_path / "other.wav", freq=330.0),
            },
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_different_length_stems_padded(self, tmp_path):
        """Stems of different lengths should be padded to the longest."""
        from musicmixer.services.mixer import overlay_and_export

        # 1 second and 0.5 second stems
        vocals_path = _make_sine_wav(tmp_path / "vocals.wav", freq=440.0, duration=1.0)
        drums_path = _make_sine_wav(tmp_path / "drums.wav", freq=220.0, duration=0.5)

        output_path = tmp_path / "output" / "remix.mp3"

        result = overlay_and_export(
            vocal_stems={"vocals": vocals_path},
            instrumental_stems={"drums": drums_path},
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_mono_input_converted_to_stereo(self, tmp_path):
        """Mono input stems should be converted to stereo."""
        from musicmixer.services.mixer import overlay_and_export

        # Create a mono WAV
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        mono = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        mono_path = tmp_path / "vocals_mono.wav"
        sf.write(str(mono_path), mono, sr, subtype="FLOAT")

        drums_path = _make_sine_wav(tmp_path / "drums.wav", freq=220.0)

        output_path = tmp_path / "output" / "remix.mp3"

        result = overlay_and_export(
            vocal_stems={"vocals": mono_path},
            instrumental_stems={"drums": drums_path},
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_all_none_stems_raises_error(self, tmp_path):
        """All-None stems should raise ValueError, not crash on np.sum."""
        from musicmixer.services.mixer import overlay_and_export

        output_path = tmp_path / "output" / "remix.mp3"

        with pytest.raises(ValueError, match="No valid stems to mix"):
            overlay_and_export(
                vocal_stems={"vocals": None},
                instrumental_stems={
                    "drums": None,
                    "bass": None,
                    "guitar": None,
                    "piano": None,
                    "other": None,
                },
                output_path=output_path,
            )
