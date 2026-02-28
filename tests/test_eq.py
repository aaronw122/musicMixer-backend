"""Tests for musicmixer.services.eq — per-stem corrective EQ.

Uses synthetic audio (sine waves), following patterns in test_processor.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicmixer.services.eq import apply_corrective_eq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100
DURATION = 2.0


def _make_stereo_sine(
    freq: float = 440.0,
    duration: float = DURATION,
    sr: int = SR,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a stereo sine wave as float32 (N, 2)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)
    return np.column_stack([mono, mono])


def _make_mono_sine(
    freq: float = 440.0,
    duration: float = DURATION,
    sr: int = SR,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave as float32 (N,)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    """Compute RMS of audio signal."""
    return float(np.sqrt(np.mean(audio**2)))


def _rms_db_change(original: np.ndarray, processed: np.ndarray) -> float:
    """Compute RMS change in dB between original and processed audio."""
    rms_in = _rms(original)
    rms_out = _rms(processed)
    if rms_in < 1e-12:
        return 0.0
    return 20 * np.log10(rms_out / rms_in)


# ---------------------------------------------------------------------------
# Preset EQ Tests
# ---------------------------------------------------------------------------

ALL_STEM_TYPES = ["vocals", "drums", "bass", "guitar", "piano", "other"]


class TestPresetEQ:
    """Test preset EQ profiles produce valid output for each stem type."""

    @pytest.mark.parametrize("stem_type", ALL_STEM_TYPES)
    def test_each_stem_type_produces_valid_output(self, stem_type: str):
        """Each stem type should produce non-zero, finite output."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, stem_type)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))
        assert _rms(result) > 0

    @pytest.mark.parametrize("stem_type", ALL_STEM_TYPES)
    def test_float32_stereo_shape_preserved(self, stem_type: str):
        """Output must be float32 with same shape as stereo input."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, stem_type)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.shape[0] == audio.shape[0]

    def test_mono_input_preserved(self):
        """Mono input should produce mono output with same length."""
        audio = _make_mono_sine(freq=1000.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, "vocals")
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert result.shape[0] == audio.shape[0]

    def test_vocals_attenuates_250hz(self):
        """Vocal preset cuts mud at 250Hz — a 250Hz sine should be quieter after EQ."""
        audio = _make_stereo_sine(freq=250.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(audio, SR, "vocals")
        change_db = _rms_db_change(audio, result)
        # The vocal preset has -1.5 dB cut at 250Hz. Expect noticeable attenuation.
        assert change_db < -1.0, (
            f"250Hz through vocals preset changed only {change_db:.1f} dB "
            f"(expected < -1.0 dB)"
        )

    def test_drums_attenuates_400hz(self):
        """Drums preset cuts box at 400Hz — a 400Hz sine should be quieter."""
        audio = _make_stereo_sine(freq=400.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(audio, SR, "drums")
        change_db = _rms_db_change(audio, result)
        assert change_db < -1.5, (
            f"400Hz through drums preset changed only {change_db:.1f} dB "
            f"(expected < -1.5 dB)"
        )

    def test_bass_attenuates_low_frequencies_via_hpf(self):
        """Bass preset has HPF at 30Hz — a 15Hz sine should be attenuated.

        pedalboard's HighpassFilter has a gentle slope, so we test a
        frequency well below the cutoff to ensure measurable attenuation.
        """
        audio = _make_stereo_sine(freq=15.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(audio, SR, "bass")
        change_db = _rms_db_change(audio, result)
        assert change_db < -3.0, (
            f"15Hz through bass preset changed only {change_db:.1f} dB "
            f"(expected HPF attenuation below -3 dB)"
        )

    def test_passband_signal_minimally_affected(self):
        """A 1kHz signal through 'other' preset should be only mildly affected."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(audio, SR, "other")
        change_db = _rms_db_change(audio, result)
        # "other" has cuts at 400Hz and boost at 2.5kHz; 1kHz is in passband
        assert abs(change_db) < 3.0, (
            f"1kHz through 'other' preset changed by {change_db:.1f} dB "
            f"(expected minimal change)"
        )


class TestUnknownStemType:
    """Unknown stem types should fall back to 'other' preset."""

    def test_unknown_falls_back_to_other(self):
        """An unknown stem type should produce the same output as 'other'."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)

        result_unknown = apply_corrective_eq(audio, SR, "theremin")
        result_other = apply_corrective_eq(audio, SR, "other")

        np.testing.assert_allclose(result_unknown, result_other, atol=1e-6)

    def test_unknown_produces_valid_output(self):
        """Unknown stem type still produces valid float32 stereo output."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, "kazoo")
        assert result.dtype == np.float32
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))


class TestSilentAudioEdgeCases:
    """Silent audio (all zeros) through apply_corrective_eq."""

    def test_silent_stereo_returns_zeros(self):
        """All-zero stereo input should return all zeros without error."""
        audio = np.zeros((int(SR * DURATION), 2), dtype=np.float32)
        result = apply_corrective_eq(audio, SR, "vocals")
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0, atol=1e-10), (
            f"Silent input should produce silent output, max abs = {np.max(np.abs(result))}"
        )

    def test_silent_mono_returns_zeros(self):
        """All-zero mono input should return all zeros without error."""
        audio = np.zeros(int(SR * DURATION), dtype=np.float32)
        result = apply_corrective_eq(audio, SR, "drums")
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0, atol=1e-10)

    def test_silent_with_no_preset(self):
        """Silent audio with apply_preset=False should return input unchanged."""
        audio = np.zeros((int(SR * DURATION), 2), dtype=np.float32)
        result = apply_corrective_eq(audio, SR, "vocals", apply_preset=False)
        assert result.shape == audio.shape
        assert result.dtype == np.float32

class TestApplyCorrectiveEQPreset:
    """Test the preset EQ application."""

    def test_preset_pass(self):
        """Preset EQ should attenuate known cut frequencies."""
        audio = _make_stereo_sine(freq=250.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(audio, SR, "vocals", apply_preset=True)
        # Should have EQ applied (250Hz cut for vocals)
        change_db = _rms_db_change(audio, result)
        assert change_db < -1.0, "Preset EQ should attenuate 250Hz for vocals"

    def test_no_processing_when_preset_false(self):
        """When apply_preset is False, output equals input."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, "vocals", apply_preset=False)
        np.testing.assert_allclose(result, audio, atol=1e-6)
