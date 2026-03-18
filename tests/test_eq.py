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


# ---------------------------------------------------------------------------
# Adaptive Correction Tests
# ---------------------------------------------------------------------------


class TestAdaptiveCorrections:
    """Test adaptive correction parameter on apply_corrective_eq."""

    def test_none_matches_current_behavior(self):
        """adaptive_corrections=None should produce identical output to default."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5, duration=3.0)
        result_default = apply_corrective_eq(audio.copy(), SR, "vocals")
        result_none = apply_corrective_eq(
            audio.copy(), SR, "vocals", adaptive_corrections=None
        )
        np.testing.assert_allclose(result_none, result_default, atol=1e-6)

    def test_empty_list_matches_current_behavior(self):
        """adaptive_corrections=[] should produce identical output to default."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5, duration=3.0)
        result_default = apply_corrective_eq(audio.copy(), SR, "vocals")
        result_empty = apply_corrective_eq(
            audio.copy(), SR, "vocals", adaptive_corrections=[]
        )
        np.testing.assert_allclose(result_empty, result_default, atol=1e-6)

    def test_adaptive_cut_produces_measurable_attenuation(self):
        """A -4 dB adaptive cut at 400Hz should measurably attenuate a 400Hz sine."""
        audio = _make_stereo_sine(freq=400.0, amplitude=0.5, duration=3.0)
        corrections = [(400.0, -4.0, 2.0)]
        result = apply_corrective_eq(
            audio, SR, "other", apply_preset=False, adaptive_corrections=corrections
        )
        change_db = _rms_db_change(audio, result)
        assert change_db < -2.0, (
            f"400Hz with -4dB adaptive cut changed only {change_db:.1f} dB "
            f"(expected < -2.0 dB)"
        )

    def test_preset_and_adaptive_combined_when_both_enabled(self):
        """Adaptive corrections should stack on top of preset EQ when both explicitly enabled."""
        audio = _make_stereo_sine(freq=400.0, amplitude=0.5, duration=3.0)

        # Preset-only result
        result_preset = apply_corrective_eq(audio.copy(), SR, "other")

        # Preset + adaptive cut at 400Hz
        corrections = [(400.0, -3.0, 2.0)]
        result_both = apply_corrective_eq(
            audio.copy(), SR, "other", adaptive_corrections=corrections
        )

        # Combined should attenuate more than preset alone
        change_preset = _rms_db_change(audio, result_preset)
        change_both = _rms_db_change(audio, result_both)
        assert change_both < change_preset, (
            f"Preset+adaptive ({change_both:.1f} dB) should attenuate more "
            f"than preset alone ({change_preset:.1f} dB)"
        )

    def test_extreme_corrections_pass_through_unclamped(self):
        """eq.py applies whatever values it receives — clamping is upstream."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=3.0)
        # Extreme -12 dB cut — well beyond the -4 dB cap that spectral.py enforces
        corrections = [(1000.0, -12.0, 2.0)]
        result = apply_corrective_eq(
            audio, SR, "other", apply_preset=False, adaptive_corrections=corrections
        )
        change_db = _rms_db_change(audio, result)
        # Should apply the full -12 dB (approximately), proving no internal clamping
        assert change_db < -8.0, (
            f"Extreme -12dB correction only achieved {change_db:.1f} dB "
            f"(expected < -8.0 dB, proving no internal clamping)"
        )

    def test_multiple_adaptive_corrections(self):
        """Multiple corrections should all be applied."""
        # Mix two sine waves at different frequencies
        t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
        sine_400 = np.sin(2 * np.pi * 400 * t) * 0.3
        sine_2000 = np.sin(2 * np.pi * 2000 * t) * 0.3
        mono = (sine_400 + sine_2000).astype(np.float32)
        audio = np.column_stack([mono, mono])

        corrections = [
            (400.0, -4.0, 2.0),
            (2000.0, -4.0, 2.0),
        ]
        result = apply_corrective_eq(
            audio, SR, "other", apply_preset=False, adaptive_corrections=corrections
        )
        change_db = _rms_db_change(audio, result)
        # Both frequencies cut — expect significant overall attenuation
        assert change_db < -2.0, (
            f"Dual-frequency correction achieved only {change_db:.1f} dB "
            f"(expected < -2.0 dB)"
        )

    def test_adaptive_only_no_preset(self):
        """Adaptive corrections without preset should still work."""
        audio = _make_stereo_sine(freq=500.0, amplitude=0.5, duration=3.0)
        corrections = [(500.0, -3.0, 1.5)]
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_preset=False, adaptive_corrections=corrections
        )
        change_db = _rms_db_change(audio, result)
        assert change_db < -1.5, (
            f"Adaptive-only at 500Hz achieved only {change_db:.1f} dB "
            f"(expected < -1.5 dB)"
        )

    def test_adaptive_preserves_shape_and_dtype(self):
        """Output shape and dtype must match input when adaptive corrections applied."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        corrections = [(440.0, -2.0, 2.0)]
        result = apply_corrective_eq(
            audio, SR, "vocals", adaptive_corrections=corrections
        )
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_adaptive_skips_preset_by_default(self):
        """With apply_preset=False + adaptive, result differs from preset+adaptive."""
        audio = _make_stereo_sine(freq=400.0, amplitude=0.5, duration=3.0)
        corrections = [(400.0, -3.0, 2.0)]

        # Adaptive-only (preset skipped — the new default when adaptive is available)
        result_adaptive_only = apply_corrective_eq(
            audio.copy(), SR, "other", apply_preset=False, adaptive_corrections=corrections
        )

        # Preset + adaptive (legacy combined path)
        result_both = apply_corrective_eq(
            audio.copy(), SR, "other", apply_preset=True, adaptive_corrections=corrections
        )

        # They must differ — preset adds extra filters on top of adaptive
        assert not np.allclose(result_adaptive_only, result_both, atol=1e-6), (
            "Adaptive-only and preset+adaptive should produce different results "
            "(preset adds additional filters)"
        )
