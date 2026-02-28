"""Tests for musicmixer.services.eq — per-stem corrective EQ.

Uses synthetic audio (sine waves), following patterns in test_processor.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicmixer.services.eq import (
    RESONANCE_ELIGIBLE_STEMS,
    apply_corrective_eq,
    detect_resonances,
)

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
        result = apply_corrective_eq(audio, SR, stem_type, apply_resonance_cuts=False)
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))
        assert _rms(result) > 0

    @pytest.mark.parametrize("stem_type", ALL_STEM_TYPES)
    def test_float32_stereo_shape_preserved(self, stem_type: str):
        """Output must be float32 with same shape as stereo input."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5)
        result = apply_corrective_eq(audio, SR, stem_type, apply_resonance_cuts=False)
        assert result.dtype == np.float32
        assert result.ndim == 2
        assert result.shape[1] == 2
        assert result.shape[0] == audio.shape[0]

    def test_mono_input_preserved(self):
        """Mono input should produce mono output with same length."""
        audio = _make_mono_sine(freq=1000.0, amplitude=0.5)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_resonance_cuts=False
        )
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert result.shape[0] == audio.shape[0]

    def test_vocals_attenuates_250hz(self):
        """Vocal preset cuts mud at 250Hz — a 250Hz sine should be quieter after EQ."""
        audio = _make_stereo_sine(freq=250.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_resonance_cuts=False
        )
        change_db = _rms_db_change(audio, result)
        # The vocal preset has -2.5 dB cut at 250Hz. Expect noticeable attenuation.
        assert change_db < -1.0, (
            f"250Hz through vocals preset changed only {change_db:.1f} dB "
            f"(expected < -1.0 dB)"
        )

    def test_drums_attenuates_400hz(self):
        """Drums preset cuts box at 400Hz — a 400Hz sine should be quieter."""
        audio = _make_stereo_sine(freq=400.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(
            audio, SR, "drums", apply_resonance_cuts=False
        )
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
        result = apply_corrective_eq(
            audio, SR, "bass", apply_resonance_cuts=False
        )
        change_db = _rms_db_change(audio, result)
        assert change_db < -3.0, (
            f"15Hz through bass preset changed only {change_db:.1f} dB "
            f"(expected HPF attenuation below -3 dB)"
        )

    def test_passband_signal_minimally_affected(self):
        """A 1kHz signal through 'other' preset should be only mildly affected."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(
            audio, SR, "other", apply_resonance_cuts=False
        )
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

        result_unknown = apply_corrective_eq(
            audio, SR, "theremin", apply_resonance_cuts=False
        )
        result_other = apply_corrective_eq(
            audio, SR, "other", apply_resonance_cuts=False
        )

        np.testing.assert_allclose(result_unknown, result_other, atol=1e-6)

    def test_unknown_produces_valid_output(self):
        """Unknown stem type still produces valid float32 stereo output."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        result = apply_corrective_eq(
            audio, SR, "kazoo", apply_resonance_cuts=False
        )
        assert result.dtype == np.float32
        assert result.shape == audio.shape
        assert np.all(np.isfinite(result))


class TestHalveHfBoosts:
    """Test the halve_hf_boosts parameter for lossy YouTube sources."""

    def test_halve_hf_boosts_reduces_gain(self):
        """With halve_hf_boosts=True, HF boost should be roughly halved.

        Compare the gain at 3kHz (vocals presence boost) with and without
        the halving option. The halved version should boost less.
        """
        audio = _make_stereo_sine(freq=3000.0, amplitude=0.3, duration=3.0)

        result_normal = apply_corrective_eq(
            audio, SR, "vocals", apply_resonance_cuts=False, halve_hf_boosts=False
        )
        result_halved = apply_corrective_eq(
            audio, SR, "vocals", apply_resonance_cuts=False, halve_hf_boosts=True
        )

        rms_normal = _rms(result_normal)
        rms_halved = _rms(result_halved)

        # The halved boost should produce a quieter result at the boost frequency
        # (less gain applied). The difference is small (0.375 dB vs 0.75 dB boost)
        # but the halved result should not be louder than the normal one.
        assert rms_halved <= rms_normal + 1e-6, (
            f"Halved HF boost ({rms_halved:.6f}) should not exceed "
            f"normal boost ({rms_normal:.6f})"
        )


# ---------------------------------------------------------------------------
# Resonance Detection Tests
# ---------------------------------------------------------------------------


class TestResonanceDetection:
    """Test resonance detection via FFT analysis."""

    def _make_signal_with_resonance(
        self,
        resonance_freq: float = 400.0,
        resonance_amplitude: float = 0.8,
        noise_amplitude: float = 0.05,
        duration: float = 3.0,
    ) -> np.ndarray:
        """Create a signal with a dominant resonance peak plus low-level noise.

        The resonance should be well above the noise baseline.
        """
        t = np.linspace(0, duration, int(SR * duration), endpoint=False)
        # Broadband noise baseline
        rng = np.random.RandomState(42)
        noise = (rng.randn(len(t)) * noise_amplitude).astype(np.float32)
        # Strong resonance peak
        resonance = (
            np.sin(2 * np.pi * resonance_freq * t) * resonance_amplitude
        ).astype(np.float32)
        mono = noise + resonance
        return np.column_stack([mono, mono])

    def test_detects_known_resonance(self):
        """A strong 400Hz peak in noise should be detected as a resonance."""
        audio = self._make_signal_with_resonance(
            resonance_freq=400.0, resonance_amplitude=0.8, noise_amplitude=0.02
        )
        resonances = detect_resonances(audio, SR, threshold_db=10.0)
        assert len(resonances) > 0, "Should detect at least one resonance"
        # The detected frequency should be close to 400Hz
        detected_freq = resonances[0][0]
        assert abs(detected_freq - 400.0) < 50.0, (
            f"Detected frequency {detected_freq:.0f}Hz is too far from 400Hz"
        )

    def test_detects_resonance_in_high_range(self):
        """A strong 3kHz peak should be detected in the 2-4kHz range."""
        audio = self._make_signal_with_resonance(
            resonance_freq=3000.0, resonance_amplitude=0.8, noise_amplitude=0.02
        )
        resonances = detect_resonances(audio, SR, threshold_db=10.0)
        assert len(resonances) > 0, "Should detect 3kHz resonance"
        detected_freq = resonances[0][0]
        assert abs(detected_freq - 3000.0) < 100.0, (
            f"Detected frequency {detected_freq:.0f}Hz is too far from 3kHz"
        )

    def test_max_resonances_limit(self):
        """Should not return more than max_resonances peaks."""
        # Create signal with multiple resonance peaks
        t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
        rng = np.random.RandomState(42)
        noise = (rng.randn(len(t)) * 0.02).astype(np.float32)
        # Add resonances at 300, 400, 500Hz (all in the 200-600Hz range)
        signal = noise.copy()
        for freq in [300.0, 400.0, 500.0]:
            signal += (np.sin(2 * np.pi * freq * t) * 0.6).astype(np.float32)
        audio = np.column_stack([signal, signal])

        resonances = detect_resonances(audio, SR, threshold_db=5.0, max_resonances=2)
        assert len(resonances) <= 2, (
            f"Should return at most 2 resonances, got {len(resonances)}"
        )

    def test_no_resonances_in_clean_signal(self):
        """A simple sine wave at a single frequency should not trigger
        resonance detection at a high threshold, since there are no
        peaks dramatically above the overall spectrum baseline.
        """
        # Use a narrowband signal — just white noise with no strong peaks
        rng = np.random.RandomState(42)
        noise = (rng.randn(int(SR * 3.0)) * 0.1).astype(np.float32)
        audio = np.column_stack([noise, noise])
        resonances = detect_resonances(audio, SR, threshold_db=15.0)
        # Uniform noise should have no dramatic peaks above baseline
        assert len(resonances) == 0, (
            f"Uniform noise should not trigger resonance detection, "
            f"got {len(resonances)} resonance(s)"
        )

    def test_outside_freq_ranges_not_detected(self):
        """A resonance at 100Hz (below 200-600Hz range) should not be detected."""
        audio = self._make_signal_with_resonance(
            resonance_freq=100.0, resonance_amplitude=0.8, noise_amplitude=0.02
        )
        resonances = detect_resonances(
            audio, SR, threshold_db=10.0,
            freq_ranges=((200, 600), (2000, 4000)),
        )
        # 100Hz is outside both frequency ranges
        for freq, _ in resonances:
            assert freq >= 200.0, (
                f"Detected resonance at {freq:.0f}Hz — should not detect below 200Hz"
            )


class TestResonanceEligibility:
    """Resonance detection respects stem type eligibility."""

    def test_pitched_instruments_skip_resonance(self):
        """Bass, guitar, piano should skip resonance detection entirely."""
        for stem_type in ("bass", "guitar", "piano"):
            assert stem_type not in RESONANCE_ELIGIBLE_STEMS

    def test_eligible_stems(self):
        """Vocals, drums, other are eligible for resonance detection."""
        for stem_type in ("vocals", "drums", "other"):
            assert stem_type in RESONANCE_ELIGIBLE_STEMS

    def test_apply_eq_skips_resonance_for_guitar(self):
        """Calling apply_corrective_eq on guitar with apply_resonance_cuts=True
        should NOT apply resonance cuts (guitar is a pitched instrument).
        """
        # Create a signal with a strong resonance that would be detected
        t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
        rng = np.random.RandomState(42)
        noise = (rng.randn(len(t)) * 0.02).astype(np.float32)
        resonance = (np.sin(2 * np.pi * 400.0 * t) * 0.8).astype(np.float32)
        mono = noise + resonance
        audio = np.column_stack([mono, mono])

        # With preset=False and resonance_cuts=True on guitar,
        # the output should be unchanged (no resonance processing)
        result = apply_corrective_eq(
            audio, SR, "guitar", apply_preset=False, apply_resonance_cuts=True
        )
        np.testing.assert_allclose(result, audio, atol=1e-6)


class TestSilentAudioEdgeCases:
    """Fix 14: Silent audio (all zeros) through apply_corrective_eq."""

    def test_silent_stereo_returns_zeros(self):
        """All-zero stereo input should return all zeros without error."""
        audio = np.zeros((int(SR * DURATION), 2), dtype=np.float32)
        result = apply_corrective_eq(audio, SR, "vocals", apply_resonance_cuts=False)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0, atol=1e-10), (
            f"Silent input should produce silent output, max abs = {np.max(np.abs(result))}"
        )

    def test_silent_mono_returns_zeros(self):
        """All-zero mono input should return all zeros without error."""
        audio = np.zeros(int(SR * DURATION), dtype=np.float32)
        result = apply_corrective_eq(audio, SR, "drums", apply_resonance_cuts=False)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0, atol=1e-10)

    def test_silent_with_resonance_detection(self):
        """Silent audio through resonance detection should not crash."""
        audio = np.zeros((int(SR * DURATION), 2), dtype=np.float32)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_preset=False, apply_resonance_cuts=True
        )
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_silent_with_halve_hf_boosts(self):
        """Silent audio with halve_hf_boosts should not crash."""
        audio = np.zeros((int(SR * DURATION), 2), dtype=np.float32)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_resonance_cuts=False, halve_hf_boosts=True
        )
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert np.allclose(result, 0.0, atol=1e-10)


class TestApplyCorrectiveEQTwoPasses:
    """Test the two-pass usage pattern (preset before stretch, resonance after)."""

    def test_preset_only_pass(self):
        """First pass: apply_preset=True, apply_resonance_cuts=False."""
        audio = _make_stereo_sine(freq=250.0, amplitude=0.5, duration=3.0)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_preset=True, apply_resonance_cuts=False
        )
        # Should have EQ applied (250Hz cut for vocals)
        change_db = _rms_db_change(audio, result)
        assert change_db < -1.0, "Preset EQ should attenuate 250Hz for vocals"

    def test_resonance_only_pass(self):
        """Second pass: apply_preset=False, apply_resonance_cuts=True on eligible stem."""
        # Create signal with a strong resonance
        t = np.linspace(0, 3.0, int(SR * 3.0), endpoint=False)
        rng = np.random.RandomState(42)
        noise = (rng.randn(len(t)) * 0.02).astype(np.float32)
        resonance = (np.sin(2 * np.pi * 400.0 * t) * 0.8).astype(np.float32)
        mono = noise + resonance
        audio = np.column_stack([mono, mono])

        result = apply_corrective_eq(
            audio, SR, "vocals", apply_preset=False, apply_resonance_cuts=True
        )
        # The resonance at 400Hz should be somewhat attenuated
        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_no_processing_when_both_false(self):
        """When both apply_preset and apply_resonance_cuts are False, output = input."""
        audio = _make_stereo_sine(freq=440.0, amplitude=0.5)
        result = apply_corrective_eq(
            audio, SR, "vocals", apply_preset=False, apply_resonance_cuts=False
        )
        np.testing.assert_allclose(result, audio, atol=1e-6)
