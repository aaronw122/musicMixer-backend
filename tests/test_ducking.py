"""Tests for musicmixer.services.ducking (spectral ducking).

Uses synthetic audio data -- no real song files needed.
Validates the spectral_duck() function against the critical requirements:
  - Full-length instrumental preservation (no truncation)
  - Mid-band energy reduction when vocals are active
  - No ducking when vocals are silent
  - Mask zero-padding for instrumental tails beyond vocal length
  - Onset threshold floor prevents perpetual ducking
  - Input array not mutated (returns a new array)
"""

from __future__ import annotations

import numpy as np
import pytest

from musicmixer.services.ducking import spectral_duck

# ---------------------------------------------------------------------------
# Fixtures and helpers
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


def _make_vocal_with_active_region(
    duration: float = DURATION,
    sr: int = SR,
    active_start: float = 0.3,
    active_end: float = 0.7,
    active_freq: float = 1000.0,
    active_amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a stereo signal that is silent except in the active region.

    The active region contains a sine tone in the vocal frequency range
    (1kHz default, well within 300-3500 Hz detection band).
    """
    n_samples = int(sr * duration)
    audio = np.zeros((n_samples, 2), dtype=np.float32)
    start_sample = int(active_start * n_samples)
    end_sample = int(active_end * n_samples)
    t = np.arange(end_sample - start_sample) / sr
    tone = (np.sin(2 * np.pi * active_freq * t) * active_amplitude).astype(
        np.float32
    )
    audio[start_sample:end_sample, 0] = tone
    audio[start_sample:end_sample, 1] = tone
    return audio


def _mid_band_rms(
    audio: np.ndarray, sr: int = SR, lo: float = 300, hi: float = 3000
) -> float:
    """Compute RMS of the mid-band (300-3000 Hz) of a stereo signal."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, audio, axis=0)
    return float(np.sqrt(np.mean(filtered**2)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpectralDuckBasic:
    """Core functionality tests."""

    def test_reduces_mid_band_when_vocals_active(self):
        """Ducking should reduce instrumental mid-band energy during vocal activity.

        Uses a vocal with clear silence/active contrast so the noise-floor-relative
        threshold detects vocal activity correctly. A constant sine has no silence
        frames, so its 10th percentile equals its RMS and onset_threshold (4x) is
        never reached.
        """
        instrumental = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=3.0)

        # Vocal with clear silence/active contrast: silent first 30%, active last 70%
        vocal = _make_vocal_with_active_region(
            duration=3.0,
            active_start=0.3,
            active_end=1.0,
            active_freq=1000.0,
            active_amplitude=0.5,
        )

        result = spectral_duck(instrumental, vocal, SR)

        # Mid-band RMS of the active region should be reduced
        active_start = int(0.4 * SR * 3.0)  # check well inside active region
        active_end = int(0.9 * SR * 3.0)
        original_rms = _mid_band_rms(instrumental[active_start:active_end])
        ducked_rms = _mid_band_rms(result[active_start:active_end])

        assert ducked_rms < original_rms, (
            f"Mid-band RMS should decrease in active region: "
            f"original={original_rms:.4f}, ducked={ducked_rms:.4f}"
        )

    def test_no_ducking_when_vocals_silent(self):
        """Silent vocals should produce no ducking (output equals input)."""
        instrumental = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=2.0)
        vocal = np.zeros((int(SR * 2.0), 2), dtype=np.float32)

        result = spectral_duck(instrumental, vocal, SR)

        # With silent vocals, the onset threshold floor (1e-5) should prevent
        # any ducking. Result should be very close to original.
        np.testing.assert_allclose(result, instrumental, atol=1e-6)

    def test_returns_float32(self):
        """Output should be float32 (pipeline convention)."""
        instrumental = _make_stereo_sine(duration=1.0)
        vocal = _make_stereo_sine(duration=1.0)

        result = spectral_duck(instrumental, vocal, SR)
        assert result.dtype == np.float32


class TestSpectralDuckLengthPreservation:
    """Tests for the critical full-length preservation requirement."""

    def test_preserves_full_instrumental_length(self):
        """Output length must match input instrumental length, not vocal length."""
        # Instrumental is longer than vocal
        instrumental = _make_stereo_sine(duration=5.0, amplitude=0.5)
        vocal = _make_stereo_sine(duration=3.0, amplitude=0.5)

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == len(instrumental), (
            f"Output length {len(result)} != instrumental length {len(instrumental)}. "
            f"The instrumental was truncated to vocal length {len(vocal)}!"
        )

    def test_preserves_full_length_when_vocal_longer(self):
        """When vocal is longer than instrumental, output matches instrumental."""
        instrumental = _make_stereo_sine(duration=3.0, amplitude=0.5)
        vocal = _make_stereo_sine(duration=5.0, amplitude=0.5)

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == len(instrumental)

    def test_preserves_full_length_when_equal(self):
        """When both are equal length, output matches."""
        instrumental = _make_stereo_sine(duration=3.0, amplitude=0.5)
        vocal = _make_stereo_sine(duration=3.0, amplitude=0.5)

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == len(instrumental)

    def test_instrumental_tail_passes_through_unducked(self):
        """Content beyond vocal length should pass through with no ducking."""
        # 5s instrumental, 2s vocal
        inst_duration = 5.0
        vocal_duration = 2.0
        instrumental = _make_stereo_sine(
            freq=1000.0, amplitude=0.5, duration=inst_duration
        )
        vocal = _make_stereo_sine(
            freq=1000.0, amplitude=0.5, duration=vocal_duration
        )

        result = spectral_duck(instrumental, vocal, SR)

        # The tail region (after vocal ends) should be unchanged
        vocal_samples = int(vocal_duration * SR)
        # Allow a small margin after vocal end for envelope release
        margin_samples = int(0.3 * SR)  # 300ms margin for release smoothing
        tail_start = vocal_samples + margin_samples
        tail_original = instrumental[tail_start:]
        tail_ducked = result[tail_start:]

        np.testing.assert_allclose(
            tail_ducked,
            tail_original,
            atol=1e-5,
            err_msg="Instrumental tail beyond vocal should be unducked",
        )


class TestSpectralDuckNoMutation:
    """Tests that input arrays are not mutated."""

    def test_does_not_mutate_instrumental(self):
        """spectral_duck must not modify the input instrumental array."""
        instrumental = _make_stereo_sine(duration=2.0, amplitude=0.5)
        original_copy = instrumental.copy()
        vocal = _make_stereo_sine(duration=2.0, amplitude=0.5)

        _ = spectral_duck(instrumental, vocal, SR)

        np.testing.assert_array_equal(
            instrumental,
            original_copy,
            err_msg="Input instrumental was mutated in-place!",
        )

    def test_returns_new_array(self):
        """The returned array must not share memory with the input."""
        instrumental = _make_stereo_sine(duration=2.0, amplitude=0.5)
        vocal = _make_stereo_sine(duration=2.0, amplitude=0.5)

        result = spectral_duck(instrumental, vocal, SR)

        # Modifying result should not affect instrumental
        result[0, 0] = 999.0
        assert instrumental[0, 0] != 999.0


class TestSpectralDuckMaskZeroPadding:
    """Tests for mask zero-padding when instrumental is longer than vocal."""

    def test_mask_is_zero_padded_beyond_vocal(self):
        """When instrumental > vocal, the ducking mask must be zero beyond min_len.

        Zero mask = no ducking = instrumental passes through cleanly.
        """
        # Create instrumental with constant mid-range content
        instrumental = _make_stereo_sine(
            freq=1000.0, amplitude=0.5, duration=4.0
        )
        # Vocal only covers first 2 seconds with active content
        vocal = _make_vocal_with_active_region(
            duration=2.0,
            active_start=0.0,
            active_end=1.0,
            active_amplitude=0.5,
        )

        result = spectral_duck(instrumental, vocal, SR)

        # Region well beyond vocal length should be identical to original
        beyond_vocal = int(2.5 * SR)
        tail_original = instrumental[beyond_vocal:]
        tail_result = result[beyond_vocal:]
        np.testing.assert_allclose(
            tail_result, tail_original, atol=1e-5,
            err_msg="Tail beyond vocal must be unducked (mask should be zero-padded)",
        )


class TestSpectralDuckOnsetThreshold:
    """Tests for the onset threshold floor behavior."""

    def test_very_quiet_vocal_does_not_trigger_ducking(self):
        """Near-silent vocal (below 1e-5 threshold) should not trigger ducking."""
        instrumental = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=2.0)
        # Extremely quiet vocal -- below the absolute floor of 1e-5
        vocal = _make_stereo_sine(freq=1000.0, amplitude=1e-7, duration=2.0)

        result = spectral_duck(instrumental, vocal, SR)

        # Should be essentially unchanged
        np.testing.assert_allclose(
            result, instrumental, atol=1e-5,
            err_msg="Very quiet vocal should not trigger ducking",
        )


class TestSpectralDuckParameters:
    """Tests for parameter variations."""

    def test_deeper_cut_produces_more_reduction(self):
        """A deeper cut_db value should produce more mid-band reduction."""
        instrumental = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=3.0)

        # Vocal with silence/active contrast so threshold is crossed
        vocal = _make_vocal_with_active_region(
            duration=3.0,
            active_start=0.3,
            active_end=1.0,
            active_freq=1000.0,
            active_amplitude=0.5,
        )

        result_mild = spectral_duck(instrumental, vocal, SR, cut_db=-2.0)
        result_deep = spectral_duck(instrumental, vocal, SR, cut_db=-6.0)

        # Measure in the active region only
        active_start = int(0.4 * SR * 3.0)
        active_end = int(0.9 * SR * 3.0)
        rms_mild = _mid_band_rms(result_mild[active_start:active_end])
        rms_deep = _mid_band_rms(result_deep[active_start:active_end])

        assert rms_deep < rms_mild, (
            f"Deeper cut should produce lower RMS: "
            f"-2dB={rms_mild:.4f}, -6dB={rms_deep:.4f}"
        )

    def test_zero_cut_produces_no_change(self):
        """cut_db=0 means no gain reduction -- output should match input."""
        instrumental = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=2.0)
        vocal = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=2.0)

        result = spectral_duck(instrumental, vocal, SR, cut_db=0.0)

        np.testing.assert_allclose(
            result, instrumental, atol=1e-5,
            err_msg="Zero cut should produce no change",
        )

    def test_stereo_shape_preserved(self):
        """Output should always be stereo (N, 2) matching input shape."""
        instrumental = _make_stereo_sine(duration=2.0)
        vocal = _make_stereo_sine(duration=2.0)

        result = spectral_duck(instrumental, vocal, SR)

        assert result.shape == instrumental.shape
        assert result.ndim == 2
        assert result.shape[1] == 2


class TestSpectralDuckEdgeCases:
    """Edge case tests."""

    def test_empty_vocal(self):
        """Empty vocal array should return instrumental unchanged."""
        instrumental = _make_stereo_sine(duration=2.0, amplitude=0.5)
        vocal = np.zeros((0, 2), dtype=np.float32)

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == len(instrumental)
        np.testing.assert_array_equal(result, instrumental)

    def test_empty_instrumental(self):
        """Empty instrumental should return empty array."""
        instrumental = np.zeros((0, 2), dtype=np.float32)
        vocal = _make_stereo_sine(duration=2.0, amplitude=0.5)

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == 0

    def test_rejects_mono_instrumental(self):
        """Mono instrumental should raise ValueError."""
        instrumental = np.zeros(int(SR * 2.0), dtype=np.float32)
        vocal = _make_stereo_sine(duration=2.0)

        with pytest.raises(ValueError, match="stereo"):
            spectral_duck(instrumental, vocal, SR)

    def test_short_audio(self):
        """Very short audio (< 1 frame) should return without error."""
        # Less than 50ms (one frame at 44.1kHz = 2205 samples)
        n_samples = 1000
        instrumental = np.random.randn(n_samples, 2).astype(np.float32) * 0.1
        vocal = np.random.randn(n_samples, 2).astype(np.float32) * 0.1

        result = spectral_duck(instrumental, vocal, SR)

        assert len(result) == n_samples

    def test_detection_band_wider_than_ducking_band(self):
        """A tone at 3200 Hz (in detection band but outside ducking band)
        should trigger ducking on the 300-3000 Hz ducking band content
        but the 3200 Hz tone in the instrumental should not be reduced.

        Uses a vocal with silence/active contrast so threshold detection works.
        """
        duration = 3.0
        n_samples = int(SR * duration)
        t = np.arange(n_samples) / SR

        # Instrumental: mix of 1kHz (in ducking band) and 3200Hz (outside)
        mid_tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.3
        high_tone = np.sin(2 * np.pi * 3200 * t).astype(np.float32) * 0.3
        instrumental = np.column_stack(
            [mid_tone + high_tone, mid_tone + high_tone]
        )

        # Vocal: 3200 Hz tone with silence/active contrast
        # Silent first 30%, active last 70%
        vocal_raw = np.zeros(n_samples, dtype=np.float32)
        active_start = int(0.3 * n_samples)
        t_active = np.arange(n_samples - active_start) / SR
        vocal_raw[active_start:] = (
            np.sin(2 * np.pi * 3200 * t_active).astype(np.float32) * 0.5
        )
        vocal = np.column_stack([vocal_raw, vocal_raw])

        result = spectral_duck(instrumental, vocal, SR)

        # The 1000 Hz content should be reduced in the active region
        from scipy.signal import butter, sosfiltfilt

        check_start = int(0.5 * n_samples)
        check_end = int(0.9 * n_samples)

        sos_mid = butter(4, [800, 1200], btype="band", fs=SR, output="sos")
        mid_original = sosfiltfilt(sos_mid, instrumental[check_start:check_end, 0])
        mid_ducked = sosfiltfilt(sos_mid, result[check_start:check_end, 0])

        rms_original = float(np.sqrt(np.mean(mid_original**2)))
        rms_ducked = float(np.sqrt(np.mean(mid_ducked**2)))

        assert rms_ducked < rms_original * 0.95, (
            f"1kHz content (in ducking band) should be reduced: "
            f"original={rms_original:.4f}, ducked={rms_ducked:.4f}"
        )
