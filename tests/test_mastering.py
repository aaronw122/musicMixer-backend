"""Tests for musicmixer.services.mastering — static mastering chain.

Uses synthetic audio (sine waves), following existing patterns in test_processor.py.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm
import pytest

from musicmixer.services.mastering import master_static
from musicmixer.services.processor import true_peak

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100
DURATION = 3.0  # 3s needed for reliable LUFS measurement (pyloudnorm needs >=400ms)


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMasterStaticShapeAndDtype:
    """Mastered output preserves float32 stereo shape."""

    def test_stereo_shape_preserved(self):
        audio = _make_stereo_sine(amplitude=0.5)
        result = master_static(audio, SR)
        assert result.shape == audio.shape
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_float32_dtype_preserved(self):
        audio = _make_stereo_sine(amplitude=0.5)
        result = master_static(audio, SR)
        assert result.dtype == np.float32


class TestMasterStaticLUFS:
    """Output LUFS is within tolerance of the target."""

    def test_output_lufs_within_tolerance(self):
        """Mastered output should be within 2 dB of -12 LUFS target."""
        audio = _make_stereo_sine(amplitude=0.5, duration=5.0)
        result = master_static(audio, SR, target_lufs=-12.0)

        meter = pyloudnorm.Meter(SR)
        result_lufs = meter.integrated_loudness(result)

        # Allow 2 dB tolerance — constrained normalization may not hit
        # the exact target if peak ceiling prevents full gain application
        assert abs(result_lufs - (-12.0)) < 2.0, (
            f"Output LUFS {result_lufs:.1f} not within 2 dB of target -12.0 LUFS"
        )

    def test_quiet_signal_boosted(self):
        """A quiet signal should be boosted toward the target LUFS."""
        audio = _make_stereo_sine(amplitude=0.05, duration=5.0)
        meter = pyloudnorm.Meter(SR)
        input_lufs = meter.integrated_loudness(audio)

        result = master_static(audio, SR, target_lufs=-12.0)
        result_lufs = meter.integrated_loudness(result)

        # Output should be louder than input
        assert result_lufs > input_lufs, (
            f"Output LUFS {result_lufs:.1f} should be louder than input {input_lufs:.1f}"
        )

    def test_loud_signal_reduced(self):
        """A loud signal should be reduced toward the target LUFS."""
        audio = _make_stereo_sine(amplitude=0.9, duration=5.0)
        meter = pyloudnorm.Meter(SR)
        input_lufs = meter.integrated_loudness(audio)

        result = master_static(audio, SR, target_lufs=-12.0)
        result_lufs = meter.integrated_loudness(result)

        # Output should be quieter than input (0.9 amplitude sine is ~-4 LUFS)
        assert result_lufs < input_lufs, (
            f"Output LUFS {result_lufs:.1f} should be quieter than input {input_lufs:.1f}"
        )


class TestMasterStaticTruePeak:
    """Output true-peak does not exceed the ceiling."""

    def test_true_peak_within_ceiling(self):
        """Output true-peak should not exceed -1.0 dBTP."""
        audio = _make_stereo_sine(amplitude=0.8, duration=5.0)
        ceiling_dbtp = -1.0
        ceiling_linear = 10 ** (ceiling_dbtp / 20.0)

        result = master_static(audio, SR, ceiling_dbtp=ceiling_dbtp)
        measured_peak = true_peak(result)

        # Allow 5% tolerance for block-based limiter + oversampled measurement
        assert measured_peak <= ceiling_linear * 1.05, (
            f"True peak {measured_peak:.4f} exceeds ceiling {ceiling_linear:.4f} "
            f"(+ 5% tolerance = {ceiling_linear * 1.05:.4f})"
        )

    def test_hot_signal_limited(self):
        """A signal peaking well above the ceiling should be brought under control."""
        audio = _make_stereo_sine(amplitude=1.5, duration=5.0)
        ceiling_dbtp = -1.0
        ceiling_linear = 10 ** (ceiling_dbtp / 20.0)

        result = master_static(audio, SR, ceiling_dbtp=ceiling_dbtp)
        measured_peak = true_peak(result)

        assert measured_peak <= ceiling_linear * 1.05, (
            f"Hot signal true peak {measured_peak:.4f} exceeds ceiling {ceiling_linear:.4f}"
        )

    def test_stricter_ceiling(self):
        """A -3 dBTP ceiling should produce lower peaks than -1 dBTP.

        Uses a signal with high crest factor so that the LUFS normalization
        is constrained by the peak ceiling. With headroom_db=3, the effective
        ceiling for normalization is ceiling_dbtp+3:
          -1 dBTP -> effective ceiling 10^(2/20) = 1.259
          -3 dBTP -> effective ceiling 10^(0/20) = 1.0

        A signal with peaks near 0.8 and low LUFS (high crest factor) forces
        the normalizer to hit the peak constraint differently at each ceiling.
        """
        n_samples = int(SR * 5.0)
        t = np.linspace(0, 5.0, n_samples, endpoint=False)
        # Very quiet bed with periodic loud clicks -- high crest factor
        # This creates a signal with peak ~0.8 but very low LUFS (~-30),
        # so LUFS normalization wants large gain. The peak constraint
        # limits the actual gain applied differently for each ceiling.
        mono = np.zeros(n_samples, dtype=np.float32)
        # Add sparse, loud clicks (just a few samples) to create high peaks
        click_len = 10  # Very short clicks
        for i in range(20):
            pos = int(i * 0.25 * SR)
            if pos + click_len <= n_samples:
                mono[pos:pos + click_len] = 0.8
        # Add very quiet noise floor so LUFS is measurable but low
        mono += (np.sin(2 * np.pi * 440 * t) * 0.001).astype(np.float32)
        audio = np.column_stack([mono, mono])

        result_1 = master_static(audio, SR, ceiling_dbtp=-1.0)
        result_3 = master_static(audio, SR, ceiling_dbtp=-3.0)

        peak_1 = np.max(np.abs(result_1))
        peak_3 = np.max(np.abs(result_3))

        # The -3 dBTP version should be constrained to a lower level
        assert peak_3 < peak_1, (
            f"-3 dBTP peak ({peak_3:.4f}) should be less than -1 dBTP peak ({peak_1:.4f})"
        )


class TestMasterStaticLossyLPF:
    """Low-pass filter for lossy sources."""

    def test_without_lpf_passes_highs(self):
        """Without lossy_lpf_hz, high-frequency content is preserved."""
        # Use a 15kHz sine wave
        audio = _make_stereo_sine(freq=15000.0, amplitude=0.3, duration=5.0)
        result = master_static(audio, SR, lossy_lpf_hz=None)

        # High-freq content should still be present (non-negligible RMS)
        output_rms = np.sqrt(np.mean(result ** 2))
        assert output_rms > 0.01, (
            f"Without LPF, 15kHz content should be preserved (RMS={output_rms:.4f})"
        )

    def test_with_lpf_attenuates_highs(self):
        """With lossy_lpf_hz=12000, high-frequency energy is reduced relative to mid.

        Uses a composite signal (1kHz + 15kHz) so LUFS normalization targets
        the overall loudness, but the high-frequency component is selectively
        attenuated by the LPF. We measure the ratio of high-freq to low-freq
        energy in both cases.
        """
        t = np.linspace(0, 5.0, int(SR * 5.0), endpoint=False)
        # Composite: 1kHz (dominant) + 15kHz (secondary)
        mono = (np.sin(2 * np.pi * 1000 * t) * 0.4
                + np.sin(2 * np.pi * 15000 * t) * 0.2).astype(np.float32)
        audio = np.column_stack([mono, mono])

        result_no_lpf = master_static(audio, SR, lossy_lpf_hz=None)
        result_with_lpf = master_static(audio, SR, lossy_lpf_hz=12000.0)

        # Measure high-frequency energy via bandpass around 15kHz
        from scipy.signal import butter, sosfiltfilt
        sos_hf = butter(4, [14000, 16000], btype='band', fs=SR, output='sos')

        hf_no_lpf = np.sqrt(np.mean(sosfiltfilt(sos_hf, result_no_lpf, axis=0) ** 2))
        hf_with_lpf = np.sqrt(np.mean(sosfiltfilt(sos_hf, result_with_lpf, axis=0) ** 2))

        # The LPF version should have noticeably less 15kHz energy
        if hf_no_lpf > 1e-6:
            attenuation_db = 20 * np.log10(hf_with_lpf / hf_no_lpf)
            assert attenuation_db < -3.0, (
                f"LPF at 12kHz only attenuated 15kHz by {attenuation_db:.1f} dB "
                f"(expected > 3 dB attenuation)"
            )

    def test_lpf_preserves_midrange(self):
        """With lossy_lpf_hz=16000, a 1kHz tone should pass through unaffected."""
        audio = _make_stereo_sine(freq=1000.0, amplitude=0.5, duration=5.0)
        result_no_lpf = master_static(audio, SR, lossy_lpf_hz=None)
        result_with_lpf = master_static(audio, SR, lossy_lpf_hz=16000.0)

        rms_no_lpf = np.sqrt(np.mean(result_no_lpf ** 2))
        rms_with_lpf = np.sqrt(np.mean(result_with_lpf ** 2))

        # 1kHz should be essentially the same with or without the 16kHz LPF
        if rms_no_lpf > 0.001:
            ratio_db = 20 * np.log10(rms_with_lpf / rms_no_lpf)
            assert abs(ratio_db) < 1.0, (
                f"LPF at 16kHz changed 1kHz content by {ratio_db:.1f} dB "
                f"(expected < 1 dB difference)"
            )


class TestMasterStaticCustomTargets:
    """Verify that custom target parameters are respected."""

    def test_custom_target_lufs(self):
        """A -16 LUFS target should produce quieter output than -12 LUFS."""
        audio = _make_stereo_sine(amplitude=0.5, duration=5.0)
        result_12 = master_static(audio, SR, target_lufs=-12.0)
        result_16 = master_static(audio, SR, target_lufs=-16.0)

        meter = pyloudnorm.Meter(SR)
        lufs_12 = meter.integrated_loudness(result_12)
        lufs_16 = meter.integrated_loudness(result_16)

        assert lufs_16 < lufs_12, (
            f"-16 LUFS target ({lufs_16:.1f}) should be quieter than "
            f"-12 LUFS target ({lufs_12:.1f})"
        )
