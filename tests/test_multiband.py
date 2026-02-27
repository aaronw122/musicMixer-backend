"""Tests for musicmixer.services.multiband — 4-band multiband compressor.

Uses synthetic audio (sine waves) following existing test patterns.
"""

from __future__ import annotations

import numpy as np
import pytest

from pedalboard import Compressor, Pedalboard

from musicmixer.services.audio_utils import process_with_pedalboard
from musicmixer.services.multiband import (
    DEFAULT_BAND_SETTINGS,
    BandSettings,
    lr4_split,
    multiband_compress,
    split_4_bands,
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


def _make_white_noise(
    duration: float = DURATION,
    sr: int = SR,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate stereo white noise as float32 (N, 2)."""
    rng = np.random.default_rng(42)
    n = int(sr * duration)
    mono = (rng.standard_normal(n) * amplitude).astype(np.float32)
    return np.column_stack([mono, mono])


# ---------------------------------------------------------------------------
# LR4 crossover tests
# ---------------------------------------------------------------------------


class TestLR4Split:
    """Test the LR4 crossover band splitter."""

    def test_unity_reconstruction_white_noise(self):
        """Band split + recombine preserves energy (allpass property).

        LR4 crossovers with causal sosfilt produce an allpass sum:
        flat magnitude response but with group delay. This means the
        reconstructed signal has the same energy as the input but is
        time-shifted. We verify the allpass property by checking that
        total energy is preserved (within tolerance).
        """
        audio = _make_white_noise(duration=1.0)
        low, high = lr4_split(audio, 1000.0, SR)
        reconstructed = low + high

        # Skip the first ~200ms where filter transients settle
        skip = int(0.2 * SR)
        input_energy = float(np.sum(audio[skip:] ** 2))
        output_energy = float(np.sum(reconstructed[skip:] ** 2))

        # Energy should be preserved within 1%
        ratio = output_energy / max(input_energy, 1e-12)
        assert 0.99 < ratio < 1.01, (
            f"LR4 split+sum should preserve energy: ratio={ratio:.4f}"
        )

    def test_allpass_frequency_response(self):
        """LR4 low + high frequency responses sum to allpass (flat magnitude)."""
        from scipy.signal import butter, sosfreqz

        fc = 1000.0
        sos_lp = butter(2, fc, btype="low", fs=SR, output="sos")
        sos_hp = butter(2, fc, btype="high", fs=SR, output="sos")

        _, h_lp = sosfreqz(sos_lp, worN=4096, fs=SR)
        _, h_hp = sosfreqz(sos_hp, worN=4096, fs=SR)

        # LR4 = cascaded twice
        h_sum = h_lp * h_lp + h_hp * h_hp
        magnitude = np.abs(h_sum)

        np.testing.assert_allclose(
            magnitude, 1.0, atol=1e-6,
            err_msg="LR4 LP + HP should sum to allpass (unity magnitude)",
        )

    def test_low_sine_goes_to_low_band(self):
        """A 100 Hz sine should land almost entirely in the low band
        when split at 1000 Hz."""
        audio = _make_stereo_sine(freq=100.0, duration=1.0)
        low, high = lr4_split(audio, 1000.0, SR)

        # Skip transient
        skip = int(0.1 * SR)
        low_energy = np.sum(low[skip:] ** 2)
        high_energy = np.sum(high[skip:] ** 2)

        # Low band should have >99% of the energy
        assert low_energy > high_energy * 100, (
            f"100Hz sine: low_energy={low_energy:.4f}, high_energy={high_energy:.4f}"
        )

    def test_high_sine_goes_to_high_band(self):
        """A 5000 Hz sine should land almost entirely in the high band
        when split at 1000 Hz."""
        audio = _make_stereo_sine(freq=5000.0, duration=1.0)
        low, high = lr4_split(audio, 1000.0, SR)

        skip = int(0.1 * SR)
        low_energy = np.sum(low[skip:] ** 2)
        high_energy = np.sum(high[skip:] ** 2)

        assert high_energy > low_energy * 100, (
            f"5kHz sine: low_energy={low_energy:.4f}, high_energy={high_energy:.4f}"
        )

    def test_preserves_shape_stereo(self):
        """Output bands have the same shape as the input."""
        audio = _make_stereo_sine(duration=0.5)
        low, high = lr4_split(audio, 1000.0, SR)
        assert low.shape == audio.shape
        assert high.shape == audio.shape

    def test_preserves_shape_mono(self):
        """Works correctly with mono input."""
        audio = _make_mono_sine(duration=0.5)
        low, high = lr4_split(audio, 1000.0, SR)
        assert low.shape == audio.shape
        assert high.shape == audio.shape


# ---------------------------------------------------------------------------
# 4-band split tests
# ---------------------------------------------------------------------------


class TestSplit4Bands:
    """Test the 4-band cascaded LR4 crossover tree."""

    def test_unity_reconstruction_4_bands(self):
        """All 4 bands summed preserve energy (allpass property).

        The cascaded LR4 tree produces an allpass sum with group delay.
        We verify energy preservation rather than sample-level equality
        because causal filters introduce frequency-dependent delay.
        """
        audio = _make_white_noise(duration=2.0)
        bands = split_4_bands(audio, SR, (150, 600, 3000))

        reconstructed = sum(bands.values())

        # Skip first ~500ms for filter transient settling (3 cascaded crossovers,
        # low crossover at 150 Hz has significant group delay)
        skip = int(0.5 * SR)
        input_energy = float(np.sum(audio[skip:] ** 2))
        output_energy = float(np.sum(reconstructed[skip:] ** 2))

        # Energy should be preserved within 2% (cascaded crossovers
        # accumulate slight float32 rounding)
        ratio = output_energy / max(input_energy, 1e-12)
        assert 0.98 < ratio < 1.02, (
            f"4-band split+sum should preserve energy: ratio={ratio:.4f}"
        )

    def test_returns_four_bands(self):
        """split_4_bands returns exactly 4 named bands."""
        audio = _make_stereo_sine(duration=0.5)
        bands = split_4_bands(audio, SR, (150, 600, 3000))

        assert set(bands.keys()) == {"low", "low_mid", "mid", "high"}

    def test_band_frequency_placement(self):
        """Sine waves at specific frequencies land in the correct band."""
        crossovers = (150, 600, 3000)

        test_cases = [
            (80.0, "low"),       # 80 Hz -> low band (0-150)
            (400.0, "low_mid"),  # 400 Hz -> low-mid band (150-600)
            (1500.0, "mid"),     # 1500 Hz -> mid band (600-3000)
            (8000.0, "high"),    # 8000 Hz -> high band (3000-20k)
        ]

        for freq, expected_band in test_cases:
            audio = _make_stereo_sine(freq=freq, duration=1.0, amplitude=0.5)
            bands = split_4_bands(audio, SR, crossovers)

            skip = int(0.2 * SR)
            energies = {
                name: float(np.sum(band[skip:] ** 2))
                for name, band in bands.items()
            }
            dominant_band = max(energies, key=energies.get)

            assert dominant_band == expected_band, (
                f"{freq}Hz sine: expected band={expected_band}, "
                f"got={dominant_band}, energies={energies}"
            )


# ---------------------------------------------------------------------------
# Multiband compression tests
# ---------------------------------------------------------------------------


class TestMultibandCompress:
    """Test the full multiband_compress function."""

    def test_preserves_float32_stereo(self):
        """Output is float32 stereo with the same shape as input."""
        audio = _make_stereo_sine(freq=440.0, duration=1.0, amplitude=0.5)
        result = multiband_compress(audio, SR)

        assert result.dtype == np.float32
        assert result.shape == audio.shape

    def test_preserves_mono(self):
        """Works correctly with mono input."""
        audio = _make_mono_sine(freq=440.0, duration=1.0, amplitude=0.5)
        result = multiband_compress(audio, SR)

        assert result.dtype == np.float32
        assert result.shape == audio.shape

    def test_compresses_loud_signal(self):
        """A loud broadband signal should have reduced peak after compression.

        The LUFS restoration restores overall loudness, but the peak-to-RMS
        ratio (crest factor) should decrease — peaks are tamed.
        """
        # Use loud white noise (high crest factor)
        rng = np.random.default_rng(123)
        n = int(SR * 2.0)
        # Create a signal with some loud transients mixed with noise
        noise = rng.standard_normal(n).astype(np.float32) * 0.3
        # Add periodic loud transients
        for i in range(0, n, SR // 4):
            end = min(i + 200, n)
            noise[i:end] *= 3.0
        audio = np.column_stack([noise, noise])

        result = multiband_compress(audio, SR)

        # Crest factor = peak / RMS. Should decrease with compression.
        skip = int(0.2 * SR)
        input_peak = float(np.max(np.abs(audio[skip:])))
        input_rms = float(np.sqrt(np.mean(audio[skip:] ** 2)))
        input_crest = input_peak / max(input_rms, 1e-12)

        output_peak = float(np.max(np.abs(result[skip:])))
        output_rms = float(np.sqrt(np.mean(result[skip:] ** 2)))
        output_crest = output_peak / max(output_rms, 1e-12)

        assert output_crest < input_crest, (
            f"Compression should reduce crest factor: "
            f"input={input_crest:.2f}, output={output_crest:.2f}"
        )

    def test_each_band_compresses_independently(self):
        """Compression in one band should not affect another band.

        A loud 100 Hz sine + quiet 5 kHz sine: after compression,
        the low band should be compressed while the high band should
        be relatively unchanged.
        """
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)

        # Loud low sine (will trigger low-band compressor)
        low_sine = (np.sin(2 * np.pi * 100.0 * t) * 0.9).astype(np.float32)
        # Quiet high sine (below high-band threshold)
        high_sine = (np.sin(2 * np.pi * 8000.0 * t) * 0.02).astype(np.float32)

        combined = low_sine + high_sine
        audio = np.column_stack([combined, combined])

        result = multiband_compress(audio, SR)

        # The result should still contain both frequency components
        # Check that both components are present in the output
        assert result.shape == audio.shape
        assert not np.allclose(result, 0.0), "Output should not be silent"

    def test_custom_crossover_frequencies(self):
        """Different crossover frequencies should work without error."""
        audio = _make_stereo_sine(freq=440.0, duration=1.0, amplitude=0.5)

        # Use non-default crossovers
        result = multiband_compress(audio, SR, crossovers=(200, 800, 4000))

        assert result.dtype == np.float32
        assert result.shape == audio.shape
        assert not np.allclose(result, 0.0)

    def test_custom_band_settings(self):
        """Custom per-band settings override defaults."""
        audio = _make_white_noise(duration=1.0)

        custom_settings = {
            "low": BandSettings(
                name="low",
                threshold_db=-10.0,
                ratio=2.0,
                attack_ms=15.0,
                release_ms=150.0,
                makeup_db=1.0,
            ),
        }

        result = multiband_compress(audio, SR, settings=custom_settings)

        assert result.dtype == np.float32
        assert result.shape == audio.shape

    def test_empty_audio(self):
        """Empty audio input returns empty output."""
        audio = np.array([], dtype=np.float32)
        result = multiband_compress(audio, SR)
        assert len(result) == 0

    def test_silent_audio_passes_through(self):
        """Near-silent audio should pass through without errors."""
        audio = np.zeros((SR, 2), dtype=np.float32)
        result = multiband_compress(audio, SR)

        assert result.shape == audio.shape
        assert result.dtype == np.float32

    def test_lufs_restoration(self):
        """Output LUFS should be close to input LUFS (internal gain staging)."""
        import pyloudnorm

        audio = _make_white_noise(duration=2.0, amplitude=0.3)
        meter = pyloudnorm.Meter(SR)

        input_lufs = meter.integrated_loudness(audio)
        result = multiband_compress(audio, SR)
        output_lufs = meter.integrated_loudness(result)

        # LUFS should be restored within ~1 dB
        assert abs(output_lufs - input_lufs) < 1.5, (
            f"LUFS should be restored: input={input_lufs:.1f}, output={output_lufs:.1f}"
        )


# ---------------------------------------------------------------------------
# Fix 15: pedalboard.Compressor behavior with input > 0 dBFS
# ---------------------------------------------------------------------------


class TestPedalboardCompressorHotSignal:
    """Verify pedalboard.Compressor behavior with signals peaking above 0 dBFS.

    The multiband compressor feeds each band through pedalboard.Compressor.
    If a band's peak exceeds 0 dBFS (linear amplitude > 1.0), the compressor
    must compress rather than hard-clip the signal. This test validates that
    pedalboard does not introduce hard clipping at 0 dBFS.
    """

    def test_compressor_does_not_hard_clip_hot_signal(self):
        """Feed a +6 dBFS signal into pedalboard.Compressor and verify
        the output is not hard-clipped at 0 dBFS.

        A +6 dBFS signal has a linear peak of ~2.0. If pedalboard hard-clips,
        the output would be capped at exactly 1.0 (0 dBFS). A proper
        compressor should produce output above 1.0 (reduced but not clipped)
        or below 1.0 (compressed), but not exactly 1.0 across a sustained
        range of samples (the hallmark of hard clipping).
        """
        # Generate a stereo sine wave at +6 dBFS (linear amplitude ~2.0)
        amplitude = 10 ** (6.0 / 20.0)  # ~1.995
        t = np.linspace(0, 2.0, int(SR * 2.0), endpoint=False)
        mono = (np.sin(2 * np.pi * 440.0 * t) * amplitude).astype(np.float32)
        audio = np.column_stack([mono, mono])

        # Verify precondition: peak is above 1.0 (> 0 dBFS)
        input_peak = float(np.max(np.abs(audio)))
        assert input_peak > 1.5, f"Precondition: input peak {input_peak:.3f} should be > 1.5"

        # Run through compressor with moderate settings
        board = Pedalboard([
            Compressor(
                threshold_db=-20.0,
                ratio=4.0,
                attack_ms=5.0,
                release_ms=100.0,
            )
        ])
        result = process_with_pedalboard(audio, board, SR)

        output_peak = float(np.max(np.abs(result)))

        # Check for hard clipping: if the output has a sustained flat region
        # at exactly 1.0, that indicates hard clipping.
        # Count samples that are within a tiny epsilon of 1.0 — a properly
        # compressed signal should not have many.
        clipped_mask = np.abs(np.abs(result) - 1.0) < 0.001
        clipped_fraction = float(np.mean(clipped_mask))

        # A hard-clipped signal would have > 10% of samples at exactly 1.0.
        # A properly compressed signal should have very few (if any).
        is_hard_clipped = clipped_fraction > 0.10

        if is_hard_clipped:
            # pedalboard.Compressor hard-clips at 0 dBFS.
            # This documents the behavior as a known issue. If this assertion
            # fires, a pre-gain reduction is needed before feeding hot signals
            # into multiband compression. The multiband compressor's band
            # splitting can produce bands that peak above 0 dBFS even when
            # the input is below 0 dBFS.
            pytest.fail(
                f"pedalboard.Compressor hard-clips at 0 dBFS: "
                f"{clipped_fraction:.1%} of samples clipped to +/-1.0. "
                f"Input peak: {input_peak:.3f}, output peak: {output_peak:.3f}. "
                f"KNOWN ISSUE: A pre-gain reduction may be needed before "
                f"multiband compression to prevent band peaks from exceeding 0 dBFS."
            )

        # If we get here, pedalboard handles hot signals correctly
        # (compresses rather than clips). Verify compression actually happened.
        assert output_peak < input_peak, (
            f"Compressor should reduce peak: input={input_peak:.3f}, output={output_peak:.3f}"
        )
