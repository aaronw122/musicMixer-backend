"""Tests for musicmixer.services.processor (Steps 3 + 4).

Uses synthetic audio data -- no real song files needed.
Tests requiring the rubberband CLI are skipped if rubberband is not installed.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from musicmixer.services.processor import (
    apply_fades,
    compute_tempo_plan,
    cross_song_level_match,
    export_mp3,
    lufs_normalize,
    rubberband_process,
    soft_clip,
    trim_audio,
    true_peak,
    true_peak_limit,
    validate_stem,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

RUBBERBAND_AVAILABLE = shutil.which("rubberband") is not None
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

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


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def stereo_wav(tmp_dir: Path) -> Path:
    """Write a stereo sine wave WAV and return its path."""
    audio = _make_stereo_sine()
    path = tmp_dir / "stereo.wav"
    sf.write(str(path), audio, SR, subtype="FLOAT")
    return path


@pytest.fixture
def mono_wav(tmp_dir: Path) -> Path:
    """Write a mono sine wave WAV and return its path."""
    audio = _make_mono_sine()
    path = tmp_dir / "mono.wav"
    sf.write(str(path), audio, SR, subtype="FLOAT")
    return path


# ---------------------------------------------------------------------------
# Step 3: Standardization + Tempo Matching
# ---------------------------------------------------------------------------


class TestValidateStem:
    def test_stereo(self, stereo_wav: Path):
        """Load a stereo WAV, verify shape and sr."""
        audio, sr = validate_stem(stereo_wav)
        assert sr == SR
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        expected_samples = int(SR * DURATION)
        assert audio.shape[0] == expected_samples
        assert audio.dtype == np.float32

    def test_mono_to_stereo(self, mono_wav: Path):
        """Load mono WAV, verify conversion to stereo."""
        audio, sr = validate_stem(mono_wav)
        assert sr == SR
        assert audio.ndim == 2
        assert audio.shape[1] == 2
        assert audio.dtype == np.float32
        # Both channels should be identical (duplicated mono)
        np.testing.assert_array_equal(audio[:, 0], audio[:, 1])


class TestTrimAudio:
    def test_trim(self):
        """Trim a 10s signal to 2-5s range, verify length."""
        sr = SR
        audio = _make_stereo_sine(duration=10.0, sr=sr)
        trimmed = trim_audio(audio, sr, start_sec=2.0, end_sec=5.0)
        expected_length = int(3.0 * sr)
        assert trimmed.shape[0] == expected_length
        assert trimmed.shape[1] == 2

    def test_trim_clamped(self):
        """Trim beyond audio length clamps to end."""
        sr = SR
        audio = _make_stereo_sine(duration=2.0, sr=sr)
        trimmed = trim_audio(audio, sr, start_sec=1.0, end_sec=10.0)
        expected_length = int(1.0 * sr)
        assert trimmed.shape[0] == expected_length


class TestRubberband:
    @pytest.mark.skipif(not RUBBERBAND_AVAILABLE, reason="rubberband CLI not installed")
    def test_skip_at_unity(self):
        """Same source/target BPM returns identical audio (no processing)."""
        audio = _make_stereo_sine()
        result = rubberband_process(audio, SR, source_bpm=120.0, target_bpm=120.0)
        # Should be the exact same object (no copy)
        assert result is audio

    @pytest.mark.skipif(not RUBBERBAND_AVAILABLE, reason="rubberband CLI not installed")
    def test_stretch(self):
        """Actual rubberband stretch changes audio length proportionally."""
        audio = _make_stereo_sine(duration=4.0)
        # 90 -> 120 BPM: time_ratio = 0.75, output should be ~75% of input length
        result = rubberband_process(audio, SR, source_bpm=90.0, target_bpm=120.0)
        expected_length = int(len(audio) * 0.75)
        # Allow 5% tolerance for rubberband's processing
        assert abs(len(result) - expected_length) < expected_length * 0.05
        assert result.ndim == 2 or result.ndim == 1


class TestComputeTempoPlan:
    def test_small_gap_dj_transparent(self):
        """118 vs 122 BPM (3.3% gap, <= 4%): matches instrumental exactly."""
        target, stretch_v, stretch_i, warnings = compute_tempo_plan(118.0, 122.0)
        assert target == 122.0  # Within DJ-transparent range
        assert stretch_v is True
        assert stretch_i is False  # inst_ratio = 122/122 = 1.0
        assert len(warnings) == 0

    def test_safe_mashup_range(self):
        """110 vs 120 BPM (8.3% gap, 4-10%): 65/35 weighted midpoint."""
        target, stretch_v, stretch_i, warnings = compute_tempo_plan(110.0, 120.0)
        # 120 * 0.65 + 110 * 0.35 = 78 + 38.5 = 116.5
        assert target == 120.0 * 0.65 + 110.0 * 0.35
        assert stretch_v is True
        assert stretch_i is True  # Instrumental also stretches
        assert len(warnings) == 0

    def test_extended_range(self):
        """100 vs 120 BPM (16.7% gap, 10-20%): 70/30 weighted midpoint."""
        target, stretch_v, stretch_i, warnings = compute_tempo_plan(100.0, 120.0)
        # 120 * 0.70 + 100 * 0.30 = 84 + 30 = 114
        assert target == 120.0 * 0.70 + 100.0 * 0.30
        assert stretch_v is True
        # vocal_stretch_pct = abs(1 - 100/114) = 12.3% > 10%: tiered limits
        # disable instrumental stretch (vocals-only in 10-25% speedup range)
        assert stretch_i is False
        assert len(warnings) == 0

    def test_large_gap_clamps_vocal_stretch(self):
        """85 vs 130 BPM (34.6% gap, > 20%): clamps so vocal stretch <= 12%."""
        target, stretch_v, stretch_i, warnings = compute_tempo_plan(85.0, 130.0)
        # Weighted: 130*0.70 + 85*0.30 = 116.5. Vocal stretch = |85-116.5|/85 = 37% > 12%
        # Clamp: vocal_bpm < target_bpm, so target = 85 * 1.12 = 95.2
        assert target == 85.0 * 1.12
        assert stretch_v is True
        # Vocal stretch at 95.2 BPM: ratio=85/95.2=0.893, pct=10.7% speedup
        # Tiered limits: 10-25% speedup -> vocals-only, no instrumental stretch
        assert stretch_i is False

    def test_very_large_gap_explicit_song_b(self):
        """70 vs 140 BPM with song_b: skips stretching entirely."""
        target, stretch_v, stretch_i, warnings = compute_tempo_plan(
            70.0, 140.0, tempo_source="song_b"
        )
        # 70->140 is 50% speedup, > 45%: skip all stretching
        assert stretch_v is False
        assert stretch_i is False
        assert any("too large" in w.lower() for w in warnings)

    def test_weighted_midpoint_is_default(self):
        """Default tempo_source is weighted_midpoint, not song_b."""
        target_wm, _, _, _ = compute_tempo_plan(100.0, 120.0)
        target_explicit, _, _, _ = compute_tempo_plan(
            100.0, 120.0, tempo_source="weighted_midpoint"
        )
        assert target_wm == target_explicit

    def test_song_b_source(self):
        """Explicit song_b source uses instrumental BPM as target."""
        target, _, _, _ = compute_tempo_plan(100.0, 120.0, tempo_source="song_b")
        assert target == 120.0

    def test_song_a_source(self):
        """Explicit song_a source uses vocal BPM as target."""
        target, _, _, _ = compute_tempo_plan(100.0, 120.0, tempo_source="song_a")
        assert target == 100.0

    def test_average_treated_as_weighted_midpoint(self):
        """Average tempo source uses same weighted midpoint logic."""
        target_avg, _, _, _ = compute_tempo_plan(110.0, 120.0, tempo_source="average")
        target_wm, _, _, _ = compute_tempo_plan(
            110.0, 120.0, tempo_source="weighted_midpoint"
        )
        assert target_avg == target_wm


# ---------------------------------------------------------------------------
# Step 4: LUFS Normalization + Peak Limiter + Fades + Export
# ---------------------------------------------------------------------------


class TestCrossSongLevelMatch:
    def test_silent_input(self):
        """Near-silent input returns unchanged audio."""
        sr = SR
        vocal = _make_stereo_sine(amplitude=0.0001)
        instrumental = _make_stereo_sine(amplitude=0.5)
        result = cross_song_level_match(vocal, instrumental, sr)
        # Should return vocal unchanged (near-silent = below LUFS_FLOOR)
        np.testing.assert_array_equal(result, vocal)

    def test_normal_matching(self):
        """Vocal audio is adjusted when both inputs are audible."""
        sr = SR
        vocal = _make_stereo_sine(amplitude=0.1)
        instrumental = _make_stereo_sine(amplitude=0.5)
        result = cross_song_level_match(vocal, instrumental, sr)
        # Result should be louder than input (boosted to match instrumental + 3dB)
        assert np.max(np.abs(result)) > np.max(np.abs(vocal))


class TestLufsNormalize:
    def test_normalize(self):
        """Loud signal normalized to -14 LUFS."""
        sr = SR
        # A loud signal (0.5 amplitude sine wave should be well above -14 LUFS)
        audio = _make_stereo_sine(amplitude=0.8, duration=3.0)
        result = lufs_normalize(audio, sr, target_lufs=-14.0)
        # After normalization, the signal should be quieter than input
        # (since a 0.8 amplitude sine is louder than -14 LUFS)
        import pyloudnorm

        meter = pyloudnorm.Meter(sr)
        result_lufs = meter.integrated_loudness(result)
        # Allow 0.5 dB tolerance
        assert abs(result_lufs - (-14.0)) < 0.5

    def test_silent_skip(self):
        """Near-silent input is returned unchanged."""
        sr = SR
        audio = _make_stereo_sine(amplitude=0.0001)
        result = lufs_normalize(audio, sr)
        np.testing.assert_array_equal(result, audio)


class TestSoftClip:
    def test_below_threshold_unchanged(self):
        """Signal below ceiling is bit-identical."""
        ceiling = 10 ** (-1.0 / 20.0)  # ~0.891
        # Signal well below the knee threshold
        audio = _make_stereo_sine(amplitude=0.3)
        result = soft_clip(audio, ceiling)
        np.testing.assert_array_equal(result, audio)

    def test_above_ceiling_limited(self):
        """Signal above ceiling is limited to ceiling."""
        ceiling = 10 ** (-1.0 / 20.0)  # ~0.891
        audio = _make_stereo_sine(amplitude=1.5)
        result = soft_clip(audio, ceiling)
        assert np.max(np.abs(result)) <= ceiling + 1e-7


def _make_impulse_signal(
    duration: float = 2.0,
    sr: int = SR,
    quiet_amp: float = 0.1,
    impulse_amp: float = 1.5,
    impulse_pos: float = 0.5,  # position as fraction of duration
) -> np.ndarray:
    """Create a stereo signal with a quiet baseline and a loud impulse."""
    n_samples = int(sr * duration)
    signal = np.ones((n_samples, 2), dtype=np.float32) * quiet_amp
    imp_sample = int(impulse_pos * n_samples)
    # Short impulse burst (10ms)
    burst_len = int(0.01 * sr)
    signal[imp_sample:imp_sample + burst_len] = impulse_amp
    return signal


class TestTruePeakLimit:
    def test_below_ceiling_passthrough(self):
        """Signal well below ceiling passes through unchanged (or nearly so)."""
        audio = _make_stereo_sine(amplitude=0.3)
        result = true_peak_limit(audio, SR, ceiling_dbtp=-1.0)
        # ceiling = ~0.891, signal peak = 0.3 -- well below
        np.testing.assert_allclose(result, audio, atol=1e-6)

    def test_peaks_reduced_to_ceiling(self):
        """Signal with peaks above ceiling is limited so true_peak(output) <= ceiling."""
        audio = _make_stereo_sine(amplitude=1.5)
        ceiling_dbtp = -1.0
        ceiling_linear = 10 ** (ceiling_dbtp / 20.0)
        result = true_peak_limit(audio, SR, ceiling_dbtp=ceiling_dbtp)
        measured_peak = true_peak(result)
        # Allow small overshoot from block-based processing + oversampled measurement
        assert measured_peak <= ceiling_linear * 1.05, (
            f"true peak {measured_peak:.4f} exceeds ceiling {ceiling_linear:.4f} by more than 5%"
        )

    def test_preserves_quiet_content(self):
        """Quiet sections away from the impulse should be largely unchanged."""
        signal = _make_impulse_signal(
            duration=2.0, quiet_amp=0.1, impulse_amp=1.5, impulse_pos=0.5
        )
        result = true_peak_limit(signal, SR, ceiling_dbtp=-1.0, release_ms=50.0)
        # Check a quiet region well before the impulse (first 20% of signal)
        n = signal.shape[0]
        quiet_end = int(0.2 * n)
        quiet_original = np.max(np.abs(signal[:quiet_end]))
        quiet_result = np.max(np.abs(result[:quiet_end]))
        # Should be within 1 dB
        if quiet_original > 1e-10:
            ratio_db = 20 * np.log10(quiet_result / quiet_original)
            assert abs(ratio_db) < 1.0, (
                f"Quiet content changed by {ratio_db:.2f} dB (should be < 1 dB)"
            )

    def test_stereo_handling(self):
        """Stereo input produces stereo output with same shape."""
        audio = _make_stereo_sine(amplitude=1.5)
        result = true_peak_limit(audio, SR, ceiling_dbtp=-1.0)
        assert result.shape == audio.shape
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_mono_handling(self):
        """Mono input produces mono output."""
        audio = _make_mono_sine(amplitude=1.5)
        result = true_peak_limit(audio, SR, ceiling_dbtp=-1.0)
        assert result.ndim == 1
        assert result.shape == audio.shape

    def test_different_ceilings(self):
        """Lower ceiling produces lower output than higher ceiling."""
        audio = _make_stereo_sine(amplitude=1.5)
        result_high = true_peak_limit(audio, SR, ceiling_dbtp=-1.0)
        result_low = true_peak_limit(audio, SR, ceiling_dbtp=-3.0)
        peak_high = np.max(np.abs(result_high))
        peak_low = np.max(np.abs(result_low))
        assert peak_low < peak_high, (
            f"Lower ceiling should produce lower output: "
            f"-3 dBTP peak={peak_low:.4f}, -1 dBTP peak={peak_high:.4f}"
        )

    def test_release_recovery(self):
        """After a loud impulse, gain should recover (release)."""
        release_ms = 50.0
        signal = _make_impulse_signal(
            duration=2.0, quiet_amp=0.1, impulse_amp=1.5, impulse_pos=0.2
        )
        result = true_peak_limit(
            signal, SR, ceiling_dbtp=-1.0, release_ms=release_ms
        )
        # Check a region well after the impulse: at 3x release time past impulse
        impulse_sample = int(0.2 * signal.shape[0])
        recovery_samples = int(release_ms * 3 * SR / 1000)
        check_start = impulse_sample + recovery_samples
        check_end = min(check_start + int(0.1 * SR), signal.shape[0])

        if check_end > check_start:
            original_quiet = np.max(np.abs(signal[check_start:check_end]))
            recovered_quiet = np.max(np.abs(result[check_start:check_end]))
            # After recovery, the quiet content should be at least 80% of original
            if original_quiet > 1e-10:
                ratio = recovered_quiet / original_quiet
                assert ratio > 0.8, (
                    f"Gain did not recover after release: ratio={ratio:.3f} (expected > 0.8)"
                )


class TestTruePeak:
    def test_stereo(self):
        """Measures peak across both channels correctly."""
        # Create stereo where right channel is louder
        t = np.linspace(0, 0.5, int(SR * 0.5), endpoint=False)
        left = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        right = (np.sin(2 * np.pi * 440 * t) * 0.7).astype(np.float32)
        stereo = np.column_stack([left, right])
        peak = true_peak(stereo)
        # Peak should be close to the right channel's amplitude
        assert peak > 0.65  # At least as loud as right channel
        assert peak < 1.0  # Shouldn't exceed 1.0

    def test_mono(self):
        """Mono true-peak measurement works."""
        audio = _make_mono_sine(amplitude=0.8)
        peak = true_peak(audio)
        assert 0.75 < peak < 0.85


class TestApplyFades:
    def test_fades(self):
        """Fade-in starts near 0 and fade-out ends near 0."""
        sr = SR
        audio = _make_stereo_sine(duration=5.0, amplitude=0.8)
        result = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0)

        # Fade-in: first samples should be near zero
        assert np.max(np.abs(result[:100])) < 0.01

        # Fade-out: last samples should be near zero
        assert np.max(np.abs(result[-100:])) < 0.01

        # Middle should be largely unaffected
        mid = len(result) // 2
        mid_range = slice(mid - 1000, mid + 1000)
        np.testing.assert_allclose(result[mid_range], audio[mid_range], atol=1e-5)

    def test_skip_fade_in(self):
        """Skip fade-in preserves start of audio."""
        sr = SR
        audio = _make_stereo_sine(duration=5.0, amplitude=0.8)
        result = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0, skip_fade_in=True)
        # Start should NOT be near zero (fade-in was skipped)
        assert np.max(np.abs(result[:100])) > 0.1
        # End should still be near zero (fade-out applied)
        assert np.max(np.abs(result[-100:])) < 0.01

    def test_skip_fade_out(self):
        """Skip fade-out preserves end of audio."""
        sr = SR
        audio = _make_stereo_sine(duration=5.0, amplitude=0.8)
        result = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0, skip_fade_out=True)
        # Start should be near zero (fade-in applied)
        assert np.max(np.abs(result[:100])) < 0.01
        # End should NOT be near zero (fade-out was skipped)
        assert np.max(np.abs(result[-100:])) > 0.1

    def test_mono(self):
        """Fades work on mono audio too."""
        sr = SR
        audio = _make_mono_sine(duration=5.0, amplitude=0.8)
        result = apply_fades(audio, sr, fade_in_sec=1.0, fade_out_sec=1.0)
        assert result.ndim == 1
        assert np.max(np.abs(result[:100])) < 0.01
        assert np.max(np.abs(result[-100:])) < 0.01


class TestExportMp3:
    def test_export_uses_s16_dither_filter_when_enabled(self, tmp_dir: Path, monkeypatch):
        """Export command includes the dither filter when enabled."""
        seen: dict[str, list[str]] = {}

        def _fake_run(cmd, capture_output, timeout):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"ID3")
            return SimpleNamespace(returncode=0, stderr=b"")

        monkeypatch.setattr("musicmixer.services.processor.subprocess.run", _fake_run)

        audio = _make_stereo_sine(duration=1.0)
        output_path = tmp_dir / "with_dither.mp3"
        export_mp3(audio, SR, output_path, use_s16_dither=True)

        assert "-af" in seen["cmd"]
        assert "aresample=osf=s16:dither_method=triangular" in seen["cmd"]

    def test_export_skips_s16_dither_filter_when_disabled(self, tmp_dir: Path, monkeypatch):
        """Export command omits the dither filter when disabled."""
        seen: dict[str, list[str]] = {}

        def _fake_run(cmd, capture_output, timeout):
            seen["cmd"] = cmd
            Path(cmd[-1]).write_bytes(b"ID3")
            return SimpleNamespace(returncode=0, stderr=b"")

        monkeypatch.setattr("musicmixer.services.processor.subprocess.run", _fake_run)

        audio = _make_stereo_sine(duration=1.0)
        output_path = tmp_dir / "without_dither.mp3"
        export_mp3(audio, SR, output_path, use_s16_dither=False)

        assert "-af" not in seen["cmd"]

    @pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg not installed")
    def test_export(self, tmp_dir: Path):
        """Verify MP3 file is created and non-empty."""
        audio = _make_stereo_sine(duration=2.0)
        output_path = tmp_dir / "output.mp3"
        result_path = export_mp3(audio, SR, output_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 0
        # MP3 files start with ID3 tag or sync bytes
        with open(result_path, "rb") as f:
            header = f.read(3)
            assert header in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
