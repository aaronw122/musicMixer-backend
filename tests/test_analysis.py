"""Tests for BPM detection and cross-song reconciliation.

Step 2 of Day 2: analysis.py
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import AudioMetadata
from musicmixer.services import analysis as analysis_module
from musicmixer.services.analysis import (
    analyze_audio,
    analyze_stems,
    reconcile_bpm,
    _compute_bar_boundaries,
    _detect_beats_neural,
    _segments_to_sections,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_click_track(path: Path, bpm: float = 120.0, duration: float = 10.0, sr: int = 22050) -> Path:
    """Generate a synthetic click track at a known BPM and save as WAV."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beat_interval = 60.0 / bpm
    signal = np.zeros_like(t)
    for beat_time in np.arange(0, duration, beat_interval):
        idx = int(beat_time * sr)
        if idx < len(signal):
            click_len = min(int(0.01 * sr), len(signal) - idx)
            signal[idx : idx + click_len] = 0.8
    sf.write(str(path), signal, sr)
    return path


def _make_metadata(bpm: float = 120.0, duration: float = 30.0) -> AudioMetadata:
    """Create a minimal AudioMetadata for reconciliation tests."""
    total_beats = round(bpm * duration / 60 / 4) * 4
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.8,
        beat_frames=np.array([0, 100, 200], dtype=np.intp),
        duration_seconds=duration,
        total_beats=max(total_beats, 4),
    )


# ---------------------------------------------------------------------------
# analyze_audio tests
# ---------------------------------------------------------------------------

class TestAnalyzeAudio:
    def test_returns_metadata(self, tmp_path: Path) -> None:
        """Generate a synthetic click track, verify analyze_audio returns
        AudioMetadata with reasonable values."""
        wav = _make_click_track(tmp_path / "click_120.wav", bpm=120.0, duration=10.0)
        meta = analyze_audio(wav)

        assert isinstance(meta, AudioMetadata)
        assert meta.bpm > 0
        assert meta.duration_seconds > 0
        assert meta.total_beats >= 4
        assert isinstance(meta.beat_frames, np.ndarray)
        assert len(meta.beat_frames) > 0

    def test_bpm_confidence_range(self, tmp_path: Path) -> None:
        """Confidence should be between 0 and 1."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        meta = analyze_audio(wav)

        assert 0.0 <= meta.bpm_confidence <= 1.0


# ---------------------------------------------------------------------------
# reconcile_bpm tests
# ---------------------------------------------------------------------------

class TestReconcileBpm:
    def test_returns_metadata_unchanged(self) -> None:
        """reconcile_bpm should return copies with BPMs, beat_frames,
        and total_beats identical to the inputs (passthrough)."""
        a = _make_metadata(bpm=95.0)
        b = _make_metadata(bpm=140.0)
        new_a, new_b = reconcile_bpm(a, b)

        assert new_a.bpm == pytest.approx(95.0)
        assert new_b.bpm == pytest.approx(140.0)
        np.testing.assert_array_equal(new_a.beat_frames, a.beat_frames)
        np.testing.assert_array_equal(new_b.beat_frames, b.beat_frames)
        assert new_a.total_beats == a.total_beats
        assert new_b.total_beats == b.total_beats

    def test_does_not_mutate(self) -> None:
        """Original metadata objects must be unchanged after reconciliation."""
        a = _make_metadata(bpm=60.0)
        b = _make_metadata(bpm=120.0)
        original_a_bpm = a.bpm
        original_b_bpm = b.bpm
        original_a_frames = a.beat_frames.copy()

        reconcile_bpm(a, b)

        assert a.bpm == original_a_bpm
        assert b.bpm == original_b_bpm
        np.testing.assert_array_equal(a.beat_frames, original_a_frames)


# ---------------------------------------------------------------------------
# beat_this neural beat detection tests
# ---------------------------------------------------------------------------

class TestNeuralBeatDetection:
    def test_detect_beats_neural_returns_none_when_unavailable(self, tmp_path: Path) -> None:
        """_detect_beats_neural returns None when beat_this raises."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        with patch.object(analysis_module, "_get_file2beats", side_effect=ImportError):
            result = _detect_beats_neural(wav)
        assert result is None

    def test_detect_beats_neural_returns_tuple_on_success(self, tmp_path: Path) -> None:
        """_detect_beats_neural returns (beat_frames, beat_times, downbeat_times, bpm, confidence)."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        beat_times = np.arange(0, 10.0, 0.5)
        downbeat_times = np.arange(0, 10.0, 2.0)

        mock_f2b = lambda path: (beat_times, downbeat_times)
        with patch.object(analysis_module, "_get_file2beats", return_value=mock_f2b):
            result = _detect_beats_neural(wav)

        assert result is not None
        frames, bt, dbt, bpm, conf = result
        assert isinstance(frames, np.ndarray)
        assert len(frames) == len(beat_times)
        np.testing.assert_array_equal(bt, beat_times)
        np.testing.assert_array_equal(dbt, downbeat_times)
        assert bpm == pytest.approx(120.0, abs=1.0)
        assert 0.0 <= conf <= 1.0

    def test_detect_beats_neural_returns_none_on_too_few_beats(self, tmp_path: Path) -> None:
        """_detect_beats_neural returns None when fewer than 2 beats are found."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        mock_f2b = lambda path: (np.array([1.0]), np.array([1.0]))
        with patch.object(analysis_module, "_get_file2beats", return_value=mock_f2b):
            result = _detect_beats_neural(wav)
        assert result is None

    def test_analyze_audio_librosa_fallback(self, tmp_path: Path) -> None:
        """analyze_audio falls back to librosa when _HAS_BEAT_THIS is False."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        with patch.object(analysis_module, "_HAS_BEAT_THIS", False):
            meta = analyze_audio(wav)

        assert isinstance(meta, AudioMetadata)
        assert meta.bpm > 0
        assert meta.beat_times is None
        assert meta.downbeat_times is None

    def test_analyze_audio_populates_beat_times_with_neural(self, tmp_path: Path) -> None:
        """analyze_audio populates beat_times and downbeat_times when beat_this succeeds."""
        wav = _make_click_track(tmp_path / "click.wav", bpm=120.0, duration=10.0)
        beat_times = np.arange(0, 10.0, 0.5)
        downbeat_times = np.arange(0, 10.0, 2.0)
        hop_length = 512
        sr = 22050
        expected_frames = np.round(beat_times * sr / hop_length).astype(int)

        mock_f2b = lambda path: (beat_times, downbeat_times)
        with patch.object(analysis_module, "_HAS_BEAT_THIS", True), \
             patch.object(analysis_module, "_get_file2beats", return_value=mock_f2b):
            meta = analyze_audio(wav)

        assert meta.beat_times is not None
        assert meta.downbeat_times is not None
        np.testing.assert_array_equal(meta.beat_times, beat_times)
        np.testing.assert_array_equal(meta.downbeat_times, downbeat_times)
        np.testing.assert_array_equal(meta.beat_frames, expected_frames)
        assert meta.bpm == pytest.approx(120.0, abs=1.0)


# ---------------------------------------------------------------------------
# analyze_stems — energy/vocal-activity helpers + tests
# ---------------------------------------------------------------------------

# Synthetic-stem grid constants
_STEM_SR: int = 22050
_N_BARS: int = 8
_BEATS_PER_BAR: int = 4
_BAR_SECONDS: float = 1.0  # 1 second per bar keeps the math simple
_BAR_FRAMES: int = int(_STEM_SR * _BAR_SECONDS)
# Exactly _N_BARS*_BEATS_PER_BAR beats. `beat_frames[::4]` then yields _N_BARS
# bar starts (0, 4, 8, ...) and _compute_bar_boundaries appends audio_length,
# giving _N_BARS bars with no spurious partial trailing bar.
_BEAT_FRAMES: np.ndarray = np.arange(
    0, (_N_BARS * _BEATS_PER_BAR)
).astype(np.intp) * (_BAR_FRAMES // _BEATS_PER_BAR)
_TOTAL_FRAMES: int = _N_BARS * _BAR_FRAMES

# Neural downbeat timestamps (seconds) that map 1:1 onto the synthetic bar grid:
# one downbeat per bar at the bar start (bar = _BAR_SECONDS seconds long). Under
# the sample-unit grid contract these become sample boundaries 0, _BAR_FRAMES,
# 2*_BAR_FRAMES, ... This is the canonical (production) grid source.
_DOWNBEAT_TIMES: np.ndarray = np.arange(_N_BARS, dtype=np.float64) * _BAR_SECONDS


def _tone(amplitude: float, n: int, freq: float = 220.0) -> np.ndarray:
    """A constant-amplitude sine tone of length n samples."""
    t = np.arange(n) / _STEM_SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _bar_signal(per_bar_amp: list[float], freq: float = 220.0) -> np.ndarray:
    """Build a full-length signal whose amplitude steps per bar."""
    assert len(per_bar_amp) == _N_BARS
    chunks = [_tone(a, _BAR_FRAMES, freq) for a in per_bar_amp]
    return np.concatenate(chunks).astype(np.float32)


def _write_wav(path: Path, signal: np.ndarray) -> Path:
    sf.write(str(path), signal, _STEM_SR, subtype="FLOAT")
    return path


# Per-bar amplitude profile for the raw mix: low->high->low so energy is
# clearly non-degenerate and varies bar to bar.
_MIX_BAR_AMPS: list[float] = [0.05, 0.2, 0.4, 0.7, 0.7, 0.4, 0.2, 0.05]
# Vocal profile with a clear gap (silent) in the middle bars.
_VOCAL_BAR_AMPS: list[float] = [0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.6, 0.6]


def _make_raw_mix(path: Path) -> Path:
    return _write_wav(path, _bar_signal(_MIX_BAR_AMPS))


class TestAnalyzeStemsEnergy:
    """analyze_stems energy + vocal-activity is stem-shape agnostic and
    anchored to the raw mix (role-independent)."""

    def _vocal_shape_paths(self, tmp_path: Path) -> dict[str, Path]:
        """Vocal-separator output shape: lead_vocals + backing_vocals + instrumental."""
        lead = _write_wav(tmp_path / "lead_vocals.wav", _bar_signal(_VOCAL_BAR_AMPS, freq=330.0))
        # backing is quiet-but-present where lead is present
        backing_amps = [a * 0.3 for a in _VOCAL_BAR_AMPS]
        backing = _write_wav(tmp_path / "backing_vocals.wav", _bar_signal(backing_amps, freq=440.0))
        instrumental = _write_wav(tmp_path / "instrumental.wav", _bar_signal(_MIX_BAR_AMPS, freq=110.0))
        return {"lead_vocals": lead, "backing_vocals": backing, "instrumental": instrumental}

    def _six_stem_paths(self, tmp_path: Path) -> dict[str, Path]:
        """6-stem instrumental separator shape."""
        # Distribute the mix energy across drums/bass; put vocals on a stem.
        vocals = _write_wav(tmp_path / "vocals.wav", _bar_signal(_VOCAL_BAR_AMPS, freq=330.0))
        drums = _write_wav(tmp_path / "drums.wav", _bar_signal([a * 0.6 for a in _MIX_BAR_AMPS], freq=110.0))
        bass = _write_wav(tmp_path / "bass.wav", _bar_signal([a * 0.4 for a in _MIX_BAR_AMPS], freq=80.0))
        guitar = _write_wav(tmp_path / "guitar.wav", _bar_signal([0.0] * _N_BARS, freq=200.0))
        piano = _write_wav(tmp_path / "piano.wav", _bar_signal([0.0] * _N_BARS, freq=260.0))
        other = _write_wav(tmp_path / "other.wav", _bar_signal([0.0] * _N_BARS, freq=300.0))
        return {
            "vocals": vocals, "drums": drums, "bass": bass,
            "guitar": guitar, "piano": piano, "other": other,
        }

    def test_vocal_shape_energy_non_degenerate(self, tmp_path: Path) -> None:
        """Vocal-shape stems + raw-mix path → non-empty, non-zero energy that
        tracks the raw-mix amplitude profile."""
        stem_paths = self._vocal_shape_paths(tmp_path)
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        stem_analysis, structure = analyze_stems(
            stem_paths=stem_paths,
            beat_frames=_BEAT_FRAMES,
            bpm=120.0,
            audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        energy = stem_analysis.combined_energy
        assert energy.size == _N_BARS
        assert np.any(energy > 0.0)
        # Energy must track the raw mix: loud bars (3,4) > quiet edge bars (0,7).
        assert energy[3] > energy[0]
        assert energy[4] > energy[7]
        assert energy.argmax() in (3, 4)
        assert structure.total_bars == _N_BARS

    def test_role_independence_energy_identical(self, tmp_path: Path) -> None:
        """KEY TEST: same raw mix, different stem shapes → identical
        combined_energy and bucket thresholds."""
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        vocal_dir = tmp_path / "vocal"
        vocal_dir.mkdir()
        six_dir = tmp_path / "six"
        six_dir.mkdir()

        vocal_paths = self._vocal_shape_paths(vocal_dir)
        six_paths = self._six_stem_paths(six_dir)

        sa_vocal, _ = analyze_stems(
            stem_paths=vocal_paths, beat_frames=_BEAT_FRAMES, bpm=120.0, audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )
        sa_six, _ = analyze_stems(
            stem_paths=six_paths, beat_frames=_BEAT_FRAMES, bpm=120.0, audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        np.testing.assert_array_equal(sa_vocal.combined_energy, sa_six.combined_energy)
        assert sa_vocal.bucket_thresholds.noise_floor == sa_six.bucket_thresholds.noise_floor
        assert sa_vocal.bucket_thresholds.p10 == sa_six.bucket_thresholds.p10
        assert sa_vocal.bucket_thresholds.p50 == sa_six.bucket_thresholds.p50
        assert sa_vocal.bucket_thresholds.p85 == sa_six.bucket_thresholds.p85

    def test_vocal_activity_from_lead_backing(self, tmp_path: Path) -> None:
        """Vocal-shape input (no `vocals` stem) still produces vocal activity
        derived from lead+backing, with the gap detected."""
        stem_paths = self._vocal_shape_paths(tmp_path)
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        stem_analysis, structure = analyze_stems(
            stem_paths=stem_paths, beat_frames=_BEAT_FRAMES, bpm=120.0, audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        vocal_active = stem_analysis.vocal_active
        assert vocal_active.size == _N_BARS
        # Absence of a `vocals` stem must NOT force all-false.
        assert vocal_active.any()
        # Vocal bars (2,3,6,7) active; gap bars (0,1,4,5) inactive.
        assert bool(vocal_active[2]) and bool(vocal_active[3])
        assert not bool(vocal_active[0]) and not bool(vocal_active[1])
        # The middle silent run (bars 4-5) should appear as a detected vocal gap.
        # VocalGap.end_bar is inclusive, so iterate through end_bar + 1.
        gap_bars = {b for gap in stem_analysis.vocal_gaps for b in range(gap.start_bar, gap.end_bar + 1)}
        assert 4 in gap_bars and 5 in gap_bars

    def test_instrumental_path_energy_tracks_mix(self, tmp_path: Path) -> None:
        """6-stem input produces a valid, non-degenerate result anchored to the mix."""
        stem_paths = self._six_stem_paths(tmp_path)
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        stem_analysis, structure = analyze_stems(
            stem_paths=stem_paths, beat_frames=_BEAT_FRAMES, bpm=120.0, audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        energy = stem_analysis.combined_energy
        assert energy.size == _N_BARS
        assert np.any(energy > 0.0)
        assert energy.argmax() in (3, 4)
        # `vocals` stem present → vocal activity from it.
        assert stem_analysis.vocal_active.any()


# ---------------------------------------------------------------------------
# Bar-grid downbeat regression (sample-unit contract)
# ---------------------------------------------------------------------------

class TestBarGridDownbeatRegression:
    """Regression for the frame/sample unit-mix bug in _compute_bar_boundaries.

    Prior behavior: the grid took ``beat_frames[::4]`` (librosa FRAME units) and
    appended ``audio_length`` (SAMPLE units), producing 1-bar-wide slivers plus a
    single giant final bar spanning the rest of the song. The fix derives the
    grid from neural ``downbeat_times × ANALYSIS_SR`` (sample indices) and
    establishes a sample-unit output contract.
    """

    def test_neural_grid_is_sample_units_no_giant_final_bar(self) -> None:
        # 83 downbeats spaced ~2.56s (the prod Hypnotize case), audio ~210s.
        spacing = 2.56
        downbeat_times = np.arange(83, dtype=np.float64) * spacing + 0.10
        audio_length = int(round((downbeat_times[-1] + spacing) * _STEM_SR))

        boundaries = _compute_bar_boundaries(
            np.array([], dtype=np.intp), audio_length, downbeat_times=downbeat_times
        )

        # Monotonic + within [0, audio_length].
        assert boundaries[0] >= 0
        assert boundaries[-1] == audio_length
        assert np.all(np.diff(boundaries) > 0)

        widths = np.diff(boundaries)
        # No giant final bar: the max bar is nowhere near the whole song.
        assert widths.max() < audio_length * 0.1
        # Bars are real (seconds-scale), not 110-sample slivers.
        assert widths.min() > _STEM_SR * 0.5
        # Bar count tracks the downbeats (one bar per downbeat interval).
        assert len(boundaries) - 1 == len(downbeat_times)

    def test_fallback_path_converts_frames_to_samples(self) -> None:
        # No downbeats → fallback uses beat_frames[::4] in FRAME units, converted
        # to samples via frames_to_samples (hop_length=512).
        hop = 512
        # 8 bars * 4 beats, one frame per beat spaced so bars are ~1s apart.
        frames_per_bar = int(round(_STEM_SR / hop))  # frames in one second
        beat_frames = np.arange(8 * 4, dtype=np.intp) * (frames_per_bar // 4)
        audio_length = 8 * _STEM_SR

        boundaries = _compute_bar_boundaries(
            beat_frames, audio_length, downbeat_times=None
        )

        assert boundaries[0] >= 0
        assert boundaries[-1] == audio_length
        assert np.all(np.diff(boundaries) > 0)
        # Converted boundaries land in sample range, not frame range.
        assert boundaries[1] > hop  # first real bar start is sample-scale

    def test_downbeats_beyond_audio_length_are_dropped(self) -> None:
        # A stem slightly shorter than the analyzed mix: a downbeat sits past the
        # stem end. It must be dropped/clamped BEFORE the assertion (no false-fire).
        downbeat_times = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        audio_length = int(round(2.5 * _STEM_SR))  # cuts off the 3.0s downbeat

        boundaries = _compute_bar_boundaries(
            np.array([], dtype=np.intp), audio_length, downbeat_times=downbeat_times
        )

        assert boundaries[-1] == audio_length
        assert np.all(np.diff(boundaries) > 0)
        assert boundaries[-1] <= audio_length

    def test_neural_path_wins_over_redetected_beat_frames(self, tmp_path: Path) -> None:
        """When downbeat_times are present, the neural grid drives the bar count
        even though beat_frames is also supplied (drum re-detection precedence)."""
        stem_paths = {
            "lead_vocals": _write_wav(
                tmp_path / "lead_vocals.wav", _bar_signal(_VOCAL_BAR_AMPS, freq=330.0)
            ),
            "instrumental": _write_wav(
                tmp_path / "instrumental.wav", _bar_signal(_MIX_BAR_AMPS, freq=110.0)
            ),
        }
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        stem_analysis, structure = analyze_stems(
            stem_paths=stem_paths,
            beat_frames=_BEAT_FRAMES,
            bpm=120.0,
            audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        # Grid tracks the 8 neural downbeats → 8 bars.
        assert structure.total_bars == _N_BARS

    def test_vocal_source_throughout_yields_non_degenerate_vocal_active(
        self, tmp_path: Path
    ) -> None:
        """A vocal stem with vocals across the whole song must not collapse to
        all-false vocal_active (the original prod symptom)."""
        # Vocals present in every bar.
        full_vocal = _write_wav(
            tmp_path / "lead_vocals.wav", _bar_signal([0.6] * _N_BARS, freq=330.0)
        )
        instrumental = _write_wav(
            tmp_path / "instrumental.wav", _bar_signal(_MIX_BAR_AMPS, freq=110.0)
        )
        stem_paths = {"lead_vocals": full_vocal, "instrumental": instrumental}
        raw_mix = _make_raw_mix(tmp_path / "mix.wav")

        stem_analysis, _ = analyze_stems(
            stem_paths=stem_paths,
            beat_frames=_BEAT_FRAMES,
            bpm=120.0,
            audio_path=raw_mix,
            downbeat_times=_DOWNBEAT_TIMES,
        )

        vocal_active = stem_analysis.vocal_active
        assert vocal_active.size == _N_BARS
        # Non-degenerate: most bars active (not the all-false prod bug).
        assert vocal_active.sum() >= _N_BARS - 1

    def test_segments_to_sections_maps_midsong_segment_to_nonzero_start_bar(self) -> None:
        """CRITICAL-1 guard: under the sample-unit grid, _segments_to_sections must
        use boundaries/sr (not frames_to_time), so a mid-song ML segment maps to a
        non-zero start_bar instead of collapsing onto bar 0."""
        # 8 one-second bars → sample boundaries 0, sr, 2sr, ..., 8sr.
        audio_length = 8 * _STEM_SR
        downbeat_times = np.arange(8, dtype=np.float64) * 1.0
        bar_boundaries = _compute_bar_boundaries(
            np.array([], dtype=np.intp), audio_length, downbeat_times=downbeat_times
        )

        # A segment in the middle of the song (4.0s -> 6.0s) → bars [4, 6).
        ml_segments = [{"label": "chorus", "start": 4.0, "end": 6.0}]
        sections = _segments_to_sections(
            ml_segments=ml_segments,
            bar_boundaries=bar_boundaries,
            sr=_STEM_SR,
            audio_length=audio_length,
        )

        assert len(sections) == 1
        assert sections[0].start_bar == 4
        assert sections[0].start_bar != 0  # did NOT collapse onto bar 0
        assert sections[0].end_bar == 6
