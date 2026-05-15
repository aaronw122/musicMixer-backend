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
    reconcile_bpm,
    _detect_beats_neural,
    _transform_beat_frames,
    _transform_total_beats,
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
    def test_same_bpm(self) -> None:
        """Two songs at 120 BPM should both stay at 120."""
        a = _make_metadata(bpm=120.0)
        b = _make_metadata(bpm=120.0)
        new_a, new_b = reconcile_bpm(a, b)

        assert new_a.bpm == pytest.approx(120.0)
        assert new_b.bpm == pytest.approx(120.0)

    def test_double_octave(self) -> None:
        """Songs at 60 and 120 BPM: 60 doubled to 120 (5% penalty) is better
        than the 50% gap at original tempos."""
        a = _make_metadata(bpm=60.0)
        b = _make_metadata(bpm=120.0)
        new_a, new_b = reconcile_bpm(a, b)

        assert new_a.bpm == pytest.approx(120.0)
        assert new_b.bpm == pytest.approx(120.0)

    def test_halved(self) -> None:
        """Songs at 140 and 70: 70 doubled to 140 (5% penalty) is better
        than the 50% gap."""
        a = _make_metadata(bpm=140.0)
        b = _make_metadata(bpm=70.0)
        new_a, new_b = reconcile_bpm(a, b)

        assert new_a.bpm == pytest.approx(140.0)
        assert new_b.bpm == pytest.approx(140.0)

    def test_within_range(self) -> None:
        """Songs at 115 and 125 BPM: small gap (~8%), both stay original
        (no penalty is better than adding 5%+ for a transformation)."""
        a = _make_metadata(bpm=115.0)
        b = _make_metadata(bpm=125.0)
        new_a, new_b = reconcile_bpm(a, b)

        assert new_a.bpm == pytest.approx(115.0)
        assert new_b.bpm == pytest.approx(125.0)

    def test_filters_out_of_range(self) -> None:
        """Song at 40 BPM: original (40) is out of 70-180 range.
        Doubled (80) should be in range and selected."""
        a = _make_metadata(bpm=40.0)
        b = _make_metadata(bpm=80.0)
        new_a, new_b = reconcile_bpm(a, b)

        # 40 is out of range, doubled 80 is in range
        assert new_a.bpm == pytest.approx(80.0)
        assert new_b.bpm == pytest.approx(80.0)

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

    def test_doubled_transforms_beat_frames(self) -> None:
        """When BPM is doubled, beat_frames should have interpolated midpoints."""
        a = _make_metadata(bpm=60.0)  # Will be doubled to 120
        b = _make_metadata(bpm=120.0)
        new_a, new_b = reconcile_bpm(a, b)

        # A was doubled: original frames [0, 100, 200] -> [0, 50, 100, 150, 200]
        expected_a = np.array([0, 50, 100, 150, 200], dtype=np.intp)
        np.testing.assert_array_equal(new_a.beat_frames, expected_a)
        # B was original: frames unchanged
        np.testing.assert_array_equal(new_b.beat_frames, b.beat_frames)

    def test_halved_transforms_beat_frames(self) -> None:
        """_transform_beat_frames with 'halved' takes every other beat."""
        frames = np.array([0, 50, 100, 150, 200, 250], dtype=np.intp)
        result = _transform_beat_frames(frames, "halved")
        expected = np.array([0, 100, 200], dtype=np.intp)
        np.testing.assert_array_equal(result, expected)

    def test_doubled_transforms_total_beats(self) -> None:
        """When BPM is doubled, total_beats should double."""
        a = _make_metadata(bpm=60.0)  # Will be doubled to 120
        b = _make_metadata(bpm=120.0)
        new_a, _ = reconcile_bpm(a, b)

        assert new_a.total_beats == a.total_beats * 2

    def test_halved_transforms_total_beats(self) -> None:
        """_transform_total_beats with 'halved' halves beat count (min 4)."""
        assert _transform_total_beats(80, "halved") == 40
        assert _transform_total_beats(6, "halved") == 4  # min 4

    def test_original_leaves_beat_frames_unchanged(self) -> None:
        """_transform_beat_frames with 'original' returns frames as-is."""
        frames = np.array([0, 100, 200], dtype=np.intp)
        result = _transform_beat_frames(frames, "original")
        np.testing.assert_array_equal(result, frames)

    def test_triplet_leaves_beat_frames_unchanged(self) -> None:
        """_transform_beat_frames with '3/2' or '2/3' returns frames as-is."""
        frames = np.array([0, 100, 200], dtype=np.intp)
        np.testing.assert_array_equal(_transform_beat_frames(frames, "3/2"), frames)
        np.testing.assert_array_equal(_transform_beat_frames(frames, "2/3"), frames)


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
