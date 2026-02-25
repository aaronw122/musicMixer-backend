"""Tests for BPM detection and cross-song reconciliation.

Step 2 of Day 2: analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import AudioMetadata
from musicmixer.services.analysis import analyze_audio, reconcile_bpm


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

        reconcile_bpm(a, b)

        assert a.bpm == original_a_bpm
        assert b.bpm == original_b_bpm
