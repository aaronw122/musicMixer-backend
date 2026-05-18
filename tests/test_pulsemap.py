"""Tests for PulseMap audio analysis module.

Tests: polyphony detection, chord detection, drum transcription,
word alignment, and AudioMetadata integration.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import (
    AudioMetadata,
    ChordEvent,
    ChordProgression,
    DrumPattern,
    LyricLine,
    LyricsData,
    PolyphonyInfo,
    WordAlignment,
    WordEvent,
)
from musicmixer.services import pulsemap as pulsemap_module
from musicmixer.services.pulsemap import (
    _build_progression_summary,
    _derive_drum_style,
    _validate_against_lrclib,
    detect_polyphony,
    transcribe_drum_pattern,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mono_sine(path: Path, freq: float = 440.0, duration: float = 2.0, sr: int = 22050) -> Path:
    """Generate a mono sine wave WAV file."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, sr)
    return path


def _make_stereo_center(path: Path, freq: float = 440.0, duration: float = 2.0, sr: int = 22050) -> Path:
    """Generate a stereo WAV with identical L/R (pure center = solo voice)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.column_stack([mono, mono])
    sf.write(str(path), stereo, sr)
    return path


def _make_stereo_wide(path: Path, duration: float = 2.0, sr: int = 22050) -> Path:
    """Generate a stereo WAV with very different L/R (wide stereo = polyphonic)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    left = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * np.pi * 660.0 * t)).astype(np.float32)
    stereo = np.column_stack([left, right])
    sf.write(str(path), stereo, sr)
    return path


def _make_drum_track(path: Path, sr: int = 22050, duration: float = 4.0) -> Path:
    """Generate a synthetic drum track with known kick, snare, and hi-hat placement.

    Kicks: low-freq pulses at 0s, 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s, 3.5s
    Snares: mid-freq noise bursts at 0.25s, 1.25s, 2.25s, 3.25s
    Hi-hats: high-freq noise at 0.125s, 0.375s, 0.625s, ... (every 0.25s offset)
    """
    n_samples = int(sr * duration)
    signal = np.zeros(n_samples, dtype=np.float32)

    pulse_len = int(0.02 * sr)  # 20ms pulse

    # Kicks: 80Hz sine bursts
    kick_times = np.arange(0, duration, 0.5)
    for kt in kick_times:
        idx = int(kt * sr)
        t = np.arange(pulse_len) / sr
        pulse = 0.8 * np.sin(2 * np.pi * 80.0 * t) * np.exp(-t * 50)
        end = min(idx + pulse_len, n_samples)
        signal[idx:end] += pulse[:end - idx].astype(np.float32)

    # Snares: band-limited noise around 500-1500Hz
    snare_times = [0.25, 1.25, 2.25, 3.25]
    for st in snare_times:
        idx = int(st * sr)
        t = np.arange(pulse_len) / sr
        noise = np.random.RandomState(42).randn(pulse_len).astype(np.float32)
        # Bandpass via simple envelope (approximate)
        pulse = 0.5 * noise * np.sin(2 * np.pi * 800 * t).astype(np.float32) * np.exp(-t * 40).astype(np.float32)
        end = min(idx + pulse_len, n_samples)
        signal[idx:end] += pulse[:end - idx]

    # Hi-hats: high-freq noise bursts >4kHz
    hihat_times = np.arange(0.125, duration, 0.25)
    short_pulse = int(0.005 * sr)  # 5ms
    for ht in hihat_times:
        idx = int(ht * sr)
        t = np.arange(short_pulse) / sr
        noise = np.random.RandomState(99).randn(short_pulse).astype(np.float32)
        pulse = 0.3 * noise * np.sin(2 * np.pi * 8000 * t).astype(np.float32)
        end = min(idx + short_pulse, n_samples)
        signal[idx:end] += pulse[:end - idx]

    sf.write(str(path), signal, sr)
    return path


# ---------------------------------------------------------------------------
# Polyphony detection tests
# ---------------------------------------------------------------------------

class TestDetectPolyphony:
    def test_mono_sine_without_essentia_defaults_solo(self, tmp_path: Path) -> None:
        """Mono sine wave without essentia should default to solo."""
        wav = _make_mono_sine(tmp_path / "mono.wav")
        with patch.object(pulsemap_module, "_HAS_ESSENTIA", False):
            result = detect_polyphony(wav)
        assert isinstance(result, PolyphonyInfo)
        assert result.polyphonic is False
        assert result.gate1_ratio is None  # mono skips gate 1

    def test_stereo_center_is_solo(self, tmp_path: Path) -> None:
        """Stereo file with identical L/R should be detected as solo (low side/mid ratio)."""
        wav = _make_stereo_center(tmp_path / "center.wav")
        # Patch essentia away so we only test gate 1
        with patch.object(pulsemap_module, "_HAS_ESSENTIA", False):
            result = detect_polyphony(wav)
        assert result.polyphonic is False
        assert result.method == "mid_side"
        assert result.gate1_ratio is not None
        assert result.gate1_ratio < 0.05

    def test_stereo_wide_is_polyphonic(self, tmp_path: Path) -> None:
        """Stereo file with very different L/R should be detected as polyphonic."""
        wav = _make_stereo_wide(tmp_path / "wide.wav")
        with patch.object(pulsemap_module, "_HAS_ESSENTIA", False):
            result = detect_polyphony(wav)
        assert result.polyphonic is True
        assert result.method == "mid_side"
        assert result.gate1_ratio is not None
        assert result.gate1_ratio > 0.15

    def test_returns_polyphony_info_type(self, tmp_path: Path) -> None:
        """Result should always be a PolyphonyInfo dataclass."""
        wav = _make_mono_sine(tmp_path / "test.wav")
        with patch.object(pulsemap_module, "_HAS_ESSENTIA", False):
            result = detect_polyphony(wav)
        assert isinstance(result, PolyphonyInfo)


# ---------------------------------------------------------------------------
# Chord detection tests (mocked lv-chordia)
# ---------------------------------------------------------------------------

class TestDetectChords:
    def test_jams_to_chord_events(self) -> None:
        """Verify JAMS annotation output is correctly converted to ChordEvents."""
        # Build a mock JAMS object
        mock_obs_1 = MagicMock()
        mock_obs_1.time = 0.0
        mock_obs_1.duration = 2.0
        mock_obs_1.value = "C"

        mock_obs_2 = MagicMock()
        mock_obs_2.time = 2.0
        mock_obs_2.duration = 2.0
        mock_obs_2.value = "Am"

        mock_obs_3 = MagicMock()
        mock_obs_3.time = 4.0
        mock_obs_3.duration = 2.0
        mock_obs_3.value = "F"

        mock_obs_n = MagicMock()
        mock_obs_n.time = 6.0
        mock_obs_n.duration = 1.0
        mock_obs_n.value = "N"  # No chord — should be skipped

        mock_annot = MagicMock()
        mock_annot.data = [mock_obs_1, mock_obs_2, mock_obs_3, mock_obs_n]

        mock_jams = MagicMock()
        mock_jams.annotations = [mock_annot]

        with patch.object(pulsemap_module, "_HAS_LV_CHORDIA", True), \
             patch.object(pulsemap_module, "lv_chordia") as mock_lvc:
            mock_lvc.chord_recognition.return_value = mock_jams
            from musicmixer.services.pulsemap import detect_chords
            result = detect_chords(Path("/fake/audio.wav"))

        assert isinstance(result, ChordProgression)
        assert len(result.chords) == 3
        assert result.chords[0] == ChordEvent(start_ms=0, end_ms=2000, chord="C")
        assert result.chords[1] == ChordEvent(start_ms=2000, end_ms=4000, chord="Am")
        assert result.chords[2] == ChordEvent(start_ms=4000, end_ms=6000, chord="F")
        assert result.unique_chords == ["C", "Am", "F"]
        assert result.most_common_chord == "C"  # all appear once, first wins
        assert "C" in result.progression_summary

    def test_no_chords_detected(self) -> None:
        """Empty JAMS annotations should produce empty ChordProgression."""
        mock_annot = MagicMock()
        mock_annot.data = []
        mock_jams = MagicMock()
        mock_jams.annotations = [mock_annot]

        with patch.object(pulsemap_module, "_HAS_LV_CHORDIA", True), \
             patch.object(pulsemap_module, "lv_chordia") as mock_lvc:
            mock_lvc.chord_recognition.return_value = mock_jams
            from musicmixer.services.pulsemap import detect_chords
            result = detect_chords(Path("/fake/audio.wav"))

        assert result.chords == []
        assert result.unique_chords == []
        assert result.most_common_chord == ""
        assert result.progression_summary == "no chords detected"

    def test_raises_without_lv_chordia(self) -> None:
        """Should raise RuntimeError when lv-chordia is not installed."""
        with patch.object(pulsemap_module, "_HAS_LV_CHORDIA", False):
            from musicmixer.services.pulsemap import detect_chords
            with pytest.raises(RuntimeError, match="lv-chordia"):
                detect_chords(Path("/fake/audio.wav"))

    def test_n_chords_skipped(self) -> None:
        """'N' (no chord) observations should be filtered out."""
        mock_obs = MagicMock()
        mock_obs.time = 0.0
        mock_obs.duration = 1.0
        mock_obs.value = "N"

        mock_annot = MagicMock()
        mock_annot.data = [mock_obs]
        mock_jams = MagicMock()
        mock_jams.annotations = [mock_annot]

        with patch.object(pulsemap_module, "_HAS_LV_CHORDIA", True), \
             patch.object(pulsemap_module, "lv_chordia") as mock_lvc:
            mock_lvc.chord_recognition.return_value = mock_jams
            from musicmixer.services.pulsemap import detect_chords
            result = detect_chords(Path("/fake/audio.wav"))

        assert result.chords == []


# ---------------------------------------------------------------------------
# Drum pattern tests
# ---------------------------------------------------------------------------

class TestTranscribeDrumPattern:
    def test_synthetic_drum_track(self, tmp_path: Path) -> None:
        """Synthetic drum track should produce non-zero hit counts."""
        wav = _make_drum_track(tmp_path / "drums.wav")
        result = transcribe_drum_pattern(wav)
        assert isinstance(result, DrumPattern)
        assert result.total_hits > 0
        assert result.duration_ms > 0
        assert result.style_hint != "silent"

    def test_silent_file(self, tmp_path: Path) -> None:
        """Silent audio should produce zero hits and 'silent' style."""
        silent = np.zeros(22050 * 2, dtype=np.float32)
        wav = tmp_path / "silent.wav"
        sf.write(str(wav), silent, 22050)
        result = transcribe_drum_pattern(wav)
        assert result.total_hits == 0
        assert result.style_hint == "silent"

    def test_drum_style_derivation(self) -> None:
        """Verify style hint logic for known ratios."""
        assert _derive_drum_style(0, 0, 0, 0) == "silent"
        assert _derive_drum_style(2, 2, 2, 6) == "sparse"
        assert _derive_drum_style(40, 10, 30, 80) == "four_on_floor"
        assert _derive_drum_style(10, 10, 60, 80) == "breakbeat"
        assert _derive_drum_style(30, 10, 50, 90) == "trap"
        assert _derive_drum_style(20, 20, 20, 60) == "standard"


# ---------------------------------------------------------------------------
# Word alignment tests (mocked whisperx)
# ---------------------------------------------------------------------------

class TestAlignWords:
    def test_whisperx_output_conversion(self) -> None:
        """Verify WhisperX output is correctly converted to WordEvents."""
        mock_aligned = {
            "segments": [
                {
                    "words": [
                        {"word": "Never", "start": 1.0, "end": 1.2},
                        {"word": "gonna", "start": 1.25, "end": 1.45},
                        {"word": "give", "start": 1.5, "end": 1.7},
                    ],
                },
                {
                    "words": [
                        {"word": "you", "start": 1.75, "end": 1.9},
                        {"word": "up", "start": 1.95, "end": 2.1},
                    ],
                },
            ],
        }

        with patch.object(pulsemap_module, "_HAS_WHISPERX", True), \
             patch.object(pulsemap_module, "_whisperx") as mock_wx:
            mock_model = MagicMock()
            mock_model.transcribe.return_value = {
                "segments": [{"text": "test"}],
                "language": "en",
            }
            mock_wx.load_model.return_value = mock_model
            mock_wx.load_audio.return_value = np.zeros(16000, dtype=np.float32)
            mock_wx.load_align_model.return_value = (MagicMock(), MagicMock())
            mock_wx.align.return_value = mock_aligned

            # Patch torch for device detection
            with patch.dict("sys.modules", {"torch": MagicMock(cuda=MagicMock(is_available=MagicMock(return_value=False)))}):
                from musicmixer.services.pulsemap import align_words
                result = align_words(Path("/fake/vocals.wav"))

        assert isinstance(result, WordAlignment)
        assert len(result.words) == 5
        assert result.words[0] == WordEvent(t=1000, text="Never", end=1200)
        assert result.words[4] == WordEvent(t=1950, text="up", end=2100)
        assert result.source == "whisperx"
        assert result.lrclib_validated is False

    def test_raises_without_whisperx(self) -> None:
        """Should raise RuntimeError when whisperx is not installed."""
        with patch.object(pulsemap_module, "_HAS_WHISPERX", False):
            from musicmixer.services.pulsemap import align_words
            with pytest.raises(RuntimeError, match="whisperx"):
                align_words(Path("/fake/vocals.wav"))


# ---------------------------------------------------------------------------
# LRCLIB validation tests
# ---------------------------------------------------------------------------

class TestLrclibValidation:
    def test_matching_timestamps_validate(self) -> None:
        """Words with matching LRCLIB timestamps should validate with low offset."""
        words = [
            WordEvent(t=1000, text="hello", end=1200),
            WordEvent(t=2000, text="world", end=2200),
            WordEvent(t=3000, text="foo", end=3200),
        ]
        lyrics = LyricsData(
            artist="Test", title="Test", source="lrclib",
            is_synced=True,
            lines=[
                LyricLine(text="hello world", timestamp_seconds=1.05),
                LyricLine(text="foo bar", timestamp_seconds=3.05),
            ],
            raw_text="hello world\nfoo bar",
        )
        validated, offset = _validate_against_lrclib(words, lyrics)
        assert validated is True
        assert offset is not None
        assert abs(offset) < 2000

    def test_mismatched_timestamps_fail(self) -> None:
        """Words with large offset from LRCLIB should fail validation."""
        words = [
            WordEvent(t=4000, text="hello", end=4200),
            WordEvent(t=6000, text="world", end=6200),
        ]
        lyrics = LyricsData(
            artist="Test", title="Test", source="lrclib",
            is_synced=True,
            lines=[
                LyricLine(text="hello world", timestamp_seconds=1.0),
                LyricLine(text="world foo", timestamp_seconds=3.0),
            ],
            raw_text="hello world\nworld foo",
        )
        validated, offset = _validate_against_lrclib(words, lyrics)
        assert validated is False
        assert offset is not None
        assert abs(offset) >= 2000

    def test_no_synced_lines(self) -> None:
        """Lines without timestamps should produce no validation."""
        words = [WordEvent(t=1000, text="hello", end=1200)]
        lyrics = LyricsData(
            artist="Test", title="Test", source="lrclib",
            is_synced=True,
            lines=[LyricLine(text="hello", timestamp_seconds=None)],
            raw_text="hello",
        )
        validated, offset = _validate_against_lrclib(words, lyrics)
        assert validated is False
        assert offset is None


# ---------------------------------------------------------------------------
# Progression summary tests
# ---------------------------------------------------------------------------

class TestProgressionSummary:
    def test_basic_progression(self) -> None:
        """Simple I-V-vi-IV should produce readable summary."""
        chords = [
            ChordEvent(0, 2000, "C"),
            ChordEvent(2000, 4000, "G"),
            ChordEvent(4000, 6000, "Am"),
            ChordEvent(6000, 8000, "F"),
        ]
        summary = _build_progression_summary(chords)
        assert "C" in summary
        # Should contain Roman numerals
        assert "I" in summary

    def test_empty_chords(self) -> None:
        summary = _build_progression_summary([])
        assert summary == "no chords detected"

    def test_consecutive_same_chords_deduplicated(self) -> None:
        """Repeated consecutive chords should be deduplicated in summary."""
        chords = [
            ChordEvent(0, 1000, "C"),
            ChordEvent(1000, 2000, "C"),
            ChordEvent(2000, 3000, "G"),
            ChordEvent(3000, 4000, "G"),
        ]
        summary = _build_progression_summary(chords)
        # Only C and G should appear, not C-C-G-G
        assert summary.count("I") <= 2  # I for C, could appear in key label


# ---------------------------------------------------------------------------
# AudioMetadata integration
# ---------------------------------------------------------------------------

class TestAudioMetadataIntegration:
    def test_new_fields_default_none(self) -> None:
        """New PulseMap fields should default to None."""
        meta = AudioMetadata(
            bpm=120.0,
            bpm_confidence=0.9,
            beat_frames=np.array([0, 100]),
            duration_seconds=30.0,
            total_beats=60,
        )
        assert meta.chord_progression is None
        assert meta.polyphony_info is None
        assert meta.drum_pattern is None
        assert meta.word_alignment is None

    def test_new_fields_accept_values(self) -> None:
        """New fields should accept proper PulseMap dataclass values."""
        meta = AudioMetadata(
            bpm=120.0,
            bpm_confidence=0.9,
            beat_frames=np.array([0, 100]),
            duration_seconds=30.0,
            total_beats=60,
            chord_progression=ChordProgression(
                chords=[ChordEvent(0, 2000, "C")],
                unique_chords=["C"],
                most_common_chord="C",
                progression_summary="I in C",
            ),
            polyphony_info=PolyphonyInfo(
                polyphonic=False, method="mid_side",
                gate1_ratio=0.02, gate2_percent=None,
            ),
            drum_pattern=DrumPattern(
                kick_count=48, snare_count=24, hihat_count=96,
                total_hits=168, duration_ms=180000,
                style_hint="four_on_floor",
            ),
            word_alignment=WordAlignment(
                words=[WordEvent(t=1000, text="hello", end=1200)],
                source="whisperx",
                lrclib_validated=True,
                lrclib_offset_ms=50,
            ),
        )
        assert meta.chord_progression is not None
        assert len(meta.chord_progression.chords) == 1
        assert meta.polyphony_info is not None
        assert meta.polyphony_info.polyphonic is False
        assert meta.drum_pattern is not None
        assert meta.drum_pattern.total_hits == 168
        assert meta.word_alignment is not None
        assert len(meta.word_alignment.words) == 1
