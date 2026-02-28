"""Tests for musicmixer.training.analyze -- mashup analysis pipeline.

Covers:
- Single mashup analysis (analyze_mashup)
- Batch analysis (analyze_all)
- Serialization of numpy arrays to JSON
- Skip logic for already-analyzed and missing WAVs
- Error handling (failed analyses continue in batch)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicmixer.models import (
    AudioMetadata,
    EnergyBuckets,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
)
from musicmixer.training.analyze import (
    _ndarray_to_list,
    _serialize_analysis,
    analyze_all,
    analyze_mashup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_metadata() -> AudioMetadata:
    """AudioMetadata with realistic values."""
    return AudioMetadata(
        bpm=120.0,
        bpm_confidence=0.85,
        beat_frames=np.array([0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]),
        duration_seconds=180.0,
        total_beats=360,
        mean_rms=0.15,
    )


@pytest.fixture
def mock_stem_analysis() -> StemAnalysis:
    """StemAnalysis with 4 bars of data."""
    return StemAnalysis(
        bar_rms={
            "vocals": np.array([0.1, 0.3, 0.5, 0.2]),
            "drums": np.array([0.4, 0.6, 0.7, 0.5]),
            "bass": np.array([0.3, 0.4, 0.5, 0.3]),
            "guitar": np.array([0.2, 0.3, 0.4, 0.2]),
            "piano": np.array([0.1, 0.2, 0.3, 0.1]),
            "other": np.array([0.05, 0.1, 0.15, 0.05]),
        },
        combined_energy=np.array([0.3, 0.5, 0.8, 0.4]),
        vocal_active=np.array([False, True, True, False]),
        vocal_gaps=[VocalGap(start_bar=0, end_bar=1, length_bars=1)],
        bucket_thresholds=EnergyBuckets(
            noise_floor=0.02, p10=0.1, p50=0.4, p85=0.7
        ),
    )


@pytest.fixture
def mock_song_structure() -> SongStructure:
    """SongStructure with 2 sections."""
    return SongStructure(
        sections=[
            SectionInfo(
                start_bar=0,
                end_bar=2,
                bar_count=2,
                start_time=0.0,
                end_time=4.0,
                label="intro",
                energy_level="low",
                energy_trajectory="low->medium",
                density="sparse",
                vocal_status="vox:no",
            ),
            SectionInfo(
                start_bar=2,
                end_bar=4,
                bar_count=2,
                start_time=4.0,
                end_time=8.0,
                label="verse",
                energy_level="medium",
                energy_trajectory="medium->high",
                density="mid",
                vocal_status="vox:yes",
            ),
        ],
        vocal_gaps=[VocalGap(start_bar=0, end_bar=1, length_bars=1)],
        total_bars=4,
    )


@pytest.fixture
def mock_stem_paths(tmp_path: Path) -> dict[str, Path]:
    """Create mock stem WAV files."""
    stems_dir = tmp_path / "stems" / "mashup_test"
    stems_dir.mkdir(parents=True)
    paths = {}
    for stem in ["vocals", "drums", "bass", "guitar", "piano", "other"]:
        p = stems_dir / f"{stem}.wav"
        p.write_bytes(b"\x00" * 100)
        paths[stem] = p
    return paths


@pytest.fixture
def manifest_data() -> list[dict]:
    """Manifest with 3 entries."""
    return [
        {"id": "mashup_001", "url": "https://youtube.com/watch?v=a", "title": "M1"},
        {"id": "mashup_002", "url": "https://youtube.com/watch?v=b", "title": "M2"},
        {"id": "mashup_003", "url": "https://youtube.com/watch?v=c", "title": "M3"},
    ]


# ===========================================================================
# _ndarray_to_list tests
# ===========================================================================


class TestNdarrayToList:
    """Test numpy-to-native-type conversion helper."""

    def test_converts_array(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        assert _ndarray_to_list(arr) == [1.0, 2.0, 3.0]

    def test_converts_nested_dict(self) -> None:
        data = {"a": np.array([1, 2]), "b": {"c": np.array([3])}}
        result = _ndarray_to_list(data)
        assert result == {"a": [1, 2], "b": {"c": [3]}}

    def test_converts_numpy_int(self) -> None:
        val = np.int64(42)
        assert _ndarray_to_list(val) == 42
        assert isinstance(_ndarray_to_list(val), int)

    def test_converts_numpy_float(self) -> None:
        val = np.float32(3.14)
        result = _ndarray_to_list(val)
        assert isinstance(result, float)

    def test_converts_numpy_bool(self) -> None:
        val = np.bool_(True)
        result = _ndarray_to_list(val)
        assert result is True
        assert isinstance(result, bool)

    def test_passthrough_native_types(self) -> None:
        assert _ndarray_to_list("hello") == "hello"
        assert _ndarray_to_list(42) == 42
        assert _ndarray_to_list(None) is None


# ===========================================================================
# _serialize_analysis tests
# ===========================================================================


class TestSerializeAnalysis:
    """Test serialization of analysis results to JSON-compatible dict."""

    def test_produces_json_serializable_output(
        self,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
    ) -> None:
        """Output can be serialized to JSON without errors."""
        result = _serialize_analysis(
            metadata=mock_metadata,
            stem_analysis=mock_stem_analysis,
            song_structure=mock_song_structure,
            key="C",
            scale="major",
            key_confidence=0.9,
        )

        # Should not raise
        json_str = json.dumps(result)
        parsed = json.loads(json_str)

        assert "audio_metadata" in parsed
        assert "stem_analysis" in parsed
        assert "song_structure" in parsed

    def test_audio_metadata_fields(
        self,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
    ) -> None:
        result = _serialize_analysis(
            metadata=mock_metadata,
            stem_analysis=mock_stem_analysis,
            song_structure=mock_song_structure,
            key="A",
            scale="minor",
            key_confidence=0.75,
        )

        meta = result["audio_metadata"]
        assert meta["bpm"] == 120.0
        assert meta["key"] == "A"
        assert meta["scale"] == "minor"
        assert meta["key_confidence"] == 0.75
        assert isinstance(meta["beat_frames"], list)

    def test_stem_analysis_fields(
        self,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
    ) -> None:
        result = _serialize_analysis(
            metadata=mock_metadata,
            stem_analysis=mock_stem_analysis,
            song_structure=mock_song_structure,
            key="C",
            scale="major",
            key_confidence=0.9,
        )

        stem = result["stem_analysis"]
        assert "vocals" in stem["bar_rms"]
        assert isinstance(stem["bar_rms"]["vocals"], list)
        assert isinstance(stem["combined_energy"], list)
        assert isinstance(stem["vocal_active"], list)

    def test_song_structure_fields(
        self,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
    ) -> None:
        result = _serialize_analysis(
            metadata=mock_metadata,
            stem_analysis=mock_stem_analysis,
            song_structure=mock_song_structure,
            key="C",
            scale="major",
            key_confidence=0.9,
        )

        structure = result["song_structure"]
        assert len(structure["sections"]) == 2
        assert structure["total_bars"] == 4
        assert structure["sections"][0]["label"] == "intro"


# ===========================================================================
# analyze_mashup tests
# ===========================================================================


class TestAnalyzeMashup:
    """Test single mashup analysis pipeline."""

    @patch("musicmixer.training.analyze.analyze_stems")
    @patch("musicmixer.training.analyze.separate_stems")
    @patch("musicmixer.training.analyze.detect_key")
    @patch("musicmixer.training.analyze.analyze_audio")
    def test_runs_full_pipeline(
        self,
        mock_analyze_audio: MagicMock,
        mock_detect_key: MagicMock,
        mock_separate: MagicMock,
        mock_analyze_stems: MagicMock,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
        mock_stem_paths: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Pipeline runs all 4 steps in order."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"\x00" * 100)
        stems_dir = tmp_path / "stems"

        mock_analyze_audio.return_value = mock_metadata
        mock_detect_key.return_value = ("C", "major", 0.9)
        mock_separate.return_value = mock_stem_paths
        mock_analyze_stems.return_value = (mock_stem_analysis, mock_song_structure)

        result = analyze_mashup("test_001", wav_path, stems_dir)

        mock_analyze_audio.assert_called_once_with(wav_path)
        mock_detect_key.assert_called_once_with(wav_path)
        mock_separate.assert_called_once()
        mock_analyze_stems.assert_called_once()

        # Result should be JSON-serializable
        json.dumps(result)

        assert result["audio_metadata"]["bpm"] == 120.0
        assert result["audio_metadata"]["key"] == "C"
        assert len(result["song_structure"]["sections"]) == 2

    def test_raises_on_missing_wav(self, tmp_path: Path) -> None:
        """FileNotFoundError when WAV does not exist."""
        with pytest.raises(FileNotFoundError, match="WAV file not found"):
            analyze_mashup(
                "test_001",
                tmp_path / "nonexistent.wav",
                tmp_path / "stems",
            )

    @patch("musicmixer.training.analyze.analyze_stems")
    @patch("musicmixer.training.analyze.separate_stems")
    @patch("musicmixer.training.analyze.detect_key")
    @patch("musicmixer.training.analyze.analyze_audio")
    def test_creates_stems_subdirectory(
        self,
        mock_analyze_audio: MagicMock,
        mock_detect_key: MagicMock,
        mock_separate: MagicMock,
        mock_analyze_stems: MagicMock,
        mock_metadata: AudioMetadata,
        mock_stem_analysis: StemAnalysis,
        mock_song_structure: SongStructure,
        mock_stem_paths: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Stems directory is created as stems_dir/{mashup_id}/."""
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(b"\x00" * 100)
        stems_dir = tmp_path / "stems"

        mock_analyze_audio.return_value = mock_metadata
        mock_detect_key.return_value = ("G", "minor", 0.8)
        mock_separate.return_value = mock_stem_paths
        mock_analyze_stems.return_value = (mock_stem_analysis, mock_song_structure)

        analyze_mashup("test_002", wav_path, stems_dir)

        # separate_stems should receive the mashup-specific subdirectory
        call_args = mock_separate.call_args
        assert call_args[0][1] == stems_dir / "test_002"


# ===========================================================================
# analyze_all tests
# ===========================================================================


class TestAnalyzeAll:
    """Test batch analysis pipeline."""

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_analyzes_all_entries(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """All entries with WAV files are analyzed."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create WAV files for all entries
        for entry in manifest_data:
            (raw_dir / f"{entry['id']}.wav").write_bytes(b"\x00" * 100)

        mock_analyze.return_value = {
            "audio_metadata": {"bpm": 120.0},
            "stem_analysis": {},
            "song_structure": {},
        }

        results = analyze_all(manifest_data, raw_dir, output_dir)

        assert len(results) == 3
        assert mock_analyze.call_count == 3

        # JSON files should exist
        for mashup_id in ["mashup_001", "mashup_002", "mashup_003"]:
            json_path = output_dir / "analysis" / f"{mashup_id}.json"
            assert json_path.exists()

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_skips_missing_wavs(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """Entries without WAV files are skipped."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"

        # Only create WAV for first entry
        (raw_dir / "mashup_001.wav").write_bytes(b"\x00" * 100)

        mock_analyze.return_value = {
            "audio_metadata": {"bpm": 120.0},
            "stem_analysis": {},
            "song_structure": {},
        }

        results = analyze_all(manifest_data, raw_dir, output_dir)

        assert len(results) == 1
        assert "mashup_001" in results
        assert mock_analyze.call_count == 1

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_skips_already_analyzed(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """Entries with existing analysis JSON are skipped."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True)

        # Create WAVs for all entries
        for entry in manifest_data:
            (raw_dir / f"{entry['id']}.wav").write_bytes(b"\x00" * 100)

        # Pre-create analysis JSON for first entry
        (analysis_dir / "mashup_001.json").write_text('{"bpm": 120}')

        mock_analyze.return_value = {
            "audio_metadata": {"bpm": 120.0},
            "stem_analysis": {},
            "song_structure": {},
        }

        results = analyze_all(manifest_data, raw_dir, output_dir)

        assert len(results) == 3
        # Only 2 analyzed (first was skipped)
        assert mock_analyze.call_count == 2

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_continues_on_failure(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """Failed analyses are logged; remaining entries continue."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"

        for entry in manifest_data:
            (raw_dir / f"{entry['id']}.wav").write_bytes(b"\x00" * 100)

        call_count = 0

        def _side_effect(mashup_id: str, wav_path: Path, stems_dir: Path) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Modal GPU error")
            return {
                "audio_metadata": {"bpm": 120.0},
                "stem_analysis": {},
                "song_structure": {},
            }

        mock_analyze.side_effect = _side_effect

        results = analyze_all(manifest_data, raw_dir, output_dir)

        # Second entry failed, 2 succeeded
        assert len(results) == 2
        assert "mashup_002" not in results

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_creates_output_directories(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """Output subdirectories (analysis/, stems/) are created."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"

        (raw_dir / "mashup_001.wav").write_bytes(b"\x00" * 100)

        mock_analyze.return_value = {
            "audio_metadata": {"bpm": 120.0},
            "stem_analysis": {},
            "song_structure": {},
        }

        analyze_all(manifest_data[:1], raw_dir, output_dir)

        assert (output_dir / "analysis").is_dir()
        assert (output_dir / "stems").is_dir()

    @patch("musicmixer.training.analyze.analyze_mashup")
    def test_skips_empty_analysis_files(
        self,
        mock_analyze: MagicMock,
        manifest_data: list[dict],
        tmp_path: Path,
    ) -> None:
        """Zero-byte analysis JSON files are re-analyzed."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        output_dir = tmp_path / "output"
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(parents=True)

        (raw_dir / "mashup_001.wav").write_bytes(b"\x00" * 100)
        (analysis_dir / "mashup_001.json").touch()  # 0 bytes

        mock_analyze.return_value = {
            "audio_metadata": {"bpm": 120.0},
            "stem_analysis": {},
            "song_structure": {},
        }

        results = analyze_all(manifest_data[:1], raw_dir, output_dir)

        assert mock_analyze.call_count == 1
        assert len(results) == 1
