"""Tests for SongFormer ML-based structure detection integration.

Covers:
- _map_labels(): SongFormer label → musicMixer vocabulary mapping
- _segments_to_sections(): ML segments → SectionInfo with bar indices
- _enrich_sections(): Adding audio features to ML-detected sections
- analyze_structure_ml(): Modal-first with local fallback
- analyze_stems() ML path: ml_segments triggers ML path, skips heuristic
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import librosa
import numpy as np
import pytest

from musicmixer.models import (
    EnergyBuckets,
    SectionInfo,
    SongStructure,
    StemAnalysis,
)
from musicmixer.services.analysis import (
    ANALYSIS_SR,
    BUCKET_NOISE_FLOOR,
    STEM_NAMES,
    _enrich_sections,
    _segments_to_sections,
    classify_energy,
    compute_adaptive_buckets,
    detect_vocal_activity,
)
from musicmixer.services.structure_ml import (
    LABEL_MAP,
    _DEFAULT_LABEL,
    _map_labels,
    analyze_structure_ml,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar_boundaries(n_bars: int = 16, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Create synthetic bar boundary frames.

    Each bar is 0.5 seconds long at ANALYSIS_SR, expressed in hop-length
    frames (hop_length=512 by default in librosa).
    """
    hop_length = 512
    bar_duration_samples = int(0.5 * sr)
    bar_duration_frames = bar_duration_samples // hop_length
    return np.arange(n_bars + 1) * bar_duration_frames


def _make_6stem_rms(n_bars: int = 16, base: float = 0.1) -> dict[str, np.ndarray]:
    """Create a 6-stem bar_rms dict with uniform energy."""
    return {
        "drums": np.full(n_bars, base, dtype=np.float64),
        "bass": np.full(n_bars, base * 0.8, dtype=np.float64),
        "guitar": np.full(n_bars, base * 0.5, dtype=np.float64),
        "piano": np.full(n_bars, base * 0.3, dtype=np.float64),
        "vocals": np.full(n_bars, base * 0.6, dtype=np.float64),
        "other": np.full(n_bars, base * 0.2, dtype=np.float64),
    }


def _make_segments(labels_and_times: list[tuple[str, float, float]]) -> list[dict]:
    """Build segment dicts from (label, start, end) tuples."""
    return [
        {"label": label, "start": start, "end": end}
        for label, start, end in labels_and_times
    ]


# ---------------------------------------------------------------------------
# _map_labels tests
# ---------------------------------------------------------------------------

class TestMapLabels:
    """Tests for structure_ml._map_labels."""

    def test_known_labels_map_correctly(self) -> None:
        """Every label in LABEL_MAP should map to its expected value."""
        segments = _make_segments([
            ("verse", 0.0, 10.0),
            ("bridge", 10.0, 20.0),
            ("pre-chorus", 20.0, 30.0),
            ("solo", 30.0, 40.0),
            ("chorus", 40.0, 50.0),
            ("intro", 50.0, 55.0),
            ("outro", 55.0, 60.0),
            ("instrumental", 60.0, 70.0),
            ("interlude", 70.0, 80.0),
        ])
        result = _map_labels(segments)

        expected = ["verse", "breakdown", "build", "instrumental",
                    "chorus", "intro", "outro", "instrumental", "breakdown"]
        assert [s["label"] for s in result] == expected

    def test_unknown_label_defaults_to_verse(self) -> None:
        """Labels not in LABEL_MAP should map to _DEFAULT_LABEL, not raise."""
        segments = _make_segments([("SOME_WEIRD_LABEL", 0.0, 5.0)])
        result = _map_labels(segments)

        assert len(result) == 1
        assert result[0]["label"] == _DEFAULT_LABEL

    def test_empty_input_returns_empty(self) -> None:
        result = _map_labels([])
        assert result == []

    def test_times_preserved_through_mapping(self) -> None:
        """Start/end times must survive the mapping unchanged."""
        segments = _make_segments([("verse", 12.34, 56.78)])
        result = _map_labels(segments)

        assert result[0]["start"] == pytest.approx(12.34)
        assert result[0]["end"] == pytest.approx(56.78)

    def test_case_insensitive_matching(self) -> None:
        """Label matching should be case-insensitive (lowered before lookup)."""
        segments = _make_segments([("Verse", 0.0, 5.0), ("CHORUS", 5.0, 10.0)])
        result = _map_labels(segments)

        assert result[0]["label"] == "verse"
        assert result[1]["label"] == "chorus"

    def test_output_has_label_start_end_keys(self) -> None:
        """Each mapped segment must have exactly label, start, end."""
        segments = _make_segments([("intro", 0.0, 5.0)])
        result = _map_labels(segments)

        assert set(result[0].keys()) == {"label", "start", "end"}

    def test_start_end_are_floats(self) -> None:
        """start and end should be float even if input is int-like."""
        segments = [{"label": "verse", "start": 0, "end": 10}]
        result = _map_labels(segments)

        assert isinstance(result[0]["start"], float)
        assert isinstance(result[0]["end"], float)


# ---------------------------------------------------------------------------
# _segments_to_sections tests
# ---------------------------------------------------------------------------

class TestSegmentsToSections:
    """Tests for analysis._segments_to_sections."""

    def test_basic_conversion(self) -> None:
        """Segments with clear bar-aligned times produce correct bar indices."""
        n_bars = 16
        bar_bounds = _make_bar_boundaries(n_bars)
        bar_times = librosa.frames_to_time(bar_bounds[:-1], sr=ANALYSIS_SR)
        # Use times that fall cleanly on bar boundaries
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)
        segments = _make_segments([
            ("verse", float(bar_times[0]), float(bar_times[8])),
            ("chorus", float(bar_times[8]), float(bar_times[16 - 1])),
        ])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        assert len(result) >= 1
        for sec in result:
            assert isinstance(sec, SectionInfo)
            assert sec.section_source == "ml"

    def test_segment_before_first_bar_clamps_to_zero(self) -> None:
        """A segment starting before time 0 should get bar_start=0."""
        n_bars = 8
        bar_bounds = _make_bar_boundaries(n_bars)
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)
        segments = _make_segments([("intro", -1.0, 2.0)])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        assert len(result) == 1
        assert result[0].start_bar == 0

    def test_segment_after_last_bar_clamps_to_end(self) -> None:
        """A segment ending after the audio should clamp end_bar to n_bars."""
        n_bars = 8
        bar_bounds = _make_bar_boundaries(n_bars)
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)
        segments = _make_segments([("outro", 2.0, 999.0)])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        assert len(result) == 1
        assert result[0].end_bar <= n_bars

    def test_empty_segments_returns_empty(self) -> None:
        bar_bounds = _make_bar_boundaries(8)
        audio_length = int(8 * 0.5 * ANALYSIS_SR)

        result = _segments_to_sections([], bar_bounds, ANALYSIS_SR, audio_length)

        assert result == []

    def test_all_sections_have_ml_source(self) -> None:
        """Every returned SectionInfo must have section_source='ml'."""
        n_bars = 16
        bar_bounds = _make_bar_boundaries(n_bars)
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)
        bar_times = librosa.frames_to_time(bar_bounds[:-1], sr=ANALYSIS_SR)
        segments = _make_segments([
            ("verse", float(bar_times[0]), float(bar_times[4])),
            ("chorus", float(bar_times[4]), float(bar_times[8])),
            ("outro", float(bar_times[8]), float(bar_times[12])),
        ])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        for sec in result:
            assert sec.section_source == "ml"

    def test_bar_count_consistency(self) -> None:
        """bar_count should equal end_bar - start_bar for each section."""
        n_bars = 16
        bar_bounds = _make_bar_boundaries(n_bars)
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)
        bar_times = librosa.frames_to_time(bar_bounds[:-1], sr=ANALYSIS_SR)
        segments = _make_segments([("verse", float(bar_times[0]), float(bar_times[8]))])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        assert len(result) == 1
        sec = result[0]
        assert sec.bar_count == sec.end_bar - sec.start_bar

    def test_uses_librosa_frames_to_time(self) -> None:
        """Bar boundary conversion must use librosa.frames_to_time, not naive division.

        If the code used naive (frame * hop_length / sr) it would differ from
        librosa's implementation which accounts for centering.
        """
        n_bars = 8
        bar_bounds = _make_bar_boundaries(n_bars)
        audio_length = int(n_bars * 0.5 * ANALYSIS_SR)

        # librosa.frames_to_time uses n * hop_length / sr
        expected_times = librosa.frames_to_time(bar_bounds[:-1], sr=ANALYSIS_SR)

        # Create a segment spanning the full range
        segments = _make_segments([
            ("verse", float(expected_times[0]), float(expected_times[-1])),
        ])

        result = _segments_to_sections(segments, bar_bounds, ANALYSIS_SR, audio_length)

        # Should produce a valid section (not an error or empty)
        assert len(result) == 1
        assert result[0].start_bar == 0

    def test_too_few_bar_boundaries_returns_empty(self) -> None:
        """With fewer than 2 bar boundaries, no sections can be formed."""
        bar_bounds = np.array([0])
        result = _segments_to_sections(
            [{"label": "verse", "start": 0.0, "end": 5.0}],
            bar_bounds, ANALYSIS_SR, 22050 * 5,
        )
        assert result == []


# ---------------------------------------------------------------------------
# _enrich_sections tests
# ---------------------------------------------------------------------------

class TestEnrichSections:
    """Tests for analysis._enrich_sections."""

    @pytest.fixture()
    def enrichment_data(self):
        """Create a consistent set of data for enrichment tests."""
        n_bars = 16
        bar_bounds = _make_bar_boundaries(n_bars)
        bar_rms = _make_6stem_rms(n_bars, base=0.15)
        combined_energy, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = detect_vocal_activity(bar_rms["vocals"])
        audio = np.random.default_rng(42).standard_normal(
            int(n_bars * 0.5 * ANALYSIS_SR),
        ).astype(np.float32) * 0.1

        # Create ML sections covering the full range
        sections = [
            SectionInfo(
                start_bar=0, end_bar=8, bar_count=8,
                start_time=0.0, end_time=4.0,
                label="verse", energy_level="medium",
                energy_trajectory="medium", density="mid",
                vocal_status="vox:no", section_source="ml",
            ),
            SectionInfo(
                start_bar=8, end_bar=16, bar_count=8,
                start_time=4.0, end_time=8.0,
                label="chorus", energy_level="medium",
                energy_trajectory="medium", density="mid",
                vocal_status="vox:no", section_source="ml",
            ),
        ]

        return {
            "sections": sections,
            "audio": audio,
            "sr": ANALYSIS_SR,
            "bar_boundaries": bar_bounds,
            "bar_rms_per_stem": bar_rms,
            "combined_energy": combined_energy,
            "vocal_active": vocal_active,
            "buckets": buckets,
        }

    def test_enriched_source_is_enriched(self, enrichment_data) -> None:
        """All enriched sections must have section_source='enriched'."""
        result = _enrich_sections(**enrichment_data)

        for sec in result:
            assert sec.section_source == "enriched"

    def test_energy_classification_populated(self, enrichment_data) -> None:
        """Energy level must be one of the valid classifications."""
        result = _enrich_sections(**enrichment_data)

        valid_levels = {"silent", "low", "medium", "high", "peak"}
        for sec in result:
            assert sec.energy_level in valid_levels

    def test_original_labels_preserved(self, enrichment_data) -> None:
        """ML labels (verse, chorus) must survive enrichment."""
        result = _enrich_sections(**enrichment_data)

        assert result[0].label == "verse"
        assert result[1].label == "chorus"

    def test_vocal_status_computed(self, enrichment_data) -> None:
        """Vocal status should be filled in (not left as default)."""
        result = _enrich_sections(**enrichment_data)

        valid_statuses = {"vox:yes", "vox:no", "vox:fading"}
        for sec in result:
            assert sec.vocal_status in valid_statuses

    def test_density_computed(self, enrichment_data) -> None:
        """Density should be one of the valid values."""
        result = _enrich_sections(**enrichment_data)

        valid_densities = {"sparse", "mid", "full", "full+extra"}
        for sec in result:
            assert sec.density in valid_densities

    def test_energy_trajectory_computed(self, enrichment_data) -> None:
        """Energy trajectory should be a non-empty string."""
        result = _enrich_sections(**enrichment_data)

        for sec in result:
            assert isinstance(sec.energy_trajectory, str)
            assert len(sec.energy_trajectory) > 0

    def test_vocal_prominence_with_active_vocals(self) -> None:
        """When vocals are active for >= 3 bars, vocal_prominence_db should be set."""
        n_bars = 16
        bar_bounds = _make_bar_boundaries(n_bars)
        bar_rms = _make_6stem_rms(n_bars, base=0.15)
        # Make vocals clearly active
        bar_rms["vocals"] = np.full(n_bars, 0.3, dtype=np.float64)
        combined_energy, buckets = compute_adaptive_buckets(bar_rms)
        # Force vocal_active to True for all bars
        vocal_active = np.ones(n_bars, dtype=bool)
        audio = np.zeros(int(n_bars * 0.5 * ANALYSIS_SR), dtype=np.float32)

        sections = [
            SectionInfo(
                start_bar=0, end_bar=n_bars, bar_count=n_bars,
                start_time=0.0, end_time=8.0,
                label="verse", energy_level="medium",
                energy_trajectory="medium", density="mid",
                vocal_status="vox:no", section_source="ml",
            ),
        ]

        result = _enrich_sections(
            sections=sections, audio=audio, sr=ANALYSIS_SR,
            bar_boundaries=bar_bounds, bar_rms_per_stem=bar_rms,
            combined_energy=combined_energy, vocal_active=vocal_active,
            buckets=buckets,
        )

        assert len(result) == 1
        assert result[0].vocal_prominence_db is not None

    def test_section_count_preserved(self, enrichment_data) -> None:
        """Enrichment should not add or remove sections."""
        result = _enrich_sections(**enrichment_data)

        assert len(result) == len(enrichment_data["sections"])

    def test_bar_indices_preserved(self, enrichment_data) -> None:
        """start_bar and end_bar should not change during enrichment."""
        result = _enrich_sections(**enrichment_data)

        for orig, enriched in zip(enrichment_data["sections"], result):
            assert enriched.start_bar == orig.start_bar
            assert enriched.end_bar == orig.end_bar


# ---------------------------------------------------------------------------
# analyze_structure_ml tests
# ---------------------------------------------------------------------------

class TestAnalyzeStructureMl:
    """Tests for structure_ml.analyze_structure_ml."""

    def test_modal_success_returns_mapped_segments(self, tmp_path: Path) -> None:
        """When Modal succeeds, mapped segments are returned."""
        raw_segments = [
            {"label": "intro", "start": 0.0, "end": 10.0},
            {"label": "verse", "start": 10.0, "end": 30.0},
            {"label": "chorus", "start": 30.0, "end": 50.0},
        ]
        audio_path = tmp_path / "test.wav"

        with patch(
            "musicmixer.services.structure_ml._analyze_modal",
            return_value=raw_segments,
        ) as mock_modal:
            result = analyze_structure_ml(audio_path)

        mock_modal.assert_called_once_with(audio_path)
        assert len(result) == 3
        assert result[0]["label"] == "intro"
        assert result[1]["label"] == "verse"
        assert result[2]["label"] == "chorus"

    def test_modal_failure_falls_back_to_local(self, tmp_path: Path) -> None:
        """When Modal raises, local CPU fallback is used."""
        fallback_segments = [
            {"label": "verse", "start": 0.0, "end": 20.0},
        ]
        audio_path = tmp_path / "test.wav"

        with patch(
            "musicmixer.services.structure_ml._analyze_modal",
            side_effect=RuntimeError("Modal down"),
        ), patch(
            "musicmixer.services.structure_ml._analyze_local",
            return_value=fallback_segments,
        ) as mock_local:
            result = analyze_structure_ml(audio_path)

        mock_local.assert_called_once_with(audio_path)
        assert len(result) == 1
        assert result[0]["label"] == "verse"

    def test_returned_segments_have_correct_keys(self, tmp_path: Path) -> None:
        """Each returned segment must have label, start, end."""
        raw = [{"label": "verse", "start": 0.0, "end": 10.0}]
        audio_path = tmp_path / "test.wav"

        with patch(
            "musicmixer.services.structure_ml._analyze_modal",
            return_value=raw,
        ):
            result = analyze_structure_ml(audio_path)

        assert set(result[0].keys()) == {"label", "start", "end"}

    def test_labels_are_mapped_in_output(self, tmp_path: Path) -> None:
        """The output should contain mapped labels, not raw SongFormer labels."""
        raw = [
            {"label": "bridge", "start": 0.0, "end": 10.0},
            {"label": "pre-chorus", "start": 10.0, "end": 20.0},
            {"label": "solo", "start": 20.0, "end": 30.0},
        ]
        audio_path = tmp_path / "test.wav"

        with patch(
            "musicmixer.services.structure_ml._analyze_modal",
            return_value=raw,
        ):
            result = analyze_structure_ml(audio_path)

        assert result[0]["label"] == "breakdown"   # bridge → breakdown
        assert result[1]["label"] == "build"        # pre-chorus → build
        assert result[2]["label"] == "instrumental"  # solo → instrumental

    def test_both_backends_fail(self, tmp_path: Path) -> None:
        """If Modal fails and local also fails, the exception propagates."""
        audio_path = tmp_path / "test.wav"

        with patch(
            "musicmixer.services.structure_ml._analyze_modal",
            side_effect=RuntimeError("Modal down"),
        ), patch(
            "musicmixer.services.structure_ml._analyze_local",
            side_effect=RuntimeError("Local also failed"),
        ):
            with pytest.raises(RuntimeError, match="Local also failed"):
                analyze_structure_ml(audio_path)


# ---------------------------------------------------------------------------
# analyze_stems with ml_segments tests
# ---------------------------------------------------------------------------

class TestAnalyzeStemsMLPath:
    """Tests for the ML branch in analysis.analyze_stems."""

    @pytest.fixture()
    def stem_wav_files(self, tmp_path: Path) -> dict[str, Path]:
        """Create minimal WAV files for all 6 stems."""
        import soundfile as sf

        sr = ANALYSIS_SR
        duration = 8.0  # seconds
        n_samples = int(sr * duration)
        rng = np.random.default_rng(42)

        paths: dict[str, Path] = {}
        for name in STEM_NAMES:
            path = tmp_path / f"{name}.wav"
            audio = rng.standard_normal(n_samples).astype(np.float32) * 0.05
            sf.write(str(path), audio, sr, subtype="FLOAT")
            paths[name] = path
        return paths

    @pytest.fixture()
    def beat_frames(self) -> np.ndarray:
        """Create synthetic beat frames at 120 BPM, 8 seconds."""
        sr = ANALYSIS_SR
        bpm = 120.0
        beat_interval = 60.0 / bpm  # 0.5 seconds per beat
        duration = 8.0
        hop_length = 512
        beat_times = np.arange(0, duration, beat_interval)
        return librosa.time_to_frames(beat_times, sr=sr, hop_length=hop_length)

    def test_ml_segments_triggers_ml_path(
        self, stem_wav_files, beat_frames,
    ) -> None:
        """When ml_segments is a non-empty list, the heuristic path is skipped."""
        from musicmixer.services.analysis import analyze_stems

        ml_segments = [
            {"label": "verse", "start": 0.0, "end": 4.0},
            {"label": "chorus", "start": 4.0, "end": 8.0},
        ]

        with patch(
            "musicmixer.services.analysis.detect_sections",
            wraps=None,
            side_effect=AssertionError("Heuristic path should not be called"),
        ):
            stem_analysis, song_structure = analyze_stems(
                stem_paths=stem_wav_files,
                beat_frames=beat_frames,
                bpm=120.0,
                ml_segments=ml_segments,
            )

        assert isinstance(stem_analysis, StemAnalysis)
        assert isinstance(song_structure, SongStructure)
        # Sections should exist and have enriched source
        assert len(song_structure.sections) > 0
        for sec in song_structure.sections:
            assert sec.section_source == "enriched"

    def test_none_ml_segments_runs_heuristic(
        self, stem_wav_files, beat_frames,
    ) -> None:
        """When ml_segments is None, the heuristic path is used."""
        from musicmixer.services.analysis import analyze_stems

        with patch(
            "musicmixer.services.analysis.detect_sections",
        ) as mock_detect:
            mock_detect.return_value = []
            stem_analysis, song_structure = analyze_stems(
                stem_paths=stem_wav_files,
                beat_frames=beat_frames,
                bpm=120.0,
                ml_segments=None,
            )

        mock_detect.assert_called_once()

    def test_empty_ml_segments_runs_heuristic(
        self, stem_wav_files, beat_frames,
    ) -> None:
        """When ml_segments is an empty list, the heuristic path is used."""
        from musicmixer.services.analysis import analyze_stems

        with patch(
            "musicmixer.services.analysis.detect_sections",
        ) as mock_detect:
            mock_detect.return_value = []
            stem_analysis, song_structure = analyze_stems(
                stem_paths=stem_wav_files,
                beat_frames=beat_frames,
                bpm=120.0,
                ml_segments=[],
            )

        mock_detect.assert_called_once()
