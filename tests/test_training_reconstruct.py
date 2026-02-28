"""Tests for musicmixer.training.reconstruct -- plan reconstruction from mashup analysis.

Covers:
  - Section label mapping (all analysis labels -> inference-time vocab)
  - Instrumental label mapping conditioned on energy level
  - Bar-to-beat boundary conversion
  - Stem gain estimation and normalization
  - Transition type inference (cut, crossfade, fade)
  - Drop-then-rise pattern detection
  - Full reconstruct_plan() end-to-end
  - reconstruct_all() batch processing
  - Edge cases: single section, empty bar_rms, unknown labels
  - Serialization round-trip
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from musicmixer.models import RemixPlan, Section, SectionInfo
from musicmixer.training.reconstruct import (
    SHARP_INCREASE_DB,
    STEM_NAMES,
    _compute_global_max_per_stem,
    _compute_section_stem_gains,
    _detect_drop_then_rise,
    _infer_transition,
    _map_section_label,
    _rms_to_db,
    _serialize_plan,
    reconstruct_all,
    reconstruct_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section_info(
    start_bar: int = 0,
    end_bar: int = 8,
    label: str = "verse",
    energy_level: str = "medium",
    energy_trajectory: str = "medium->medium",
    density: str = "mid",
    vocal_status: str = "vox:yes",
    annotations: list[str] | None = None,
) -> SectionInfo:
    """Create a SectionInfo for testing."""
    return SectionInfo(
        start_bar=start_bar,
        end_bar=end_bar,
        bar_count=end_bar - start_bar,
        start_time=0.0,
        end_time=0.0,
        label=label,
        energy_level=energy_level,
        energy_trajectory=energy_trajectory,
        density=density,
        vocal_status=vocal_status,
        annotations=annotations or [],
    )


def _make_analysis(
    sections: list[dict] | None = None,
    total_bars: int = 32,
    duration: float = 120.0,
    bpm: float = 120.0,
    bar_rms: dict[str, list[float]] | None = None,
    combined_energy: list[float] | None = None,
    title: str = "Test Mashup",
) -> dict:
    """Create a minimal analysis dict for testing."""
    if sections is None:
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "intro", "energy_level": "low"},
            {"start_bar": 8, "end_bar": 24, "label": "verse", "energy_level": "medium"},
            {"start_bar": 24, "end_bar": 32, "label": "outro", "energy_level": "low"},
        ]

    if bar_rms is None:
        bar_rms = {name: [0.1] * total_bars for name in STEM_NAMES}

    if combined_energy is None:
        combined_energy = [0.5] * total_bars

    return {
        "audio_metadata": {
            "duration_seconds": duration,
            "bpm": bpm,
        },
        "song_structure": {
            "sections": sections,
            "total_bars": total_bars,
        },
        "stem_analysis": {
            "bar_rms": bar_rms,
            "combined_energy": combined_energy,
        },
        "title": title,
    }


# ---------------------------------------------------------------------------
# Section Label Mapping
# ---------------------------------------------------------------------------

class TestSectionLabelMapping:
    """Tests for _map_section_label."""

    def test_intro_maps_to_intro(self):
        sec = _make_section_info(label="intro")
        assert _map_section_label(sec) == "intro"

    def test_verse_maps_to_verse(self):
        sec = _make_section_info(label="verse")
        assert _map_section_label(sec) == "verse"

    def test_chorus_maps_to_drop(self):
        sec = _make_section_info(label="chorus")
        assert _map_section_label(sec) == "drop"

    def test_build_maps_to_verse(self):
        sec = _make_section_info(label="build")
        assert _map_section_label(sec) == "verse"

    def test_breakdown_maps_to_breakdown(self):
        sec = _make_section_info(label="breakdown")
        assert _map_section_label(sec) == "breakdown"

    def test_outro_maps_to_outro(self):
        sec = _make_section_info(label="outro")
        assert _map_section_label(sec) == "outro"

    def test_instrumental_low_energy_maps_to_breakdown(self):
        sec = _make_section_info(label="instrumental", energy_level="low")
        assert _map_section_label(sec) == "breakdown"

    def test_instrumental_medium_energy_maps_to_breakdown(self):
        sec = _make_section_info(label="instrumental", energy_level="medium")
        assert _map_section_label(sec) == "breakdown"

    def test_instrumental_high_energy_maps_to_drop(self):
        sec = _make_section_info(label="instrumental", energy_level="high")
        assert _map_section_label(sec) == "drop"

    def test_instrumental_peak_energy_maps_to_drop(self):
        sec = _make_section_info(label="instrumental", energy_level="peak")
        assert _map_section_label(sec) == "drop"

    def test_unknown_label_falls_back_to_verse(self):
        sec = _make_section_info(label="bridge")
        assert _map_section_label(sec) == "verse"

    def test_all_output_labels_in_vocabulary(self):
        """Verify all possible outputs are in the inference-time vocabulary."""
        valid_labels = {"intro", "verse", "breakdown", "drop", "outro"}
        test_cases = [
            ("intro", "low"),
            ("verse", "medium"),
            ("chorus", "high"),
            ("build", "medium"),
            ("breakdown", "low"),
            ("outro", "low"),
            ("instrumental", "low"),
            ("instrumental", "high"),
            ("unknown_label", "medium"),
        ]
        for label, energy in test_cases:
            sec = _make_section_info(label=label, energy_level=energy)
            result = _map_section_label(sec)
            assert result in valid_labels, f"Label '{result}' from ({label}, {energy}) not in vocabulary"


# ---------------------------------------------------------------------------
# Stem Gain Estimation
# ---------------------------------------------------------------------------

class TestStemGainEstimation:
    """Tests for stem gain computation and normalization."""

    def test_uniform_rms_gives_gain_of_one(self):
        """When a stem has uniform RMS, all sections get gain=1.0."""
        bar_rms = {"vocals": np.array([0.5, 0.5, 0.5, 0.5])}
        sections = [_make_section_info(start_bar=0, end_bar=4)]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        gains = _compute_section_stem_gains(bar_rms, 0, 4, global_max)
        assert gains["vocals"] == pytest.approx(1.0)

    def test_half_rms_gives_gain_of_half(self):
        """Section with half the max RMS gets gain=0.5."""
        bar_rms = {
            "vocals": np.array([1.0, 1.0, 0.5, 0.5]),
        }
        sections = [
            _make_section_info(start_bar=0, end_bar=2),
            _make_section_info(start_bar=2, end_bar=4),
        ]
        global_max = _compute_global_max_per_stem(bar_rms, sections)
        assert global_max["vocals"] == pytest.approx(1.0)

        gains = _compute_section_stem_gains(bar_rms, 2, 4, global_max)
        assert gains["vocals"] == pytest.approx(0.5)

    def test_zero_rms_gives_gain_of_zero(self):
        """Silent stem gets gain=0.0."""
        bar_rms = {"vocals": np.array([0.0, 0.0, 0.0, 0.0])}
        sections = [_make_section_info(start_bar=0, end_bar=4)]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        gains = _compute_section_stem_gains(bar_rms, 0, 4, global_max)
        assert gains["vocals"] == 0.0

    def test_gains_capped_at_one(self):
        """Gains should never exceed 1.0."""
        bar_rms = {"vocals": np.array([1.0, 1.0, 2.0, 2.0])}
        sections = [
            _make_section_info(start_bar=0, end_bar=2),
            _make_section_info(start_bar=2, end_bar=4),
        ]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        # The loud section (bars 2-4) should have gain=1.0, not >1.0
        gains = _compute_section_stem_gains(bar_rms, 2, 4, global_max)
        assert gains["vocals"] == pytest.approx(1.0)

    def test_all_stems_present_in_gains(self):
        """Output should include all 6 stem names."""
        bar_rms = {name: np.array([0.1, 0.1]) for name in STEM_NAMES}
        sections = [_make_section_info(start_bar=0, end_bar=2)]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        gains = _compute_section_stem_gains(bar_rms, 0, 2, global_max)
        for name in STEM_NAMES:
            assert name in gains

    def test_missing_stem_gets_zero_gain(self):
        """Stems not present in bar_rms get gain=0.0."""
        bar_rms = {"vocals": np.array([0.5, 0.5])}
        sections = [_make_section_info(start_bar=0, end_bar=2)]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        gains = _compute_section_stem_gains(bar_rms, 0, 2, global_max)
        assert gains["drums"] == 0.0

    def test_empty_bar_rms_gives_all_zeros(self):
        """Empty arrays produce zero gains."""
        bar_rms = {"vocals": np.array([])}
        sections = [_make_section_info(start_bar=0, end_bar=4)]
        global_max = _compute_global_max_per_stem(bar_rms, sections)

        gains = _compute_section_stem_gains(bar_rms, 0, 4, global_max)
        assert gains["vocals"] == 0.0

    def test_global_max_across_multiple_sections(self):
        """Global max reflects the loudest section across the whole song."""
        bar_rms = {
            "drums": np.array([0.2, 0.2, 0.8, 0.8, 0.4, 0.4]),
        }
        sections = [
            _make_section_info(start_bar=0, end_bar=2),
            _make_section_info(start_bar=2, end_bar=4),
            _make_section_info(start_bar=4, end_bar=6),
        ]
        global_max = _compute_global_max_per_stem(bar_rms, sections)
        assert global_max["drums"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Transition Inference
# ---------------------------------------------------------------------------

class TestTransitionInference:
    """Tests for _infer_transition and related helpers."""

    def test_first_section_gets_fade(self):
        """First section (no predecessor) should get 'fade' transition."""
        sec = _make_section_info(start_bar=0, end_bar=8)
        combined = np.array([0.5] * 16)
        t_type, t_beats = _infer_transition(combined, sec, None)
        assert t_type == "fade"
        assert t_beats == 4

    def test_sharp_increase_gets_cut(self):
        """Energy jump >3dB should produce a 'cut' transition."""
        # Create energy: low in prev section, high in current
        combined = np.array([0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)

        # 0.1 -> 0.5 is 20*log10(0.5/0.1) = 20*log10(5) ~= 14dB, definitely >3dB
        t_type, t_beats = _infer_transition(combined, curr, prev)
        assert t_type == "cut"
        assert t_beats == 2

    def test_gradual_change_gets_crossfade(self):
        """Energy change <3dB should produce a 'crossfade' transition."""
        # 0.45 -> 0.5 is 20*log10(0.5/0.45) ~= 0.9dB
        combined = np.array([0.45, 0.45, 0.45, 0.45, 0.50, 0.50, 0.50, 0.50])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)

        t_type, t_beats = _infer_transition(combined, curr, prev)
        assert t_type == "crossfade"
        assert t_beats == 4

    def test_energy_decrease_gets_crossfade(self):
        """Energy decrease should produce 'crossfade'."""
        combined = np.array([0.8, 0.8, 0.8, 0.8, 0.3, 0.3, 0.3, 0.3])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)

        t_type, t_beats = _infer_transition(combined, curr, prev)
        assert t_type == "crossfade"
        assert t_beats == 4

    def test_empty_combined_energy_gets_crossfade_default(self):
        """Empty combined_energy should default to crossfade."""
        sec = _make_section_info(start_bar=0, end_bar=4)
        prev = _make_section_info(start_bar=0, end_bar=0)
        t_type, t_beats = _infer_transition(np.array([]), sec, prev)
        assert t_type == "crossfade"
        assert t_beats == 4

    def test_all_transition_types_in_vocabulary(self):
        """All transition types produced must be in {fade, crossfade, cut}."""
        valid = {"fade", "crossfade", "cut"}
        # Test various scenarios
        combined = np.array([0.5] * 8)
        sec = _make_section_info(start_bar=4, end_bar=8)
        prev = _make_section_info(start_bar=0, end_bar=4)

        t_type, _ = _infer_transition(combined, sec, None)
        assert t_type in valid

        t_type, _ = _infer_transition(combined, sec, prev)
        assert t_type in valid


class TestDropThenRise:
    """Tests for _detect_drop_then_rise pattern detection."""

    def test_clear_drop_then_rise(self):
        """Energy dips at boundary then rises on both sides."""
        # Bars: 0.8 0.8 | 0.1 0.1 | 0.8 0.8
        # Boundary at bar 3 (between prev ending at 3, curr starting at 3)
        combined = np.array([0.8, 0.8, 0.8, 0.05, 0.05, 0.8, 0.8, 0.8])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)
        assert _detect_drop_then_rise(combined, prev, curr) is True

    def test_no_drop(self):
        """Flat energy should not trigger drop-then-rise."""
        combined = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)
        assert _detect_drop_then_rise(combined, prev, curr) is False

    def test_only_increase_no_drop(self):
        """Steady increase is not a drop-then-rise."""
        combined = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        prev = _make_section_info(start_bar=0, end_bar=4)
        curr = _make_section_info(start_bar=4, end_bar=8)
        assert _detect_drop_then_rise(combined, prev, curr) is False


# ---------------------------------------------------------------------------
# RMS to dB Conversion
# ---------------------------------------------------------------------------

class TestRmsToDb:
    """Tests for _rms_to_db helper."""

    def test_unity_rms_is_zero_db(self):
        assert _rms_to_db(1.0) == pytest.approx(0.0)

    def test_half_rms(self):
        assert _rms_to_db(0.5) == pytest.approx(20 * math.log10(0.5))

    def test_zero_rms_is_neg_inf(self):
        assert _rms_to_db(0.0) == float("-inf")

    def test_negative_rms_is_neg_inf(self):
        assert _rms_to_db(-0.1) == float("-inf")


# ---------------------------------------------------------------------------
# Full Reconstruction (End-to-End)
# ---------------------------------------------------------------------------

class TestReconstructPlan:
    """Tests for reconstruct_plan end-to-end."""

    def test_basic_reconstruction(self):
        """Basic 3-section mashup reconstructs correctly."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_001")

        assert isinstance(plan, RemixPlan)
        assert len(plan.sections) == 3
        assert plan.vocal_source == "song_a"
        assert plan.tempo_source == "average"
        assert plan.key_source == "none"
        assert plan.used_fallback is False
        assert "test_001" not in plan.explanation  # uses title instead
        assert "Test Mashup" in plan.explanation

    def test_section_labels_mapped_correctly(self):
        """Sections are mapped to inference-time vocabulary."""
        sections = [
            {"start_bar": 0, "end_bar": 4, "label": "intro", "energy_level": "low"},
            {"start_bar": 4, "end_bar": 12, "label": "chorus", "energy_level": "high"},
            {"start_bar": 12, "end_bar": 20, "label": "instrumental", "energy_level": "low"},
            {"start_bar": 20, "end_bar": 28, "label": "build", "energy_level": "medium"},
            {"start_bar": 28, "end_bar": 32, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(sections=sections)
        plan = reconstruct_plan(analysis, "test_002")

        labels = [s.label for s in plan.sections]
        assert labels == ["intro", "drop", "breakdown", "verse", "outro"]

    def test_beat_boundaries_from_bar_boundaries(self):
        """Beat boundaries are 4x the bar boundaries."""
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "intro", "energy_level": "low"},
            {"start_bar": 8, "end_bar": 16, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(sections=sections, total_bars=16)
        plan = reconstruct_plan(analysis, "test_003")

        assert plan.sections[0].start_beat == 0
        assert plan.sections[0].end_beat == 32
        assert plan.sections[1].start_beat == 32
        assert plan.sections[1].end_beat == 64

    def test_stem_gains_are_normalized(self):
        """Stem gains are in [0, 1] range."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_004")

        for section in plan.sections:
            for stem_name, gain in section.stem_gains.items():
                assert 0.0 <= gain <= 1.0, f"Gain {gain} for {stem_name} out of range"

    def test_all_stems_have_gains(self):
        """Every section has gains for all 6 stems."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_005")

        for section in plan.sections:
            for name in STEM_NAMES:
                assert name in section.stem_gains

    def test_transition_types_valid(self):
        """All transition types are in the allowed vocabulary."""
        valid_transitions = {"fade", "crossfade", "cut"}
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_006")

        for section in plan.sections:
            assert section.transition_in in valid_transitions

    def test_first_section_has_fade(self):
        """First section should have 'fade' transition."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_007")
        assert plan.sections[0].transition_in == "fade"

    def test_duration_propagated(self):
        """Duration from audio metadata appears in plan time boundaries."""
        analysis = _make_analysis(duration=180.0)
        plan = reconstruct_plan(analysis, "test_008")
        assert plan.end_time_vocal == 180.0
        assert plan.end_time_instrumental == 180.0

    def test_missing_sections_raises_value_error(self):
        """Empty sections list should raise ValueError."""
        analysis = _make_analysis(sections=[])
        with pytest.raises(ValueError, match="No sections"):
            reconstruct_plan(analysis, "test_009")

    def test_missing_audio_metadata_raises_key_error(self):
        """Missing audio_metadata key should raise KeyError."""
        analysis = {"song_structure": {"sections": [{"start_bar": 0, "end_bar": 4, "label": "intro"}]},
                     "stem_analysis": {"bar_rms": {}, "combined_energy": []}}
        with pytest.raises(KeyError):
            reconstruct_plan(analysis, "test_010")

    def test_uses_title_from_analysis(self):
        """Explanation should include the mashup title."""
        analysis = _make_analysis(title="Girl Talk - All Day")
        plan = reconstruct_plan(analysis, "mashup_042")
        assert "Girl Talk - All Day" in plan.explanation

    def test_uses_mashup_id_when_no_title(self):
        """When title is absent, explanation uses mashup_id."""
        analysis = _make_analysis()
        del analysis["title"]
        plan = reconstruct_plan(analysis, "mashup_042")
        assert "mashup_042" in plan.explanation

    def test_single_section(self):
        """Single-section mashup reconstructs without errors."""
        sections = [
            {"start_bar": 0, "end_bar": 16, "label": "verse", "energy_level": "medium"},
        ]
        analysis = _make_analysis(sections=sections, total_bars=16)
        plan = reconstruct_plan(analysis, "test_single")

        assert len(plan.sections) == 1
        assert plan.sections[0].label == "verse"
        assert plan.sections[0].transition_in == "fade"

    def test_instrumental_high_energy_section(self):
        """Instrumental section with high energy maps to 'drop'."""
        sections = [
            {"start_bar": 0, "end_bar": 4, "label": "intro", "energy_level": "low"},
            {"start_bar": 4, "end_bar": 12, "label": "instrumental", "energy_level": "high"},
            {"start_bar": 12, "end_bar": 16, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(sections=sections, total_bars=16)
        plan = reconstruct_plan(analysis, "test_inst_high")
        assert plan.sections[1].label == "drop"

    def test_instrumental_low_energy_section(self):
        """Instrumental section with low energy maps to 'breakdown'."""
        sections = [
            {"start_bar": 0, "end_bar": 4, "label": "intro", "energy_level": "low"},
            {"start_bar": 4, "end_bar": 12, "label": "instrumental", "energy_level": "low"},
            {"start_bar": 12, "end_bar": 16, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(sections=sections, total_bars=16)
        plan = reconstruct_plan(analysis, "test_inst_low")
        assert plan.sections[1].label == "breakdown"


class TestStemGainsVariation:
    """Tests that stem gains vary correctly across sections with differing energy."""

    def test_loud_section_has_higher_gains(self):
        """Louder section (higher RMS) should have higher stem gains."""
        bar_rms_data = {
            "vocals": [0.1] * 8 + [0.8] * 8,
            "drums": [0.2] * 8 + [0.9] * 8,
            "bass": [0.1] * 16,
            "guitar": [0.1] * 16,
            "piano": [0.1] * 16,
            "other": [0.1] * 16,
        }
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "verse", "energy_level": "low"},
            {"start_bar": 8, "end_bar": 16, "label": "chorus", "energy_level": "high"},
        ]
        analysis = _make_analysis(
            sections=sections,
            total_bars=16,
            bar_rms=bar_rms_data,
            combined_energy=[0.3] * 8 + [0.9] * 8,
        )
        plan = reconstruct_plan(analysis, "test_gains")

        # Second section (chorus/drop) should have gain=1.0 for vocals
        assert plan.sections[1].stem_gains["vocals"] == pytest.approx(1.0)
        # First section should have lower gain
        assert plan.sections[0].stem_gains["vocals"] < plan.sections[1].stem_gains["vocals"]


class TestTransitionInContext:
    """Tests for transition inference in the context of full reconstruction."""

    def test_sharp_energy_jump_produces_cut(self):
        """A sharp energy increase between sections should produce a cut."""
        # Low energy in first section, high in second
        combined = [0.05] * 8 + [0.8] * 8 + [0.05] * 8
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "intro", "energy_level": "low"},
            {"start_bar": 8, "end_bar": 16, "label": "chorus", "energy_level": "peak"},
            {"start_bar": 16, "end_bar": 24, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(
            sections=sections,
            total_bars=24,
            combined_energy=combined,
        )
        plan = reconstruct_plan(analysis, "test_cut")
        # Second section should have a cut (sharp increase from 0.05 to 0.8)
        assert plan.sections[1].transition_in == "cut"
        assert plan.sections[1].transition_beats == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for _serialize_plan."""

    def test_round_trip(self):
        """Serialized plan contains all required fields."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_serial")
        serialized = _serialize_plan(plan)

        assert serialized["vocal_source"] == "song_a"
        assert serialized["tempo_source"] == "average"
        assert serialized["key_source"] == "none"
        assert isinstance(serialized["sections"], list)
        assert len(serialized["sections"]) == 3

        for sec in serialized["sections"]:
            assert "label" in sec
            assert "start_beat" in sec
            assert "end_beat" in sec
            assert "stem_gains" in sec
            assert "transition_in" in sec
            assert "transition_beats" in sec

    def test_json_serializable(self):
        """Serialized plan can be round-tripped through JSON."""
        analysis = _make_analysis()
        plan = reconstruct_plan(analysis, "test_json")
        serialized = _serialize_plan(plan)

        json_str = json.dumps(serialized)
        loaded = json.loads(json_str)
        assert loaded["vocal_source"] == "song_a"
        assert len(loaded["sections"]) == 3


# ---------------------------------------------------------------------------
# Batch Reconstruction
# ---------------------------------------------------------------------------

class TestReconstructAll:
    """Tests for reconstruct_all batch processing."""

    def test_processes_all_files(self, tmp_path: Path):
        """reconstruct_all processes all JSON files in the directory."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans"

        # Write 3 analysis files
        for i in range(3):
            analysis = _make_analysis(title=f"Mashup {i}")
            path = analysis_dir / f"mashup_{i:03d}.json"
            path.write_text(json.dumps(analysis))

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 3
        assert output_dir.exists()

        for path in written:
            assert path.exists()
            data = json.loads(path.read_text())
            assert "sections" in data
            assert data["vocal_source"] == "song_a"

    def test_skips_invalid_files(self, tmp_path: Path):
        """Invalid JSON files are skipped with error logging."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans"

        # One valid, one invalid
        valid = _make_analysis(title="Good Mashup")
        (analysis_dir / "good.json").write_text(json.dumps(valid))
        (analysis_dir / "bad.json").write_text("not valid json{{{")

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 1
        assert written[0].name == "good.json"

    def test_skips_missing_sections(self, tmp_path: Path):
        """Analysis files with empty sections are skipped."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans"

        analysis = _make_analysis(sections=[])
        (analysis_dir / "empty.json").write_text(json.dumps(analysis))

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 0

    def test_empty_directory(self, tmp_path: Path):
        """Empty analysis directory produces no output."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans"

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 0

    def test_creates_output_dir(self, tmp_path: Path):
        """Output directory is created if it doesn't exist."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans" / "nested"

        analysis = _make_analysis()
        (analysis_dir / "test.json").write_text(json.dumps(analysis))

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 1
        assert output_dir.exists()

    def test_output_filenames_match_input(self, tmp_path: Path):
        """Output files have the same stem as input files."""
        analysis_dir = tmp_path / "analysis"
        analysis_dir.mkdir()
        output_dir = tmp_path / "plans"

        analysis = _make_analysis()
        (analysis_dir / "mashup_042.json").write_text(json.dumps(analysis))

        written = reconstruct_all(analysis_dir, output_dir)
        assert len(written) == 1
        assert written[0].name == "mashup_042.json"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_section_at_end_of_bar_rms(self):
        """Section boundaries at the exact end of bar_rms array."""
        bar_rms_data = {name: [0.5] * 8 for name in STEM_NAMES}
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "verse", "energy_level": "medium"},
        ]
        analysis = _make_analysis(
            sections=sections,
            total_bars=8,
            bar_rms=bar_rms_data,
            combined_energy=[0.5] * 8,
        )
        plan = reconstruct_plan(analysis, "test_edge")
        assert len(plan.sections) == 1

    def test_section_bar_rms_shorter_than_sections(self):
        """bar_rms shorter than section boundaries should still work."""
        bar_rms_data = {name: [0.5] * 4 for name in STEM_NAMES}
        sections = [
            {"start_bar": 0, "end_bar": 4, "label": "intro", "energy_level": "low"},
            {"start_bar": 4, "end_bar": 8, "label": "outro", "energy_level": "low"},
        ]
        analysis = _make_analysis(
            sections=sections,
            total_bars=8,
            bar_rms=bar_rms_data,
            combined_energy=[0.5] * 4,
        )
        # Should not crash, but sections beyond bar_rms get zero gains
        plan = reconstruct_plan(analysis, "test_short_rms")
        assert len(plan.sections) == 2

    def test_very_many_sections(self):
        """Handles many small sections without errors."""
        n_bars = 64
        sections = [
            {"start_bar": i * 4, "end_bar": (i + 1) * 4, "label": "verse", "energy_level": "medium"}
            for i in range(n_bars // 4)
        ]
        analysis = _make_analysis(
            sections=sections,
            total_bars=n_bars,
            bar_rms={name: [0.5] * n_bars for name in STEM_NAMES},
            combined_energy=[0.5] * n_bars,
        )
        plan = reconstruct_plan(analysis, "test_many")
        assert len(plan.sections) == n_bars // 4

    def test_default_energy_level_when_missing(self):
        """Missing energy_level defaults to 'medium'."""
        sections = [
            {"start_bar": 0, "end_bar": 8, "label": "verse"},  # no energy_level
        ]
        analysis = _make_analysis(sections=sections, total_bars=8)
        plan = reconstruct_plan(analysis, "test_default")
        assert plan.sections[0].label == "verse"

    def test_optional_fields_default_gracefully(self):
        """Sections with minimal fields still reconstruct."""
        sections = [
            {"start_bar": 0, "end_bar": 16, "label": "chorus", "energy_level": "high"},
        ]
        analysis = _make_analysis(sections=sections, total_bars=16)
        plan = reconstruct_plan(analysis, "test_minimal")
        assert plan.sections[0].label == "drop"  # chorus -> drop

    def test_zero_duration(self):
        """Zero duration doesn't crash."""
        analysis = _make_analysis(duration=0.0)
        plan = reconstruct_plan(analysis, "test_zero_dur")
        assert plan.end_time_vocal == 0.0
