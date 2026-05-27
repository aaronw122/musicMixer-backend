"""Tests for musicmixer.services.taste_constraints — hard constraint validation.

Each constraint is tested individually with a plan that violates only that
constraint, plus integration tests for valid plans, multiple violations,
and audio-dependent constraint skipping.
"""

from __future__ import annotations

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_constraints import (
    ConstraintCode,
    ConstraintViolation,
    check_beat_grid_alignment,
    check_contiguous_sections,
    check_contrast_requirement,
    check_lra_floor,
    check_lufs_window,
    check_mvp_source_split,
    check_no_dual_lead_vocals,
    check_outro_quality,
    check_section_min_length,
    check_stem_quality_gate,
    check_tempo_stretch_safety,
    check_transition_bounds,
    check_true_peak_ceiling,
    validate_candidate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(
    label: str = "main",
    start_beat: int = 0,
    end_beat: int = 16,
    stem_gains: dict[str, float] | None = None,
    transition_in: str = "crossfade",
    transition_beats: int = 4,
) -> Section:
    """Create a Section with sensible defaults."""
    if stem_gains is None:
        stem_gains = {"vocals": 1.0, "drums": 0.8, "bass": 0.7}
    return Section(
        label=label,
        start_beat=start_beat,
        end_beat=end_beat,
        stem_gains=stem_gains,
        transition_in=transition_in,
        transition_beats=transition_beats,
    )


def _make_valid_plan(
    sections: list[Section] | None = None,
    vocal_source: str = "song_a",
    tempo_source: str = "song_a",
) -> RemixPlan:
    """Create a valid RemixPlan that passes all plan-level constraints.

    Default plan: intro(0-16) -> build(16-32) -> breakdown(32-48) ->
                  main(48-80) -> outro(80-96)

    The breakdown before main provides the required contrast event
    (stem count drop >= 2 or energy drop >= 20%).
    """
    if sections is None:
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 0.0, "drums": 0.5, "bass": 0.3},
                          transition_beats=4),
            _make_section("build", 16, 32,
                          {"vocals": 0.5, "drums": 0.7, "bass": 0.5},
                          transition_beats=4),
            _make_section("breakdown", 32, 48,
                          {"vocals": 0.0, "drums": 0.0, "bass": 0.2},
                          transition_beats=4),
            _make_section("main", 48, 80,
                          {"vocals": 1.0, "drums": 1.0, "bass": 0.8},
                          transition_beats=4),
            _make_section("outro", 80, 96,
                          {"vocals": 0.0, "drums": 0.3, "bass": 0.2},
                          transition_beats=4),
        ]
    return RemixPlan(
        vocal_source=vocal_source,
        start_time_vocal=0.0,
        end_time_vocal=60.0,
        start_time_instrumental=0.0,
        end_time_instrumental=60.0,
        sections=sections,
        tempo_source=tempo_source,
        explanation="Test plan",
    )


def _make_audio_metadata(bpm: float = 120.0, key: str | None = None) -> AudioMetadata:
    """Create minimal AudioMetadata for testing."""
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=np.array([0, 100, 200]),
        duration_seconds=180.0,
        total_beats=360,
        key=key,
    )


# ---------------------------------------------------------------------------
# Constraint 1: Contiguous non-overlapping sections
# ---------------------------------------------------------------------------


class TestContiguousSections:

    def test_valid_contiguous_sections(self):
        plan = _make_valid_plan()
        violations = check_contiguous_sections(plan)
        assert violations == []

    def test_empty_sections(self):
        plan = _make_valid_plan(sections=[])
        violations = check_contiguous_sections(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.SECTION_OVERLAP

    def test_first_section_not_at_zero(self):
        sections = [
            _make_section("intro", 4, 16, transition_beats=4),
            _make_section("outro", 16, 32, {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contiguous_sections(plan)
        assert any(v.code == ConstraintCode.SECTION_OVERLAP for v in violations)
        assert any("starts at beat 4" in v.message for v in violations)

    def test_gap_between_sections(self):
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("outro", 20, 32, {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contiguous_sections(plan)
        assert any(v.code == ConstraintCode.SECTION_OVERLAP for v in violations)
        assert any("Gap/overlap" in v.message for v in violations)

    def test_overlapping_sections(self):
        sections = [
            _make_section("intro", 0, 20, transition_beats=4),
            _make_section("outro", 16, 32, {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contiguous_sections(plan)
        assert any(v.code == ConstraintCode.SECTION_OVERLAP for v in violations)

    def test_end_before_start(self):
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("main", 16, 12, transition_beats=2),  # end < start
            _make_section("outro", 12, 24, {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contiguous_sections(plan)
        assert any(v.code == ConstraintCode.SECTION_OVERLAP for v in violations)


# ---------------------------------------------------------------------------
# Constraint 2: Section minimum length
# ---------------------------------------------------------------------------


class TestSectionMinLength:

    def test_valid_section_lengths(self):
        plan = _make_valid_plan()
        violations = check_section_min_length(plan)
        assert violations == []

    def test_major_section_too_short(self):
        """Major section (intro) with only 4 beats should fail (needs >= 8)."""
        sections = [
            _make_section("intro", 0, 4, transition_beats=2),
            _make_section("main", 4, 20, transition_beats=4),
            _make_section("outro", 20, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_section_min_length(plan)
        assert len(violations) >= 1
        assert violations[0].code == ConstraintCode.SECTION_TOO_SHORT
        assert violations[0].section_index == 0

    def test_minor_section_at_minimum(self):
        """A non-major section at exactly 4 beats should pass."""
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("fill", 16, 20, transition_beats=2),  # 4 beats, non-major
            _make_section("main", 20, 36, transition_beats=4),
            _make_section("outro", 36, 48,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_section_min_length(plan)
        # The only violations should NOT be for the 'fill' section
        fill_violations = [v for v in violations if v.section_index == 1]
        assert fill_violations == []

    def test_minor_section_below_minimum(self):
        """A non-major section below 4 beats should fail."""
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("fill", 16, 18, transition_beats=1),  # 2 beats
            _make_section("main", 18, 34, transition_beats=4),
            _make_section("outro", 34, 48,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_section_min_length(plan)
        fill_violations = [v for v in violations if v.section_index == 1]
        assert len(fill_violations) == 1
        assert fill_violations[0].code == ConstraintCode.SECTION_TOO_SHORT

    def test_major_section_at_eight_beats_passes(self):
        """A major section at exactly 8 beats should pass."""
        sections = [
            _make_section("intro", 0, 8, transition_beats=4),
            _make_section("main", 8, 24, transition_beats=4),
            _make_section("outro", 24, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_section_min_length(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 3: Beat grid alignment
# ---------------------------------------------------------------------------


class TestBeatGridAlignment:

    def test_valid_grid_alignment(self):
        plan = _make_valid_plan()
        violations = check_beat_grid_alignment(plan)
        assert violations == []

    def test_start_beat_off_grid(self):
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("main", 17, 32, transition_beats=4),  # start off grid
            _make_section("outro", 32, 48,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_beat_grid_alignment(plan)
        assert any(v.code == ConstraintCode.BEAT_GRID_MISALIGN for v in violations)
        assert any("start_beat 17" in v.message for v in violations)

    def test_end_beat_off_grid(self):
        sections = [
            _make_section("intro", 0, 14, transition_beats=4),  # end off grid
            _make_section("outro", 14, 28,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_beat_grid_alignment(plan)
        assert any(v.code == ConstraintCode.BEAT_GRID_MISALIGN for v in violations)

    def test_all_on_grid(self):
        sections = [
            _make_section("intro", 0, 8, transition_beats=4),
            _make_section("main", 8, 24, transition_beats=4),
            _make_section("outro", 24, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_beat_grid_alignment(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 4: MVP source split
# ---------------------------------------------------------------------------


class TestMvpSourceSplit:

    def test_valid_song_a(self):
        plan = _make_valid_plan(vocal_source="song_a")
        violations = check_mvp_source_split(plan)
        assert violations == []

    def test_valid_song_b(self):
        plan = _make_valid_plan(vocal_source="song_b")
        violations = check_mvp_source_split(plan)
        assert violations == []

    def test_invalid_source(self):
        plan = _make_valid_plan(vocal_source="both")
        violations = check_mvp_source_split(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.MVP_SOURCE_SPLIT


# ---------------------------------------------------------------------------
# Constraint 5: Tempo stretch safety
# ---------------------------------------------------------------------------


class TestTempoStretchSafety:

    def test_safe_stretch(self):
        """10% stretch is safe for drums and vocals."""
        plan = _make_valid_plan(vocal_source="song_a", tempo_source="song_b")
        meta_a = _make_audio_metadata(bpm=120.0)
        meta_b = _make_audio_metadata(bpm=130.0)
        violations = check_tempo_stretch_safety(plan, meta_a, meta_b)
        # Vocal stretch: |130 - 120| / 120 = 8.3% (safe)
        # Drum stretch: |130 - 130| / 130 = 0% (safe, inst is song_b at target)
        assert violations == []

    def test_drum_stretch_exceeds_limit(self):
        """Drum stretch > 12% should fail."""
        plan = _make_valid_plan(vocal_source="song_a", tempo_source="song_a")
        meta_a = _make_audio_metadata(bpm=120.0)
        meta_b = _make_audio_metadata(bpm=150.0)  # inst stretch = |120-150|/150 = 20%
        violations = check_tempo_stretch_safety(plan, meta_a, meta_b)
        assert any(
            v.code == ConstraintCode.TEMPO_STRETCH_UNSAFE and "Drum" in v.message
            for v in violations
        )

    def test_vocal_stretch_exceeds_limit(self):
        """Vocal stretch > 35% should fail."""
        plan = _make_valid_plan(vocal_source="song_a", tempo_source="song_b")
        meta_a = _make_audio_metadata(bpm=80.0)
        meta_b = _make_audio_metadata(bpm=120.0)
        # Vocal stretch: |120 - 80| / 80 = 50% (exceeds 35%)
        violations = check_tempo_stretch_safety(plan, meta_a, meta_b)
        assert any(
            v.code == ConstraintCode.TEMPO_STRETCH_UNSAFE and "Vocal" in v.message
            for v in violations
        )

    def test_skips_when_meta_none(self):
        """Should skip validation when metadata is None."""
        plan = _make_valid_plan()
        violations = check_tempo_stretch_safety(plan, None, None)
        assert violations == []

    def test_skips_when_one_meta_none(self):
        """Should skip when only one metadata is available."""
        plan = _make_valid_plan()
        meta_a = _make_audio_metadata(bpm=120.0)
        violations = check_tempo_stretch_safety(plan, meta_a, None)
        assert violations == []

    def test_average_tempo_source(self):
        """Average tempo source should use mean BPM."""
        plan = _make_valid_plan(
            vocal_source="song_a", tempo_source="average"
        )
        meta_a = _make_audio_metadata(bpm=100.0)
        meta_b = _make_audio_metadata(bpm=114.0)
        # Target = 107, vocal stretch = |107-100|/100 = 7%, drum stretch = |107-114|/114 = 6.1%
        violations = check_tempo_stretch_safety(plan, meta_a, meta_b)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 7: Transition bounds
# ---------------------------------------------------------------------------


class TestTransitionBounds:

    def test_valid_transitions(self):
        plan = _make_valid_plan()
        violations = check_transition_bounds(plan)
        assert violations == []

    def test_transition_exceeds_half_section(self):
        """transition_beats=12 on a 16-beat section (half=8) should fail."""
        sections = [
            _make_section("intro", 0, 16, transition_beats=12),  # 12 > 8
            _make_section("main", 16, 32, transition_beats=4),
            _make_section("outro", 32, 48,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_transition_bounds(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.TRANSITION_TOO_LONG
        assert violations[0].section_index == 0

    def test_transition_at_half_passes(self):
        """transition_beats exactly == half section length should pass."""
        sections = [
            _make_section("intro", 0, 16, transition_beats=8),  # 8 == 16/2
            _make_section("main", 16, 32, transition_beats=4),
            _make_section("outro", 32, 48,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_transition_bounds(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 8: True peak ceiling
# ---------------------------------------------------------------------------


class TestTruePeakCeiling:

    def test_within_ceiling(self):
        violations = check_true_peak_ceiling(-2.0)
        assert violations == []

    def test_at_ceiling(self):
        violations = check_true_peak_ceiling(-1.0)
        assert violations == []

    def test_exceeds_ceiling(self):
        violations = check_true_peak_ceiling(-0.5)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.TRUE_PEAK_EXCEEDED

    def test_skips_when_none(self):
        violations = check_true_peak_ceiling(None)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 9: LUFS window
# ---------------------------------------------------------------------------


class TestLufsWindow:

    def test_default_genre_in_range(self):
        violations = check_lufs_window(-12.0, genre=None)
        assert violations == []

    def test_default_genre_out_of_range(self):
        violations = check_lufs_window(-8.0, genre=None)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.LUFS_OUT_OF_RANGE

    def test_lofi_in_range(self):
        violations = check_lufs_window(-14.0, genre="lo-fi")
        assert violations == []

    def test_lofi_too_loud(self):
        violations = check_lufs_window(-10.0, genre="lo-fi")
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.LUFS_OUT_OF_RANGE

    def test_edm_in_range(self):
        violations = check_lufs_window(-10.0, genre="EDM")
        assert violations == []

    def test_edm_too_quiet(self):
        violations = check_lufs_window(-14.0, genre="EDM")
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.LUFS_OUT_OF_RANGE

    def test_skips_when_none(self):
        violations = check_lufs_window(None, genre="EDM")
        assert violations == []

    def test_electronic_genre_alias(self):
        """'electronic' should use EDM range."""
        violations = check_lufs_window(-10.0, genre="electronic")
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 10: LRA floor
# ---------------------------------------------------------------------------


class TestLraFloor:

    def test_above_floor(self):
        violations = check_lra_floor(6.0)
        assert violations == []

    def test_at_floor(self):
        violations = check_lra_floor(4.0)
        assert violations == []

    def test_below_floor(self):
        violations = check_lra_floor(3.0)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.LRA_TOO_LOW

    def test_skips_when_none(self):
        violations = check_lra_floor(None)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 11: Contrast requirement
# ---------------------------------------------------------------------------


class TestContrastRequirement:

    def test_valid_contrast_via_stem_drop(self):
        """Stem count dropping by 2 before peak is valid contrast."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0, "drums": 0.8, "bass": 0.7},
                          transition_beats=4),
            _make_section("breakdown", 16, 32,
                          {"vocals": 0.0, "drums": 0.0, "bass": 0.5},
                          transition_beats=4),  # stem drop of 2 (vocals, drums)
            _make_section("main", 32, 48,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),  # peak
            _make_section("outro", 48, 64,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contrast_requirement(plan)
        assert violations == []

    def test_valid_contrast_via_energy_drop(self):
        """Energy dropping by >= 20% before peak is valid contrast."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),  # total = 3.0
            _make_section("breakdown", 16, 32,
                          {"vocals": 0.5, "drums": 0.5, "bass": 0.5},
                          transition_beats=4),  # total = 1.5 -> 50% drop
            _make_section("main", 32, 48,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.2},
                          transition_beats=4),  # peak total = 3.2
            _make_section("outro", 48, 64,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contrast_requirement(plan)
        assert violations == []

    def test_no_contrast_fails(self):
        """Steadily increasing energy with no drops should fail."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 0.5, "drums": 0.5, "bass": 0.5},
                          transition_beats=4),
            _make_section("build", 16, 32,
                          {"vocals": 0.7, "drums": 0.7, "bass": 0.7},
                          transition_beats=4),
            _make_section("main", 32, 48,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),
            _make_section("outro", 48, 64,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contrast_requirement(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.NO_CONTRAST

    def test_peak_is_first_section_fails(self):
        """If peak is the first section, no contrast is possible."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contrast_requirement(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.NO_CONTRAST

    def test_single_section_passes(self):
        """A plan with a single section vacuously passes."""
        sections = [
            _make_section("outro", 0, 16,
                          {"vocals": 0.0, "drums": 0.3, "bass": 0.2},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_contrast_requirement(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 12: No dual lead vocals
# ---------------------------------------------------------------------------


class TestNoDualLeadVocals:

    def test_single_vocal_stem_passes(self):
        plan = _make_valid_plan()
        violations = check_no_dual_lead_vocals(plan)
        assert violations == []

    def test_cross_song_vocals_fails(self):
        """Song A lead_vocals + Song B vocals active should fail."""
        sections = [
            _make_section("intro", 0, 16,
                          {"lead_vocals": 1.0, "vocals": 0.8, "drums": 0.5},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"lead_vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_no_dual_lead_vocals(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.DUAL_LEAD_VOCALS
        assert violations[0].section_index == 0

    def test_same_song_vocal_layering_allowed(self):
        """Song A lead_vocals + backing_vocals active together is allowed."""
        sections = [
            _make_section("chorus", 0, 16,
                          {"lead_vocals": 1.0, "backing_vocals": 0.6, "drums": 0.5},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"lead_vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_no_dual_lead_vocals(plan)
        assert violations == []

    def test_zero_gain_vocal_ignored(self):
        """A vocal stem with gain=0 should not count as active."""
        sections = [
            _make_section("intro", 0, 16,
                          {"lead_vocals": 1.0, "vocals": 0.0, "drums": 0.5},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"lead_vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_no_dual_lead_vocals(plan)
        assert violations == []


# ---------------------------------------------------------------------------
# Constraint 13: Outro quality
# ---------------------------------------------------------------------------


class TestOutroQuality:

    def test_valid_outro(self):
        plan = _make_valid_plan()
        violations = check_outro_quality(plan)
        assert violations == []

    def test_final_not_labeled_outro(self):
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("main", 16, 32,
                          {"vocals": 1.0, "drums": 1.0, "bass": 0.8},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_outro_quality(plan)
        assert any(
            v.code == ConstraintCode.OUTRO_QUALITY and "expected 'outro'" in v.message
            for v in violations
        )

    def test_outro_too_short(self):
        sections = [
            _make_section("intro", 0, 16, transition_beats=4),
            _make_section("main", 16, 32,
                          {"vocals": 1.0, "drums": 1.0, "bass": 0.8},
                          transition_beats=4),
            _make_section("outro", 32, 36,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=2),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_outro_quality(plan)
        assert any(
            v.code == ConstraintCode.OUTRO_QUALITY and "4 beats" in v.message
            for v in violations
        )

    def test_outro_energy_not_below_peak(self):
        """Outro with same energy as peak section should fail."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_outro_quality(plan)
        assert any(
            v.code == ConstraintCode.OUTRO_QUALITY and "not below peak" in v.message
            for v in violations
        )

    def test_empty_sections(self):
        plan = _make_valid_plan(sections=[])
        violations = check_outro_quality(plan)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.OUTRO_QUALITY


# ---------------------------------------------------------------------------
# Constraint 14: Stem quality gate
# ---------------------------------------------------------------------------


class TestStemQualityGate:

    def test_good_quality_allows_solo_vocal(self):
        """High stem quality should allow solo vocal sections."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_stem_quality_gate(plan, stem_quality=0.85)
        assert violations == []

    def test_low_quality_blocks_solo_vocal(self):
        """Low stem quality should block solo vocal sections."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_stem_quality_gate(plan, stem_quality=0.5)
        assert len(violations) == 1
        assert violations[0].code == ConstraintCode.STEM_QUALITY_GATE
        assert violations[0].section_index == 0

    def test_low_quality_allows_multi_stem(self):
        """Low quality should be fine when vocals aren't solo."""
        plan = _make_valid_plan()  # all sections have multiple stems
        violations = check_stem_quality_gate(plan, stem_quality=0.5)
        assert violations == []

    def test_skips_when_quality_none(self):
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_stem_quality_gate(plan, stem_quality=None)
        assert violations == []

    def test_at_threshold_passes(self):
        """Quality exactly at 0.7 should pass."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0},
                          transition_beats=4),
            _make_section("outro", 16, 32,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        violations = check_stem_quality_gate(plan, stem_quality=0.7)
        assert violations == []


# ---------------------------------------------------------------------------
# Integration: validate_candidate
# ---------------------------------------------------------------------------


class TestValidateCandidate:

    def test_valid_plan_passes(self):
        """A well-formed plan should pass all constraints."""
        plan = _make_valid_plan()
        passed, violations = validate_candidate(plan)
        assert passed is True
        assert violations == []

    def test_valid_plan_with_audio_data_passes(self):
        """A valid plan with compatible audio data passes."""
        plan = _make_valid_plan(
            vocal_source="song_a",
            tempo_source="song_a",
        )
        meta_a = _make_audio_metadata(bpm=120.0, key="C")
        meta_b = _make_audio_metadata(bpm=125.0, key="D")
        passed, violations = validate_candidate(
            plan, meta_a=meta_a, meta_b=meta_b,
            genre="rock", stem_quality=0.8,
            true_peak_dbtp=-2.0, lufs=-12.0, lra=6.0,
        )
        assert passed is True
        assert violations == []

    def test_audio_dependent_constraints_skipped_when_none(self):
        """Audio-dependent constraints should be skipped when data is None."""
        plan = _make_valid_plan()
        passed, violations = validate_candidate(
            plan,
            meta_a=None, meta_b=None,
            genre=None, stem_quality=None,
            true_peak_dbtp=None, lufs=None, lra=None,
        )
        assert passed is True
        assert violations == []

    def test_multiple_violations_collected(self):
        """All violations should be collected, not short-circuited."""
        sections = [
            # Section starts at beat 2 (not on grid, not at 0)
            _make_section("intro", 2, 6,
                          {"vocals": 1.0, "vocals_b": 1.0, "drums": 0.5},
                          transition_beats=4),
            # Main section: too short, off grid
            _make_section("main", 7, 11,
                          {"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                          transition_beats=3),
        ]
        plan = _make_valid_plan(sections=sections, vocal_source="both")
        passed, violations = validate_candidate(plan)

        assert passed is False
        # Should have violations from multiple constraints
        codes = {v.code for v in violations}
        assert len(codes) >= 3, (
            f"Expected >= 3 different violation types, got {len(codes)}: {codes}"
        )

    def test_only_audio_constraints_fail(self):
        """A valid plan that only fails audio constraints."""
        plan = _make_valid_plan()
        passed, violations = validate_candidate(
            plan,
            true_peak_dbtp=0.0,   # exceeds -1.0 dBTP
            lufs=-5.0,            # too loud for default
            lra=2.0,              # below 4.0 dB floor
        )
        assert passed is False
        codes = {v.code for v in violations}
        assert ConstraintCode.TRUE_PEAK_EXCEEDED in codes
        assert ConstraintCode.LUFS_OUT_OF_RANGE in codes
        assert ConstraintCode.LRA_TOO_LOW in codes

    def test_stem_quality_gate_in_validate(self):
        """Stem quality gate should fire through validate_candidate."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 1.0},
                          transition_beats=4),
            _make_section("main", 16, 32,
                          {"vocals": 1.0, "drums": 1.0, "bass": 0.8},
                          transition_beats=4),
            _make_section("breakdown", 32, 48,
                          {"vocals": 0.3, "drums": 0.0, "bass": 0.2},
                          transition_beats=4),
            _make_section("outro", 48, 64,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        passed, violations = validate_candidate(plan, stem_quality=0.5)
        # Should catch the solo vocal section 0
        assert not passed
        assert any(v.code == ConstraintCode.STEM_QUALITY_GATE for v in violations)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_single_outro_section_plan(self):
        """A plan with only an outro section should pass structural constraints."""
        sections = [
            _make_section("outro", 0, 16,
                          {"vocals": 0.0, "drums": 0.3, "bass": 0.2},
                          transition_beats=4),
        ]
        plan = _make_valid_plan(sections=sections)
        passed, violations = validate_candidate(plan)
        # Should pass: contiguous, on grid, long enough, outro at end
        assert passed is True

    def test_many_sections(self):
        """A plan with many sections should still validate correctly."""
        sections = []
        for i in range(10):
            label = "outro" if i == 9 else "main"
            gains = {"vocals": 0.0, "drums": 0.2, "bass": 0.1} if i == 9 else {
                "vocals": float(i % 2), "drums": 0.8, "bass": 0.7
            }
            sections.append(_make_section(
                label, i * 16, (i + 1) * 16,
                stem_gains=gains,
                transition_beats=4,
            ))
        plan = _make_valid_plan(sections=sections)
        passed, violations = validate_candidate(plan)
        # Should pass if there's proper contrast before peak
        # Sections alternate vocals on/off, so stem count varies
        # which should provide contrast
        if not passed:
            # The only possible failure is contrast if the energy pattern
            # doesn't include a drop. Let's verify which codes failed.
            non_contrast = [v for v in violations if v.code != ConstraintCode.NO_CONTRAST]
            assert non_contrast == [], f"Unexpected violations: {non_contrast}"
