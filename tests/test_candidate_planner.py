"""Tests for musicmixer.services.candidate_planner -- candidate plan generation."""

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.candidate_planner import (
    _apply_gain_delta,
    _compute_total_beats,
    _deduplicate,
    _ensure_boundaries,
    _get_family_name,
    _snap_to_grid,
    _structure_hash,
    generate_candidates,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_metadata(bpm: float = 120.0, duration: float = 240.0) -> AudioMetadata:
    """Create a synthetic AudioMetadata for testing."""
    beat_interval_frames = int(60 / bpm * 44100 / 512)
    num_beats = int(bpm * duration / 60)
    beat_frames = np.arange(0, num_beats) * beat_interval_frames
    total_beats = round(bpm * duration / 60 / 4) * 4
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=total_beats,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestSnapToGrid:
    """Tests for _snap_to_grid."""

    def test_exact_multiple(self):
        assert _snap_to_grid(16) == 16
        assert _snap_to_grid(32) == 32

    def test_rounds_down(self):
        assert _snap_to_grid(17) == 16
        assert _snap_to_grid(19) == 16
        assert _snap_to_grid(31) == 28

    def test_zero(self):
        assert _snap_to_grid(0) == 0

    def test_negative_clamps_to_zero(self):
        assert _snap_to_grid(-5) == 0

    def test_custom_grid(self):
        assert _snap_to_grid(17, grid=8) == 16
        assert _snap_to_grid(23, grid=8) == 16
        assert _snap_to_grid(24, grid=8) == 24


class TestEnsureBoundaries:
    """Tests for _ensure_boundaries."""

    def test_already_valid(self):
        result = _ensure_boundaries([16, 32, 48, 64], 80)
        assert result == [16, 32, 48, 64]

    def test_fixes_non_monotonic(self):
        result = _ensure_boundaries([16, 16, 48, 64], 80)
        # Second element should be pushed past first
        assert result[0] < result[1] < result[2] < result[3]

    def test_respects_total_beats(self):
        result = _ensure_boundaries([16, 32, 48, 64], 68)
        # Last boundary must leave room for final section
        assert result[-1] < 68

    def test_all_boundaries_grid_aligned(self):
        result = _ensure_boundaries([10, 25, 47, 63], 80)
        for b in result:
            assert b % 4 == 0, f"Boundary {b} not on 4-beat grid"

    def test_empty_list(self):
        result = _ensure_boundaries([], 80)
        assert result == []


class TestApplyGainDelta:
    """Tests for _apply_gain_delta."""

    def test_zero_delta_preserves_gains(self):
        base = {"vocals": 0.8, "drums": 0.7, "bass": 0.6}
        result = _apply_gain_delta(base, 0.0)
        for key in base:
            assert abs(result[key] - base[key]) < 0.01

    def test_positive_delta_boosts_vocals(self):
        base = {"vocals": 0.6, "drums": 0.7, "bass": 0.6}
        result = _apply_gain_delta(base, 6.0)
        assert result["vocals"] > base["vocals"]

    def test_negative_delta_reduces_vocals(self):
        base = {"vocals": 0.8, "drums": 0.7, "bass": 0.6}
        result = _apply_gain_delta(base, -6.0)
        assert result["vocals"] < base["vocals"]

    def test_gains_clamped_to_0_1(self):
        base = {"vocals": 0.9, "drums": 0.1, "bass": 0.1}
        result = _apply_gain_delta(base, 20.0)
        assert result["vocals"] <= 1.0
        assert all(v >= 0.0 for v in result.values())


class TestStructureHash:
    """Tests for _structure_hash."""

    def test_identical_plans_same_hash(self):
        meta_a = _make_metadata()
        meta_b = _make_metadata()
        sections = [
            Section(label="intro", start_beat=0, end_beat=16,
                    stem_gains={"vocals": 0.0, "drums": 0.6},
                    transition_in="fade", transition_beats=4),
            Section(label="main", start_beat=16, end_beat=80,
                    stem_gains={"vocals": 1.0, "drums": 0.7},
                    transition_in="crossfade", transition_beats=4),
        ]
        plan1 = RemixPlan(
            vocal_source="song_a", start_time_vocal=0, end_time_vocal=60,
            start_time_instrumental=0, end_time_instrumental=60,
            sections=sections, tempo_source="average", key_source="none",
            explanation="test", used_fallback=False,
        )
        plan2 = RemixPlan(
            vocal_source="song_a", start_time_vocal=0, end_time_vocal=60,
            start_time_instrumental=0, end_time_instrumental=60,
            sections=sections, tempo_source="average", key_source="none",
            explanation="different explanation", used_fallback=False,
        )
        assert _structure_hash(plan1) == _structure_hash(plan2)

    def test_different_sections_different_hash(self):
        sections_a = [
            Section(label="intro", start_beat=0, end_beat=16,
                    stem_gains={"vocals": 0.0}, transition_in="fade",
                    transition_beats=4),
        ]
        sections_b = [
            Section(label="intro", start_beat=0, end_beat=32,
                    stem_gains={"vocals": 0.0}, transition_in="fade",
                    transition_beats=4),
        ]
        plan_a = RemixPlan(
            vocal_source="song_a", start_time_vocal=0, end_time_vocal=60,
            start_time_instrumental=0, end_time_instrumental=60,
            sections=sections_a, tempo_source="average", key_source="none",
            explanation="test", used_fallback=False,
        )
        plan_b = RemixPlan(
            vocal_source="song_a", start_time_vocal=0, end_time_vocal=60,
            start_time_instrumental=0, end_time_instrumental=60,
            sections=sections_b, tempo_source="average", key_source="none",
            explanation="test", used_fallback=False,
        )
        assert _structure_hash(plan_a) != _structure_hash(plan_b)


class TestComputeTotalBeats:
    """Tests for _compute_total_beats."""

    def test_returns_bpm_and_beats(self):
        meta_a = _make_metadata(bpm=120.0)
        meta_b = _make_metadata(bpm=120.0)
        bpm, beats = _compute_total_beats(meta_a, meta_b)
        assert bpm > 0
        assert beats > 0

    def test_beats_on_grid(self):
        meta_a = _make_metadata(bpm=95.0)
        meta_b = _make_metadata(bpm=130.0)
        _, beats = _compute_total_beats(meta_a, meta_b)
        assert beats % 4 == 0

    def test_minimum_beats(self):
        meta_a = _make_metadata(bpm=30.0, duration=30.0)
        meta_b = _make_metadata(bpm=30.0, duration=30.0)
        _, beats = _compute_total_beats(meta_a, meta_b)
        assert beats >= 32


# ---------------------------------------------------------------------------
# Main generate_candidates tests
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    """Tests for generate_candidates -- the public API."""

    def test_candidate_count_within_range(self):
        """Candidate count is within [min_count, max_count] range."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b, min_count=8, max_count=16)
        assert 8 <= len(candidates) <= 16

    def test_all_families_represented(self):
        """All 4 arrangement families are represented in the output."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)

        families = {_get_family_name(c) for c in candidates}
        assert "standard_arc" in families
        assert "hook_first" in families
        assert "dj_lift" in families
        assert "quick_hit" in families

    def test_all_plans_are_remix_plans(self):
        """Every candidate is a valid RemixPlan."""
        meta_a = _make_metadata(bpm=100.0, duration=200.0)
        meta_b = _make_metadata(bpm=110.0, duration=220.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            assert isinstance(plan, RemixPlan)

    def test_used_fallback_is_false(self):
        """All candidates have used_fallback=False."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            assert plan.used_fallback is False

    def test_explanation_includes_family_name(self):
        """All candidates include the arrangement family name in explanation."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        family_names = {"standard_arc", "hook_first", "dj_lift", "quick_hit"}
        for plan in candidates:
            assert any(name in plan.explanation for name in family_names), (
                f"No family name found in explanation: {plan.explanation}"
            )

    def test_section_boundaries_monotonic(self):
        """Section boundaries are monotonically increasing and start at 0."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            assert plan.sections[0].start_beat == 0, (
                f"First section doesn't start at 0: {plan.sections[0]}"
            )
            for i in range(len(plan.sections) - 1):
                assert plan.sections[i].end_beat == plan.sections[i + 1].start_beat, (
                    f"Gap between sections {i} and {i+1}: "
                    f"end={plan.sections[i].end_beat}, start={plan.sections[i+1].start_beat}"
                )
            for i, s in enumerate(plan.sections):
                assert s.start_beat < s.end_beat, (
                    f"Section {i} ({s.label}) has invalid bounds: "
                    f"start={s.start_beat}, end={s.end_beat}"
                )

    def test_beat_grid_alignment(self):
        """Section boundaries are on 4-beat multiples."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            for s in plan.sections:
                assert s.start_beat % 4 == 0, (
                    f"Section {s.label} start_beat={s.start_beat} not on 4-beat grid"
                )
                # end_beat = total_beats (grid-aligned) or start of next section (grid-aligned)
                # The last section's end_beat is total_beats which is grid-snapped
                assert s.end_beat % 4 == 0, (
                    f"Section {s.label} end_beat={s.end_beat} not on 4-beat grid"
                )

    def test_minimum_section_length(self):
        """All sections meet minimum length requirements (>= 4 beats)."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            for s in plan.sections:
                length = s.end_beat - s.start_beat
                assert length >= 4, (
                    f"Section {s.label} too short: {length} beats "
                    f"(start={s.start_beat}, end={s.end_beat})"
                )

    def test_transition_beats_bounded(self):
        """Transition beats don't exceed half the section length."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            for s in plan.sections:
                section_len = s.end_beat - s.start_beat
                assert s.transition_beats <= section_len // 2, (
                    f"Section {s.label}: transition_beats={s.transition_beats} > "
                    f"half section_len={section_len // 2}"
                )

    def test_stem_gains_valid_range(self):
        """All stem gains are in [0.0, 1.0]."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)
        for plan in candidates:
            for s in plan.sections:
                for stem, gain in s.stem_gains.items():
                    assert 0.0 <= gain <= 1.0, (
                        f"Section {s.label} stem {stem} gain {gain} out of range"
                    )


class TestDeduplication:
    """Tests for deduplication logic."""

    def test_removes_actual_duplicates(self):
        """Deduplication removes plans with identical structure hashes."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)

        # Create duplicates by doubling the list
        doubled = candidates + candidates
        deduped = _deduplicate(doubled)
        assert len(deduped) == len(candidates)

    def test_preserves_unique(self):
        """Deduplication keeps unique candidates."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        candidates = generate_candidates(meta_a, meta_b)

        deduped = _deduplicate(candidates)
        # All should be unique already
        assert len(deduped) == len(candidates)

    def test_empty_input(self):
        """Deduplication handles empty list."""
        assert _deduplicate([]) == []


class TestBackfill:
    """Tests for backfill when too few candidates survive dedup."""

    def test_backfill_reaches_min_count(self):
        """After backfill, at least min_count candidates exist."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=125.0, duration=250.0)
        # Request many candidates to test backfill
        candidates = generate_candidates(meta_a, meta_b, min_count=12, max_count=16)
        assert len(candidates) >= 12


class TestDifferentBPMs:
    """Test candidate generation with varying BPM/duration combinations."""

    def test_slow_songs(self):
        """Works with slow BPMs."""
        meta_a = _make_metadata(bpm=70.0, duration=300.0)
        meta_b = _make_metadata(bpm=75.0, duration=280.0)
        candidates = generate_candidates(meta_a, meta_b)
        assert len(candidates) >= 8
        # Verify basic validity
        for plan in candidates:
            assert plan.sections[0].start_beat == 0

    def test_fast_songs(self):
        """Works with fast BPMs."""
        meta_a = _make_metadata(bpm=170.0, duration=180.0)
        meta_b = _make_metadata(bpm=175.0, duration=200.0)
        candidates = generate_candidates(meta_a, meta_b)
        assert len(candidates) >= 8

    def test_mismatched_bpms(self):
        """Works with significantly different BPMs."""
        meta_a = _make_metadata(bpm=80.0, duration=240.0)
        meta_b = _make_metadata(bpm=140.0, duration=240.0)
        candidates = generate_candidates(meta_a, meta_b)
        assert len(candidates) >= 8
        for plan in candidates:
            assert plan.sections[0].start_beat == 0
            for s in plan.sections:
                assert s.end_beat > s.start_beat

    def test_identical_bpms(self):
        """Works when both songs have identical BPMs."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)
        candidates = generate_candidates(meta_a, meta_b)
        assert len(candidates) >= 8

    def test_short_songs(self):
        """Works with short duration songs."""
        meta_a = _make_metadata(bpm=120.0, duration=60.0)
        meta_b = _make_metadata(bpm=120.0, duration=60.0)
        candidates = generate_candidates(meta_a, meta_b)
        assert len(candidates) >= 8
        for plan in candidates:
            assert plan.sections[0].start_beat == 0
            for s in plan.sections:
                assert s.end_beat > s.start_beat


class TestCustomTargetCounts:
    """Test with different target/min/max counts."""

    def test_small_target(self):
        """Can generate as few as 4 candidates if min allows."""
        meta_a = _make_metadata()
        meta_b = _make_metadata()
        candidates = generate_candidates(meta_a, meta_b, min_count=4, max_count=6)
        assert 4 <= len(candidates) <= 6

    def test_large_target(self):
        """Can handle large target counts (up to max)."""
        meta_a = _make_metadata()
        meta_b = _make_metadata()
        candidates = generate_candidates(meta_a, meta_b, target_count=20, min_count=8, max_count=16)
        assert len(candidates) <= 16

    def test_max_count_caps_output(self):
        """Output is always <= max_count."""
        meta_a = _make_metadata()
        meta_b = _make_metadata()
        candidates = generate_candidates(meta_a, meta_b, max_count=10)
        assert len(candidates) <= 10
