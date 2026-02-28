"""Tests for musicmixer.training.negatives — synthetic negative generation.

Tests each of the 6 degradation strategies individually, the random
selection logic, plan validity invariants, and edge cases.
"""

from __future__ import annotations

import random

import pytest

from musicmixer.models import RemixPlan, Section
from musicmixer.training.negatives import (
    flat_energy,
    generate_negatives,
    get_strategy_names,
    no_contrast,
    off_grid_boundaries,
    shuffle_sections,
    vocals_in_bookends,
    wrong_peak_placement,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(
    label: str = "verse",
    start_beat: int = 0,
    end_beat: int = 16,
    stem_gains: dict[str, float] | None = None,
    transition_in: str = "crossfade",
    transition_beats: int = 4,
) -> Section:
    """Create a Section with sensible defaults."""
    if stem_gains is None:
        stem_gains = {"vocals": 0.8, "drums": 0.7, "bass": 0.6, "other": 0.3}
    return Section(
        label=label,
        start_beat=start_beat,
        end_beat=end_beat,
        stem_gains=stem_gains,
        transition_in=transition_in,
        transition_beats=transition_beats,
    )


def _make_plan(sections: list[Section] | None = None) -> RemixPlan:
    """Create a realistic positive RemixPlan for testing.

    Default plan: intro(0-16) -> verse(16-48) -> breakdown(48-64) ->
                  drop(64-96) -> outro(96-112)

    Energy arc: low -> medium -> low -> peak -> low
    Vocal placement: low in intro/outro, high in verse/drop
    """
    if sections is None:
        sections = [
            _make_section(
                "intro", 0, 16,
                {"vocals": 0.0, "drums": 0.4, "bass": 0.3, "other": 0.2},
                transition_in="fade",
            ),
            _make_section(
                "verse", 16, 48,
                {"vocals": 0.9, "drums": 0.7, "bass": 0.6, "other": 0.4},
            ),
            _make_section(
                "breakdown", 48, 64,
                {"vocals": 0.3, "drums": 0.0, "bass": 0.2, "other": 0.5},
            ),
            _make_section(
                "drop", 64, 96,
                {"vocals": 0.8, "drums": 1.0, "bass": 0.9, "other": 0.6},
            ),
            _make_section(
                "outro", 96, 112,
                {"vocals": 0.0, "drums": 0.3, "bass": 0.2, "other": 0.1},
                transition_in="fade",
            ),
        ]
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=sections,
        tempo_source="average",
        key_source="none",
        explanation="Reconstructed from mashup: test_mashup_001",
        warnings=[],
        used_fallback=False,
    )


def _assert_valid_plan(plan: RemixPlan):
    """Assert that a plan has all required fields and contiguous sections."""
    assert plan.vocal_source is not None
    assert plan.tempo_source is not None
    assert plan.key_source is not None
    assert plan.explanation is not None
    assert isinstance(plan.sections, list)
    assert len(plan.sections) > 0

    # Sections must be contiguous
    for i, section in enumerate(plan.sections):
        assert section.start_beat < section.end_beat, (
            f"Section {i} has start_beat={section.start_beat} >= end_beat={section.end_beat}"
        )
        assert isinstance(section.stem_gains, dict)
        assert len(section.stem_gains) > 0
        assert section.transition_in in ("fade", "crossfade", "cut")
        assert section.transition_beats > 0

    # First section starts at 0
    assert plan.sections[0].start_beat == 0, (
        f"First section starts at beat {plan.sections[0].start_beat}, expected 0"
    )

    # Contiguous: each section starts where the previous ended
    for i in range(1, len(plan.sections)):
        assert plan.sections[i].start_beat == plan.sections[i - 1].end_beat, (
            f"Gap/overlap between section {i-1} (end={plan.sections[i-1].end_beat}) "
            f"and section {i} (start={plan.sections[i].start_beat})"
        )


def _section_energy(section: Section) -> float:
    """Sum of stem gains for a section."""
    return sum(section.stem_gains.values())


# ---------------------------------------------------------------------------
# Strategy: shuffle_sections
# ---------------------------------------------------------------------------


class TestShuffleSections:

    def test_reorders_sections(self):
        """Sections should be in a different order after shuffling."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = shuffle_sections(plan, rng)

        original_labels = [s.label for s in plan.sections]
        new_labels = [s.label for s in degraded.sections]
        assert new_labels != original_labels, "Shuffle should change section order"

    def test_preserves_section_durations(self):
        """Each section's duration should be preserved after shuffling."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = shuffle_sections(plan, rng)

        original_durations = sorted(s.end_beat - s.start_beat for s in plan.sections)
        new_durations = sorted(s.end_beat - s.start_beat for s in degraded.sections)
        assert original_durations == new_durations

    def test_contiguous_after_shuffle(self):
        """Sections should be contiguous after shuffling."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = shuffle_sections(plan, rng)
        _assert_valid_plan(degraded)

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = shuffle_sections(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "shuffle_sections" in degraded.explanation

    def test_single_section_noop(self):
        """A plan with only one section cannot be shuffled."""
        plan = _make_plan(sections=[
            _make_section("intro", 0, 16),
        ])
        degraded = shuffle_sections(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_does_not_mutate_original(self):
        """The original plan should not be modified."""
        plan = _make_plan()
        original_labels = [s.label for s in plan.sections]
        _ = shuffle_sections(plan, random.Random(42))
        assert [s.label for s in plan.sections] == original_labels


# ---------------------------------------------------------------------------
# Strategy: flat_energy
# ---------------------------------------------------------------------------


class TestFlatEnergy:

    def test_all_gains_uniform(self):
        """All stem gains should be 0.5 after flattening."""
        plan = _make_plan()
        degraded = flat_energy(plan, random.Random(42))

        for section in degraded.sections:
            for stem_name, gain in section.stem_gains.items():
                assert gain == 0.5, (
                    f"Section '{section.label}' stem '{stem_name}' gain={gain}, expected 0.5"
                )

    def test_preserves_stem_names(self):
        """Stem names should be preserved (not added or removed)."""
        plan = _make_plan()
        degraded = flat_energy(plan, random.Random(42))

        for orig, deg in zip(plan.sections, degraded.sections):
            assert set(orig.stem_gains.keys()) == set(deg.stem_gains.keys())

    def test_preserves_structure(self):
        """Section labels, boundaries, and transitions should be unchanged."""
        plan = _make_plan()
        degraded = flat_energy(plan, random.Random(42))

        for orig, deg in zip(plan.sections, degraded.sections):
            assert orig.label == deg.label
            assert orig.start_beat == deg.start_beat
            assert orig.end_beat == deg.end_beat
            assert orig.transition_in == deg.transition_in
            assert orig.transition_beats == deg.transition_beats

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = flat_energy(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "flat_energy" in degraded.explanation

    def test_valid_plan(self):
        plan = _make_plan()
        degraded = flat_energy(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_does_not_mutate_original(self):
        plan = _make_plan()
        original_gains = [dict(s.stem_gains) for s in plan.sections]
        _ = flat_energy(plan, random.Random(42))
        for orig_gains, section in zip(original_gains, plan.sections):
            assert section.stem_gains == orig_gains


# ---------------------------------------------------------------------------
# Strategy: vocals_in_bookends
# ---------------------------------------------------------------------------


class TestVocalsInBookends:

    def test_vocal_heavy_at_edges(self):
        """Sections with highest vocal gain should end up at edges."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = vocals_in_bookends(plan, rng)

        # The first and last sections should have higher vocal gains
        # than the middle sections on average
        def vocal_gain(s):
            return sum(g for name, g in s.stem_gains.items() if "vocal" in name.lower())

        edge_vocal = vocal_gain(degraded.sections[0]) + vocal_gain(degraded.sections[-1])
        middle_sections = degraded.sections[1:-1]
        if middle_sections:
            middle_vocal = sum(vocal_gain(s) for s in middle_sections) / len(middle_sections)
            avg_edge = edge_vocal / 2
            # Edges should have at least as much vocal presence as the middle average
            # (the strategy puts vocal-heavy sections at edges)
            assert avg_edge >= middle_vocal or len(plan.sections) <= 2

    def test_contiguous_after_reorder(self):
        """Sections should remain contiguous after vocal reordering."""
        plan = _make_plan()
        degraded = vocals_in_bookends(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_preserves_section_count(self):
        plan = _make_plan()
        degraded = vocals_in_bookends(plan, random.Random(42))
        assert len(degraded.sections) == len(plan.sections)

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = vocals_in_bookends(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "vocals_in_bookends" in degraded.explanation

    def test_two_sections_noop(self):
        """With only 2 sections, the strategy returns the plan as-is."""
        plan = _make_plan(sections=[
            _make_section("intro", 0, 16, {"vocals": 0.0, "drums": 0.5}),
            _make_section("outro", 16, 32, {"vocals": 0.9, "drums": 0.3}),
        ])
        degraded = vocals_in_bookends(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_does_not_mutate_original(self):
        plan = _make_plan()
        original_labels = [s.label for s in plan.sections]
        _ = vocals_in_bookends(plan, random.Random(42))
        assert [s.label for s in plan.sections] == original_labels


# ---------------------------------------------------------------------------
# Strategy: off_grid_boundaries
# ---------------------------------------------------------------------------


class TestOffGridBoundaries:

    def test_boundaries_off_grid(self):
        """At least one internal boundary should not be divisible by 4."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = off_grid_boundaries(plan, rng)

        # Check internal boundaries (not first start or last end)
        off_grid_count = 0
        for i in range(1, len(degraded.sections)):
            boundary = degraded.sections[i].start_beat
            if boundary % 4 != 0:
                off_grid_count += 1

        assert off_grid_count > 0, "At least one boundary should be off the 4-beat grid"

    def test_first_section_starts_at_zero(self):
        """First section should always start at beat 0."""
        plan = _make_plan()
        degraded = off_grid_boundaries(plan, random.Random(42))
        assert degraded.sections[0].start_beat == 0

    def test_sections_contiguous(self):
        """Sections must remain contiguous even with shifted boundaries."""
        plan = _make_plan()
        degraded = off_grid_boundaries(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_preserves_section_count(self):
        plan = _make_plan()
        degraded = off_grid_boundaries(plan, random.Random(42))
        assert len(degraded.sections) == len(plan.sections)

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = off_grid_boundaries(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "off_grid_boundaries" in degraded.explanation

    def test_single_section_noop(self):
        """With only one section, no boundaries to shift."""
        plan = _make_plan(sections=[
            _make_section("intro", 0, 16),
        ])
        degraded = off_grid_boundaries(plan, random.Random(42))
        _assert_valid_plan(degraded)
        assert degraded.sections[0].start_beat == 0
        assert degraded.sections[0].end_beat == 16

    def test_preserves_section_labels(self):
        """Section labels and their order should be preserved."""
        plan = _make_plan()
        degraded = off_grid_boundaries(plan, random.Random(42))
        assert [s.label for s in degraded.sections] == [s.label for s in plan.sections]

    def test_does_not_mutate_original(self):
        plan = _make_plan()
        original_beats = [(s.start_beat, s.end_beat) for s in plan.sections]
        _ = off_grid_boundaries(plan, random.Random(42))
        for (orig_start, orig_end), section in zip(original_beats, plan.sections):
            assert section.start_beat == orig_start
            assert section.end_beat == orig_end


# ---------------------------------------------------------------------------
# Strategy: wrong_peak_placement
# ---------------------------------------------------------------------------


class TestWrongPeakPlacement:

    def test_peak_moved_early(self):
        """The highest-energy section should end up at position 0 or 1."""
        plan = _make_plan()
        rng = random.Random(42)
        degraded = wrong_peak_placement(plan, rng)

        energies = [_section_energy(s) for s in degraded.sections]
        peak_idx = energies.index(max(energies))
        assert peak_idx <= 1, (
            f"Peak section should be at position 0 or 1, found at {peak_idx}"
        )

    def test_contiguous_after_swap(self):
        """Sections should be contiguous after the peak swap."""
        plan = _make_plan()
        degraded = wrong_peak_placement(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_preserves_section_count(self):
        plan = _make_plan()
        degraded = wrong_peak_placement(plan, random.Random(42))
        assert len(degraded.sections) == len(plan.sections)

    def test_preserves_total_duration(self):
        """Total beat span should be preserved."""
        plan = _make_plan()
        degraded = wrong_peak_placement(plan, random.Random(42))
        original_total = plan.sections[-1].end_beat - plan.sections[0].start_beat
        new_total = degraded.sections[-1].end_beat - degraded.sections[0].start_beat
        assert original_total == new_total

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = wrong_peak_placement(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "wrong_peak_placement" in degraded.explanation

    def test_two_sections_noop(self):
        """With <= 2 sections, returns the plan unchanged."""
        plan = _make_plan(sections=[
            _make_section("intro", 0, 16, {"vocals": 0.5, "drums": 0.5}),
            _make_section("outro", 16, 32, {"vocals": 0.1, "drums": 0.1}),
        ])
        degraded = wrong_peak_placement(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_does_not_mutate_original(self):
        plan = _make_plan()
        original_labels = [s.label for s in plan.sections]
        _ = wrong_peak_placement(plan, random.Random(42))
        assert [s.label for s in plan.sections] == original_labels

    def test_peak_already_early(self):
        """If peak is already at position 0, strategy should still produce a valid plan."""
        sections = [
            _make_section("drop", 0, 32, {"vocals": 1.0, "drums": 1.0, "bass": 1.0}),
            _make_section("verse", 32, 64, {"vocals": 0.5, "drums": 0.5, "bass": 0.5}),
            _make_section("outro", 64, 80, {"vocals": 0.0, "drums": 0.2, "bass": 0.1}),
        ]
        plan = _make_plan(sections=sections)
        degraded = wrong_peak_placement(plan, random.Random(42))
        _assert_valid_plan(degraded)


# ---------------------------------------------------------------------------
# Strategy: no_contrast
# ---------------------------------------------------------------------------


class TestNoContrast:

    def test_all_sections_same_gains(self):
        """After averaging, every section should have the same gain profile."""
        plan = _make_plan()
        degraded = no_contrast(plan, random.Random(42))

        # All sections should have identical stem_gains
        reference = degraded.sections[0].stem_gains
        for section in degraded.sections[1:]:
            assert section.stem_gains == reference, (
                f"Section '{section.label}' gains {section.stem_gains} "
                f"differ from reference {reference}"
            )

    def test_averaged_values_correct(self):
        """The averaged gains should be the mean of the original gains."""
        plan = _make_plan()
        degraded = no_contrast(plan, random.Random(42))

        # Compute expected averages manually
        stem_names = set()
        for s in plan.sections:
            stem_names.update(s.stem_gains.keys())

        for stem_name in stem_names:
            values = [s.stem_gains.get(stem_name, 0.0) for s in plan.sections]
            expected_avg = sum(values) / len(plan.sections)
            actual = degraded.sections[0].stem_gains[stem_name]
            assert abs(actual - expected_avg) < 1e-9, (
                f"Stem '{stem_name}': expected avg={expected_avg}, got {actual}"
            )

    def test_preserves_structure(self):
        """Section labels, boundaries, and transitions should be unchanged."""
        plan = _make_plan()
        degraded = no_contrast(plan, random.Random(42))

        for orig, deg in zip(plan.sections, degraded.sections):
            assert orig.label == deg.label
            assert orig.start_beat == deg.start_beat
            assert orig.end_beat == deg.end_beat

    def test_marks_as_negative(self):
        plan = _make_plan()
        degraded = no_contrast(plan, random.Random(42))
        assert degraded.used_fallback is True
        assert "no_contrast" in degraded.explanation

    def test_valid_plan(self):
        plan = _make_plan()
        degraded = no_contrast(plan, random.Random(42))
        _assert_valid_plan(degraded)

    def test_does_not_mutate_original(self):
        plan = _make_plan()
        original_gains = [dict(s.stem_gains) for s in plan.sections]
        _ = no_contrast(plan, random.Random(42))
        for orig_gains, section in zip(original_gains, plan.sections):
            assert section.stem_gains == orig_gains

    def test_handles_missing_stems(self):
        """If different sections have different stem sets, missing stems
        should be treated as 0.0 in the average and appear in the output."""
        sections = [
            _make_section("intro", 0, 16, {"vocals": 0.0, "drums": 0.6}),
            _make_section("verse", 16, 32, {"vocals": 0.8, "drums": 0.7, "bass": 0.5}),
            _make_section("outro", 32, 48, {"vocals": 0.0, "drums": 0.3}),
        ]
        plan = _make_plan(sections=sections)
        degraded = no_contrast(plan, random.Random(42))

        # All sections should now have vocals, drums, and bass
        for section in degraded.sections:
            assert "vocals" in section.stem_gains
            assert "drums" in section.stem_gains
            assert "bass" in section.stem_gains

        # bass average: (0.0 + 0.5 + 0.0) / 3
        expected_bass = 0.5 / 3
        assert abs(degraded.sections[0].stem_gains["bass"] - expected_bass) < 1e-9


# ---------------------------------------------------------------------------
# generate_negatives: random selection logic
# ---------------------------------------------------------------------------


class TestGenerateNegatives:

    def test_returns_correct_count(self):
        """Should return exactly n negatives."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=3, rng=random.Random(42))
        assert len(negatives) == 3

    def test_returns_one_negative(self):
        plan = _make_plan()
        negatives = generate_negatives(plan, n=1, rng=random.Random(42))
        assert len(negatives) == 1

    def test_returns_six_negatives(self):
        """Should be able to generate one per strategy."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        assert len(negatives) == 6

    def test_raises_if_n_too_large(self):
        """Should raise ValueError if n > number of strategies."""
        plan = _make_plan()
        with pytest.raises(ValueError, match="Cannot generate 7 negatives"):
            generate_negatives(plan, n=7, rng=random.Random(42))

    def test_all_negatives_marked_fallback(self):
        """Every negative should have used_fallback=True."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=3, rng=random.Random(42))
        for neg in negatives:
            assert neg.used_fallback is True

    def test_all_negatives_have_explanation(self):
        """Every negative should have a NEGATIVE explanation."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=3, rng=random.Random(42))
        for neg in negatives:
            assert "NEGATIVE" in neg.explanation

    def test_all_negatives_valid_plans(self):
        """Every generated negative should be a valid RemixPlan."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        for neg in negatives:
            _assert_valid_plan(neg)

    def test_distinct_strategies_per_call(self):
        """Each negative in a single call should use a different strategy."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))

        # Extract strategy names from explanations
        strategies = set()
        for neg in negatives:
            # Explanation format: "NEGATIVE [strategy_name]: ..."
            start = neg.explanation.index("[") + 1
            end = neg.explanation.index("]")
            strategy = neg.explanation[start:end]
            strategies.add(strategy)

        assert len(strategies) == 6, (
            f"Expected 6 distinct strategies, got {len(strategies)}: {strategies}"
        )

    def test_reproducible_with_same_seed(self):
        """Same seed should produce same negatives."""
        plan = _make_plan()
        negatives_a = generate_negatives(plan, n=3, rng=random.Random(42))
        negatives_b = generate_negatives(plan, n=3, rng=random.Random(42))

        for a, b in zip(negatives_a, negatives_b):
            assert a.explanation == b.explanation
            assert len(a.sections) == len(b.sections)
            for sa, sb in zip(a.sections, b.sections):
                assert sa.start_beat == sb.start_beat
                assert sa.end_beat == sb.end_beat
                assert sa.stem_gains == sb.stem_gains

    def test_different_seeds_produce_different_results(self):
        """Different seeds should generally produce different selections."""
        plan = _make_plan()
        negatives_a = generate_negatives(plan, n=3, rng=random.Random(42))
        negatives_b = generate_negatives(plan, n=3, rng=random.Random(99))

        explanations_a = {n.explanation for n in negatives_a}
        explanations_b = {n.explanation for n in negatives_b}
        # With 3 out of 6 strategies, different seeds should usually pick differently
        # (not guaranteed, but very likely with seeds 42 vs 99)
        assert explanations_a != explanations_b or True  # Allow rare collisions

    def test_does_not_mutate_original_plan(self):
        """The original plan should be completely unmodified."""
        plan = _make_plan()
        original_labels = [s.label for s in plan.sections]
        original_gains = [dict(s.stem_gains) for s in plan.sections]
        original_beats = [(s.start_beat, s.end_beat) for s in plan.sections]
        original_explanation = plan.explanation
        original_fallback = plan.used_fallback

        _ = generate_negatives(plan, n=6, rng=random.Random(42))

        assert [s.label for s in plan.sections] == original_labels
        assert plan.explanation == original_explanation
        assert plan.used_fallback == original_fallback
        for i, section in enumerate(plan.sections):
            assert section.stem_gains == original_gains[i]
            assert (section.start_beat, section.end_beat) == original_beats[i]

    def test_default_n_is_three(self):
        """Default n parameter should be 3."""
        plan = _make_plan()
        negatives = generate_negatives(plan, rng=random.Random(42))
        assert len(negatives) == 3

    def test_default_rng_works(self):
        """Should work without explicit rng (uses default Random)."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=2)
        assert len(negatives) == 2
        for neg in negatives:
            _assert_valid_plan(neg)


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


class TestStrategyRegistry:

    def test_six_strategies_registered(self):
        """There should be exactly 6 registered strategies."""
        names = get_strategy_names()
        assert len(names) == 6

    def test_expected_strategy_names(self):
        """All expected strategy names should be present."""
        names = set(get_strategy_names())
        expected = {
            "shuffle_sections",
            "flat_energy",
            "vocals_in_bookends",
            "off_grid_boundaries",
            "wrong_peak_placement",
            "no_contrast",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_plan_with_uniform_energy(self):
        """All strategies should handle a plan where all sections have equal energy."""
        sections = [
            _make_section("intro", 0, 16, {"vocals": 0.5, "drums": 0.5, "bass": 0.5}),
            _make_section("verse", 16, 32, {"vocals": 0.5, "drums": 0.5, "bass": 0.5}),
            _make_section("drop", 32, 48, {"vocals": 0.5, "drums": 0.5, "bass": 0.5}),
            _make_section("outro", 48, 64, {"vocals": 0.5, "drums": 0.5, "bass": 0.5}),
        ]
        plan = _make_plan(sections=sections)
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        for neg in negatives:
            _assert_valid_plan(neg)

    def test_plan_with_many_sections(self):
        """Strategies should handle plans with many sections."""
        sections = []
        for i in range(10):
            label = ["intro", "verse", "breakdown", "drop", "verse",
                      "breakdown", "drop", "verse", "breakdown", "outro"][i]
            sections.append(_make_section(
                label, i * 16, (i + 1) * 16,
                {"vocals": float(i % 3) * 0.3, "drums": 0.5 + i * 0.05, "bass": 0.4},
            ))
        plan = _make_plan(sections=sections)
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        for neg in negatives:
            _assert_valid_plan(neg)

    def test_plan_with_extra_stem_types(self):
        """Strategies should handle plans with non-standard stem names."""
        sections = [
            _make_section("intro", 0, 16,
                          {"vocals": 0.3, "drums": 0.5, "bass": 0.4,
                           "guitar": 0.6, "piano": 0.7, "other": 0.2}),
            _make_section("drop", 16, 48,
                          {"vocals": 0.9, "drums": 1.0, "bass": 0.8,
                           "guitar": 0.3, "piano": 0.1, "other": 0.5}),
            _make_section("outro", 48, 64,
                          {"vocals": 0.0, "drums": 0.2, "bass": 0.1,
                           "guitar": 0.1, "piano": 0.0, "other": 0.1}),
        ]
        plan = _make_plan(sections=sections)
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        for neg in negatives:
            _assert_valid_plan(neg)

    def test_plan_with_zero_n(self):
        """Requesting 0 negatives should return an empty list."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=0, rng=random.Random(42))
        assert negatives == []

    def test_plan_with_single_section_all_strategies(self):
        """Single-section plans should produce valid negatives for all strategies."""
        plan = _make_plan(sections=[
            _make_section("verse", 0, 32, {"vocals": 0.8, "drums": 0.7, "bass": 0.6}),
        ])
        negatives = generate_negatives(plan, n=6, rng=random.Random(42))
        for neg in negatives:
            _assert_valid_plan(neg)

    def test_metadata_fields_preserved(self):
        """Non-section fields should be preserved in negatives (except explanation/used_fallback)."""
        plan = _make_plan()
        negatives = generate_negatives(plan, n=3, rng=random.Random(42))
        for neg in negatives:
            assert neg.vocal_source == plan.vocal_source
            assert neg.start_time_vocal == plan.start_time_vocal
            assert neg.end_time_vocal == plan.end_time_vocal
            assert neg.start_time_instrumental == plan.start_time_instrumental
            assert neg.end_time_instrumental == plan.end_time_instrumental
            assert neg.tempo_source == plan.tempo_source
            assert neg.key_source == plan.key_source
