"""Tests for musicmixer.services.gain_mapper -- intent-to-gains conversion."""

import math

import pytest

from musicmixer.models import (
    IntentPlan,
    IntentSection,
    RemixPlan,
    Section,
    STEM_ROLES,
)
from musicmixer.services.gain_mapper import (
    ALL_STEMS,
    ENERGY_BUDGET_TARGETS,
    ENERGY_MULTIPLIERS,
    INSTRUMENTAL_STEMS,
    MAX_MUTING_FRACTION,
    MIN_ACTIVE_GAIN_THRESHOLD,
    MIN_ACTIVE_INSTRUMENTAL_STEMS,
    ROLE_BASE_GAINS,
    ROLE_GAIN_FLOORS,
    _compute_section_gains,
    _enforce_min_active_instrumentals,
    _enforce_muting_budget,
    _enforce_no_globally_muted,
    _lufs_adjustment,
    _validate_energy_budgets,
    intent_section_to_section,
    map_intent_to_gains,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_intent_section(
    label: str = "verse",
    start_beat: int = 0,
    end_beat: int = 32,
    energy: str = "medium",
    stem_roles: dict[str, str] | None = None,
    transition_in: str = "crossfade",
    transition_beats: int = 4,
) -> IntentSection:
    """Create a synthetic IntentSection for testing."""
    if stem_roles is None:
        stem_roles = {
            "vocals": "lead",
            "drums": "support",
            "bass": "support",
            "guitar": "background",
            "piano": "texture",
            "other": "background",
        }
    return IntentSection(
        label=label,
        start_beat=start_beat,
        end_beat=end_beat,
        energy=energy,
        stem_roles=stem_roles,
        transition_in=transition_in,
        transition_beats=transition_beats,
    )


def _make_intent_plan(
    sections: list[IntentSection] | None = None,
    key_source: str = "song_b",
    explanation: str = "Test plan",
    warnings: list[str] | None = None,
) -> IntentPlan:
    """Create a synthetic IntentPlan for testing."""
    if sections is None:
        sections = [
            _make_intent_section("intro", 0, 16, "low"),
            _make_intent_section("verse", 16, 48, "medium"),
            _make_intent_section("chorus", 48, 80, "high"),
            _make_intent_section("breakdown", 80, 96, "low"),
            _make_intent_section("drop", 96, 128, "peak"),
            _make_intent_section("outro", 128, 144, "low"),
        ]
    return IntentPlan(
        start_time_vocal=10.0,
        end_time_vocal=80.0,
        start_time_instrumental=0.0,
        end_time_instrumental=90.0,
        sections=sections,
        key_source=key_source,
        explanation=explanation,
        warnings=warnings or [],
    )


# ---------------------------------------------------------------------------
# 1. TestRoleToGainMapping
# ---------------------------------------------------------------------------

class TestRoleToGainMapping:
    """Each role produces an expected gain range."""

    def test_lead_is_highest(self):
        gains = _compute_section_gains(
            _make_intent_section(energy="high", stem_roles={
                "vocals": "lead", "drums": "silent", "bass": "silent",
                "guitar": "silent", "piano": "silent", "other": "silent",
            }),
            None, None,
        )
        assert gains["vocals"] == pytest.approx(0.92, abs=0.01)

    def test_support_mid_range(self):
        gains = _compute_section_gains(
            _make_intent_section(energy="high", stem_roles={
                "vocals": "silent", "drums": "support", "bass": "silent",
                "guitar": "silent", "piano": "silent", "other": "silent",
            }),
            None, None,
        )
        assert 0.60 <= gains["drums"] <= 0.90

    def test_background_lower_than_support(self):
        roles = {s: "silent" for s in ALL_STEMS}
        roles["drums"] = "support"
        roles["bass"] = "background"
        gains = _compute_section_gains(
            _make_intent_section(energy="high", stem_roles=roles),
            None, None,
        )
        assert gains["bass"] < gains["drums"]

    def test_texture_lowest_active(self):
        roles = {s: "silent" for s in ALL_STEMS}
        roles["piano"] = "texture"
        gains = _compute_section_gains(
            _make_intent_section(energy="high", stem_roles=roles),
            None, None,
        )
        assert 0.15 <= gains["piano"] <= 0.35

    def test_silent_is_zero(self):
        roles = {s: "silent" for s in ALL_STEMS}
        gains = _compute_section_gains(
            _make_intent_section(energy="high", stem_roles=roles),
            None, None,
        )
        for stem in ALL_STEMS:
            assert gains[stem] == 0.0

    def test_role_ordering(self):
        """lead > support > background > texture > silent."""
        section = _make_intent_section(
            energy="high",
            stem_roles={
                "vocals": "lead",
                "drums": "support",
                "bass": "background",
                "guitar": "texture",
                "piano": "silent",
                "other": "silent",
            },
        )
        gains = _compute_section_gains(section, None, None)
        assert gains["vocals"] > gains["drums"] > gains["bass"] > gains["guitar"]
        assert gains["piano"] == 0.0


# ---------------------------------------------------------------------------
# 2. TestEnergyScaling
# ---------------------------------------------------------------------------

class TestEnergyScaling:
    """Low/medium/high/peak produce progressively higher total gains."""

    def _total_gain(self, energy: str) -> float:
        section = _make_intent_section(energy=energy)
        gains = _compute_section_gains(section, None, None)
        return sum(gains.values())

    def test_low_less_than_medium(self):
        assert self._total_gain("low") < self._total_gain("medium")

    def test_medium_less_than_high(self):
        assert self._total_gain("medium") < self._total_gain("high")

    def test_high_less_than_peak(self):
        assert self._total_gain("high") < self._total_gain("peak")

    def test_all_energies_positive(self):
        for energy in ("low", "medium", "high", "peak"):
            assert self._total_gain(energy) > 0.0

    def test_peak_multiplier_above_one(self):
        """Peak should push gains slightly above the base."""
        assert ENERGY_MULTIPLIERS["peak"] > 1.0

    def test_gains_clamped_at_one(self):
        """Even peak energy shouldn't push any gain above 1.0."""
        section = _make_intent_section(energy="peak", stem_roles={
            "vocals": "lead", "drums": "lead", "bass": "lead",
            "guitar": "lead", "piano": "lead", "other": "lead",
        })
        gains = _compute_section_gains(section, None, None)
        for stem, val in gains.items():
            assert val <= 1.0, f"{stem} gain {val} exceeds 1.0"


# ---------------------------------------------------------------------------
# 3. TestLufsAdjustment
# ---------------------------------------------------------------------------

class TestLufsAdjustment:
    """Quiet stems boosted, loud stems attenuated, NaN/missing handled."""

    def test_very_quiet_stem_boosted(self):
        factor = _lufs_adjustment(-35.0)
        assert factor > 1.0

    def test_neutral_reference(self):
        factor = _lufs_adjustment(-18.0)
        assert factor == pytest.approx(1.0)

    def test_loud_stem_attenuated(self):
        factor = _lufs_adjustment(-10.0)
        assert factor < 1.0

    def test_nan_returns_neutral(self):
        factor = _lufs_adjustment(float("nan"))
        assert factor == 1.0

    def test_very_low_lufs_neutral(self):
        """LUFS below -60 treated as no data."""
        factor = _lufs_adjustment(-70.0)
        assert factor == 1.0

    def test_lufs_applied_to_vocal_stem(self):
        """Vocal LUFS data should affect vocals gain."""
        section = _make_intent_section(energy="high", stem_roles={
            "vocals": "lead", "drums": "silent", "bass": "silent",
            "guitar": "silent", "piano": "silent", "other": "silent",
        })
        # Quiet vocal stem should get boosted
        gains_boosted = _compute_section_gains(
            section, vocal_stem_lufs={"vocals": -35.0}, inst_stem_lufs=None,
        )
        gains_neutral = _compute_section_gains(section, None, None)
        assert gains_boosted["vocals"] > gains_neutral["vocals"]

    def test_lufs_applied_to_instrumental_stem(self):
        """Instrumental LUFS data should affect instrumental stems."""
        section = _make_intent_section(energy="high", stem_roles={
            "vocals": "silent", "drums": "lead", "bass": "silent",
            "guitar": "silent", "piano": "silent", "other": "silent",
        })
        # Loud drums should get attenuated
        gains_attenuated = _compute_section_gains(
            section, vocal_stem_lufs=None, inst_stem_lufs={"drums": -10.0},
        )
        gains_neutral = _compute_section_gains(section, None, None)
        assert gains_attenuated["drums"] < gains_neutral["drums"]

    def test_lufs_ranges(self):
        """Verify the full range of LUFS adjustments (neutral ref = -18)."""
        assert _lufs_adjustment(-35.0) == 1.3
        assert _lufs_adjustment(-25.0) == 1.15
        assert _lufs_adjustment(-20.0) == 1.0
        assert _lufs_adjustment(-15.0) == 0.90
        assert _lufs_adjustment(-10.0) == 0.80


# ---------------------------------------------------------------------------
# 4. TestConstraintMinActiveStems
# ---------------------------------------------------------------------------

class TestConstraintMinActiveStems:
    """Section with many silent instrumentals gets corrected to have 3+ active."""

    def test_four_silent_instrumentals_corrected(self):
        """Verse with only 1 active instrumental gets bumped to 3."""
        section = _make_intent_section(
            label="verse", energy="medium",
            stem_roles={
                "vocals": "lead",
                "drums": "support",
                "bass": "silent",
                "guitar": "silent",
                "piano": "silent",
                "other": "silent",
            },
        )
        gains = [_compute_section_gains(section, None, None)]
        _enforce_min_active_instrumentals(gains, ["verse"])

        active_inst = [
            s for s in INSTRUMENTAL_STEMS
            if gains[0][s] > MIN_ACTIVE_GAIN_THRESHOLD
        ]
        assert len(active_inst) >= MIN_ACTIVE_INSTRUMENTAL_STEMS

    def test_intro_exempt(self):
        """Intros are not required to have minimum active instrumentals."""
        section = _make_intent_section(
            label="intro", energy="low",
            stem_roles={s: "silent" for s in ALL_STEMS},
        )
        gains = [_compute_section_gains(section, None, None)]
        _enforce_min_active_instrumentals(gains, ["intro"])

        # All should still be 0.0 — intro is exempt
        active_inst = [
            s for s in INSTRUMENTAL_STEMS
            if gains[0][s] > MIN_ACTIVE_GAIN_THRESHOLD
        ]
        assert len(active_inst) == 0

    def test_outro_exempt(self):
        """Outros are not required to have minimum active instrumentals."""
        section = _make_intent_section(
            label="outro", energy="low",
            stem_roles={s: "silent" for s in ALL_STEMS},
        )
        gains = [_compute_section_gains(section, None, None)]
        _enforce_min_active_instrumentals(gains, ["outro"])

        active_inst = [
            s for s in INSTRUMENTAL_STEMS
            if gains[0][s] > MIN_ACTIVE_GAIN_THRESHOLD
        ]
        assert len(active_inst) == 0

    def test_already_meets_minimum(self):
        """Section with 3+ active instrumentals is not modified."""
        section = _make_intent_section(
            label="chorus", energy="high",
            stem_roles={
                "vocals": "lead",
                "drums": "support",
                "bass": "support",
                "guitar": "background",
                "piano": "silent",
                "other": "silent",
            },
        )
        gains = [_compute_section_gains(section, None, None)]
        original_piano = gains[0]["piano"]
        _enforce_min_active_instrumentals(gains, ["chorus"])
        # Piano was silent and should stay silent since constraint is met
        assert gains[0]["piano"] == original_piano


# ---------------------------------------------------------------------------
# 5. TestConstraintNoGloballyMuted
# ---------------------------------------------------------------------------

class TestConstraintNoGloballyMuted:
    """Plan with a stem silent everywhere gets it boosted in at least one section."""

    def test_piano_silent_everywhere_gets_boosted(self):
        """Piano set to silent in all sections gets 0.30 in highest-energy section."""
        sections = [
            _make_intent_section("verse", 0, 32, "medium", stem_roles={
                "vocals": "lead", "drums": "support", "bass": "support",
                "guitar": "background", "piano": "silent", "other": "background",
            }),
            _make_intent_section("chorus", 32, 64, "high", stem_roles={
                "vocals": "lead", "drums": "support", "bass": "support",
                "guitar": "background", "piano": "silent", "other": "background",
            }),
        ]
        gains = [_compute_section_gains(s, None, None) for s in sections]
        energies = [s.energy for s in sections]

        _enforce_no_globally_muted(gains, energies)

        # Piano should be non-zero in at least one section
        piano_gains = [g["piano"] for g in gains]
        assert any(g > 0.0 for g in piano_gains)
        # Specifically, the high-energy section (chorus = index 1) should be bumped
        assert gains[1]["piano"] == pytest.approx(0.30)

    def test_already_active_stem_unchanged(self):
        """Stem active in at least one section should not be modified."""
        sections = [
            _make_intent_section("verse", 0, 32, "medium", stem_roles={
                "vocals": "lead", "drums": "support", "bass": "support",
                "guitar": "background", "piano": "texture", "other": "background",
            }),
            _make_intent_section("chorus", 32, 64, "high", stem_roles={
                "vocals": "lead", "drums": "support", "bass": "support",
                "guitar": "background", "piano": "silent", "other": "background",
            }),
        ]
        gains = [_compute_section_gains(s, None, None) for s in sections]
        original_chorus_piano = gains[1]["piano"]
        energies = [s.energy for s in sections]

        _enforce_no_globally_muted(gains, energies)

        # Piano was already active in verse, so chorus piano stays unchanged
        assert gains[1]["piano"] == original_chorus_piano


# ---------------------------------------------------------------------------
# 6. TestConstraintMutingBudget
# ---------------------------------------------------------------------------

class TestConstraintMutingBudget:
    """Plans with >15% muted instrumental entries get corrected."""

    def test_excessive_muting_corrected(self):
        """Create a plan with many silent instrumentals and verify correction."""
        # 4 sections, each with 3 silent instrumentals = 12 zeros out of 20 entries = 60%
        roles_heavy_muting = {
            "vocals": "lead", "drums": "support", "bass": "support",
            "guitar": "silent", "piano": "silent", "other": "silent",
        }
        sections = [
            _make_intent_section(f"verse", i * 32, (i + 1) * 32, "medium",
                                 stem_roles=dict(roles_heavy_muting))
            for i in range(4)
        ]
        gains = [_compute_section_gains(s, None, None) for s in sections]
        energies = [s.energy for s in sections]

        total_entries = len(sections) * len(INSTRUMENTAL_STEMS)
        max_zeros = int(total_entries * MAX_MUTING_FRACTION)

        _enforce_muting_budget(gains, energies)

        # Count remaining zeros
        actual_zeros = sum(
            1 for g in gains for s in INSTRUMENTAL_STEMS if g[s] == 0.0
        )
        assert actual_zeros <= max_zeros

    def test_within_budget_unchanged(self):
        """Plan within muting budget should not be modified."""
        # All stems active — no zeros
        sections = [
            _make_intent_section("verse", 0, 32, "medium"),
            _make_intent_section("chorus", 32, 64, "high"),
        ]
        gains = [_compute_section_gains(s, None, None) for s in sections]
        original_gains = [dict(g) for g in gains]
        energies = [s.energy for s in sections]

        _enforce_muting_budget(gains, energies)

        for orig, modified in zip(original_gains, gains):
            assert orig == modified


# ---------------------------------------------------------------------------
# 7. TestGainFloors
# ---------------------------------------------------------------------------

class TestGainFloors:
    """Support stems clamped to 0.50 min, background to 0.30, texture to 0.15."""

    def test_support_floor_at_low_energy(self):
        """Even at low energy, support should not drop below 0.60."""
        section = _make_intent_section(energy="low", stem_roles={
            "vocals": "silent", "drums": "support", "bass": "silent",
            "guitar": "silent", "piano": "silent", "other": "silent",
        })
        gains = _compute_section_gains(section, None, None)
        assert gains["drums"] >= 0.60

    def test_background_floor_at_low_energy(self):
        """Background should not drop below 0.40."""
        section = _make_intent_section(energy="low", stem_roles={
            "vocals": "silent", "drums": "silent", "bass": "background",
            "guitar": "silent", "piano": "silent", "other": "silent",
        })
        gains = _compute_section_gains(section, None, None)
        assert gains["bass"] >= 0.40

    def test_texture_floor_at_low_energy(self):
        """Texture should not drop below 0.20."""
        section = _make_intent_section(energy="low", stem_roles={
            "vocals": "silent", "drums": "silent", "bass": "silent",
            "guitar": "silent", "piano": "texture", "other": "silent",
        })
        gains = _compute_section_gains(section, None, None)
        assert gains["piano"] >= 0.20

    def test_silent_not_floored(self):
        """Silent stems should stay at 0.0 regardless of floors."""
        section = _make_intent_section(energy="low", stem_roles={
            "vocals": "silent", "drums": "silent", "bass": "silent",
            "guitar": "silent", "piano": "silent", "other": "silent",
        })
        gains = _compute_section_gains(section, None, None)
        for stem in ALL_STEMS:
            assert gains[stem] == 0.0

    def test_floor_values_match_spec(self):
        """Verify the documented floor values."""
        assert ROLE_GAIN_FLOORS["support"] == 0.60
        assert ROLE_GAIN_FLOORS["background"] == 0.40
        assert ROLE_GAIN_FLOORS["texture"] == 0.20


# ---------------------------------------------------------------------------
# 8. TestOutputShape
# ---------------------------------------------------------------------------

class TestOutputShape:
    """Produces valid RemixPlan with all 6 stems in every section."""

    def test_all_stems_present(self):
        plan = map_intent_to_gains(_make_intent_plan())
        for section in plan.sections:
            for stem in ALL_STEMS:
                assert stem in section.stem_gains, (
                    f"Missing stem '{stem}' in section '{section.label}'"
                )

    def test_correct_number_of_sections(self):
        intent = _make_intent_plan()
        plan = map_intent_to_gains(intent)
        assert len(plan.sections) == len(intent.sections)

    def test_returns_remix_plan(self):
        plan = map_intent_to_gains(_make_intent_plan())
        assert isinstance(plan, RemixPlan)

    def test_section_types(self):
        plan = map_intent_to_gains(_make_intent_plan())
        for section in plan.sections:
            assert isinstance(section, Section)

    def test_gains_in_valid_range(self):
        plan = map_intent_to_gains(_make_intent_plan())
        for section in plan.sections:
            for stem, gain in section.stem_gains.items():
                assert 0.0 <= gain <= 1.0, (
                    f"Gain {gain} out of range for '{stem}' in '{section.label}'"
                )

    def test_time_fields_copied(self):
        intent = _make_intent_plan()
        plan = map_intent_to_gains(intent)
        assert plan.start_time_vocal == intent.start_time_vocal
        assert plan.end_time_vocal == intent.end_time_vocal
        assert plan.start_time_instrumental == intent.start_time_instrumental
        assert plan.end_time_instrumental == intent.end_time_instrumental

    def test_key_source_copied(self):
        intent = _make_intent_plan(key_source="song_a")
        plan = map_intent_to_gains(intent)
        assert plan.key_source == "song_a"

    def test_explanation_copied(self):
        intent = _make_intent_plan(explanation="Custom explanation")
        plan = map_intent_to_gains(intent)
        assert plan.explanation == "Custom explanation"

    def test_section_labels_preserved(self):
        intent = _make_intent_plan()
        plan = map_intent_to_gains(intent)
        for intent_sec, sec in zip(intent.sections, plan.sections):
            assert sec.label == intent_sec.label

    def test_beat_ranges_preserved(self):
        intent = _make_intent_plan()
        plan = map_intent_to_gains(intent)
        for intent_sec, sec in zip(intent.sections, plan.sections):
            assert sec.start_beat == intent_sec.start_beat
            assert sec.end_beat == intent_sec.end_beat


# ---------------------------------------------------------------------------
# 9. TestNullLufs
# ---------------------------------------------------------------------------

class TestNullLufs:
    """Works correctly when no LUFS data is provided."""

    def test_none_lufs_produces_valid_plan(self):
        plan = map_intent_to_gains(_make_intent_plan(), None, None)
        assert isinstance(plan, RemixPlan)
        assert len(plan.sections) > 0

    def test_empty_dict_lufs(self):
        """Empty dicts should behave like None — neutral adjustment."""
        plan_none = map_intent_to_gains(_make_intent_plan(), None, None)
        plan_empty = map_intent_to_gains(_make_intent_plan(), {}, {})
        # Gains should be identical
        for sec_none, sec_empty in zip(plan_none.sections, plan_empty.sections):
            for stem in ALL_STEMS:
                assert sec_none.stem_gains[stem] == pytest.approx(
                    sec_empty.stem_gains[stem], abs=0.001
                )

    def test_partial_lufs_data(self):
        """Only some stems have LUFS — others get neutral adjustment."""
        plan = map_intent_to_gains(
            _make_intent_plan(),
            vocal_stem_lufs={"vocals": -20.0},  # neutral
            inst_stem_lufs={"drums": -35.0},     # quiet, should boost
        )
        assert isinstance(plan, RemixPlan)

    def test_nan_lufs_values(self):
        """NaN LUFS values should be treated as neutral."""
        plan_nan = map_intent_to_gains(
            _make_intent_plan(),
            vocal_stem_lufs={"vocals": float("nan")},
            inst_stem_lufs={"drums": float("nan")},
        )
        plan_none = map_intent_to_gains(_make_intent_plan(), None, None)
        # Should produce identical results
        for sec_nan, sec_none in zip(plan_nan.sections, plan_none.sections):
            for stem in ALL_STEMS:
                assert sec_nan.stem_gains[stem] == pytest.approx(
                    sec_none.stem_gains[stem], abs=0.001
                )


# ---------------------------------------------------------------------------
# 10. TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full IntentPlan -> RemixPlan conversion with realistic data."""

    def _realistic_plan(self) -> IntentPlan:
        """Build a realistic 6-section intent plan."""
        return IntentPlan(
            start_time_vocal=15.0,
            end_time_vocal=120.0,
            start_time_instrumental=0.0,
            end_time_instrumental=130.0,
            sections=[
                IntentSection(
                    label="intro", start_beat=0, end_beat=16, energy="low",
                    stem_roles={
                        "vocals": "silent", "drums": "texture",
                        "bass": "background", "guitar": "lead",
                        "piano": "texture", "other": "silent",
                    },
                    transition_in="fade", transition_beats=8,
                ),
                IntentSection(
                    label="verse", start_beat=16, end_beat=48, energy="medium",
                    stem_roles={
                        "vocals": "lead", "drums": "support",
                        "bass": "support", "guitar": "background",
                        "piano": "texture", "other": "texture",
                    },
                    transition_in="crossfade", transition_beats=4,
                ),
                IntentSection(
                    label="chorus", start_beat=48, end_beat=80, energy="high",
                    stem_roles={
                        "vocals": "lead", "drums": "support",
                        "bass": "support", "guitar": "support",
                        "piano": "background", "other": "background",
                    },
                    transition_in="cut", transition_beats=0,
                ),
                IntentSection(
                    label="breakdown", start_beat=80, end_beat=96, energy="low",
                    stem_roles={
                        "vocals": "background", "drums": "silent",
                        "bass": "texture", "guitar": "lead",
                        "piano": "support", "other": "texture",
                    },
                    transition_in="crossfade", transition_beats=4,
                ),
                IntentSection(
                    label="drop", start_beat=96, end_beat=128, energy="peak",
                    stem_roles={
                        "vocals": "lead", "drums": "lead",
                        "bass": "support", "guitar": "support",
                        "piano": "background", "other": "support",
                    },
                    transition_in="cut", transition_beats=0,
                ),
                IntentSection(
                    label="outro", start_beat=128, end_beat=144, energy="low",
                    stem_roles={
                        "vocals": "texture", "drums": "silent",
                        "bass": "texture", "guitar": "background",
                        "piano": "lead", "other": "silent",
                    },
                    transition_in="fade", transition_beats=8,
                ),
            ],
            key_source="song_b",
            explanation="Vocal-forward remix with a heavy drop section.",
            warnings=["Key conflict between songs may cause dissonance."],
        )

    def test_realistic_plan_converts(self):
        """Full conversion should produce a valid RemixPlan."""
        intent = self._realistic_plan()
        plan = map_intent_to_gains(intent)
        assert isinstance(plan, RemixPlan)
        assert len(plan.sections) == 6

    def test_chorus_louder_than_verse(self):
        """Chorus (high energy) should have higher total gains than verse (medium)."""
        plan = map_intent_to_gains(self._realistic_plan())
        verse_total = sum(plan.sections[1].stem_gains.values())
        chorus_total = sum(plan.sections[2].stem_gains.values())
        assert chorus_total > verse_total

    def test_drop_loudest_section(self):
        """Drop (peak energy) should be the loudest section."""
        plan = map_intent_to_gains(self._realistic_plan())
        totals = [sum(s.stem_gains.values()) for s in plan.sections]
        drop_idx = 4  # "drop" section
        assert totals[drop_idx] == max(totals)

    def test_intro_quietest_section(self):
        """Intro (low energy, few active stems) should be among the quietest."""
        plan = map_intent_to_gains(self._realistic_plan())
        totals = [sum(s.stem_gains.values()) for s in plan.sections]
        intro_total = totals[0]
        # Intro should be less than chorus and drop
        assert intro_total < totals[2]  # chorus
        assert intro_total < totals[4]  # drop

    def test_lead_stems_dominate(self):
        """In each section, the 'lead' stem should have the highest gain."""
        intent = self._realistic_plan()
        plan = map_intent_to_gains(intent)
        for intent_sec, sec in zip(intent.sections, plan.sections):
            lead_stems = [
                s for s, role in intent_sec.stem_roles.items() if role == "lead"
            ]
            if not lead_stems:
                continue
            max_lead_gain = max(sec.stem_gains[s] for s in lead_stems)
            non_lead_gains = [
                sec.stem_gains[s] for s in ALL_STEMS if s not in lead_stems
            ]
            if non_lead_gains:
                max_non_lead = max(non_lead_gains)
                assert max_lead_gain >= max_non_lead, (
                    f"Section '{sec.label}': lead gain {max_lead_gain} < "
                    f"non-lead {max_non_lead}"
                )

    def test_with_lufs_data(self):
        """Conversion with realistic LUFS data should still produce valid output."""
        intent = self._realistic_plan()
        plan = map_intent_to_gains(
            intent,
            vocal_stem_lufs={"vocals": -22.0},
            inst_stem_lufs={
                "drums": -18.0,  # slightly loud
                "bass": -25.0,   # slightly quiet
                "guitar": -20.0, # neutral
                "piano": -30.0,  # quiet, will boost
                "other": -20.0,  # neutral
            },
        )
        assert isinstance(plan, RemixPlan)
        # Piano should be boosted compared to no-LUFS version
        plan_no_lufs = map_intent_to_gains(intent)
        # Check verse section (index 1) where piano is "texture"
        assert plan.sections[1].stem_gains["piano"] >= plan_no_lufs.sections[1].stem_gains["piano"]

    def test_warnings_propagated(self):
        """Intent warnings plus energy budget warnings should appear in output."""
        intent = self._realistic_plan()
        plan = map_intent_to_gains(intent)
        # The original warning should be present
        assert any("Key conflict" in w for w in plan.warnings)

    def test_vocal_source_set(self):
        """RemixPlan should use the standard vocal source convention."""
        plan = map_intent_to_gains(self._realistic_plan())
        assert plan.vocal_source == "song_a"

    def test_transition_info_preserved(self):
        """Transition type and beats should be preserved from intent."""
        intent = self._realistic_plan()
        plan = map_intent_to_gains(intent)
        for intent_sec, sec in zip(intent.sections, plan.sections):
            assert sec.transition_in == intent_sec.transition_in
            assert sec.transition_beats == intent_sec.transition_beats


# ---------------------------------------------------------------------------
# 11. TestEnergyBudgetAutoCorrect
# ---------------------------------------------------------------------------

class TestEnergyBudgetAutoCorrect:
    """Energy budget auto-correction scales gains up when total is below target."""

    def test_below_target_gets_scaled_up(self):
        """Section below energy budget minimum gets non-silent gains scaled up."""
        # Build a chorus with deliberately low gains (all texture)
        section = _make_intent_section(
            label="chorus", energy="high",
            stem_roles={
                "vocals": "texture", "drums": "texture", "bass": "texture",
                "guitar": "texture", "piano": "silent", "other": "silent",
            },
        )
        gains = [_compute_section_gains(section, None, None)]
        total_before = sum(gains[0].values())
        lo, hi = ENERGY_BUDGET_TARGETS["chorus"]
        assert total_before < lo, "Pre-condition: total should be below target"

        warnings = _validate_energy_budgets(gains, ["chorus"])
        total_after = sum(gains[0].values())

        assert total_after >= lo, "Auto-correct should bring total to at least the minimum"
        assert len(warnings) == 0, "No warnings expected after successful auto-correct"

    def test_auto_correct_preserves_ordering(self):
        """After scaling, relative ordering of gains is preserved (below cap)."""
        # Use a verse target (3.0-4.5) with low energy so the scale factor
        # is moderate and doesn't push everything to the 1.0 cap.
        section = _make_intent_section(
            label="verse", energy="low",
            stem_roles={
                "vocals": "lead", "drums": "support", "bass": "background",
                "guitar": "texture", "piano": "texture", "other": "silent",
            },
        )
        gains = [_compute_section_gains(section, None, None)]
        _validate_energy_budgets(gains, ["verse"])

        assert gains[0]["vocals"] >= gains[0]["drums"]
        assert gains[0]["drums"] >= gains[0]["bass"]
        assert gains[0]["bass"] >= gains[0]["guitar"]
        assert gains[0]["other"] == 0.0  # silent stays silent

    def test_auto_correct_caps_at_one(self):
        """Individual gains never exceed 1.0 after auto-correct."""
        # Chorus with a single active stem forces a large scale factor
        section = _make_intent_section(
            label="chorus", energy="low",
            stem_roles={
                "vocals": "texture", "drums": "silent", "bass": "silent",
                "guitar": "silent", "piano": "silent", "other": "silent",
            },
        )
        gains = [_compute_section_gains(section, None, None)]
        _validate_energy_budgets(gains, ["chorus"])

        for stem, val in gains[0].items():
            assert val <= 1.0, f"{stem} gain {val} exceeds 1.0 after auto-correct"

    def test_above_target_produces_warning(self):
        """Section above energy budget still produces a warning (no downward correction)."""
        # Intro with all stems at lead + peak energy — will be way above intro target
        section = _make_intent_section(
            label="intro", energy="peak",
            stem_roles={s: "lead" for s in ALL_STEMS},
        )
        gains = [_compute_section_gains(section, None, None)]
        total = sum(gains[0].values())
        _, hi = ENERGY_BUDGET_TARGETS["intro"]
        assert total > hi, "Pre-condition: total should be above target"

        warnings = _validate_energy_budgets(gains, ["intro"])
        assert len(warnings) == 1
        assert "above" in warnings[0]

    def test_within_target_no_change(self):
        """Section within budget range is not modified."""
        section = _make_intent_section(
            label="verse", energy="medium",
        )
        gains = [_compute_section_gains(section, None, None)]
        original = dict(gains[0])

        warnings = _validate_energy_budgets(gains, ["verse"])

        assert len(warnings) == 0
        for stem in ALL_STEMS:
            assert gains[0][stem] == pytest.approx(original[stem], abs=0.001)

    def test_all_silent_below_target_warns(self):
        """All-silent section below budget warns instead of scaling (nothing to scale)."""
        gains = [{"vocals": 0.0, "drums": 0.0, "bass": 0.0,
                  "guitar": 0.0, "piano": 0.0, "other": 0.0}]
        warnings = _validate_energy_budgets(gains, ["chorus"])
        assert len(warnings) == 1
        assert "all stems silent" in warnings[0]


# ---------------------------------------------------------------------------
# Extra: intent_section_to_section helper
# ---------------------------------------------------------------------------

class TestIntentSectionToSection:
    """Tests for the single-section conversion helper."""

    def test_returns_section(self):
        result = intent_section_to_section(_make_intent_section())
        assert isinstance(result, Section)

    def test_all_stems_present(self):
        result = intent_section_to_section(_make_intent_section())
        for stem in ALL_STEMS:
            assert stem in result.stem_gains

    def test_with_lufs(self):
        result = intent_section_to_section(
            _make_intent_section(),
            vocal_lufs={"vocals": -20.0},
            inst_lufs={"drums": -20.0},
        )
        assert isinstance(result, Section)

    def test_label_preserved(self):
        result = intent_section_to_section(
            _make_intent_section(label="bridge")
        )
        assert result.label == "bridge"
