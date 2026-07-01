"""Unit tests for the key convergence algorithm."""

from __future__ import annotations

import pytest

from musicmixer.services.key_matching import (
    KeyPlan,
    chromatic_distance,
    compute_key_plan,
    note_to_semitone,
    signed_shift,
)


# ---------------------------------------------------------------------------
# note_to_semitone
# ---------------------------------------------------------------------------


class TestNoteToSemitone:
    """Test note name → semitone mapping."""

    def test_natural_notes(self):
        assert note_to_semitone("C") == 0
        assert note_to_semitone("D") == 2
        assert note_to_semitone("E") == 4
        assert note_to_semitone("F") == 5
        assert note_to_semitone("G") == 7
        assert note_to_semitone("A") == 9
        assert note_to_semitone("B") == 11

    def test_sharps(self):
        assert note_to_semitone("C#") == 1
        assert note_to_semitone("D#") == 3
        assert note_to_semitone("F#") == 6
        assert note_to_semitone("G#") == 8
        assert note_to_semitone("A#") == 10

    def test_flats(self):
        assert note_to_semitone("Db") == 1
        assert note_to_semitone("Eb") == 3
        assert note_to_semitone("Gb") == 6
        assert note_to_semitone("Ab") == 8
        assert note_to_semitone("Bb") == 10

    def test_enharmonic_e_sharp(self):
        assert note_to_semitone("E#") == 5
        assert note_to_semitone("E#") == note_to_semitone("F")

    def test_enharmonic_b_sharp(self):
        assert note_to_semitone("B#") == 0
        assert note_to_semitone("B#") == note_to_semitone("C")

    def test_enharmonic_fb(self):
        assert note_to_semitone("Fb") == 4
        assert note_to_semitone("Fb") == note_to_semitone("E")

    def test_enharmonic_cb(self):
        assert note_to_semitone("Cb") == 11
        assert note_to_semitone("Cb") == note_to_semitone("B")

    def test_enharmonic_equivalents_csharp_db(self):
        """C# and Db should map to the same semitone."""
        assert note_to_semitone("C#") == note_to_semitone("Db")

    def test_enharmonic_equivalents_gsharp_ab(self):
        """G# and Ab should map to the same semitone."""
        assert note_to_semitone("G#") == note_to_semitone("Ab")

    def test_unrecognized_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized key"):
            note_to_semitone("X")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            note_to_semitone("")


# ---------------------------------------------------------------------------
# chromatic_distance
# ---------------------------------------------------------------------------


class TestChromaticDistance:
    """Test minimum distance on the chromatic circle."""

    def test_same_note_is_zero(self):
        assert chromatic_distance(0, 0) == 0

    def test_adjacent_notes(self):
        assert chromatic_distance(0, 1) == 1

    def test_symmetric(self):
        assert chromatic_distance(0, 4) == chromatic_distance(4, 0)

    def test_max_distance_is_6(self):
        # C to F# (tritone) = 6
        assert chromatic_distance(0, 6) == 6

    def test_wraps_around(self):
        # C(0) to B(11): clockwise=11, counter=1 → min=1
        assert chromatic_distance(0, 11) == 1

    def test_distance_5(self):
        # C(0) to F(5) = 5
        assert chromatic_distance(0, 5) == 5
        # C(0) to G#(8): clockwise=8, counter=4 → min=4
        assert chromatic_distance(0, 8) == 4

    def test_all_distances_in_range(self):
        """Every pair should produce a distance in [0, 6]."""
        for a in range(12):
            for b in range(12):
                d = chromatic_distance(a, b)
                assert 0 <= d <= 6


# ---------------------------------------------------------------------------
# signed_shift
# ---------------------------------------------------------------------------


class TestSignedShift:
    """Test shortest signed shift between semitones."""

    def test_same_note_is_zero(self):
        assert signed_shift(0, 0) == 0

    def test_up_one(self):
        assert signed_shift(0, 1) == 1

    def test_down_one(self):
        # C(0) to B(11): shortest is -1
        assert signed_shift(0, 11) == -1

    def test_tritone_returns_6(self):
        # C(0) to F#(6): exactly 6, returned as +6
        assert signed_shift(0, 6) == 6

    def test_range_minus6_to_plus6(self):
        for a in range(12):
            for b in range(12):
                s = signed_shift(a, b)
                assert -6 <= s <= 6


# ---------------------------------------------------------------------------
# compute_key_plan: gate conditions
# ---------------------------------------------------------------------------


class TestKeyPlanGates:
    """Test early-exit (skip) conditions."""

    def test_rap_vocals_skips(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "F#", "major", 0.90, False,
            rap_vocals=True,
        )
        assert plan.action == "skip"
        assert "rap vocals" in plan.reason

    def test_missing_key_a_skips(self):
        plan = compute_key_plan(
            None, "major", 0.90, False,
            "C", "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert "missing key data" in plan.reason

    def test_missing_key_b_skips(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            None, "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert "missing key data" in plan.reason

    def test_missing_scale_a_skips(self):
        plan = compute_key_plan(
            "C", None, 0.90, False,
            "D", "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert "missing key data" in plan.reason

    def test_missing_scale_b_skips(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D", None, 0.90, False,
        )
        assert plan.action == "skip"
        assert "missing key data" in plan.reason

    def test_low_confidence_a_skips(self):
        plan = compute_key_plan(
            "C", "major", 0.30, False,
            "D", "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert "low confidence" in plan.reason

    def test_low_confidence_b_skips(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D", "major", 0.35, False,
        )
        assert plan.action == "skip"
        assert "low confidence" in plan.reason

    def test_confidence_exactly_040_does_not_skip(self):
        """Confidence at threshold should proceed, not skip."""
        plan = compute_key_plan(
            "C", "major", 0.40, False,
            "D", "major", 0.40, False,
        )
        assert plan.action != "skip" or "low confidence" not in plan.reason

    def test_none_confidence_proceeds(self):
        """If confidence is None, proceed (don't skip)."""
        plan = compute_key_plan(
            "C", "major", None, False,
            "D", "major", None, False,
        )
        assert plan.action != "skip" or "low confidence" not in plan.reason

    def test_modulation_detected_proceeds(self):
        """Modulation flag should NOT cause a skip."""
        plan = compute_key_plan(
            "C", "major", 0.85, True,
            "D", "major", 0.85, True,
        )
        assert plan.action != "skip"
        assert plan.distance == 2


# ---------------------------------------------------------------------------
# compute_key_plan: same mode, various distances
# ---------------------------------------------------------------------------


class TestSameModeDistances:
    """Test shift allocation for same-mode key pairs."""

    def test_distance_0_same_key(self):
        """Same key → skip."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "C", "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert plan.shift_a == 0
        assert plan.shift_b == 0
        assert plan.distance == 0

    def test_distance_1(self):
        """C major vs C# major → instrumental shifts 1, vocal shifts 0."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "C#", "major", 0.90, False,
        )
        assert plan.action == "shift"
        assert plan.distance == 1
        assert abs(plan.shift_a) == 0
        assert abs(plan.shift_b) == 1

    def test_distance_2(self):
        """C major vs D major → instrumental shifts 2, vocal shifts 0."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D", "major", 0.90, False,
        )
        assert plan.action == "shift"
        assert plan.distance == 2
        assert abs(plan.shift_a) == 0
        assert abs(plan.shift_b) == 2

    def test_distance_3(self):
        """C major vs D# major → instrumental shifts 3, vocal shifts 0."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D#", "major", 0.90, False,
        )
        assert plan.action == "shift"
        assert plan.distance == 3
        assert abs(plan.shift_a) == 0
        assert abs(plan.shift_b) == 3

    def test_distance_4(self):
        """C major vs E major → instrumental shifts 4, vocal shifts 0."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "E", "major", 0.90, False,
        )
        assert plan.action == "shift"
        assert plan.distance == 4
        assert abs(plan.shift_a) == 0
        assert abs(plan.shift_b) == 4

    def test_distance_5(self):
        """C major vs F major → instrumental shifts 4, vocal shifts 1."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "F", "major", 0.90, False,
        )
        assert plan.action == "shift"
        assert plan.distance == 5
        assert abs(plan.shift_a) == 1
        assert abs(plan.shift_b) == 4

    def test_distance_6_warning(self):
        """C major vs F# major → warning action, instrumental 4, vocal 2."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "F#", "major", 0.90, False,
        )
        assert plan.action == "warning"
        assert plan.distance == 6
        assert abs(plan.shift_a) == 2
        assert abs(plan.shift_b) == 4

    def test_shifts_toward_each_other(self):
        """Songs should shift TOWARD each other, not away."""
        # C(0) vs E(4): distance 4
        # B should shift toward A (down from E toward C) = negative
        # A stays at 0 (vocal_mag=0 for dist 4)
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "E", "major", 0.90, False,
        )
        assert plan.shift_b < 0  # E shifts down toward C
        assert plan.shift_a == 0

    def test_wrapping_direction(self):
        """B(11) vs C(0): distance 1. B should shift up (+1) to reach C."""
        plan = compute_key_plan(
            "B", "major", 0.90, False,
            "C", "major", 0.90, False,
        )
        # A=B(11), B_song=C(0), dist=1
        # B_song(C) should shift toward A(B): signed_shift(0, 11) = -1
        # So shift_b = -1 (C shifts down to B)
        assert plan.distance == 1
        assert abs(plan.shift_b) == 1


# ---------------------------------------------------------------------------
# compute_key_plan: different modes
# ---------------------------------------------------------------------------


class TestDifferentModes:
    """Test relative key conversion for mixed major/minor pairs."""

    def test_relative_keys_distance_0(self):
        """A minor + C major → relative keys → distance 0 → skip.

        A minor's relative major is C (A + 3 = C). So after conversion
        they're both at C, distance 0.
        """
        plan = compute_key_plan(
            "A", "minor", 0.90, False,
            "C", "major", 0.90, False,
        )
        assert plan.action == "skip"
        assert plan.distance == 0

    def test_relative_keys_reverse(self):
        """C major + A minor → also relative keys → distance 0."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "A", "minor", 0.90, False,
        )
        assert plan.action == "skip"
        assert plan.distance == 0

    def test_different_mode_picks_smaller_path(self):
        """Should try both conversion paths and pick the smaller distance."""
        # D minor(2) vs G major(7)
        # Path 1: convert D minor → F major (2+3=5), dist from F(5) to G(7) = 2
        # Path 2: convert G major → E minor (7-3=4), dist from D(2) to E(4) = 2
        # Both equal, either is fine
        plan = compute_key_plan(
            "D", "minor", 0.90, False,
            "G", "major", 0.90, False,
        )
        assert plan.distance == 2
        assert plan.action == "shift"

    def test_different_mode_nonzero_distance(self):
        """E minor + A major: not relative keys, should produce a shift."""
        # E minor(4) vs A major(9)
        # Path 1: E minor → G major (4+3=7), dist(7, 9) = 2
        # Path 2: A major → F# minor (9-3=6), dist(4, 6) = 2
        plan = compute_key_plan(
            "E", "minor", 0.90, False,
            "A", "major", 0.90, False,
        )
        assert plan.distance == 2
        assert plan.action == "shift"


# ---------------------------------------------------------------------------
# chromatic_distance invariant: distance never exceeds 6
# ---------------------------------------------------------------------------


class TestChromaticDistanceInvariant:
    """Chromatic distance is bounded to 0-6, so no two songs are ever
    key-incompatible. This test enforces that contract."""

    def test_max_chromatic_distance_is_6(self):
        """Verify that no pair of semitones can produce distance > 6."""
        for a in range(12):
            for b in range(12):
                assert chromatic_distance(a, b) <= 6


# ---------------------------------------------------------------------------
# compute_key_plan: rap mode overrides everything
# ---------------------------------------------------------------------------


class TestRapMode:
    """Rap vocals should always skip, regardless of key distance."""

    def test_rap_with_distance_0(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "C", "major", 0.90, False,
            rap_vocals=True,
        )
        assert plan.action == "skip"
        assert "rap vocals" in plan.reason

    def test_rap_with_distance_6(self):
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "F#", "major", 0.90, False,
            rap_vocals=True,
        )
        assert plan.action == "skip"
        assert "rap vocals" in plan.reason

    def test_rap_with_missing_data(self):
        """Rap skip takes priority over missing data skip."""
        plan = compute_key_plan(
            None, None, None, None,
            None, None, None, None,
            rap_vocals=True,
        )
        assert plan.action == "skip"
        assert "rap vocals" in plan.reason


# ---------------------------------------------------------------------------
# compute_key_plan: target key correctness
# ---------------------------------------------------------------------------


class TestTargetKey:
    """Verify the target_key field is populated correctly."""

    def test_distance_0_target_key(self):
        plan = compute_key_plan(
            "G", "major", 0.90, False,
            "G", "major", 0.90, False,
        )
        assert plan.target_key == "G"
        assert plan.target_scale == "major"

    def test_shift_target_key_is_convergence_point(self):
        """After shifting, both songs should converge to target_key."""
        # C(0) vs D(2), dist=2, inst shifts 2 toward C
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D", "major", 0.90, False,
        )
        assert plan.target_key == "C"
        assert plan.target_scale == "major"

    def test_relative_keys_target(self):
        """A minor + C major → skip with target key populated."""
        plan = compute_key_plan(
            "A", "minor", 0.90, False,
            "C", "major", 0.90, False,
        )
        # After conversion, both map to same semitone. Target should be set.
        assert plan.target_key != ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_enharmonic_keys_same_distance(self):
        """C# and Db are enharmonic — distance to any key should be identical."""
        plan_sharp = compute_key_plan(
            "C#", "major", 0.90, False,
            "E", "major", 0.90, False,
        )
        plan_flat = compute_key_plan(
            "Db", "major", 0.90, False,
            "E", "major", 0.90, False,
        )
        assert plan_sharp.distance == plan_flat.distance
        assert plan_sharp.action == plan_flat.action
        assert abs(plan_sharp.shift_a) == abs(plan_flat.shift_a)
        assert abs(plan_sharp.shift_b) == abs(plan_flat.shift_b)

    def test_all_keys_produce_valid_plan(self):
        """Every valid key pair should produce a non-error plan."""
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for k_a in keys:
            for k_b in keys:
                plan = compute_key_plan(
                    k_a, "major", 0.90, False,
                    k_b, "major", 0.90, False,
                )
                assert plan.action in ("skip", "shift", "warning")

    def test_minor_keys_same_distance_as_major(self):
        """Distance between two minor keys should match same pair as major."""
        plan_major = compute_key_plan(
            "C", "major", 0.90, False,
            "E", "major", 0.90, False,
        )
        plan_minor = compute_key_plan(
            "C", "minor", 0.90, False,
            "E", "minor", 0.90, False,
        )
        assert plan_major.distance == plan_minor.distance

    def test_keyplan_dataclass_fields(self):
        """Verify KeyPlan has all expected fields."""
        plan = compute_key_plan(
            "C", "major", 0.90, False,
            "D", "major", 0.90, False,
        )
        assert isinstance(plan, KeyPlan)
        assert isinstance(plan.action, str)
        assert isinstance(plan.shift_a, float)
        assert isinstance(plan.shift_b, float)
        assert isinstance(plan.target_key, str)
        assert isinstance(plan.target_scale, str)
        assert isinstance(plan.reason, str)
        assert isinstance(plan.distance, int)
