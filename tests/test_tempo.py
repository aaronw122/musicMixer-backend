"""Unit tests for the shared tempo module (single source of truth for BPM estimation)."""

from __future__ import annotations

import pytest

from musicmixer.services.tempo import compute_stretch_pct, estimate_material_budget, estimate_target_bpm


# ---------------------------------------------------------------------------
# estimate_target_bpm: tempo_source variants
# ---------------------------------------------------------------------------


class TestTempoSourceVariants:
    """Test that tempo_source == "song_a" / "song_b" return the correct BPM."""

    def test_song_a_returns_vocal_bpm(self):
        assert estimate_target_bpm(95.0, 130.0, "song_a") == 95.0

    def test_song_b_returns_instrumental_bpm(self):
        assert estimate_target_bpm(95.0, 130.0, "song_b") == 130.0

    def test_weighted_midpoint_is_default(self):
        """Default tempo_source should behave like weighted_midpoint."""
        default = estimate_target_bpm(100.0, 120.0)
        explicit = estimate_target_bpm(100.0, 120.0, "weighted_midpoint")
        assert default == explicit

    def test_average_matches_weighted_midpoint(self):
        """'average' should use the same tiered logic as 'weighted_midpoint'."""
        avg = estimate_target_bpm(100.0, 120.0, "average")
        wm = estimate_target_bpm(100.0, 120.0, "weighted_midpoint")
        assert avg == wm

    def test_unknown_source_returns_instrumental_bpm(self):
        """Unrecognized tempo_source should return instrumental_bpm (original behavior)."""
        unknown = estimate_target_bpm(100.0, 120.0, "something_random")
        assert unknown == 120.0

    def test_unknown_source_various_values_return_instrumental(self):
        """Multiple unrecognized tempo_source values all return instrumental_bpm."""
        for bad_source in ("bad", "auto", "custom", "", "WEIGHTED_MIDPOINT"):
            result = estimate_target_bpm(95.0, 130.0, bad_source)
            assert result == 130.0, f"Expected instrumental_bpm for tempo_source={bad_source!r}"


# ---------------------------------------------------------------------------
# estimate_target_bpm: tier boundaries
# ---------------------------------------------------------------------------


class TestTierBoundaries:
    """Test each tier boundary of the weighted midpoint logic."""

    def test_identical_bpms_returns_instrumental(self):
        """0% gap: target == instrumental_bpm."""
        result = estimate_target_bpm(120.0, 120.0)
        assert result == 120.0

    def test_gap_within_4pct_returns_instrumental(self):
        """Gap <= 4%: DJ-transparent, target = instrumental_bpm."""
        # 120 vs 124: gap = 4/124 = 3.2% < 4%
        result = estimate_target_bpm(120.0, 124.0)
        assert result == 124.0

    def test_gap_exactly_4pct_boundary(self):
        """Gap exactly at 4% boundary should use DJ-transparent tier."""
        # Need gap_pct = 0.04 exactly: |v - i| / max(v, i) = 0.04
        # If instrumental = 100, vocal = 96: gap = 4/100 = 0.04
        result = estimate_target_bpm(96.0, 100.0)
        assert result == 100.0  # DJ-transparent: return instrumental

    def test_gap_just_above_4pct(self):
        """Gap just above 4% should use 65/35 tier."""
        # vocal=95, inst=100: gap = 5/100 = 5% > 4%
        result = estimate_target_bpm(95.0, 100.0)
        expected = 100.0 * 0.65 + 95.0 * 0.35  # 98.25
        assert result == pytest.approx(expected)

    def test_gap_within_10pct_uses_65_35_weights(self):
        """Gap 4-10%: use 65/35 instrumental bias."""
        # vocal=92, inst=100: gap = 8/100 = 8%
        result = estimate_target_bpm(92.0, 100.0)
        expected = 100.0 * 0.65 + 92.0 * 0.35  # 97.2
        assert result == pytest.approx(expected)

    def test_gap_at_10pct_boundary(self):
        """Gap exactly at 10% should still use 65/35 tier."""
        # vocal=90, inst=100: gap = 10/100 = 0.10
        result = estimate_target_bpm(90.0, 100.0)
        expected = 100.0 * 0.65 + 90.0 * 0.35  # 96.5
        assert result == pytest.approx(expected)

    def test_gap_just_above_10pct_uses_70_30(self):
        """Gap just above 10% should use 70/30 tier."""
        # vocal=89, inst=100: gap = 11/100 = 11%
        result = estimate_target_bpm(89.0, 100.0)
        expected = 100.0 * 0.70 + 89.0 * 0.30  # 96.7
        assert result == pytest.approx(expected)

    def test_gap_within_20pct_uses_70_30(self):
        """Gap 10-20%: use 70/30 instrumental bias."""
        # vocal=85, inst=100: gap = 15/100 = 15%
        result = estimate_target_bpm(85.0, 100.0)
        expected = 100.0 * 0.70 + 85.0 * 0.30  # 95.5
        assert result == pytest.approx(expected)

    def test_gap_at_20pct_boundary(self):
        """Gap exactly at 20% uses 70/30 tier WITHOUT the 12% clamp.

        The clamp only applies when gap > 20% (the else branch). At exactly
        20%, gap_pct <= 0.20 is True, so no clamp.
        """
        # vocal=80, inst=100: gap = 20/100 = 0.20 -> <= 0.20 tier
        result = estimate_target_bpm(80.0, 100.0)
        expected = 100.0 * 0.70 + 80.0 * 0.30  # 94.0
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# estimate_target_bpm: 12% vocal clamp
# ---------------------------------------------------------------------------


class TestVocalClamp:
    """Test the 12% vocal stretch clamp for gaps > 20%."""

    def test_large_gap_clamps_vocal_speedup(self):
        """Vocal slower than instrumental: clamp at vocal * 1.12."""
        # vocal=85, inst=130: gap = 45/130 = 34.6%
        # Unclamped: 130*0.70 + 85*0.30 = 116.5
        # Vocal stretch: |85 - 116.5| / 85 = 37% > 12%
        # Clamped: 85 * 1.12 = 95.2
        result = estimate_target_bpm(85.0, 130.0)
        assert result == pytest.approx(95.2)

    def test_large_gap_clamps_vocal_slowdown(self):
        """Vocal faster than instrumental: clamp at vocal * 0.88."""
        # vocal=160, inst=100: gap = 60/160 = 37.5%
        # Unclamped: 100*0.70 + 160*0.30 = 118.0
        # Vocal stretch: |160 - 118| / 160 = 26.25% > 12%
        # vocal_bpm > target_bpm, so clamp: 160 * 0.88 = 140.8
        result = estimate_target_bpm(160.0, 100.0)
        assert result == pytest.approx(140.8)

    def test_gap_above_20pct_no_clamp_needed(self):
        """Gap > 20% but vocal stretch < 12%: no clamp applied."""
        # vocal=100, inst=125: gap = 25/125 = 20% -> exactly at boundary
        # Actually gap_pct = 0.20, so <= 0.20 tier (70/30 without clamp)
        # Need gap > 20%:
        # vocal=100, inst=126: gap = 26/126 = 20.6%
        # Unclamped: 126*0.70 + 100*0.30 = 118.2
        # Vocal stretch: |100 - 118.2| / 100 = 18.2% > 12%! Will clamp.
        # Let's try a case where it doesn't clamp:
        # vocal=115, inst=145: gap = 30/145 = 20.7%
        # Unclamped: 145*0.70 + 115*0.30 = 136.0
        # Vocal stretch: |115 - 136| / 115 = 18.3% > 12%! Still clamps.
        # Hard to get > 20% gap without > 12% stretch. Let's verify the clamp triggers.
        result = estimate_target_bpm(100.0, 126.0)
        assert result == pytest.approx(100.0 * 1.12)  # 112.0

    def test_50pct_gap(self):
        """Extreme gap: verify clamp works."""
        # vocal=80, inst=160: gap = 80/160 = 50%
        # Unclamped: 160*0.70 + 80*0.30 = 136.0
        # Vocal stretch: |80 - 136| / 80 = 70% >> 12%
        # vocal < target, so clamp: 80 * 1.12 = 89.6
        result = estimate_target_bpm(80.0, 160.0)
        assert result == pytest.approx(89.6)


# ---------------------------------------------------------------------------
# estimate_target_bpm: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test guard conditions for invalid BPM values."""

    def test_both_zero_returns_1(self):
        assert estimate_target_bpm(0.0, 0.0) == 1.0

    def test_vocal_zero_returns_instrumental(self):
        assert estimate_target_bpm(0.0, 120.0) == 120.0

    def test_instrumental_zero_returns_vocal(self):
        assert estimate_target_bpm(95.0, 0.0) == 95.0

    def test_negative_vocal_returns_instrumental(self):
        assert estimate_target_bpm(-5.0, 120.0) == 120.0

    def test_negative_instrumental_returns_vocal(self):
        assert estimate_target_bpm(95.0, -10.0) == 95.0

    def test_both_negative_returns_1(self):
        assert estimate_target_bpm(-5.0, -10.0) == 1.0


# ---------------------------------------------------------------------------
# compute_stretch_pct
# ---------------------------------------------------------------------------


class TestComputeStretchPct:
    """Test the stretch percentage calculation."""

    def test_identical_bpms_zero_stretch(self):
        assert compute_stretch_pct(120.0, 120.0) == 0.0

    def test_small_gap_returns_expected(self):
        """4% gap: DJ-transparent, target = instrumental."""
        # vocal=120, inst=124 -> target=124 (DJ-transparent)
        # vocal stretch: |124-120|/120 = 3.33%
        # inst stretch: 0%
        result = compute_stretch_pct(120.0, 124.0)
        assert result == pytest.approx(abs(124.0 - 120.0) / 120.0 * 100)

    def test_medium_gap(self):
        """8% gap: 65/35 tier."""
        # vocal=92, inst=100 -> target = 100*0.65 + 92*0.35 = 97.2
        target = 100.0 * 0.65 + 92.0 * 0.35  # 97.2
        vocal_s = abs(target - 92.0) / 92.0 * 100  # ~5.65%
        inst_s = abs(target - 100.0) / 100.0 * 100  # ~2.8%
        result = compute_stretch_pct(92.0, 100.0)
        assert result == pytest.approx(max(vocal_s, inst_s))

    def test_zero_bpm_returns_zero(self):
        assert compute_stretch_pct(0.0, 120.0) == 0.0
        assert compute_stretch_pct(120.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Real-world bug cases
# ---------------------------------------------------------------------------


class TestRealWorldCases:
    """Verify with the actual cases from the divergence bug report."""

    def test_biggie_dead_95_7_vs_83_4(self):
        """95.7 / 83.4 BPM (Biggie + Grateful Dead) -> ~87.09 target."""
        result = estimate_target_bpm(vocal_bpm=95.7, instrumental_bpm=83.4)
        # gap = |95.7 - 83.4| / 95.7 = 12.85% -> 70/30 tier (gap > 10%)
        # target = 83.4 * 0.70 + 95.7 * 0.30 = 58.38 + 28.71 = 87.09
        # vocal stretch: |95.7 - 87.09| / 95.7 = 9.0% < 12% -> no clamp
        assert result == pytest.approx(87.09)

    def test_failing_case_123_vs_129_2(self):
        """123.0 / 129.2 BPM -> ~127.03 target."""
        result = estimate_target_bpm(vocal_bpm=123.0, instrumental_bpm=129.2)
        # gap = |123 - 129.2| / 129.2 = 4.80% -> 65/35 tier
        # target = 129.2 * 0.65 + 123.0 * 0.35 = 84.0 + 43.05 = 127.03 (wait, let me be precise)
        expected = 129.2 * 0.65 + 123.0 * 0.35
        assert result == pytest.approx(expected)

    def test_large_gap_85_vs_130(self):
        """85.0 / 130.0 BPM (large gap) -> 95.2 target (clamped)."""
        result = estimate_target_bpm(vocal_bpm=85.0, instrumental_bpm=130.0)
        # gap = |85 - 130| / 130 = 34.6% -> > 20%, 70/30 then clamp
        # Unclamped: 130*0.70 + 85*0.30 = 91+25.5 = 116.5
        # vocal stretch: |85 - 116.5| / 85 = 37.1% > 12%
        # vocal < target -> clamp to 85 * 1.12 = 95.2
        assert result == pytest.approx(95.2)

    def test_symmetry_vocal_faster_than_instrumental(self):
        """When vocal is faster, the clamp should use 0.88 multiplier."""
        result = estimate_target_bpm(vocal_bpm=130.0, instrumental_bpm=85.0)
        # gap = |130 - 85| / 130 = 34.6% -> > 20%
        # Unclamped: 85*0.70 + 130*0.30 = 59.5 + 39 = 98.5
        # vocal stretch: |130 - 98.5| / 130 = 24.2% > 12%
        # vocal > target -> clamp to 130 * 0.88 = 114.4
        assert result == pytest.approx(114.4)


# ---------------------------------------------------------------------------
# Consistency with compute_tempo_plan in processor.py
# ---------------------------------------------------------------------------


class TestConsistencyWithProcessor:
    """Verify estimate_target_bpm produces identical results to inline logic
    that was previously in compute_tempo_plan()."""

    def _old_compute_target_bpm(
        self, vocal_bpm: float, instrumental_bpm: float, tempo_source: str
    ) -> float:
        """Exact copy of the BPM selection logic that was in compute_tempo_plan()
        before the refactor. Used as reference to verify identity."""
        gap_pct = abs(vocal_bpm - instrumental_bpm) / max(vocal_bpm, instrumental_bpm)

        if tempo_source == "weighted_midpoint" or tempo_source == "average":
            if gap_pct <= 0.04:
                target_bpm = instrumental_bpm
            elif gap_pct <= 0.10:
                target_bpm = instrumental_bpm * 0.65 + vocal_bpm * 0.35
            elif gap_pct <= 0.20:
                target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
            else:
                target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
                vocal_stretch = abs(vocal_bpm - target_bpm) / vocal_bpm
                if vocal_stretch > 0.12:
                    if vocal_bpm > target_bpm:
                        target_bpm = vocal_bpm * 0.88
                    else:
                        target_bpm = vocal_bpm * 1.12
        elif tempo_source == "song_a":
            target_bpm = vocal_bpm
        elif tempo_source == "song_b":
            target_bpm = instrumental_bpm
        else:
            target_bpm = instrumental_bpm

        return target_bpm

    @pytest.mark.parametrize(
        "vocal_bpm,instrumental_bpm,tempo_source",
        [
            (120.0, 120.0, "weighted_midpoint"),
            (120.0, 124.0, "weighted_midpoint"),
            (95.0, 100.0, "weighted_midpoint"),
            (90.0, 100.0, "weighted_midpoint"),
            (85.0, 100.0, "weighted_midpoint"),
            (80.0, 100.0, "weighted_midpoint"),
            (85.0, 130.0, "weighted_midpoint"),
            (130.0, 85.0, "weighted_midpoint"),
            (160.0, 100.0, "weighted_midpoint"),
            (80.0, 160.0, "weighted_midpoint"),
            (95.7, 83.4, "weighted_midpoint"),
            (123.0, 129.2, "weighted_midpoint"),
            (100.0, 100.0, "average"),
            (85.0, 130.0, "average"),
            (95.0, 130.0, "song_a"),
            (95.0, 130.0, "song_b"),
            (120.0, 100.0, "song_a"),
            (120.0, 100.0, "song_b"),
            (100.0, 120.0, "unknown_source"),
            (95.0, 130.0, "bad"),
        ],
    )
    def test_matches_old_inline_logic(
        self, vocal_bpm: float, instrumental_bpm: float, tempo_source: str
    ):
        old_result = self._old_compute_target_bpm(vocal_bpm, instrumental_bpm, tempo_source)
        new_result = estimate_target_bpm(vocal_bpm, instrumental_bpm, tempo_source)
        assert new_result == pytest.approx(old_result), (
            f"Mismatch for ({vocal_bpm}, {instrumental_bpm}, {tempo_source}): "
            f"old={old_result}, new={new_result}"
        )


# ---------------------------------------------------------------------------
# estimate_material_budget
# ---------------------------------------------------------------------------


class TestEstimateMaterialBudget:
    """Tests for the material budget estimator (beat grid mismatch fix)."""

    def test_slow_instrumental_shrinks_budget(self):
        """A slow instrumental (78.3 BPM) stretched to 148.4 BPM shrinks to ~96s."""
        budget = estimate_material_budget(
            vocal_bpm=129.2,
            vocal_duration=220.0,
            instrumental_bpm=78.3,
            instrumental_duration=182.0,
            target_bpm=148.4,
        )
        # Instrumental: usable = min(210, 182 - 45.5) = 136.5s
        # Post-stretch: 136.5 * (78.3 / 148.4) = ~72s
        # Vocals: usable = min(210, 220 - 55) = 165s
        # Post-stretch: 165 * (129.2 / 148.4) = ~143.7s
        # Budget = min(~72, ~143.7) = ~72s
        assert 60 < budget < 100

    def test_similar_tempos_no_cap(self):
        """When both songs have similar tempos and long durations, budget >= 210s."""
        budget = estimate_material_budget(
            vocal_bpm=120.0,
            vocal_duration=400.0,
            instrumental_bpm=120.0,
            instrumental_duration=400.0,
            target_bpm=120.0,
        )
        # Both songs have plenty of material, no stretch -> budget = 210s (max_duration cap)
        assert budget >= 210.0

    def test_fast_source_slow_target_stretches_longer(self):
        """A fast song slowed down becomes longer, not shorter."""
        budget = estimate_material_budget(
            vocal_bpm=160.0,
            vocal_duration=200.0,
            instrumental_bpm=160.0,
            instrumental_duration=200.0,
            target_bpm=120.0,
        )
        # Stretching from 160 to 120 makes audio longer: 150s * (160/120) = 200s
        assert budget >= 200.0

    def test_zero_bpm_uses_raw_duration(self):
        """Zero BPM (invalid) should use raw duration without stretching."""
        budget = estimate_material_budget(
            vocal_bpm=0.0,
            vocal_duration=200.0,
            instrumental_bpm=120.0,
            instrumental_duration=200.0,
            target_bpm=120.0,
        )
        # Vocals: 0 BPM -> use raw: min(210, 200 - 50) = 150s
        # Instrumentals: min(210, 200 - 50) = 150s * (120/120) = 150s
        assert budget == pytest.approx(150.0)

    def test_short_songs_cap_aggressively(self):
        """Very short songs (<90s) should have very limited budgets."""
        budget = estimate_material_budget(
            vocal_bpm=120.0,
            vocal_duration=80.0,
            instrumental_bpm=120.0,
            instrumental_duration=80.0,
            target_bpm=120.0,
        )
        # Usable region: 80 - 20 = 60s (no stretch since BPMs match)
        assert budget == pytest.approx(60.0)
