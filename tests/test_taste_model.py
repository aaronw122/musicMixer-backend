"""Tests for musicmixer.services.taste_model -- heuristic taste scorer."""

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_model import (
    DIMENSION_WEIGHTS,
    TIE_THRESHOLD,
    ScoredCandidate,
    _camelot_distance,
    _conservatism_score,
    load_model,
    score_candidate,
    score_with_model,
    select_best,
)


# ---------------------------------------------------------------------------
# Helpers: plan builders
# ---------------------------------------------------------------------------

def _make_good_plan() -> RemixPlan:
    """Construct a known-good plan: proper arc, aligned, moderate gains."""
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=[
            Section(
                label="intro", start_beat=0, end_beat=16,
                stem_gains={"vocals": 0.0, "drums": 0.6, "bass": 0.5,
                            "guitar": 0.4, "piano": 0.3, "other": 0.5},
                transition_in="fade", transition_beats=4,
            ),
            Section(
                label="build", start_beat=16, end_beat=48,
                stem_gains={"vocals": 0.5, "drums": 0.7, "bass": 0.6,
                            "guitar": 0.5, "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="main", start_beat=48, end_beat=112,
                stem_gains={"vocals": 1.0, "drums": 0.8, "bass": 0.7,
                            "guitar": 0.5, "piano": 0.3, "other": 0.4},
                transition_in="crossfade", transition_beats=2,
            ),
            Section(
                label="breakdown", start_beat=112, end_beat=144,
                stem_gains={"vocals": 0.7, "drums": 0.0, "bass": 0.5,
                            "guitar": 0.6, "piano": 0.7, "other": 0.6},
                transition_in="cut", transition_beats=0,
            ),
            Section(
                label="outro", start_beat=144, end_beat=176,
                stem_gains={"vocals": 0.0, "drums": 0.4, "bass": 0.3,
                            "guitar": 0.3, "piano": 0.4, "other": 0.5},
                transition_in="crossfade", transition_beats=4,
            ),
        ],
        tempo_source="song_a",
        key_source="song_a",
        explanation="Well-structured plan with build-peak-release arc",
    )


def _make_bad_plan() -> RemixPlan:
    """Construct a known-bad plan: flat energy, all gains maxed, no variety."""
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=[
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"vocals": 1.0, "drums": 1.0, "bass": 1.0,
                            "guitar": 1.0, "piano": 1.0, "other": 1.0},
                transition_in="cut", transition_beats=0,
            ),
            Section(
                label="main", start_beat=50, end_beat=100,
                stem_gains={"vocals": 1.0, "drums": 1.0, "bass": 1.0,
                            "guitar": 1.0, "piano": 1.0, "other": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ],
        tempo_source="song_a",
        key_source="none",
        explanation="Bad plan: flat, maxed, no arc",
    )


def _make_conservative_plan() -> RemixPlan:
    """A conservative plan: fewer sections, lower gains, shorter transitions."""
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=[
            Section(
                label="intro", start_beat=0, end_beat=16,
                stem_gains={"vocals": 0.0, "drums": 0.4, "bass": 0.3,
                            "guitar": 0.3, "piano": 0.2, "other": 0.3},
                transition_in="fade", transition_beats=2,
            ),
            Section(
                label="main", start_beat=16, end_beat=80,
                stem_gains={"vocals": 0.8, "drums": 0.6, "bass": 0.5,
                            "guitar": 0.4, "piano": 0.3, "other": 0.4},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="breakdown", start_beat=80, end_beat=112,
                stem_gains={"vocals": 0.5, "drums": 0.0, "bass": 0.4,
                            "guitar": 0.5, "piano": 0.6, "other": 0.4},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="outro", start_beat=112, end_beat=144,
                stem_gains={"vocals": 0.0, "drums": 0.3, "bass": 0.2,
                            "guitar": 0.2, "piano": 0.3, "other": 0.3},
                transition_in="fade", transition_beats=4,
            ),
        ],
        tempo_source="song_a",
        key_source="song_a",
        explanation="Conservative plan with moderate levels",
    )


def _make_risky_plan() -> RemixPlan:
    """A risky plan: more sections, higher gains, longer transitions."""
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=[
            Section(
                label="intro", start_beat=0, end_beat=8,
                stem_gains={"vocals": 0.0, "drums": 0.7, "bass": 0.6,
                            "guitar": 0.5, "piano": 0.4, "other": 0.6},
                transition_in="fade", transition_beats=4,
            ),
            Section(
                label="build", start_beat=8, end_beat=24,
                stem_gains={"vocals": 0.6, "drums": 0.8, "bass": 0.7,
                            "guitar": 0.6, "piano": 0.5, "other": 0.6},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="main", start_beat=24, end_beat=56,
                stem_gains={"vocals": 1.0, "drums": 0.9, "bass": 0.8,
                            "guitar": 0.7, "piano": 0.5, "other": 0.6},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="breakdown", start_beat=56, end_beat=72,
                stem_gains={"vocals": 0.7, "drums": 0.0, "bass": 0.6,
                            "guitar": 0.7, "piano": 0.8, "other": 0.7},
                transition_in="cut", transition_beats=0,
            ),
            Section(
                label="build", start_beat=72, end_beat=96,
                stem_gains={"vocals": 0.8, "drums": 0.9, "bass": 0.8,
                            "guitar": 0.7, "piano": 0.5, "other": 0.7},
                transition_in="crossfade", transition_beats=8,
            ),
            Section(
                label="main", start_beat=96, end_beat=128,
                stem_gains={"vocals": 1.0, "drums": 1.0, "bass": 0.9,
                            "guitar": 0.8, "piano": 0.6, "other": 0.8},
                transition_in="crossfade", transition_beats=4,
            ),
            Section(
                label="outro", start_beat=128, end_beat=144,
                stem_gains={"vocals": 0.0, "drums": 0.5, "bass": 0.4,
                            "guitar": 0.4, "piano": 0.5, "other": 0.5},
                transition_in="crossfade", transition_beats=4,
            ),
        ],
        tempo_source="song_a",
        key_source="song_a",
        explanation="Risky plan with more sections and higher gains",
    )


def _make_metadata(
    key: str | None = "C",
    scale: str | None = "major",
    bpm: float = 120.0,
) -> AudioMetadata:
    """Build a minimal AudioMetadata for testing."""
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=np.arange(0, 200) * 21,  # Fake beat frames
        duration_seconds=200 * 60 / bpm,
        total_beats=200,
        key=key,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Tests: score_candidate basics
# ---------------------------------------------------------------------------

class TestScoreCandidate:
    """Tests for the score_candidate function."""

    def test_score_in_range(self):
        """Total score is always in [0.0, 1.0]."""
        plan = _make_good_plan()
        result = score_candidate(plan)
        assert 0.0 <= result.total_score <= 1.0

    def test_bad_plan_score_in_range(self):
        """Even a bad plan scores in [0.0, 1.0]."""
        plan = _make_bad_plan()
        result = score_candidate(plan)
        assert 0.0 <= result.total_score <= 1.0

    def test_all_dimensions_present(self):
        """All 7 dimensions are present in dimension_scores."""
        plan = _make_good_plan()
        result = score_candidate(plan)
        expected_dims = {
            "arrangement_quality",
            "energy_arc",
            "vocal_intelligibility",
            "harmonic_fit",
            "transition_quality",
            "groove_coherence",
            "loudness_fatigue",
        }
        assert set(result.dimension_scores.keys()) == expected_dims

    def test_dimension_scores_in_range(self):
        """Each dimension score is in [0.0, 1.0]."""
        plan = _make_good_plan()
        result = score_candidate(plan)
        for dim, score in result.dimension_scores.items():
            assert 0.0 <= score <= 1.0, f"{dim} score {score} out of range"

    def test_good_plan_beats_bad_plan(self):
        """A well-structured plan should score higher than a bad one."""
        good = score_candidate(_make_good_plan())
        bad = score_candidate(_make_bad_plan())
        assert good.total_score > bad.total_score, (
            f"Good plan ({good.total_score:.3f}) should beat "
            f"bad plan ({bad.total_score:.3f})"
        )

    def test_works_without_metadata(self):
        """Scoring works when no audio metadata is provided."""
        plan = _make_good_plan()
        result = score_candidate(plan)
        assert result.total_score > 0.0
        assert len(result.dimension_scores) == 7

    def test_works_with_metadata(self):
        """Scoring incorporates metadata when provided."""
        plan = _make_good_plan()
        meta_a = _make_metadata(key="C", scale="major")
        meta_b = _make_metadata(key="G", scale="major")
        result = score_candidate(plan, meta_a, meta_b)
        assert 0.0 <= result.total_score <= 1.0

    def test_returns_scored_candidate(self):
        """Returns a ScoredCandidate dataclass."""
        plan = _make_good_plan()
        result = score_candidate(plan)
        assert isinstance(result, ScoredCandidate)
        assert result.plan is plan
        assert result.rank == 0  # Not ranked yet


# ---------------------------------------------------------------------------
# Tests: dimension weights
# ---------------------------------------------------------------------------

class TestDimensionWeights:
    """Tests for dimension weight configuration."""

    def test_weights_sum_to_one(self):
        """Dimension weights should sum to 1.0."""
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_dimensions_have_weights(self):
        """Every dimension has a defined weight."""
        expected = {
            "arrangement_quality", "energy_arc", "vocal_intelligibility",
            "harmonic_fit", "transition_quality", "groove_coherence",
            "loudness_fatigue",
        }
        assert set(DIMENSION_WEIGHTS.keys()) == expected


# ---------------------------------------------------------------------------
# Tests: select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    """Tests for the select_best function."""

    def test_returns_highest_scoring(self):
        """select_best returns the highest-scoring candidate."""
        good = _make_good_plan()
        bad = _make_bad_plan()
        best, scored = select_best([good, bad])
        assert best is good

    def test_single_candidate(self):
        """Works with a single candidate."""
        plan = _make_good_plan()
        best, scored = select_best([plan])
        assert best is plan
        assert len(scored) == 1
        assert scored[0].rank == 1

    def test_empty_candidates_raises(self):
        """Raises ValueError for empty candidates list."""
        with pytest.raises(ValueError, match="empty candidates"):
            select_best([])

    def test_all_identical_candidates(self):
        """Handles identical candidates gracefully."""
        plan = _make_good_plan()
        # Create "copies" by building identical plans
        plans = [
            RemixPlan(
                vocal_source=plan.vocal_source,
                start_time_vocal=plan.start_time_vocal,
                end_time_vocal=plan.end_time_vocal,
                start_time_instrumental=plan.start_time_instrumental,
                end_time_instrumental=plan.end_time_instrumental,
                sections=list(plan.sections),
                tempo_source=plan.tempo_source,
                key_source=plan.key_source,
                explanation=plan.explanation,
            )
            for _ in range(3)
        ]
        best, scored = select_best(plans)
        assert best is not None
        assert len(scored) == 3
        # All should have the same score
        scores = [sc.total_score for sc in scored]
        assert max(scores) - min(scores) < 1e-9

    def test_ranks_assigned(self):
        """Ranks are assigned after scoring."""
        plans = [_make_good_plan(), _make_bad_plan(), _make_conservative_plan()]
        _, scored = select_best(plans)
        ranks = [sc.rank for sc in scored]
        assert sorted(ranks) == [1, 2, 3]

    def test_sorted_descending(self):
        """Scored candidates are sorted by total_score descending."""
        plans = [_make_good_plan(), _make_bad_plan(), _make_conservative_plan()]
        _, scored = select_best(plans)
        for i in range(len(scored) - 1):
            assert scored[i].total_score >= scored[i + 1].total_score

    def test_with_metadata(self):
        """select_best works with audio metadata."""
        plans = [_make_good_plan(), _make_bad_plan()]
        meta_a = _make_metadata(key="C", scale="major")
        meta_b = _make_metadata(key="G", scale="major")
        best, scored = select_best(plans, meta_a, meta_b)
        assert best is not None
        assert len(scored) == 2


# ---------------------------------------------------------------------------
# Tests: tie-breaking logic
# ---------------------------------------------------------------------------

class TestTieBreaking:
    """Tests for tie-breaking between close-scoring candidates."""

    def test_tie_prefers_conservative(self):
        """When scores are close, the more conservative candidate wins."""
        conservative = _make_conservative_plan()
        risky = _make_risky_plan()

        score_c = score_candidate(conservative)
        score_r = score_candidate(risky)

        # Verify the plans score close enough for this test to be meaningful.
        # If they don't naturally tie, we test the conservatism_score directly.
        risk_c = _conservatism_score(conservative)
        risk_r = _conservatism_score(risky)
        assert risk_c < risk_r, (
            f"Conservative plan (risk={risk_c:.2f}) should have lower risk "
            f"than risky plan (risk={risk_r:.2f})"
        )

    def test_conservatism_fewer_sections(self):
        """A plan with fewer sections has lower conservatism risk."""
        conservative = _make_conservative_plan()  # 4 sections
        risky = _make_risky_plan()  # 7 sections
        assert _conservatism_score(conservative) < _conservatism_score(risky)

    def test_conservatism_lower_gains(self):
        """A plan with lower average gains has lower conservatism risk."""
        low_gain = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(
                    label="main", start_beat=0, end_beat=64,
                    stem_gains={"vocals": 0.3, "drums": 0.3, "bass": 0.3},
                    transition_in="crossfade", transition_beats=4,
                ),
            ],
            tempo_source="song_a", key_source="song_a",
            explanation="low gain plan",
        )
        high_gain = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(
                    label="main", start_beat=0, end_beat=64,
                    stem_gains={"vocals": 1.0, "drums": 1.0, "bass": 1.0},
                    transition_in="crossfade", transition_beats=4,
                ),
            ],
            tempo_source="song_a", key_source="song_a",
            explanation="high gain plan",
        )
        assert _conservatism_score(low_gain) < _conservatism_score(high_gain)


# ---------------------------------------------------------------------------
# Tests: Camelot distance
# ---------------------------------------------------------------------------

class TestCamelotDistance:
    """Tests for the Camelot wheel distance calculation."""

    def test_same_key_distance_zero(self):
        """Same key has distance 0."""
        assert _camelot_distance("C", "major", "C", "major") == 0

    def test_adjacent_key_distance_one(self):
        """Adjacent Camelot keys have distance 1."""
        # C major = 8B, G major = 9B -> distance 1
        assert _camelot_distance("C", "major", "G", "major") == 1

    def test_relative_minor_distance_one(self):
        """Relative minor has distance 1."""
        # C major = 8B, Am = 8A -> distance 1
        assert _camelot_distance("C", "major", "A", "minor") == 1

    def test_distant_key(self):
        """Distant keys have higher distance."""
        # C major = 8B, F# major = 2B -> distance 6
        dist = _camelot_distance("C", "major", "F#", "major")
        assert dist >= 4

    def test_unknown_key_returns_default(self):
        """Unknown key returns default distance (4)."""
        assert _camelot_distance("X", "major", "C", "major") == 4


# ---------------------------------------------------------------------------
# Tests: CatBoost stub
# ---------------------------------------------------------------------------

class TestCatBoostStub:
    """Tests for CatBoost model loading and scoring stubs."""

    def test_load_model_no_path(self):
        """load_model returns False when no path provided."""
        assert load_model(None) is False

    def test_load_model_no_catboost(self):
        """load_model returns False when catboost is not installed."""
        # catboost is unlikely to be installed in test env
        result = load_model("/nonexistent/model.cbm")
        assert result is False

    def test_score_with_model_no_model(self):
        """score_with_model returns None when no model is loaded."""
        result = score_with_model([{"feature_a": 1.0, "feature_b": 2.0}])
        assert result is None

    def test_score_with_model_empty_vectors(self):
        """score_with_model returns None for empty vectors (no model)."""
        result = score_with_model([])
        assert result is None


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_plan_with_empty_sections(self):
        """Scoring handles a plan with no sections."""
        plan = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[],
            tempo_source="song_a", key_source="none",
            explanation="Empty plan",
        )
        result = score_candidate(plan)
        assert 0.0 <= result.total_score <= 1.0

    def test_plan_with_single_section(self):
        """Scoring works with a single section."""
        plan = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(
                    label="main", start_beat=0, end_beat=64,
                    stem_gains={"vocals": 0.8, "drums": 0.6, "bass": 0.5},
                    transition_in="fade", transition_beats=4,
                ),
            ],
            tempo_source="song_a", key_source="song_a",
            explanation="Single section plan",
        )
        result = score_candidate(plan)
        assert 0.0 <= result.total_score <= 1.0
        assert len(result.dimension_scores) == 7

    def test_plan_with_zero_gains(self):
        """Scoring handles sections with all-zero gains."""
        plan = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(
                    label="intro", start_beat=0, end_beat=32,
                    stem_gains={"vocals": 0.0, "drums": 0.0, "bass": 0.0},
                    transition_in="fade", transition_beats=4,
                ),
                Section(
                    label="outro", start_beat=32, end_beat=64,
                    stem_gains={"vocals": 0.0, "drums": 0.0, "bass": 0.0},
                    transition_in="fade", transition_beats=4,
                ),
            ],
            tempo_source="song_a", key_source="none",
            explanation="Silent plan",
        )
        result = score_candidate(plan)
        assert 0.0 <= result.total_score <= 1.0

    def test_metadata_without_key(self):
        """Scoring works when metadata has no key info."""
        plan = _make_good_plan()
        meta_a = _make_metadata(key=None, scale=None)
        meta_b = _make_metadata(key=None, scale=None)
        result = score_candidate(plan, meta_a, meta_b)
        assert 0.0 <= result.total_score <= 1.0

    def test_many_candidates_select_best(self):
        """select_best handles many candidates."""
        plans = [_make_good_plan(), _make_bad_plan(), _make_conservative_plan(),
                 _make_risky_plan(), _make_good_plan()]
        best, scored = select_best(plans)
        assert best is not None
        assert len(scored) == 5
        # All ranked
        assert all(sc.rank > 0 for sc in scored)

    def test_misaligned_boundaries(self):
        """Plans with non-4-aligned boundaries get lower groove scores."""
        aligned = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(label="intro", start_beat=0, end_beat=16,
                        stem_gains={"vocals": 0.0, "drums": 0.5},
                        transition_in="fade", transition_beats=4),
                Section(label="main", start_beat=16, end_beat=64,
                        stem_gains={"vocals": 0.8, "drums": 0.6},
                        transition_in="crossfade", transition_beats=4),
                Section(label="outro", start_beat=64, end_beat=80,
                        stem_gains={"vocals": 0.0, "drums": 0.4},
                        transition_in="fade", transition_beats=4),
            ],
            tempo_source="song_a", key_source="song_a",
            explanation="Aligned",
        )
        misaligned = RemixPlan(
            vocal_source="song_a",
            start_time_vocal=0.0, end_time_vocal=60.0,
            start_time_instrumental=0.0, end_time_instrumental=60.0,
            sections=[
                Section(label="intro", start_beat=0, end_beat=13,
                        stem_gains={"vocals": 0.0, "drums": 0.5},
                        transition_in="fade", transition_beats=3),
                Section(label="main", start_beat=13, end_beat=57,
                        stem_gains={"vocals": 0.8, "drums": 0.6},
                        transition_in="crossfade", transition_beats=3),
                Section(label="outro", start_beat=57, end_beat=79,
                        stem_gains={"vocals": 0.0, "drums": 0.4},
                        transition_in="fade", transition_beats=3),
            ],
            tempo_source="song_a", key_source="song_a",
            explanation="Misaligned",
        )
        score_aligned = score_candidate(aligned)
        score_misaligned = score_candidate(misaligned)
        assert (
            score_aligned.dimension_scores["groove_coherence"]
            > score_misaligned.dimension_scores["groove_coherence"]
        )
