"""Tests for musicmixer.training.validate -- domain alignment validation.

Covers:
  - _make_synthetic_metadata: creates valid AudioMetadata
  - _collect_labels: extracts Section.label vocabulary from plans
  - _compute_feature_stats: computes mean/std/min/max per feature
  - validate_domain_alignment: end-to-end validation with CSV input
  - Label vocabulary consistency check (hard gate)
  - Feature divergence flagging
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_features import get_manifest
from musicmixer.training.validate import (
    _collect_labels,
    _compute_feature_stats,
    _make_synthetic_metadata,
    validate_domain_alignment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_plan(
    labels: list[str] | None = None,
) -> RemixPlan:
    """Create a simple RemixPlan for testing."""
    if labels is None:
        labels = ["intro", "verse", "drop", "breakdown", "drop", "outro"]

    sections = []
    current_beat = 0
    for label in labels:
        duration = 32
        sections.append(Section(
            label=label,
            start_beat=current_beat,
            end_beat=current_beat + duration,
            stem_gains={
                "vocals": 0.0 if label in ("intro", "outro") else 0.8,
                "drums": 0.6,
                "bass": 0.5,
                "guitar": 0.3,
                "piano": 0.2,
                "other": 0.1,
            },
            transition_in="crossfade",
            transition_beats=4,
        ))
        current_beat += duration

    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=210.0,
        start_time_instrumental=0.0,
        end_time_instrumental=210.0,
        sections=sections,
        tempo_source="average",
        key_source="none",
        explanation="Test plan",
    )


def _write_training_csv(
    output_path: Path,
    n_positives: int = 5,
    n_negatives_per_positive: int = 3,
) -> None:
    """Create a training CSV with synthetic data for testing."""
    from musicmixer.services.taste_features import extract_features
    from musicmixer.services.taste_model import score_candidate

    manifest = get_manifest()
    feature_names = sorted(manifest.feature_names)

    heuristic_dimensions = [
        "arrangement_quality",
        "energy_arc",
        "vocal_intelligibility",
        "harmonic_fit",
        "transition_quality",
        "groove_coherence",
        "loudness_fatigue",
    ]
    heuristic_columns = [f"heuristic_{dim}" for dim in heuristic_dimensions]

    header = (
        ["group_id", "plan_id", "label", "manifest_version"]
        + feature_names
        + heuristic_columns
        + ["heuristic_total"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(n_positives):
            group_id = f"mashup_{i:03d}"

            # Positive plan
            plan = _make_test_plan()
            fv = extract_features(plan)
            scored = score_candidate(plan)

            row = [group_id, f"mashup_{i:03d}", 1, manifest.version]
            for fname in feature_names:
                row.append(fv.features.get(fname, 0.0))
            for dim in heuristic_dimensions:
                row.append(scored.dimension_scores.get(dim, 0.0))
            row.append(scored.total_score)
            writer.writerow(row)

            # Negative plans
            for j in range(n_negatives_per_positive):
                neg_plan = _make_test_plan()
                # Make negatives different
                for sec in neg_plan.sections:
                    sec.stem_gains = {k: 0.5 for k in sec.stem_gains}

                fv_neg = extract_features(neg_plan)
                scored_neg = score_candidate(neg_plan)

                row = [group_id, f"mashup_{i:03d}_neg{j}", 0, manifest.version]
                for fname in feature_names:
                    row.append(fv_neg.features.get(fname, 0.0))
                for dim in heuristic_dimensions:
                    row.append(scored_neg.dimension_scores.get(dim, 0.0))
                row.append(scored_neg.total_score)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Tests: _make_synthetic_metadata
# ---------------------------------------------------------------------------

class TestMakeSyntheticMetadata:

    def test_basic_metadata(self):
        meta = _make_synthetic_metadata()
        assert isinstance(meta, AudioMetadata)
        assert meta.bpm == 120.0
        assert meta.duration_seconds == 210.0
        assert meta.key == "C"
        assert meta.scale == "major"
        assert meta.total_beats > 0
        assert len(meta.beat_frames) > 0

    def test_custom_bpm(self):
        meta = _make_synthetic_metadata(bpm=140.0)
        assert meta.bpm == 140.0
        assert meta.total_beats > 0

    def test_total_beats_on_bar_boundary(self):
        meta = _make_synthetic_metadata(bpm=120.0, duration=210.0)
        assert meta.total_beats % 4 == 0


# ---------------------------------------------------------------------------
# Tests: _collect_labels
# ---------------------------------------------------------------------------

class TestCollectLabels:

    def test_basic_collection(self):
        plans = [_make_test_plan(["intro", "verse", "drop", "outro"])]
        labels = _collect_labels(plans)
        assert labels == {"intro", "verse", "drop", "outro"}

    def test_multiple_plans(self):
        plans = [
            _make_test_plan(["intro", "verse", "outro"]),
            _make_test_plan(["intro", "drop", "breakdown", "outro"]),
        ]
        labels = _collect_labels(plans)
        assert labels == {"intro", "verse", "drop", "breakdown", "outro"}

    def test_empty_plans(self):
        labels = _collect_labels([])
        assert labels == set()


# ---------------------------------------------------------------------------
# Tests: _compute_feature_stats
# ---------------------------------------------------------------------------

class TestComputeFeatureStats:

    def test_basic_stats(self):
        vectors = [
            {"a": 1.0, "b": 2.0},
            {"a": 3.0, "b": 4.0},
            {"a": 5.0, "b": 6.0},
        ]
        stats = _compute_feature_stats(vectors)

        assert abs(stats["a"]["mean"] - 3.0) < 1e-10
        assert abs(stats["b"]["mean"] - 4.0) < 1e-10
        assert stats["a"]["min"] == 1.0
        assert stats["a"]["max"] == 5.0

    def test_single_vector(self):
        vectors = [{"a": 5.0}]
        stats = _compute_feature_stats(vectors)
        assert stats["a"]["mean"] == 5.0
        assert stats["a"]["std"] == 0.0

    def test_empty_list(self):
        stats = _compute_feature_stats([])
        assert stats == {}


# ---------------------------------------------------------------------------
# Tests: validate_domain_alignment
# ---------------------------------------------------------------------------

class TestValidateDomainAlignment:

    def test_basic_validation(self, tmp_path: Path):
        csv_path = tmp_path / "training" / "training_data.csv"
        _write_training_csv(csv_path, n_positives=3)

        result = validate_domain_alignment(
            training_csv=csv_path,
            n_candidates=10,
        )

        assert "label_check" in result
        assert "flagged_features" in result
        assert "total_features" in result
        assert "total_flagged" in result
        assert "mashup_stats" in result
        assert "candidate_stats" in result

    def test_returns_flagged_features(self, tmp_path: Path):
        csv_path = tmp_path / "training" / "training_data.csv"
        _write_training_csv(csv_path, n_positives=3)

        result = validate_domain_alignment(
            training_csv=csv_path,
            n_candidates=10,
        )

        # Some features may be flagged (especially harmonic/tempo ones)
        assert isinstance(result["flagged_features"], list)
        for flagged in result["flagged_features"]:
            assert "feature" in flagged
            assert "divergence_ratio" in flagged
            assert flagged["divergence_ratio"] > 1.0

    def test_candidate_labels_populated(self, tmp_path: Path):
        csv_path = tmp_path / "training" / "training_data.csv"
        _write_training_csv(csv_path, n_positives=3)

        result = validate_domain_alignment(
            training_csv=csv_path,
            n_candidates=10,
        )

        label_check = result["label_check"]
        assert len(label_check["candidate_labels"]) > 0

    def test_missing_csv_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            validate_domain_alignment(
                training_csv=tmp_path / "nonexistent.csv",
            )

    def test_stats_have_expected_keys(self, tmp_path: Path):
        csv_path = tmp_path / "training" / "training_data.csv"
        _write_training_csv(csv_path, n_positives=3)

        result = validate_domain_alignment(
            training_csv=csv_path,
            n_candidates=10,
        )

        for stats in [result["mashup_stats"], result["candidate_stats"]]:
            for fname, stat in stats.items():
                assert "mean" in stat
                assert "std" in stat
                assert "min" in stat
                assert "max" in stat
