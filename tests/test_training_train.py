"""Tests for musicmixer.training.train -- CatBoost model training.

Covers:
  - _compute_data_hash: deterministic hashing
  - _compute_config_hash: deterministic config hashing
  - _load_training_data: CSV loading with correct column parsing
  - _split_by_group: group-level train/val split
  - train_model: end-to-end training (when catboost is available)
  - Error handling: missing CSV, insufficient data

Note: CatBoost-dependent tests are skipped when catboost is not installed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from musicmixer.services.taste_features import get_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_training_csv(
    output_path: Path,
    n_groups: int = 5,
    n_per_group: int = 4,
) -> None:
    """Create a training CSV with synthetic feature values."""
    from musicmixer.models import RemixPlan, Section
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

        for i in range(n_groups):
            group_id = f"mashup_{i:03d}"
            for j in range(n_per_group):
                label = 1 if j == 0 else 0
                plan_id = group_id if j == 0 else f"{group_id}_neg{j - 1}"

                # Create a plan with varying characteristics
                sections = []
                current_beat = 0
                section_labels = ["intro", "verse", "drop", "breakdown", "drop", "outro"]
                for slabel in section_labels:
                    duration = 32
                    # Vary gains based on label/plan index for diversity
                    vocal_gain = 0.0 if slabel in ("intro", "outro") else 0.8
                    if label == 0:
                        vocal_gain = 0.5  # flat for negatives
                    sections.append(Section(
                        label=slabel,
                        start_beat=current_beat,
                        end_beat=current_beat + duration,
                        stem_gains={
                            "vocals": vocal_gain,
                            "drums": 0.6 + (j * 0.05),
                            "bass": 0.5,
                            "guitar": 0.3,
                            "piano": 0.2,
                            "other": 0.1,
                        },
                        transition_in="crossfade",
                        transition_beats=4,
                    ))
                    current_beat += duration

                plan = RemixPlan(
                    vocal_source="song_a",
                    start_time_vocal=0.0,
                    end_time_vocal=210.0,
                    start_time_instrumental=0.0,
                    end_time_instrumental=210.0,
                    sections=sections,
                    tempo_source="average",
                    key_source="none",
                    explanation=f"Test plan {plan_id}",
                )

                fv = extract_features(plan)
                scored = score_candidate(plan)

                row = [group_id, plan_id, label, manifest.version]
                for fname in feature_names:
                    row.append(fv.features.get(fname, 0.0))
                for dim in heuristic_dimensions:
                    row.append(scored.dimension_scores.get(dim, 0.0))
                row.append(scored.total_score)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Tests: hash functions
# ---------------------------------------------------------------------------

class TestHashFunctions:

    def test_data_hash_deterministic(self, tmp_path: Path):
        from musicmixer.training.train import _compute_data_hash

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("a,b,c\n1,2,3\n")

        hash1 = _compute_data_hash(csv_path)
        hash2 = _compute_data_hash(csv_path)
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_config_hash_deterministic(self):
        from musicmixer.training.train import _compute_config_hash

        params = {"a": 1, "b": 2.0, "c": "three"}
        hash1 = _compute_config_hash(params)
        hash2 = _compute_config_hash(params)
        assert hash1 == hash2
        assert len(hash1) == 12

    def test_config_hash_changes_with_params(self):
        from musicmixer.training.train import _compute_config_hash

        hash1 = _compute_config_hash({"a": 1})
        hash2 = _compute_config_hash({"a": 2})
        assert hash1 != hash2


# ---------------------------------------------------------------------------
# Tests: _load_training_data
# ---------------------------------------------------------------------------

class TestLoadTrainingData:

    def test_load_basic(self, tmp_path: Path):
        from musicmixer.training.train import _load_training_data

        csv_path = tmp_path / "training_data.csv"
        _write_training_csv(csv_path, n_groups=3, n_per_group=4)

        feature_matrix, labels, group_ids, feature_names = _load_training_data(csv_path)

        assert len(feature_matrix) == 12  # 3 groups * 4 plans
        assert len(labels) == 12
        assert len(group_ids) == 12
        assert len(feature_names) > 0

        # Check label distribution
        assert labels.count(1) == 3  # 1 positive per group
        assert labels.count(0) == 9  # 3 negatives per group

    def test_feature_matrix_shape(self, tmp_path: Path):
        from musicmixer.training.train import _load_training_data

        csv_path = tmp_path / "training_data.csv"
        _write_training_csv(csv_path, n_groups=2, n_per_group=4)

        feature_matrix, _, _, feature_names = _load_training_data(csv_path)

        # All rows should have the same number of features
        for row in feature_matrix:
            assert len(row) == len(feature_names)

    def test_feature_values_numeric(self, tmp_path: Path):
        from musicmixer.training.train import _load_training_data

        csv_path = tmp_path / "training_data.csv"
        _write_training_csv(csv_path, n_groups=2, n_per_group=4)

        feature_matrix, _, _, _ = _load_training_data(csv_path)

        for row in feature_matrix:
            for val in row:
                assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Tests: _split_by_group
# ---------------------------------------------------------------------------

class TestSplitByGroup:

    def test_basic_split(self):
        from musicmixer.training.train import _split_by_group

        features = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        labels = [1, 0, 1, 0, 1, 0]
        group_ids = ["a", "a", "b", "b", "c", "c"]

        tf, tl, tg, vf, vl, vg = _split_by_group(
            features, labels, group_ids, train_ratio=0.67, seed=42
        )

        # All entries for a group should be in the same split
        train_groups = set(tg)
        val_groups = set(vg)
        assert len(train_groups & val_groups) == 0, "Groups should not overlap"

        # Total should match
        assert len(tf) + len(vf) == 6

    def test_no_leakage(self):
        from musicmixer.training.train import _split_by_group

        features = [[i] for i in range(20)]
        labels = [i % 2 for i in range(20)]
        group_ids = [f"g{i // 4}" for i in range(20)]

        tf, tl, tg, vf, vl, vg = _split_by_group(
            features, labels, group_ids, train_ratio=0.8
        )

        train_groups = set(tg)
        val_groups = set(vg)
        assert len(train_groups & val_groups) == 0


# ---------------------------------------------------------------------------
# Tests: train_model (requires catboost)
# ---------------------------------------------------------------------------

# Check if catboost is available
try:
    import catboost  # noqa: F401
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
class TestTrainModel:

    def test_basic_training(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        csv_path = tmp_path / "training_data.csv"
        output_dir = tmp_path / "models"
        _write_training_csv(csv_path, n_groups=10, n_per_group=4)

        result = train_model(csv_path=csv_path, output_dir=output_dir)

        assert "model_path" in result
        assert "eval_metrics" in result
        assert "config_hash" in result
        assert "data_hash" in result
        assert "feature_manifest_version" in result

        # Model file should exist
        model_path = Path(result["model_path"])
        assert model_path.exists()

        # Training manifest should exist
        manifest_path = output_dir / "training_manifest.json"
        assert manifest_path.exists()

    def test_eval_metrics_structure(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        csv_path = tmp_path / "training_data.csv"
        output_dir = tmp_path / "models"
        _write_training_csv(csv_path, n_groups=10, n_per_group=4)

        result = train_model(csv_path=csv_path, output_dir=output_dir)

        metrics = result["eval_metrics"]
        assert "pairwise_accuracy" in metrics
        assert "top1_preferred_rate" in metrics
        assert "total_pairs" in metrics
        assert "total_groups" in metrics

        assert 0.0 <= metrics["pairwise_accuracy"] <= 1.0
        assert 0.0 <= metrics["top1_preferred_rate"] <= 1.0

    def test_model_loadable(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        csv_path = tmp_path / "training_data.csv"
        output_dir = tmp_path / "models"
        _write_training_csv(csv_path, n_groups=10, n_per_group=4)

        result = train_model(csv_path=csv_path, output_dir=output_dir)

        # Verify model can be loaded
        model = catboost.CatBoost()
        model.load_model(result["model_path"])

    def test_missing_csv_raises(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        with pytest.raises(FileNotFoundError):
            train_model(
                csv_path=tmp_path / "nonexistent.csv",
                output_dir=tmp_path / "models",
            )

    def test_insufficient_data_raises(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        csv_path = tmp_path / "training_data.csv"
        output_dir = tmp_path / "models"

        # Write a CSV with too few rows
        _write_training_csv(csv_path, n_groups=1, n_per_group=2)

        with pytest.raises(ValueError, match="Insufficient"):
            train_model(csv_path=csv_path, output_dir=output_dir)

    def test_training_manifest_json_valid(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        csv_path = tmp_path / "training_data.csv"
        output_dir = tmp_path / "models"
        _write_training_csv(csv_path, n_groups=10, n_per_group=4)

        train_model(csv_path=csv_path, output_dir=output_dir)

        manifest_path = output_dir / "training_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["train_rows"] > 0
        assert manifest["feature_count"] > 0
        assert isinstance(manifest["catboost_params"], dict)


@pytest.mark.skipif(HAS_CATBOOST, reason="Only runs when catboost is NOT installed")
class TestTrainModelNoCatboost:

    def test_import_error_raised(self, tmp_path: Path):
        from musicmixer.training.train import train_model

        with pytest.raises(ImportError, match="catboost"):
            train_model(
                csv_path=tmp_path / "dummy.csv",
                output_dir=tmp_path / "models",
            )
