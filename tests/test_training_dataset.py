"""Tests for musicmixer.training.dataset -- feature extraction and dataset building.

Covers:
  - _load_plan_from_json: deserialization from JSON format
  - _extract_group_id: group ID extraction from plan IDs
  - build_dataset: end-to-end CSV building
  - CSV schema validation (correct columns, values, types)
  - Edge cases: missing files, empty directories, malformed JSON
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from musicmixer.models import RemixPlan, Section
from musicmixer.services.taste_features import get_manifest
from musicmixer.training.dataset import (
    _extract_group_id,
    _load_plan_from_json,
    build_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan_dict(
    label_sequence: list[str] | None = None,
    vocal_source: str = "song_a",
) -> dict:
    """Create a minimal serialized plan dict for testing."""
    if label_sequence is None:
        label_sequence = ["intro", "verse", "drop", "breakdown", "drop", "outro"]

    sections = []
    current_beat = 0
    for label in label_sequence:
        duration = 32
        sections.append({
            "label": label,
            "start_beat": current_beat,
            "end_beat": current_beat + duration,
            "stem_gains": {
                "vocals": 0.0 if label in ("intro", "outro") else 0.8,
                "drums": 0.6,
                "bass": 0.5,
                "guitar": 0.3,
                "piano": 0.2,
                "other": 0.1,
            },
            "transition_in": "crossfade",
            "transition_beats": 4,
        })
        current_beat += duration

    return {
        "vocal_source": vocal_source,
        "tempo_source": "average",
        "key_source": "none",
        "start_time_vocal": 0.0,
        "end_time_vocal": 210.0,
        "start_time_instrumental": 0.0,
        "end_time_instrumental": 210.0,
        "explanation": "Test plan",
        "used_fallback": False,
        "warnings": [],
        "sections": sections,
    }


def _write_plan_json(dir_path: Path, filename: str, plan_dict: dict) -> Path:
    """Write a plan dict as a JSON file."""
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / filename
    with open(path, "w") as f:
        json.dump(plan_dict, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Tests: _extract_group_id
# ---------------------------------------------------------------------------

class TestExtractGroupId:

    def test_positive_plan_id(self):
        assert _extract_group_id("mashup_001") == "mashup_001"

    def test_negative_plan_id(self):
        assert _extract_group_id("mashup_001_neg0") == "mashup_001"
        assert _extract_group_id("mashup_001_neg1") == "mashup_001"
        assert _extract_group_id("mashup_001_neg2") == "mashup_001"

    def test_complex_id(self):
        assert _extract_group_id("girl_talk_001_neg0") == "girl_talk_001"

    def test_no_suffix(self):
        assert _extract_group_id("simple") == "simple"


# ---------------------------------------------------------------------------
# Tests: _load_plan_from_json
# ---------------------------------------------------------------------------

class TestLoadPlanFromJson:

    def test_round_trip(self, tmp_path: Path):
        plan_dict = _make_plan_dict()
        path = _write_plan_json(tmp_path, "test.json", plan_dict)

        plan = _load_plan_from_json(path)

        assert isinstance(plan, RemixPlan)
        assert plan.vocal_source == "song_a"
        assert len(plan.sections) == 6
        assert plan.sections[0].label == "intro"
        assert plan.sections[0].start_beat == 0
        assert plan.sections[0].end_beat == 32
        assert plan.sections[0].stem_gains["vocals"] == 0.0
        assert plan.sections[1].stem_gains["vocals"] == 0.8
        assert plan.explanation == "Test plan"

    def test_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        with pytest.raises(json.JSONDecodeError):
            _load_plan_from_json(path)

    def test_missing_sections_key(self, tmp_path: Path):
        path = _write_plan_json(tmp_path, "no_sections.json", {"vocal_source": "song_a"})
        with pytest.raises(KeyError):
            _load_plan_from_json(path)


# ---------------------------------------------------------------------------
# Tests: build_dataset
# ---------------------------------------------------------------------------

class TestBuildDataset:

    def _setup_dirs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Create plans and negatives directories with test data."""
        plans_dir = tmp_path / "plans"
        negatives_dir = tmp_path / "negatives"
        output_path = tmp_path / "training" / "training_data.csv"

        # Write positive plans
        for i in range(3):
            plan_dict = _make_plan_dict()
            plan_dict["explanation"] = f"Positive plan {i}"
            _write_plan_json(plans_dir, f"mashup_{i:03d}.json", plan_dict)

        # Write negative plans
        for i in range(3):
            for j in range(3):
                plan_dict = _make_plan_dict()
                plan_dict["explanation"] = f"Negative plan {i}_{j}"
                plan_dict["used_fallback"] = True
                # Make negatives slightly different (flat energy)
                for sec in plan_dict["sections"]:
                    sec["stem_gains"] = {k: 0.5 for k in sec["stem_gains"]}
                _write_plan_json(negatives_dir, f"mashup_{i:03d}_neg{j}.json", plan_dict)

        return plans_dir, negatives_dir, output_path

    def test_basic_build(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)

        result = build_dataset(plans_dir, negatives_dir, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_csv_row_count(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # 1 header + 3 positive + 9 negative = 13 rows
        assert len(rows) == 13

    def test_csv_columns(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

        # Check required metadata columns
        assert "group_id" in fieldnames
        assert "plan_id" in fieldnames
        assert "label" in fieldnames
        assert "manifest_version" in fieldnames

        # Check feature columns
        manifest = get_manifest()
        for fname in manifest.feature_names:
            assert fname in fieldnames, f"Missing feature column: {fname}"

        # Check heuristic columns
        heuristic_cols = [
            "heuristic_arrangement_quality",
            "heuristic_energy_arc",
            "heuristic_vocal_intelligibility",
            "heuristic_harmonic_fit",
            "heuristic_transition_quality",
            "heuristic_groove_coherence",
            "heuristic_loudness_fatigue",
            "heuristic_total",
        ]
        for col in heuristic_cols:
            assert col in fieldnames, f"Missing heuristic column: {col}"

    def test_csv_labels(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        labels = [int(r["label"]) for r in rows]
        assert labels.count(1) == 3  # 3 positive plans
        assert labels.count(0) == 9  # 9 negative plans

    def test_csv_group_ids(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # All rows for mashup_000 should have group_id = "mashup_000"
        mashup_000_rows = [r for r in rows if r["group_id"] == "mashup_000"]
        assert len(mashup_000_rows) == 4  # 1 positive + 3 negatives

    def test_manifest_version_consistent(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        manifest = get_manifest()
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["manifest_version"] == manifest.version

    def test_feature_values_are_numeric(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        manifest = get_manifest()
        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for fname in manifest.feature_names:
                    val = float(row[fname])
                    assert isinstance(val, float)
                    assert not (val != val)  # check for NaN

    def test_no_plans_dir_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            build_dataset(
                tmp_path / "nonexistent",
                tmp_path / "also_nonexistent",
                tmp_path / "out.csv",
            )

    def test_empty_plans_dir_raises(self, tmp_path: Path):
        plans_dir = tmp_path / "plans"
        negatives_dir = tmp_path / "negatives"
        plans_dir.mkdir(parents=True)
        negatives_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="No positive plan files"):
            build_dataset(plans_dir, negatives_dir, tmp_path / "out.csv")

    def test_malformed_plan_skipped(self, tmp_path: Path):
        plans_dir = tmp_path / "plans"
        negatives_dir = tmp_path / "negatives"
        negatives_dir.mkdir(parents=True)

        # Write one good plan and one bad plan
        _write_plan_json(plans_dir, "good.json", _make_plan_dict())
        _write_plan_json(plans_dir, "bad.json", {"bad": "data"})

        output_path = tmp_path / "out.csv"
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Header + 1 good plan (bad one skipped)
        assert len(rows) == 2

    def test_heuristic_scores_in_range(self, tmp_path: Path):
        plans_dir, negatives_dir, output_path = self._setup_dirs(tmp_path)
        build_dataset(plans_dir, negatives_dir, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                total = float(row["heuristic_total"])
                assert 0.0 <= total <= 1.0, f"Total score out of range: {total}"
