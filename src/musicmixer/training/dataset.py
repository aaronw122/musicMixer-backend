"""Feature extraction and training dataset building.

Step 6 of the mashup training pipeline. Loads positive (mashup-derived) and
negative (synthetic) RemixPlan JSON files, extracts Tier 1 features using
taste_features.extract_features(), scores each plan with the heuristic
scorer, and builds a CSV suitable for CatBoost PairLogit training.

CSV schema:
    group_id, plan_id, label, manifest_version,
    [28 feature columns], [7 heuristic dimension columns], heuristic_total
"""

from __future__ import annotations

import csv
import logging
import json
import re
from pathlib import Path

from musicmixer.models import RemixPlan, Section
from musicmixer.services.taste_features import extract_features, get_manifest
from musicmixer.services.taste_model import score_candidate

logger = logging.getLogger(__name__)


def _load_plan_from_json(path: Path) -> RemixPlan:
    """Deserialize a RemixPlan from a JSON file (reconstruct.py format).

    Args:
        path: Path to a plan JSON file.

    Returns:
        Deserialized RemixPlan object.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields are missing.
    """
    with open(path) as f:
        data = json.load(f)

    sections = []
    for s in data["sections"]:
        sections.append(Section(
            label=str(s["label"]),
            start_beat=int(s["start_beat"]),
            end_beat=int(s["end_beat"]),
            stem_gains={k: float(v) for k, v in s["stem_gains"].items()},
            transition_in=str(s["transition_in"]),
            transition_beats=int(s["transition_beats"]),
        ))

    return RemixPlan(
        vocal_source=str(data.get("vocal_source", "song_a")),
        start_time_vocal=float(data.get("start_time_vocal", 0.0)),
        end_time_vocal=float(data.get("end_time_vocal", 0.0)),
        start_time_instrumental=float(data.get("start_time_instrumental", 0.0)),
        end_time_instrumental=float(data.get("end_time_instrumental", 0.0)),
        sections=sections,
        tempo_source=str(data.get("tempo_source", "average")),
        key_source=str(data.get("key_source", "none")),
        explanation=str(data.get("explanation", "")),
        warnings=list(data.get("warnings", [])),
        used_fallback=bool(data.get("used_fallback", False)),
    )


def _extract_group_id(plan_id: str) -> str:
    """Extract the mashup group ID from a plan ID.

    Positive plans have IDs like "mashup_001".
    Negative plans have IDs like "mashup_001_neg0", "mashup_001_neg1", etc.
    The group_id is the base mashup ID (without _neg suffix).
    """
    match = re.match(r"^(.+?)(_neg\d+)?$", plan_id)
    if match:
        return match.group(1)
    return plan_id


def build_dataset(
    plans_dir: Path,
    negatives_dir: Path,
    output_path: Path,
) -> Path:
    """Extract features from all plans and build training CSV.

    Process:
        1. Load all positive plans from plans_dir
        2. Load all negative plans from negatives_dir
        3. Extract features for each plan (meta_a=None, meta_b=None)
        4. Score each plan with the heuristic scorer
        5. Build CSV with feature vectors, labels, and group IDs
        6. Tag with manifest_version from get_manifest()

    Args:
        plans_dir: Directory containing positive plan JSON files.
        negatives_dir: Directory containing negative plan JSON files.
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.

    Raises:
        FileNotFoundError: If plans_dir or negatives_dir does not exist.
        ValueError: If no plans are loaded.
    """
    if not plans_dir.exists():
        raise FileNotFoundError(f"Plans directory not found: {plans_dir}")
    if not negatives_dir.exists():
        raise FileNotFoundError(f"Negatives directory not found: {negatives_dir}")

    manifest = get_manifest()

    # Collect all plans with their metadata
    plan_entries: list[dict] = []

    # Load positive plans
    positive_files = sorted(plans_dir.glob("*.json"))
    if not positive_files:
        raise ValueError(f"No positive plan files found in {plans_dir}")

    for plan_path in positive_files:
        plan_id = plan_path.stem
        try:
            plan = _load_plan_from_json(plan_path)
            plan_entries.append({
                "plan_id": plan_id,
                "group_id": _extract_group_id(plan_id),
                "label": 1,
                "plan": plan,
            })
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load positive plan %s: %s", plan_path, exc)
            continue

    # Load negative plans
    negative_files = sorted(negatives_dir.glob("*.json"))
    if not negative_files:
        logger.warning("No negative plan files found in %s", negatives_dir)

    for plan_path in negative_files:
        plan_id = plan_path.stem
        try:
            plan = _load_plan_from_json(plan_path)
            plan_entries.append({
                "plan_id": plan_id,
                "group_id": _extract_group_id(plan_id),
                "label": 0,
                "plan": plan,
            })
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load negative plan %s: %s", plan_path, exc)
            continue

    if not plan_entries:
        raise ValueError("No plans loaded from either directory")

    logger.info(
        "Loaded %d plans (%d positive, %d negative)",
        len(plan_entries),
        sum(1 for e in plan_entries if e["label"] == 1),
        sum(1 for e in plan_entries if e["label"] == 0),
    )

    # Build CSV header: metadata columns + feature columns + heuristic columns
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

    # Extract features and scores for each plan
    rows: list[list] = []
    for entry in plan_entries:
        plan = entry["plan"]
        plan_id = entry["plan_id"]

        # Extract Tier 1 features (no audio metadata available for mashup-derived plans)
        fv = extract_features(plan, meta_a=None, meta_b=None)

        # Score with heuristic scorer
        scored = score_candidate(plan, meta_a=None, meta_b=None)

        row = [
            entry["group_id"],
            plan_id,
            entry["label"],
            manifest.version,
        ]

        # Feature values in sorted order
        for fname in feature_names:
            row.append(fv.features.get(fname, 0.0))

        # Heuristic dimension scores
        for dim in heuristic_dimensions:
            row.append(scored.dimension_scores.get(dim, 0.0))

        # Heuristic total
        row.append(scored.total_score)

        rows.append(row)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    logger.info(
        "Built training dataset: %d rows, %d features + %d heuristic dims -> %s",
        len(rows),
        len(feature_names),
        len(heuristic_columns),
        output_path,
    )

    return output_path
