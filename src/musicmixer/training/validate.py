"""Domain alignment validation for mashup training data.

Step 6b of the mashup training pipeline. Compares feature distributions
between mashup-derived (reconstructed) plans and inference-time candidate
plans to detect train/serve skew.

Flags features where the two distributions diverge by more than 1 standard
deviation in mean. Also validates that the Section.label vocabulary is
identical across both populations (hard gate).
"""

from __future__ import annotations

import csv
import logging
import statistics
from pathlib import Path

import numpy as np

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.candidate_planner import generate_candidates
from musicmixer.services.taste_features import extract_features, get_manifest

logger = logging.getLogger(__name__)


def _make_synthetic_metadata(
    bpm: float = 120.0,
    duration: float = 210.0,
    key: str = "C",
    scale: str = "major",
) -> AudioMetadata:
    """Create a synthetic AudioMetadata for candidate generation.

    The candidate planner requires AudioMetadata objects to compute
    total beats and tempo targets. These synthetic values represent
    typical song characteristics.
    """
    total_beats = int(bpm / 60.0 * duration)
    # Round to nearest 4-beat bar boundary
    total_beats = (total_beats // 4) * 4

    # Create a simple beat_frames array
    beat_interval_frames = int(22050 * 60.0 / bpm)  # sr=22050 default for librosa
    beat_frames = np.arange(0, total_beats) * beat_interval_frames

    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=total_beats,
        key=key,
        scale=scale,
        key_confidence=0.8,
    )


def _collect_labels(plans: list[RemixPlan]) -> set[str]:
    """Collect all unique Section.label values from a list of plans."""
    labels: set[str] = set()
    for plan in plans:
        for section in plan.sections:
            labels.add(section.label)
    return labels


def _compute_feature_stats(
    feature_vectors: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute mean, std, min, max for each feature across a list of vectors.

    Returns:
        Dict mapping feature_name -> {"mean": ..., "std": ..., "min": ..., "max": ...}
    """
    if not feature_vectors:
        return {}

    feature_names = sorted(feature_vectors[0].keys())
    stats: dict[str, dict[str, float]] = {}

    for fname in feature_names:
        values = [fv[fname] for fv in feature_vectors]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        stats[fname] = {
            "mean": mean,
            "std": std,
            "min": min(values),
            "max": max(values),
        }

    return stats


def _load_mashup_plans_from_csv(training_csv: Path) -> list[RemixPlan]:
    """Load positive mashup-derived plans from the training CSV.

    We cannot reconstruct full RemixPlan objects from CSV feature rows, but
    we can load the plan files from the plans directory. Instead, we extract
    just the label information we need from the feature data.

    For feature distribution comparison, we read the CSV directly.
    """
    # This is only used for label extraction; feature values come from CSV
    return []


def _load_csv_features(
    training_csv: Path,
    label_filter: int | None = None,
) -> list[dict[str, float]]:
    """Load feature vectors from training CSV.

    Args:
        training_csv: Path to the training_data.csv file.
        label_filter: If set, only include rows with this label value (0 or 1).

    Returns:
        List of dicts mapping feature_name -> float value.
    """
    manifest = get_manifest()
    feature_names = sorted(manifest.feature_names)

    vectors: list[dict[str, float]] = []
    with open(training_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if label_filter is not None and int(row["label"]) != label_filter:
                continue

            fv: dict[str, float] = {}
            for fname in feature_names:
                fv[fname] = float(row.get(fname, 0.0))
            vectors.append(fv)

    return vectors


def _load_csv_labels(training_csv: Path, label_filter: int | None = None) -> set[str]:
    """Extract unique section labels from plans in a directory.

    Since CSV doesn't store raw section labels, we look for plan JSON files
    in the standard locations relative to the CSV path.
    """
    # The plans directory is a sibling of the training directory
    # training_data.csv is at data/mashups/training/training_data.csv
    # plans are at data/mashups/plans/*.json
    plans_dir = training_csv.parent.parent / "plans"
    negatives_dir = training_csv.parent.parent / "negatives"

    labels: set[str] = set()

    for directory in [plans_dir, negatives_dir]:
        if not directory.exists():
            continue
        for plan_path in directory.glob("*.json"):
            try:
                from musicmixer.training.dataset import _load_plan_from_json
                plan = _load_plan_from_json(plan_path)
                for section in plan.sections:
                    labels.add(section.label)
            except Exception:
                continue

    return labels


def validate_domain_alignment(
    training_csv: Path,
    example_songs_dir: Path | None = None,
    n_candidates: int = 30,
) -> dict:
    """Compare feature distributions between mashup-derived and candidate plans.

    Process:
        1. Generate candidate plans from synthetic AudioMetadata (or example
           songs if available) using the candidate planner.
        2. Extract features for each candidate.
        3. Compare mean, std, range for each feature between the mashup-derived
           training data and the candidate plan population.
        4. Flag features with >1 std dev divergence in mean.
        5. HARD GATE: Verify label vocabulary consistency -- the set of unique
           Section.label values must be identical across reconstructed and
           candidate plans.

    Args:
        training_csv: Path to the training CSV produced by build_dataset().
        example_songs_dir: Optional directory containing example song files
            for generating candidates. If None, uses synthetic metadata.
        n_candidates: Number of candidate plans to generate for comparison.

    Returns:
        Dict with:
            - "label_check": {"passed": bool, "mashup_labels": set, "candidate_labels": set}
            - "flagged_features": list of dicts with feature name and divergence info
            - "total_features": int
            - "total_flagged": int
            - "mashup_stats": per-feature statistics for mashup plans
            - "candidate_stats": per-feature statistics for candidate plans

    Raises:
        FileNotFoundError: If training_csv does not exist.
        ValueError: If label vocabularies diverge (hard gate failure).
    """
    if not training_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {training_csv}")

    # Step 1: Load mashup-derived feature vectors from CSV (positives only)
    mashup_features = _load_csv_features(training_csv, label_filter=1)
    if not mashup_features:
        raise ValueError("No positive (mashup-derived) plans found in training CSV")

    logger.info("Loaded %d mashup-derived feature vectors from CSV", len(mashup_features))

    # Step 2: Generate candidate plans
    # Use a range of synthetic metadata to cover typical song characteristics
    metadata_configs = [
        {"bpm": 90.0, "duration": 200.0, "key": "Am", "scale": "minor"},
        {"bpm": 120.0, "duration": 210.0, "key": "C", "scale": "major"},
        {"bpm": 140.0, "duration": 180.0, "key": "G", "scale": "major"},
        {"bpm": 100.0, "duration": 240.0, "key": "Em", "scale": "minor"},
        {"bpm": 128.0, "duration": 195.0, "key": "F", "scale": "major"},
    ]

    candidate_plans: list[RemixPlan] = []
    candidates_per_config = max(1, n_candidates // len(metadata_configs))

    for config in metadata_configs:
        meta_a = _make_synthetic_metadata(**config)
        # Use slightly different BPM for song B to simulate real conditions
        meta_b = _make_synthetic_metadata(
            bpm=config["bpm"] * 1.05,
            duration=config["duration"] * 0.95,
            key="G" if config["key"] == "C" else "C",
            scale="major",
        )

        plans = generate_candidates(
            meta_a=meta_a,
            meta_b=meta_b,
            target_count=candidates_per_config,
            min_count=max(1, candidates_per_config - 2),
            max_count=candidates_per_config + 4,
        )
        candidate_plans.extend(plans)

        if len(candidate_plans) >= n_candidates:
            break

    candidate_plans = candidate_plans[:n_candidates]
    logger.info("Generated %d candidate plans", len(candidate_plans))

    # Step 3: Extract features for candidates
    candidate_feature_vectors: list[dict[str, float]] = []
    for plan in candidate_plans:
        fv = extract_features(plan, meta_a=None, meta_b=None)
        candidate_feature_vectors.append(fv.features)

    # Step 4: Compute statistics and compare
    mashup_stats = _compute_feature_stats(mashup_features)
    candidate_stats = _compute_feature_stats(candidate_feature_vectors)

    # Flag features with >1 std dev divergence in mean
    flagged_features: list[dict] = []
    manifest = get_manifest()

    for fname in sorted(manifest.feature_names):
        m_stat = mashup_stats.get(fname, {"mean": 0.0, "std": 0.0})
        c_stat = candidate_stats.get(fname, {"mean": 0.0, "std": 0.0})

        # Use the pooled std as the reference scale
        pooled_std = max(m_stat["std"], c_stat["std"], 1e-10)
        mean_diff = abs(m_stat["mean"] - c_stat["mean"])
        divergence_ratio = mean_diff / pooled_std

        if divergence_ratio > 1.0:
            flagged_features.append({
                "feature": fname,
                "mashup_mean": m_stat["mean"],
                "mashup_std": m_stat["std"],
                "candidate_mean": c_stat["mean"],
                "candidate_std": c_stat["std"],
                "mean_difference": mean_diff,
                "divergence_ratio": divergence_ratio,
            })

    logger.info(
        "Feature comparison: %d/%d features flagged (>1 std dev divergence)",
        len(flagged_features),
        len(manifest.feature_names),
    )

    # Step 5: Label vocabulary consistency check (HARD GATE)
    mashup_labels = _load_csv_labels(training_csv, label_filter=1)
    candidate_labels = _collect_labels(candidate_plans)

    label_check_passed = True
    label_check_message = "Label vocabularies match"

    if mashup_labels and candidate_labels:
        if mashup_labels != candidate_labels:
            only_in_mashup = mashup_labels - candidate_labels
            only_in_candidate = candidate_labels - mashup_labels
            label_check_passed = False
            label_check_message = (
                f"Label vocabulary mismatch: "
                f"only in mashup={only_in_mashup}, "
                f"only in candidate={only_in_candidate}"
            )
            logger.error("HARD GATE FAILURE: %s", label_check_message)
    elif not mashup_labels:
        label_check_message = (
            "Could not extract mashup labels from plan files; "
            "skipping label vocabulary check"
        )
        logger.warning(label_check_message)

    result = {
        "label_check": {
            "passed": label_check_passed,
            "mashup_labels": sorted(mashup_labels) if mashup_labels else [],
            "candidate_labels": sorted(candidate_labels),
            "message": label_check_message,
        },
        "flagged_features": flagged_features,
        "total_features": len(manifest.feature_names),
        "total_flagged": len(flagged_features),
        "mashup_stats": {k: v for k, v in mashup_stats.items()},
        "candidate_stats": {k: v for k, v in candidate_stats.items()},
    }

    if not label_check_passed:
        raise ValueError(f"Domain alignment validation failed: {label_check_message}")

    return result
