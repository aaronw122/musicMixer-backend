"""CatBoost pairwise ranker training for the taste model.

Step 7 of the mashup training pipeline. Trains a CatBoost model using
PairLogit loss on the training CSV produced by Step 6. The model learns
to rank positive (mashup-derived) plans higher than negative (synthetic)
plans within each group.

CatBoost is an optional dependency -- this module guards imports and
raises a clear error if catboost is not installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

from musicmixer.services.taste_features import get_manifest

logger = logging.getLogger(__name__)

# CatBoost training configuration
CATBOOST_PARAMS: dict = {
    "loss_function": "PairLogit",
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "od_type": "Iter",
    "od_wait": 50,
    "random_seed": 42,
    "verbose": 50,
}


def _compute_data_hash(csv_path: Path) -> str:
    """Compute a SHA-256 hash of the training CSV for reproducibility tracking."""
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _compute_config_hash(params: dict) -> str:
    """Compute a hash of the training configuration."""
    payload = json.dumps(params, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _load_training_data(csv_path: Path) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """Load training data from CSV.

    Returns:
        Tuple of (feature_matrix, labels, group_ids, feature_names).
        feature_matrix: List of feature value lists (one per row).
        labels: List of label values (0 or 1).
        group_ids: List of group ID strings.
        feature_names: Sorted list of feature column names.
    """
    manifest = get_manifest()
    feature_names = sorted(manifest.feature_names)

    # Also include heuristic dimension columns
    heuristic_dimensions = [
        "heuristic_arrangement_quality",
        "heuristic_energy_arc",
        "heuristic_vocal_intelligibility",
        "heuristic_harmonic_fit",
        "heuristic_transition_quality",
        "heuristic_groove_coherence",
        "heuristic_loudness_fatigue",
        "heuristic_total",
    ]

    all_feature_cols = feature_names + heuristic_dimensions

    feature_matrix: list[list[float]] = []
    labels: list[int] = []
    group_ids: list[str] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = []
            for col in all_feature_cols:
                features.append(float(row.get(col, 0.0)))
            feature_matrix.append(features)
            labels.append(int(row["label"]))
            group_ids.append(row["group_id"])

    return feature_matrix, labels, group_ids, all_feature_cols


def _split_by_group(
    feature_matrix: list[list[float]],
    labels: list[int],
    group_ids: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[
    list[list[float]], list[int], list[str],
    list[list[float]], list[int], list[str],
]:
    """Split data by group_id into train/validation sets.

    Ensures all rows with the same group_id go to the same split,
    preventing data leakage.

    Returns:
        (train_features, train_labels, train_groups,
         val_features, val_labels, val_groups)
    """
    import random

    unique_groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    split_idx = max(1, int(len(unique_groups) * train_ratio))
    train_groups_set = set(unique_groups[:split_idx])

    train_features: list[list[float]] = []
    train_labels: list[int] = []
    train_groups: list[str] = []
    val_features: list[list[float]] = []
    val_labels: list[int] = []
    val_groups: list[str] = []

    for features, label, group in zip(feature_matrix, labels, group_ids):
        if group in train_groups_set:
            train_features.append(features)
            train_labels.append(label)
            train_groups.append(group)
        else:
            val_features.append(features)
            val_labels.append(label)
            val_groups.append(group)

    return (
        train_features, train_labels, train_groups,
        val_features, val_labels, val_groups,
    )


def _evaluate_pairwise(
    model: object,
    features: list[list[float]],
    labels: list[int],
    group_ids: list[str],
    feature_names: list[str],
) -> dict:
    """Evaluate pairwise accuracy and top-1 preferred rate on holdout.

    For each group, form all (positive, negative) pairs and check if the
    model scores the positive higher.

    Returns:
        Dict with "pairwise_accuracy", "top1_preferred_rate", "total_pairs",
        "total_groups".
    """
    try:
        import catboost  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "catboost is required for evaluation. Install with: "
            "uv add --optional ml catboost"
        ) from exc

    pool = catboost.Pool(data=features, feature_names=feature_names)
    predictions = model.predict(pool)

    # Group predictions by group_id
    groups: dict[str, list[tuple[float, int]]] = {}
    for pred, label, gid in zip(predictions, labels, group_ids):
        if gid not in groups:
            groups[gid] = []
        groups[gid].append((float(pred), label))

    correct_pairs = 0
    total_pairs = 0
    top1_correct = 0
    total_groups = 0

    for gid, entries in groups.items():
        positives = [(pred, lab) for pred, lab in entries if lab == 1]
        negatives = [(pred, lab) for pred, lab in entries if lab == 0]

        if not positives or not negatives:
            continue

        total_groups += 1

        # Pairwise accuracy: for each (pos, neg) pair, check if pos > neg
        for pos_pred, _ in positives:
            for neg_pred, _ in negatives:
                total_pairs += 1
                if pos_pred > neg_pred:
                    correct_pairs += 1

        # Top-1 preferred rate: does the highest-scored plan in this group
        # have label=1?
        best_pred, best_label = max(entries, key=lambda x: x[0])
        if best_label == 1:
            top1_correct += 1

    return {
        "pairwise_accuracy": correct_pairs / total_pairs if total_pairs > 0 else 0.0,
        "top1_preferred_rate": top1_correct / total_groups if total_groups > 0 else 0.0,
        "total_pairs": total_pairs,
        "total_groups": total_groups,
    }


def train_model(csv_path: Path, output_dir: Path) -> dict:
    """Train CatBoost pairwise ranker. Returns eval metrics dict.

    Process:
        1. Load training_data.csv
        2. Split by group_id (80/20)
        3. Create CatBoost Pool objects with group_id
        4. Train with early stopping
        5. Evaluate: pairwise accuracy, top-1 preferred rate on holdout
        6. Save model to output_dir/taste_v1.cbm
        7. Save training manifest with config, data hash, metrics

    Args:
        csv_path: Path to training_data.csv.
        output_dir: Directory to save model and manifest.

    Returns:
        Dict with training configuration, eval metrics, and file paths.

    Raises:
        ImportError: If catboost is not installed.
        FileNotFoundError: If csv_path does not exist.
        ValueError: If training data is insufficient.
    """
    try:
        import catboost  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "catboost is required for training. Install with: "
            "uv add --optional ml catboost"
        ) from exc

    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    # Step 1: Load data
    feature_matrix, labels, group_ids, feature_names = _load_training_data(csv_path)

    if len(feature_matrix) < 4:
        raise ValueError(
            f"Insufficient training data: {len(feature_matrix)} rows "
            f"(need at least 4 for train/val split)"
        )

    unique_groups = set(group_ids)
    logger.info(
        "Loaded training data: %d rows, %d groups, %d features",
        len(feature_matrix),
        len(unique_groups),
        len(feature_names),
    )

    # Step 2: Split by group_id (80/20)
    (
        train_features, train_labels, train_groups,
        val_features, val_labels, val_groups,
    ) = _split_by_group(feature_matrix, labels, group_ids)

    logger.info(
        "Train/val split: %d train rows (%d groups), %d val rows (%d groups)",
        len(train_features),
        len(set(train_groups)),
        len(val_features),
        len(set(val_groups)),
    )

    if not val_features:
        logger.warning(
            "No validation data after split. Training without early stopping validation."
        )

    # Step 3: Create CatBoost Pool objects
    train_pool = catboost.Pool(
        data=train_features,
        label=train_labels,
        group_id=train_groups,
        feature_names=feature_names,
    )

    eval_pool = None
    if val_features:
        eval_pool = catboost.Pool(
            data=val_features,
            label=val_labels,
            group_id=val_groups,
            feature_names=feature_names,
        )

    # Step 4: Train
    model = catboost.CatBoost(CATBOOST_PARAMS)

    logger.info("Training CatBoost with params: %s", CATBOOST_PARAMS)

    if eval_pool is not None:
        model.fit(train_pool, eval_set=eval_pool, verbose=CATBOOST_PARAMS.get("verbose", 50))
    else:
        model.fit(train_pool, verbose=CATBOOST_PARAMS.get("verbose", 50))

    # Step 5: Evaluate
    eval_metrics: dict = {}

    if val_features:
        eval_metrics = _evaluate_pairwise(
            model, val_features, val_labels, val_groups, feature_names
        )
        logger.info(
            "Evaluation: pairwise_accuracy=%.3f, top1_preferred_rate=%.3f "
            "(%d pairs, %d groups)",
            eval_metrics["pairwise_accuracy"],
            eval_metrics["top1_preferred_rate"],
            eval_metrics["total_pairs"],
            eval_metrics["total_groups"],
        )
    else:
        # Evaluate on training data as a sanity check
        eval_metrics = _evaluate_pairwise(
            model, train_features, train_labels, train_groups, feature_names
        )
        eval_metrics["note"] = "Evaluated on training data (no validation split)"
        logger.warning("Evaluated on training data (no validation split)")

    # Step 6: Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "taste_v1.cbm"
    model.save_model(str(model_path))
    logger.info("Saved model to %s", model_path)

    # Step 7: Save training manifest
    manifest = get_manifest()
    data_hash = _compute_data_hash(csv_path)
    config_hash = _compute_config_hash(CATBOOST_PARAMS)

    training_manifest = {
        "model_path": str(model_path),
        "csv_path": str(csv_path),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "feature_manifest_version": manifest.version,
        "catboost_params": CATBOOST_PARAMS,
        "train_rows": len(train_features),
        "val_rows": len(val_features),
        "train_groups": len(set(train_groups)),
        "val_groups": len(set(val_groups)),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "eval_metrics": eval_metrics,
    }

    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(training_manifest, f, indent=2)
    logger.info("Saved training manifest to %s", manifest_path)

    return training_manifest
