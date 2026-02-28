#!/usr/bin/env python3
"""Mashup reverse-engineering training pipeline CLI.

Provides subcommands for each step of the mashup training pipeline:
    download          Download mashups from curated manifest
    analyze           Run audio analysis and stem separation
    reconstruct       Reconstruct RemixPlan objects from analysis
    generate-negatives Generate synthetic negative training examples
    build-dataset     Extract features and build training CSV
    validate          Run domain alignment validation
    train             Train CatBoost pairwise ranker
    run-all           Execute the full pipeline end-to-end

Usage:
    python scripts/mashup_pipeline.py download --manifest data/mashup_manifest.json
    python scripts/mashup_pipeline.py build-dataset
    python scripts/mashup_pipeline.py run-all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Default paths (match the plan's file layout)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "mashup_manifest.json"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "mashups" / "raw"
DEFAULT_MASHUPS_DIR = DEFAULT_DATA_DIR / "mashups"
DEFAULT_ANALYSIS_DIR = DEFAULT_DATA_DIR / "mashups" / "analysis"
DEFAULT_STEMS_DIR = DEFAULT_DATA_DIR / "mashups" / "stems"
DEFAULT_PLANS_DIR = DEFAULT_DATA_DIR / "mashups" / "plans"
DEFAULT_NEGATIVES_DIR = DEFAULT_DATA_DIR / "mashups" / "negatives"
DEFAULT_TRAINING_DIR = DEFAULT_DATA_DIR / "mashups" / "training"
DEFAULT_TRAINING_CSV = DEFAULT_TRAINING_DIR / "training_data.csv"
DEFAULT_MODELS_DIR = DEFAULT_DATA_DIR / "models"

logger = logging.getLogger("mashup_pipeline")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    """Download mashups from the curated manifest."""
    from musicmixer.training.download import download_mashups

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    logger.info("Downloading mashups from %s to %s", manifest_path, output_dir)
    results = download_mashups(
        manifest_path=manifest_path,
        output_dir=output_dir,
    )
    logger.info("Downloaded %d mashups", len(results))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run audio analysis and stem separation on downloaded mashups."""
    from musicmixer.training.analyze import analyze_all
    from musicmixer.training.download import load_manifest

    manifest_path = Path(args.manifest)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    manifest = load_manifest(manifest_path)
    logger.info(
        "Analyzing %d mashups: raw_dir=%s, output_dir=%s",
        len(manifest),
        raw_dir,
        output_dir,
    )

    results = analyze_all(
        manifest=manifest,
        raw_dir=raw_dir,
        output_dir=output_dir,
    )
    logger.info("Analyzed %d mashups", len(results))


def cmd_reconstruct(args: argparse.Namespace) -> None:
    """Reconstruct RemixPlan objects from analysis results."""
    from musicmixer.training.reconstruct import reconstruct_all

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)

    logger.info("Reconstructing plans from %s to %s", analysis_dir, output_dir)
    written = reconstruct_all(analysis_dir=analysis_dir, output_dir=output_dir)
    logger.info("Reconstructed %d plans", len(written))


def cmd_generate_negatives(args: argparse.Namespace) -> None:
    """Generate synthetic negative training examples from positive plans."""
    from musicmixer.training.dataset import _load_plan_from_json
    from musicmixer.training.negatives import generate_negatives
    from musicmixer.training.reconstruct import _serialize_plan

    plans_dir = Path(args.plans_dir)
    output_dir = Path(args.output_dir)
    n_per_plan = args.n_per_plan

    output_dir.mkdir(parents=True, exist_ok=True)

    plan_files = sorted(plans_dir.glob("*.json"))
    if not plan_files:
        logger.error("No plan files found in %s", plans_dir)
        return

    logger.info(
        "Generating %d negatives per plan from %d plans",
        n_per_plan,
        len(plan_files),
    )

    total_written = 0
    for plan_path in plan_files:
        plan_id = plan_path.stem

        try:
            plan = _load_plan_from_json(plan_path)
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.error("Failed to load plan %s: %s", plan_path, exc)
            continue

        negatives = generate_negatives(plan, n=n_per_plan)

        for i, neg_plan in enumerate(negatives):
            neg_id = f"{plan_id}_neg{i}"
            neg_path = output_dir / f"{neg_id}.json"
            serialized = _serialize_plan(neg_plan)

            with open(neg_path, "w") as f:
                json.dump(serialized, f, indent=2)

            total_written += 1

    logger.info("Generated %d negative plan files in %s", total_written, output_dir)


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Extract features and build the training CSV."""
    from musicmixer.training.dataset import build_dataset

    plans_dir = Path(args.plans_dir)
    negatives_dir = Path(args.negatives_dir)
    output_path = Path(args.output)

    logger.info(
        "Building dataset: plans=%s, negatives=%s, output=%s",
        plans_dir,
        negatives_dir,
        output_path,
    )

    result_path = build_dataset(
        plans_dir=plans_dir,
        negatives_dir=negatives_dir,
        output_path=output_path,
    )
    logger.info("Dataset written to %s", result_path)


def cmd_validate(args: argparse.Namespace) -> None:
    """Run domain alignment validation."""
    from musicmixer.training.validate import validate_domain_alignment

    training_csv = Path(args.training_csv)
    n_candidates = args.n_candidates

    logger.info(
        "Running domain alignment validation: csv=%s, n_candidates=%d",
        training_csv,
        n_candidates,
    )

    try:
        result = validate_domain_alignment(
            training_csv=training_csv,
            n_candidates=n_candidates,
        )
    except ValueError as exc:
        logger.error("Validation failed (hard gate): %s", exc)
        sys.exit(1)

    # Report results
    label_check = result["label_check"]
    print(f"\nLabel vocabulary check: {'PASSED' if label_check['passed'] else 'FAILED'}")
    print(f"  Mashup labels: {label_check['mashup_labels']}")
    print(f"  Candidate labels: {label_check['candidate_labels']}")

    flagged = result["flagged_features"]
    print(f"\nFeature divergence: {result['total_flagged']}/{result['total_features']} flagged")

    if flagged:
        print("\nFlagged features (>1 std dev divergence in mean):")
        for f in flagged:
            print(
                f"  {f['feature']}: "
                f"mashup_mean={f['mashup_mean']:.4f} vs "
                f"candidate_mean={f['candidate_mean']:.4f} "
                f"(divergence={f['divergence_ratio']:.2f}x std)"
            )

    # Save full results
    output_path = Path(args.output) if args.output else training_csv.parent / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2, default=str)
    logger.info("Validation report saved to %s", output_path)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the CatBoost pairwise ranker."""
    from musicmixer.training.train import train_model

    csv_path = Path(args.training_csv)
    output_dir = Path(args.output_dir)

    logger.info("Training model: csv=%s, output_dir=%s", csv_path, output_dir)

    result = train_model(csv_path=csv_path, output_dir=output_dir)

    # Report results
    metrics = result.get("eval_metrics", {})
    print(f"\nTraining complete:")
    print(f"  Model: {result['model_path']}")
    print(f"  Config hash: {result['config_hash']}")
    print(f"  Data hash: {result['data_hash']}")
    print(f"  Feature manifest: {result['feature_manifest_version']}")
    print(f"  Train: {result['train_rows']} rows ({result['train_groups']} groups)")
    print(f"  Val: {result['val_rows']} rows ({result['val_groups']} groups)")
    print(f"\nEvaluation metrics:")
    print(f"  Pairwise accuracy: {metrics.get('pairwise_accuracy', 0):.3f}")
    print(f"  Top-1 preferred rate: {metrics.get('top1_preferred_rate', 0):.3f}")
    print(f"  Total pairs: {metrics.get('total_pairs', 0)}")
    print(f"  Total groups: {metrics.get('total_groups', 0)}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end."""
    logger.info("Running full mashup training pipeline")

    # Step 2: Download
    print("\n=== Step 2: Download Mashups ===")
    cmd_download(argparse.Namespace(
        manifest=args.manifest,
        output_dir=str(DEFAULT_RAW_DIR),
    ))

    # Step 3: Analyze
    print("\n=== Step 3: Analyze Mashups ===")
    cmd_analyze(argparse.Namespace(
        manifest=args.manifest,
        raw_dir=str(DEFAULT_RAW_DIR),
        output_dir=str(DEFAULT_MASHUPS_DIR),
    ))

    # Step 4: Reconstruct
    print("\n=== Step 4: Reconstruct Plans ===")
    cmd_reconstruct(argparse.Namespace(
        analysis_dir=str(DEFAULT_ANALYSIS_DIR),
        output_dir=str(DEFAULT_PLANS_DIR),
    ))

    # Step 5: Generate negatives
    print("\n=== Step 5: Generate Negatives ===")
    cmd_generate_negatives(argparse.Namespace(
        plans_dir=str(DEFAULT_PLANS_DIR),
        output_dir=str(DEFAULT_NEGATIVES_DIR),
        n_per_plan=3,
    ))

    # Step 6: Build dataset
    print("\n=== Step 6: Build Dataset ===")
    cmd_build_dataset(argparse.Namespace(
        plans_dir=str(DEFAULT_PLANS_DIR),
        negatives_dir=str(DEFAULT_NEGATIVES_DIR),
        output=str(DEFAULT_TRAINING_CSV),
    ))

    # Step 6b: Validate
    print("\n=== Step 6b: Domain Alignment Validation ===")
    cmd_validate(argparse.Namespace(
        training_csv=str(DEFAULT_TRAINING_CSV),
        n_candidates=30,
        output=str(DEFAULT_TRAINING_DIR / "validation_report.json"),
    ))

    # Step 7: Train
    print("\n=== Step 7: Train Model ===")
    cmd_train(argparse.Namespace(
        training_csv=str(DEFAULT_TRAINING_CSV),
        output_dir=str(DEFAULT_MODELS_DIR),
    ))

    print("\n=== Pipeline Complete ===")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="mashup_pipeline",
        description="Mashup reverse-engineering training pipeline",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # download
    dl = subparsers.add_parser("download", help="Download mashups from manifest")
    dl.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to mashup manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    dl.add_argument(
        "--output-dir",
        default=str(DEFAULT_RAW_DIR),
        help=f"Output directory for WAV files (default: {DEFAULT_RAW_DIR})",
    )

    # analyze
    an = subparsers.add_parser("analyze", help="Analyze downloaded mashups")
    an.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to mashup manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    an.add_argument(
        "--raw-dir",
        default=str(DEFAULT_RAW_DIR),
        help=f"Directory with downloaded WAV files (default: {DEFAULT_RAW_DIR})",
    )
    an.add_argument(
        "--output-dir",
        default=str(DEFAULT_MASHUPS_DIR),
        help=f"Root output directory (default: {DEFAULT_MASHUPS_DIR})",
    )

    # reconstruct
    rc = subparsers.add_parser("reconstruct", help="Reconstruct RemixPlan objects")
    rc.add_argument(
        "--analysis-dir",
        default=str(DEFAULT_ANALYSIS_DIR),
        help=f"Directory with analysis JSONs (default: {DEFAULT_ANALYSIS_DIR})",
    )
    rc.add_argument(
        "--output-dir",
        default=str(DEFAULT_PLANS_DIR),
        help=f"Output directory for plan JSONs (default: {DEFAULT_PLANS_DIR})",
    )

    # generate-negatives
    gn = subparsers.add_parser("generate-negatives", help="Generate negative examples")
    gn.add_argument(
        "--plans-dir",
        default=str(DEFAULT_PLANS_DIR),
        help=f"Directory with positive plan JSONs (default: {DEFAULT_PLANS_DIR})",
    )
    gn.add_argument(
        "--output-dir",
        default=str(DEFAULT_NEGATIVES_DIR),
        help=f"Output directory for negative plan JSONs (default: {DEFAULT_NEGATIVES_DIR})",
    )
    gn.add_argument(
        "--n-per-plan",
        type=int,
        default=3,
        help="Number of negatives per positive plan (default: 3)",
    )

    # build-dataset
    bd = subparsers.add_parser("build-dataset", help="Build training CSV")
    bd.add_argument(
        "--plans-dir",
        default=str(DEFAULT_PLANS_DIR),
        help=f"Directory with positive plan JSONs (default: {DEFAULT_PLANS_DIR})",
    )
    bd.add_argument(
        "--negatives-dir",
        default=str(DEFAULT_NEGATIVES_DIR),
        help=f"Directory with negative plan JSONs (default: {DEFAULT_NEGATIVES_DIR})",
    )
    bd.add_argument(
        "--output",
        default=str(DEFAULT_TRAINING_CSV),
        help=f"Output CSV path (default: {DEFAULT_TRAINING_CSV})",
    )

    # validate
    vl = subparsers.add_parser("validate", help="Run domain alignment validation")
    vl.add_argument(
        "--training-csv",
        default=str(DEFAULT_TRAINING_CSV),
        help=f"Path to training CSV (default: {DEFAULT_TRAINING_CSV})",
    )
    vl.add_argument(
        "--n-candidates",
        type=int,
        default=30,
        help="Number of candidate plans to generate for comparison (default: 30)",
    )
    vl.add_argument(
        "--output",
        default=None,
        help="Output path for validation report JSON (default: alongside CSV)",
    )

    # train
    tr = subparsers.add_parser("train", help="Train CatBoost model")
    tr.add_argument(
        "--training-csv",
        default=str(DEFAULT_TRAINING_CSV),
        help=f"Path to training CSV (default: {DEFAULT_TRAINING_CSV})",
    )
    tr.add_argument(
        "--output-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=f"Output directory for model files (default: {DEFAULT_MODELS_DIR})",
    )

    # run-all
    ra = subparsers.add_parser("run-all", help="Run full pipeline end-to-end")
    ra.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to mashup manifest JSON (default: {DEFAULT_MANIFEST})",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to subcommand handler
    handlers = {
        "download": cmd_download,
        "analyze": cmd_analyze,
        "reconstruct": cmd_reconstruct,
        "generate-negatives": cmd_generate_negatives,
        "build-dataset": cmd_build_dataset,
        "validate": cmd_validate,
        "train": cmd_train,
        "run-all": cmd_run_all,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
