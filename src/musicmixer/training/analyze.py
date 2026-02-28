"""Analyze downloaded mashups: audio analysis, stem separation, and structure detection.

Reuses the existing analysis and separation services to process each
downloaded mashup into structured analysis data (BPM, key, stems, sections).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from musicmixer.models import (
    AudioMetadata,
    EnergyBuckets,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
)
from musicmixer.services.analysis import analyze_audio, analyze_stems, detect_key
from musicmixer.services.separation import separate_stems

logger = logging.getLogger(__name__)


def _ndarray_to_list(obj: object) -> object:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_ndarray_to_list(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _serialize_analysis(
    metadata: AudioMetadata,
    stem_analysis: StemAnalysis,
    song_structure: SongStructure,
    key: str,
    scale: str,
    key_confidence: float,
) -> dict:
    """Convert analysis results to a JSON-serializable dict.

    Handles numpy arrays by converting them to lists.
    """
    # Serialize StemAnalysis manually to handle numpy arrays
    stem_analysis_dict = {
        "bar_rms": {
            name: arr.tolist() for name, arr in stem_analysis.bar_rms.items()
        },
        "combined_energy": stem_analysis.combined_energy.tolist(),
        "vocal_active": stem_analysis.vocal_active.tolist(),
        "vocal_gaps": [asdict(vg) for vg in stem_analysis.vocal_gaps],
        "bucket_thresholds": asdict(stem_analysis.bucket_thresholds),
    }

    # Serialize SongStructure
    song_structure_dict = {
        "sections": [asdict(s) for s in song_structure.sections],
        "vocal_gaps": [asdict(vg) for vg in song_structure.vocal_gaps],
        "total_bars": song_structure.total_bars,
    }

    # Serialize AudioMetadata (excluding nested analysis objects that
    # we serialize separately, and numpy arrays)
    metadata_dict = {
        "bpm": metadata.bpm,
        "bpm_confidence": metadata.bpm_confidence,
        "beat_frames": metadata.beat_frames.tolist(),
        "duration_seconds": metadata.duration_seconds,
        "total_beats": metadata.total_beats,
        "key": key,
        "scale": scale,
        "key_confidence": key_confidence,
        "mean_rms": metadata.mean_rms,
        "source_quality": metadata.source_quality,
    }

    return {
        "audio_metadata": metadata_dict,
        "stem_analysis": stem_analysis_dict,
        "song_structure": song_structure_dict,
    }


def analyze_mashup(
    mashup_id: str,
    wav_path: Path,
    stems_dir: Path,
) -> dict:
    """Full analysis pipeline for one mashup. Returns serializable dict.

    Pipeline:
        1. analyze_audio() -> BPM, beat_frames, duration, mean_rms
        2. detect_key() -> key, scale, confidence
        3. separate_stems() -> 6 stems via Modal
        4. analyze_stems() -> per-bar energy, vocal activity, section boundaries

    Args:
        mashup_id: Unique identifier for the mashup.
        wav_path: Path to the downloaded WAV file.
        stems_dir: Directory to write separated stem WAV files.
            Stems are saved to stems_dir/{mashup_id}/.

    Returns:
        JSON-serializable dict containing audio_metadata, stem_analysis,
        and song_structure.

    Raises:
        FileNotFoundError: If wav_path does not exist.
        RuntimeError: On stem separation or analysis failure.
    """
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    logger.info("Analyzing mashup %s: %s", mashup_id, wav_path)

    # Step 1: Audio analysis (BPM, beats, duration)
    logger.info("[%s] Running audio analysis...", mashup_id)
    metadata = analyze_audio(wav_path)
    logger.info(
        "[%s] Audio: bpm=%.1f, duration=%.1fs, beats=%d",
        mashup_id,
        metadata.bpm,
        metadata.duration_seconds,
        metadata.total_beats,
    )

    # Step 2: Key detection
    logger.info("[%s] Detecting key...", mashup_id)
    key, scale, key_confidence = detect_key(wav_path)
    logger.info(
        "[%s] Key: %s %s (confidence=%.2f)",
        mashup_id,
        key,
        scale,
        key_confidence,
    )

    # Step 3: Stem separation
    mashup_stems_dir = stems_dir / mashup_id
    mashup_stems_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Separating stems...", mashup_id)
    stem_paths = separate_stems(wav_path, mashup_stems_dir)
    logger.info(
        "[%s] Separated %d stems: %s",
        mashup_id,
        len(stem_paths),
        list(stem_paths.keys()),
    )

    # Step 4: Stem analysis (energy, vocal activity, sections)
    logger.info("[%s] Analyzing stems...", mashup_id)
    stem_analysis, song_structure = analyze_stems(
        stem_paths=stem_paths,
        beat_frames=metadata.beat_frames,
        bpm=metadata.bpm,
        audio_path=wav_path,
    )
    logger.info(
        "[%s] Structure: %d bars, %d sections, %d vocal gaps",
        mashup_id,
        song_structure.total_bars,
        len(song_structure.sections),
        len(song_structure.vocal_gaps),
    )

    result = _serialize_analysis(
        metadata=metadata,
        stem_analysis=stem_analysis,
        song_structure=song_structure,
        key=key,
        scale=scale,
        key_confidence=key_confidence,
    )

    logger.info("[%s] Analysis complete", mashup_id)
    return result


def analyze_all(
    manifest: list[dict],
    raw_dir: Path,
    output_dir: Path,
    max_concurrent: int = 3,
) -> dict[str, Path]:
    """Batch analysis with concurrent Modal separation.

    Analyzes all mashups in the manifest that have been downloaded.
    Skips mashups that have already been analyzed (output JSON exists).
    Logs failures and continues with the remaining entries.

    Note: Stem separation via Modal is the bottleneck. The max_concurrent
    parameter limits how many mashups are processed simultaneously to
    avoid overwhelming the Modal GPU pool. Currently runs sequentially
    since separate_stems() is synchronous and Modal handles its own
    concurrency internally.

    Args:
        manifest: List of manifest entry dicts (from load_manifest).
        raw_dir: Directory containing downloaded WAV files ({id}.wav).
        output_dir: Root output directory. Analysis results go to
            output_dir/analysis/{id}.json and stems to
            output_dir/stems/{id}/.
        max_concurrent: Maximum concurrent analyses. Currently runs
            sequentially; reserved for future parallelization.

    Returns:
        Dict mapping mashup ID to the Path of the analysis JSON file.
        Only includes successfully analyzed entries.
    """
    analysis_dir = output_dir / "analysis"
    stems_dir = output_dir / "stems"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stems_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path] = {}
    skipped = 0
    failed = 0
    missing = 0

    logger.info(
        "Starting batch analysis: %d entries, raw_dir=%s, output_dir=%s",
        len(manifest),
        raw_dir,
        output_dir,
    )

    for i, entry in enumerate(manifest):
        mashup_id = entry["id"]
        title = entry.get("title", mashup_id)

        # Check for source WAV
        wav_path = raw_dir / f"{mashup_id}.wav"
        if not wav_path.exists():
            logger.warning(
                "[%d/%d] Skipping %s (%s) — WAV not found at %s",
                i + 1,
                len(manifest),
                mashup_id,
                title,
                wav_path,
            )
            missing += 1
            continue

        # Check for existing analysis
        analysis_path = analysis_dir / f"{mashup_id}.json"
        if analysis_path.exists() and analysis_path.stat().st_size > 0:
            logger.info(
                "[%d/%d] Skipping %s (%s) — already analyzed",
                i + 1,
                len(manifest),
                mashup_id,
                title,
            )
            results[mashup_id] = analysis_path
            skipped += 1
            continue

        logger.info(
            "[%d/%d] Analyzing %s (%s)...",
            i + 1,
            len(manifest),
            mashup_id,
            title,
        )

        try:
            analysis = analyze_mashup(
                mashup_id=mashup_id,
                wav_path=wav_path,
                stems_dir=stems_dir,
            )

            # Write analysis JSON
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)

            results[mashup_id] = analysis_path

            logger.info(
                "[%d/%d] Analysis saved for %s: %s",
                i + 1,
                len(manifest),
                mashup_id,
                analysis_path,
            )

        except Exception as e:
            logger.error(
                "[%d/%d] Failed to analyze %s (%s): %s",
                i + 1,
                len(manifest),
                mashup_id,
                title,
                e,
                exc_info=True,
            )
            failed += 1
            continue

    logger.info(
        "Batch analysis complete: %d succeeded, %d skipped, "
        "%d missing WAV, %d failed (of %d total)",
        len(results) - skipped,
        skipped,
        missing,
        failed,
        len(manifest),
    )

    return results
