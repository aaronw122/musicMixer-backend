"""Plan reconstruction: convert mashup analysis results into RemixPlan objects.

Step 4 of the mashup training pipeline. Takes the output of Step 3 (audio
analysis + stem analysis + song structure) and reconstructs a RemixPlan that
represents the arrangement decisions implicit in the original mashup.

The reconstructed plan uses the inference-time label vocabulary:
    intro | verse | breakdown | drop | outro
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from musicmixer.models import (
    AudioMetadata,
    EnergyBuckets,
    RemixPlan,
    Section,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
)

logger = logging.getLogger(__name__)

# Stem names matching analysis.py STEM_NAMES
STEM_NAMES: list[str] = ["drums", "bass", "guitar", "piano", "vocals", "other"]

# Energy change threshold in dB for transition type inference
SHARP_INCREASE_DB: float = 3.0

# Section label mapping from analysis labels to inference-time vocabulary.
# The analysis pipeline (detect_sections) emits:
#   intro, verse, chorus, instrumental, breakdown, build, outro
# The inference-time vocabulary (LLM interpreter) uses:
#   intro, verse, breakdown, drop, outro
_LABEL_MAP: dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "chorus": "drop",
    "build": "verse",
    "breakdown": "breakdown",
    "outro": "outro",
    # "instrumental" is handled conditionally based on energy level
}


def _map_section_label(section: SectionInfo) -> str:
    """Map a SectionInfo label to the inference-time vocabulary.

    For "instrumental" sections, the mapping depends on energy level:
    - low/medium energy -> "breakdown"
    - high/peak energy -> "drop"

    All other labels use the static _LABEL_MAP.
    """
    if section.label == "instrumental":
        if section.energy_level in ("high", "peak"):
            return "drop"
        return "breakdown"
    return _LABEL_MAP.get(section.label, "verse")


def _compute_section_stem_gains(
    bar_rms: dict[str, np.ndarray],
    start_bar: int,
    end_bar: int,
    global_max_per_stem: dict[str, float],
) -> dict[str, float]:
    """Compute normalized stem gains for a section.

    For each stem, computes the mean RMS across bars within the section,
    then normalizes to [0, 1] relative to the stem's global max RMS across
    all sections.

    Args:
        bar_rms: Per-stem bar RMS arrays from StemAnalysis.
        start_bar: Section start bar (inclusive).
        end_bar: Section end bar (exclusive).
        global_max_per_stem: Pre-computed global max mean-RMS per stem.

    Returns:
        Dict mapping stem name to normalized gain in [0, 1].
    """
    gains: dict[str, float] = {}
    for stem_name in STEM_NAMES:
        rms = bar_rms.get(stem_name)
        if rms is None or len(rms) == 0:
            gains[stem_name] = 0.0
            continue

        # Clamp indices to valid range
        s = max(0, min(start_bar, len(rms)))
        e = max(0, min(end_bar, len(rms)))
        if s >= e:
            gains[stem_name] = 0.0
            continue

        section_mean = float(np.mean(rms[s:e]))
        global_max = global_max_per_stem.get(stem_name, 0.0)
        if global_max > 0:
            gains[stem_name] = min(section_mean / global_max, 1.0)
        else:
            gains[stem_name] = 0.0

    return gains


def _compute_global_max_per_stem(
    bar_rms: dict[str, np.ndarray],
    sections: list[SectionInfo],
) -> dict[str, float]:
    """Pre-compute global max of per-section mean RMS for each stem.

    For each stem, iterate over all sections, compute mean RMS within each
    section, and track the maximum. This becomes the normalization factor
    for stem_gains.
    """
    global_max: dict[str, float] = {name: 0.0 for name in STEM_NAMES}

    for section in sections:
        for stem_name in STEM_NAMES:
            rms = bar_rms.get(stem_name)
            if rms is None or len(rms) == 0:
                continue
            s = max(0, min(section.start_bar, len(rms)))
            e = max(0, min(section.end_bar, len(rms)))
            if s >= e:
                continue
            section_mean = float(np.mean(rms[s:e]))
            if section_mean > global_max[stem_name]:
                global_max[stem_name] = section_mean

    return global_max


def _rms_to_db(rms: float) -> float:
    """Convert RMS amplitude to dB. Returns -inf for zero/negative values."""
    if rms <= 0:
        return float("-inf")
    return 20.0 * np.log10(rms)


def _infer_transition(
    combined_energy: np.ndarray,
    section: SectionInfo,
    prev_section: SectionInfo | None,
) -> tuple[str, int]:
    """Infer transition type and beat length from energy trajectory at boundary.

    Rules:
    - Sharp increase (>3dB) at boundary -> cut, 2 beats
    - Gradual change (<3dB) at boundary -> crossfade, 4 beats
    - Drop then rise pattern -> crossfade, 8 beats
    - Default -> crossfade, 4 beats

    For the first section (no prev_section), defaults to "fade", 4 beats.

    Args:
        combined_energy: Per-bar normalized combined energy array.
        section: Current section.
        prev_section: Previous section (None for first section).

    Returns:
        (transition_in, transition_beats) tuple.
    """
    if prev_section is None:
        return "fade", 4

    if len(combined_energy) == 0:
        return "crossfade", 4

    # Get energy at the boundary: last bar of prev section vs first bar of current
    prev_end_bar = prev_section.end_bar - 1
    curr_start_bar = section.start_bar

    if prev_end_bar < 0 or prev_end_bar >= len(combined_energy):
        return "crossfade", 4
    if curr_start_bar >= len(combined_energy):
        return "crossfade", 4

    prev_energy = float(combined_energy[prev_end_bar])
    curr_energy = float(combined_energy[curr_start_bar])

    prev_db = _rms_to_db(prev_energy)
    curr_db = _rms_to_db(curr_energy)

    # Check for drop-then-rise pattern: look at a few bars around the boundary.
    # A "drop then rise" means there's a dip in energy near the boundary.
    drop_then_rise = _detect_drop_then_rise(
        combined_energy, prev_section, section
    )

    if drop_then_rise:
        return "crossfade", 8

    db_change = curr_db - prev_db

    if db_change > SHARP_INCREASE_DB:
        return "cut", 2

    # Gradual change (including decreases and small increases)
    return "crossfade", 4


def _detect_drop_then_rise(
    combined_energy: np.ndarray,
    prev_section: SectionInfo,
    curr_section: SectionInfo,
) -> bool:
    """Detect a drop-then-rise energy pattern around a section boundary.

    Looks at the last 2 bars of the previous section and first 2 bars of the
    current section. If there's a dip (the boundary bars are lower than both
    the bars before and after), this indicates a drop-then-rise pattern.
    """
    boundary_bar = prev_section.end_bar  # = curr_section.start_bar

    # Need at least 2 bars before and after the boundary
    look_back = 2
    look_ahead = 2

    before_start = max(0, boundary_bar - look_back)
    after_end = min(len(combined_energy), boundary_bar + look_ahead)

    if before_start >= boundary_bar or boundary_bar >= after_end:
        return False

    # Energy before the boundary (last bars of prev section)
    energy_before = float(np.mean(combined_energy[before_start:boundary_bar]))

    # Energy after the boundary (first bars of curr section)
    energy_after = float(np.mean(combined_energy[boundary_bar:after_end]))

    # Energy right at the boundary (1 bar on each side)
    boundary_start = max(0, boundary_bar - 1)
    boundary_end = min(len(combined_energy), boundary_bar + 1)
    if boundary_start >= boundary_end:
        return False
    energy_at_boundary = float(np.mean(combined_energy[boundary_start:boundary_end]))

    # Drop-then-rise: boundary energy is significantly lower than both neighbors
    if energy_before <= 0 or energy_after <= 0:
        return False

    before_db = _rms_to_db(energy_before)
    at_boundary_db = _rms_to_db(energy_at_boundary)
    after_db = _rms_to_db(energy_after)

    # The boundary dips below both sides by at least 3dB
    return bool(
        before_db - at_boundary_db > SHARP_INCREASE_DB
        and after_db - at_boundary_db > SHARP_INCREASE_DB
    )


def reconstruct_plan(analysis: dict, mashup_id: str) -> RemixPlan:
    """Convert mashup analysis to a RemixPlan object.

    Takes a serialized analysis dict (from Step 3 output) and reconstructs a
    RemixPlan that captures the arrangement decisions in the original mashup.

    The analysis dict must contain:
    - "audio_metadata": dict with at least "duration_seconds" and "bpm"
    - "song_structure": dict with "sections" list and "total_bars"
    - "stem_analysis": dict with "bar_rms" (stem_name -> list of floats)
      and "combined_energy" (list of floats)

    Args:
        analysis: Serialized analysis dict from Step 3.
        mashup_id: Unique identifier for the mashup.

    Returns:
        Reconstructed RemixPlan.

    Raises:
        KeyError: If required fields are missing from the analysis dict.
        ValueError: If the analysis contains no sections.
    """
    # Extract components from analysis dict
    audio_meta = analysis["audio_metadata"]
    structure = analysis["song_structure"]
    stem_data = analysis["stem_analysis"]

    duration = float(audio_meta.get("duration_seconds", 0.0))
    title = analysis.get("title", mashup_id)

    # Reconstruct SectionInfo objects from serialized data
    raw_sections: list[SectionInfo] = []
    for s in structure.get("sections", []):
        raw_sections.append(SectionInfo(
            start_bar=int(s["start_bar"]),
            end_bar=int(s["end_bar"]),
            bar_count=int(s.get("bar_count", s["end_bar"] - s["start_bar"])),
            start_time=float(s.get("start_time", 0.0)),
            end_time=float(s.get("end_time", 0.0)),
            label=str(s["label"]),
            energy_level=str(s.get("energy_level", "medium")),
            energy_trajectory=str(s.get("energy_trajectory", "")),
            density=str(s.get("density", "mid")),
            vocal_status=str(s.get("vocal_status", "vox:no")),
            annotations=list(s.get("annotations", [])),
        ))

    if not raw_sections:
        raise ValueError(f"No sections found in analysis for mashup {mashup_id}")

    # Reconstruct bar_rms as numpy arrays
    bar_rms: dict[str, np.ndarray] = {}
    raw_bar_rms = stem_data.get("bar_rms", {})
    for stem_name in STEM_NAMES:
        values = raw_bar_rms.get(stem_name, [])
        bar_rms[stem_name] = np.array(values, dtype=np.float64)

    # Reconstruct combined_energy
    combined_energy = np.array(
        stem_data.get("combined_energy", []), dtype=np.float64
    )

    # Pre-compute global max per stem (for gain normalization)
    global_max_per_stem = _compute_global_max_per_stem(bar_rms, raw_sections)

    # Build reconstructed sections
    reconstructed_sections: list[Section] = []
    for i, raw_sec in enumerate(raw_sections):
        # 1. Map label to inference-time vocabulary
        label = _map_section_label(raw_sec)

        # 2. Convert bar-based to beat-based boundaries
        start_beat = raw_sec.start_bar * 4
        end_beat = raw_sec.end_bar * 4

        # 3. Estimate stem gains
        stem_gains = _compute_section_stem_gains(
            bar_rms, raw_sec.start_bar, raw_sec.end_bar, global_max_per_stem
        )

        # 4. Infer transition type
        prev_section = raw_sections[i - 1] if i > 0 else None
        transition_in, transition_beats = _infer_transition(
            combined_energy, raw_sec, prev_section
        )

        reconstructed_sections.append(Section(
            label=label,
            start_beat=start_beat,
            end_beat=end_beat,
            stem_gains=stem_gains,
            transition_in=transition_in,
            transition_beats=transition_beats,
        ))

    # Build the RemixPlan
    plan = RemixPlan(
        vocal_source="song_a",
        tempo_source="average",
        key_source="none",
        start_time_vocal=0.0,
        end_time_vocal=duration,
        start_time_instrumental=0.0,
        end_time_instrumental=duration,
        sections=reconstructed_sections,
        explanation=f"Reconstructed from mashup: {title}",
        used_fallback=False,
    )

    logger.info(
        "Reconstructed plan for %s: %d sections, duration=%.1fs",
        mashup_id,
        len(reconstructed_sections),
        duration,
    )

    return plan


def _serialize_plan(plan: RemixPlan) -> dict:
    """Serialize a RemixPlan to a JSON-compatible dict."""
    return {
        "vocal_source": plan.vocal_source,
        "tempo_source": plan.tempo_source,
        "key_source": plan.key_source,
        "start_time_vocal": plan.start_time_vocal,
        "end_time_vocal": plan.end_time_vocal,
        "start_time_instrumental": plan.start_time_instrumental,
        "end_time_instrumental": plan.end_time_instrumental,
        "explanation": plan.explanation,
        "used_fallback": plan.used_fallback,
        "warnings": plan.warnings,
        "sections": [
            {
                "label": s.label,
                "start_beat": s.start_beat,
                "end_beat": s.end_beat,
                "stem_gains": s.stem_gains,
                "transition_in": s.transition_in,
                "transition_beats": s.transition_beats,
            }
            for s in plan.sections
        ],
    }


def reconstruct_all(analysis_dir: Path, output_dir: Path) -> list[Path]:
    """Batch reconstruction: convert all analysis JSONs to RemixPlan JSONs.

    Reads every .json file in analysis_dir, runs reconstruct_plan on each,
    and writes the resulting RemixPlan to output_dir as {id}.json.

    Args:
        analysis_dir: Directory containing analysis JSON files from Step 3.
        output_dir: Directory to write reconstructed plan JSON files.

    Returns:
        List of paths to successfully written plan files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_files = sorted(analysis_dir.glob("*.json"))
    if not analysis_files:
        logger.warning("No analysis files found in %s", analysis_dir)
        return []

    written: list[Path] = []
    for analysis_path in analysis_files:
        mashup_id = analysis_path.stem

        try:
            with open(analysis_path) as f:
                analysis = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load analysis %s: %s", analysis_path, exc)
            continue

        try:
            plan = reconstruct_plan(analysis, mashup_id)
        except (KeyError, ValueError) as exc:
            logger.error("Failed to reconstruct plan for %s: %s", mashup_id, exc)
            continue

        output_path = output_dir / f"{mashup_id}.json"
        serialized = _serialize_plan(plan)

        try:
            with open(output_path, "w") as f:
                json.dump(serialized, f, indent=2)
        except OSError as exc:
            logger.error("Failed to write plan %s: %s", output_path, exc)
            continue

        written.append(output_path)
        logger.info("Wrote plan: %s", output_path)

    logger.info(
        "Reconstruction complete: %d/%d successful",
        len(written),
        len(analysis_files),
    )
    return written
