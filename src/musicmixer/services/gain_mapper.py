"""Intent-to-gains mapper: converts musical intent (roles) to concrete stem gains.

Pure Python, no LLM calls. Transforms an IntentPlan (coarse roles like "lead",
"support", "background") into a RemixPlan with exact float gain values per stem.

Pipeline:
  A. Base role-to-gain mapping
  B. Energy scaling
  C. LUFS-informed adjustment
  D. Constraint enforcement (min active stems, muting budget, gain floors, etc.)
  E. Build RemixPlan from IntentPlan + computed gains
"""

from __future__ import annotations

import logging
import math

from musicmixer.models import (
    VOCAL_SOURCE,
    IntentPlan,
    IntentSection,
    RemixPlan,
    Section,
    STEM_ROLES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
INSTRUMENTAL_STEMS = ["drums", "bass", "guitar", "piano", "other"]

# Step A: base role-to-gain mapping
# NOTE: support/background/texture raised in energy-budget-tuning so that
# after LUFS attenuation (~0.88x at -18 LUFS neutral ref) the effective
# gains still land inside energy-budget target ranges.
ROLE_BASE_GAINS: dict[str, float] = {
    "lead":       0.92,
    "support":    0.82,
    "background": 0.55,
    "texture":    0.30,
    "silent":     0.0,
}

# Step B: energy multipliers
ENERGY_MULTIPLIERS: dict[str, float] = {
    "low":    0.75,
    "medium": 0.90,
    "high":   1.0,
    "peak":   1.05,
}

# Step D: gain floors per role
# Floors raised to sit below the new base gains but above the old ones.
ROLE_GAIN_FLOORS: dict[str, float] = {
    "support":    0.60,
    "background": 0.40,
    "texture":    0.20,
}

# Step D: energy budget targets (total gain sum per section)
ENERGY_BUDGET_TARGETS: dict[str, tuple[float, float]] = {
    "intro":     (1.5, 3.5),
    "outro":     (1.5, 3.5),
    "verse":     (3.0, 4.5),
    "bridge":    (3.0, 4.5),
    "chorus":    (4.0, 5.5),
    "drop":      (4.0, 5.5),
    "breakdown": (2.0, 3.5),
}

# Minimum active instrumental stems for non-intro/outro sections
MIN_ACTIVE_INSTRUMENTAL_STEMS = 3
MIN_ACTIVE_GAIN_THRESHOLD = 0.10

# Muting budget: max fraction of instrumental stem entries that can be 0.0
MAX_MUTING_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Step C: LUFS adjustment
# ---------------------------------------------------------------------------

def _lufs_adjustment(lufs: float) -> float:
    """LUFS-based gain correction. Reference: -18 LUFS = neutral.

    Neutral reference shifted from -20 to -18 (energy-budget-tuning) so
    that stems around -14 to -15 LUFS — common for drums and bass — receive
    gentler attenuation (~0.90x instead of ~0.80x).
    """
    if math.isnan(lufs) or lufs < -60:
        return 1.0  # no data, neutral
    if lufs < -28:
        return 1.3   # very quiet stem, boost
    if lufs < -23:
        return 1.15
    if lufs <= -18:
        return 1.0   # neutral reference
    if lufs < -13:
        return 0.90
    return 0.80       # very loud stem, attenuate


# ---------------------------------------------------------------------------
# Section-level conversion (Steps A-C + gain floors)
# ---------------------------------------------------------------------------

def _compute_section_gains(
    intent_section: IntentSection,
    vocal_stem_lufs: dict[str, float] | None,
    inst_stem_lufs: dict[str, float] | None,
) -> dict[str, float]:
    """Compute raw stem gains for a single section (Steps A-C + floors).

    Does NOT apply cross-section constraints (min active stems, muting budget).
    Those are handled in the plan-level pass.
    """
    energy_mult = ENERGY_MULTIPLIERS.get(intent_section.energy, 0.90)
    gains: dict[str, float] = {}

    for stem_name in ALL_STEMS:
        role = intent_section.stem_roles.get(stem_name, "silent")

        # Step A: base gain from role
        base_gain = ROLE_BASE_GAINS.get(role, 0.0)

        # Step B: energy scaling
        gain = base_gain * energy_mult

        # Step C: LUFS adjustment (skip silent stems)
        if role != "silent":
            lufs_data = None
            if stem_name == "vocals" and vocal_stem_lufs is not None:
                lufs_data = vocal_stem_lufs.get(stem_name)
            elif stem_name in INSTRUMENTAL_STEMS and inst_stem_lufs is not None:
                lufs_data = inst_stem_lufs.get(stem_name)

            if lufs_data is not None:
                gain *= _lufs_adjustment(lufs_data)

        # Clamp to [0.0, 1.0]
        gain = max(0.0, min(1.0, gain))

        # Apply gain floors for active roles
        floor = ROLE_GAIN_FLOORS.get(role, 0.0)
        if role != "silent" and gain < floor:
            gain = floor

        gains[stem_name] = gain

    return gains


# ---------------------------------------------------------------------------
# Step D: constraint enforcement (plan-level)
# ---------------------------------------------------------------------------

def _enforce_min_active_instrumentals(
    sections_gains: list[dict[str, float]],
    section_labels: list[str],
) -> None:
    """Constraint 1: non-intro/outro sections need >= 3 active instrumental stems."""
    for i, (gains, label) in enumerate(zip(sections_gains, section_labels)):
        if label in ("intro", "outro"):
            continue

        active_inst = [
            s for s in INSTRUMENTAL_STEMS
            if gains.get(s, 0.0) > MIN_ACTIVE_GAIN_THRESHOLD
        ]
        deficit = MIN_ACTIVE_INSTRUMENTAL_STEMS - len(active_inst)
        if deficit <= 0:
            continue

        # Find inactive instrumental stems sorted by current gain (ascending)
        inactive = sorted(
            [s for s in INSTRUMENTAL_STEMS if s not in active_inst],
            key=lambda s: gains.get(s, 0.0),
        )
        bumped = []
        for s in inactive[:deficit]:
            gains[s] = 0.30
            bumped.append(s)
        if bumped:
            logger.info(
                "Constraint: section %d (%s) — bumped %s to 0.30 "
                "(min active instrumentals)",
                i, label, bumped,
            )


def _enforce_no_globally_muted(
    sections_gains: list[dict[str, float]],
    section_energies: list[str],
) -> None:
    """Constraint 2: no instrumental stem can be 0.0 in ALL sections."""
    if not sections_gains:
        return

    for stem in INSTRUMENTAL_STEMS:
        all_zero = all(
            gains.get(stem, 0.0) == 0.0
            for gains in sections_gains
        )
        if not all_zero:
            continue

        # Find the highest-energy section
        energy_rank = {"peak": 4, "high": 3, "medium": 2, "low": 1}
        best_idx = max(
            range(len(sections_gains)),
            key=lambda idx: energy_rank.get(section_energies[idx], 0),
        )
        sections_gains[best_idx][stem] = 0.30
        logger.info(
            "Constraint: stem '%s' was globally muted — set to 0.30 in section %d",
            stem, best_idx,
        )


def _enforce_muting_budget(
    sections_gains: list[dict[str, float]],
    section_energies: list[str],
) -> None:
    """Constraint 3: no more than 15% of instrumental stem entries can be 0.0."""
    if not sections_gains:
        return

    total_entries = len(sections_gains) * len(INSTRUMENTAL_STEMS)
    if total_entries == 0:
        return

    # Collect all zero entries with their section index and stem name
    zero_entries: list[tuple[int, str]] = []
    for i, gains in enumerate(sections_gains):
        for stem in INSTRUMENTAL_STEMS:
            if gains.get(stem, 0.0) == 0.0:
                zero_entries.append((i, stem))

    max_zeros = int(total_entries * MAX_MUTING_FRACTION)
    excess = len(zero_entries) - max_zeros
    if excess <= 0:
        return

    # Sort: prefer converting zeros in highest-energy sections first
    energy_rank = {"peak": 4, "high": 3, "medium": 2, "low": 1}
    zero_entries.sort(
        key=lambda entry: energy_rank.get(section_energies[entry[0]], 0),
        reverse=True,
    )

    converted = []
    for section_idx, stem in zero_entries[:excess]:
        sections_gains[section_idx][stem] = 0.25
        converted.append(f"{stem}@section{section_idx}")

    if converted:
        logger.info(
            "Constraint: muting budget exceeded (%d/%d zeros, max %d) — "
            "converted %d entries to 0.25: %s",
            len(zero_entries), total_entries, max_zeros, len(converted), converted,
        )


def _validate_energy_budgets(
    sections_gains: list[dict[str, float]],
    section_labels: list[str],
) -> list[str]:
    """Constraint 4: auto-correct sections whose total gain is below target.

    When a section's total gain falls below the energy budget minimum, all
    non-silent stems in that section are scaled proportionally upward so the
    total hits the minimum.  Individual gains are capped at 1.0.

    Sections above the target range still produce warnings (no downward
    auto-correction — clipping is safer to flag than to silently fix).
    """
    warnings: list[str] = []
    for i, (gains, label) in enumerate(zip(sections_gains, section_labels)):
        total = sum(gains.values())
        targets = ENERGY_BUDGET_TARGETS.get(label)
        if targets is None:
            continue
        lo, hi = targets

        if total < lo:
            # --- auto-correct: scale non-silent gains up to hit lo ---
            non_silent = {s: g for s, g in gains.items() if g > 0.0}
            ns_total = sum(non_silent.values())
            if ns_total > 0:
                scale = lo / ns_total
                for stem in non_silent:
                    gains[stem] = min(1.0, gains[stem] * scale)
                new_total = sum(gains.values())
                logger.info(
                    "Energy budget: section %d (%s) adjusted total gain "
                    "from %.2f to %.2f (target min %.1f)",
                    i, label, total, new_total, lo,
                )
            else:
                # All stems silent — nothing to scale; emit warning
                msg = (
                    f"Section {i} ({label}): total gain {total:.2f} below "
                    f"target range [{lo:.1f}, {hi:.1f}] (all stems silent)"
                )
                logger.warning("Energy budget: %s", msg)
                warnings.append(msg)
        elif total > hi:
            msg = (
                f"Section {i} ({label}): total gain {total:.2f} above "
                f"target range [{lo:.1f}, {hi:.1f}]"
            )
            logger.warning("Energy budget: %s", msg)
            warnings.append(msg)
    return warnings


def _apply_constraints(
    sections_gains: list[dict[str, float]],
    intent_sections: list[IntentSection],
) -> list[str]:
    """Apply all plan-level constraints (Step D). Returns energy budget warnings."""
    labels = [s.label for s in intent_sections]
    energies = [s.energy for s in intent_sections]

    _enforce_min_active_instrumentals(sections_gains, labels)
    _enforce_no_globally_muted(sections_gains, energies)
    _enforce_muting_budget(sections_gains, energies)
    return _validate_energy_budgets(sections_gains, labels)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def intent_section_to_section(
    intent_section: IntentSection,
    vocal_lufs: dict[str, float] | None = None,
    inst_lufs: dict[str, float] | None = None,
) -> Section:
    """Convert a single IntentSection to a Section (gains computed, no plan-level constraints)."""
    gains = _compute_section_gains(intent_section, vocal_lufs, inst_lufs)
    return Section(
        label=intent_section.label,
        start_beat=intent_section.start_beat,
        end_beat=intent_section.end_beat,
        stem_gains=gains,
        transition_in=intent_section.transition_in,
        transition_beats=intent_section.transition_beats,
    )


def map_intent_to_gains(
    intent: IntentPlan,
    vocal_stem_lufs: dict[str, float] | None = None,
    inst_stem_lufs: dict[str, float] | None = None,
) -> RemixPlan:
    """Convert an IntentPlan to a RemixPlan with concrete gain values.

    Pipeline:
      A. Map stem roles to base gains
      B. Scale by energy level
      C. Adjust for LUFS loudness differences
      D. Enforce hard constraints (min active stems, muting budget, floors)
      E. Build RemixPlan

    Args:
        intent: The musical intent plan from the LLM.
        vocal_stem_lufs: Optional LUFS measurements for vocal-source stems.
            Keys are stem names (e.g. "vocals"), values are LUFS floats.
        inst_stem_lufs: Optional LUFS measurements for instrumental-source stems.
            Keys are stem names (e.g. "drums", "bass"), values are LUFS floats.

    Returns:
        A complete RemixPlan with float gains per stem per section.
    """
    # Steps A-C: compute per-section gains
    sections_gains: list[dict[str, float]] = []
    for intent_section in intent.sections:
        gains = _compute_section_gains(intent_section, vocal_stem_lufs, inst_stem_lufs)
        sections_gains.append(gains)

    # Step D: plan-level constraints
    energy_warnings = _apply_constraints(sections_gains, intent.sections)

    # Step E: build Section objects
    sections: list[Section] = []
    for intent_section, gains in zip(intent.sections, sections_gains):
        sections.append(Section(
            label=intent_section.label,
            start_beat=intent_section.start_beat,
            end_beat=intent_section.end_beat,
            stem_gains=gains,
            transition_in=intent_section.transition_in,
            transition_beats=intent_section.transition_beats,
        ))

    # Combine warnings
    all_warnings = list(intent.warnings) + energy_warnings

    return RemixPlan(
        vocal_source=VOCAL_SOURCE,
        start_time_vocal=intent.start_time_vocal,
        end_time_vocal=intent.end_time_vocal,
        start_time_instrumental=intent.start_time_instrumental,
        end_time_instrumental=intent.end_time_instrumental,
        sections=sections,
        tempo_source="weighted_midpoint",
        key_source=intent.key_source,
        explanation=intent.explanation,
        warnings=all_warnings,
    )
