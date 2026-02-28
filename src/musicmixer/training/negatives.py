"""Synthetic negative example generation for mashup training.

Given a positive (mashup-derived) RemixPlan, generates degraded versions
that break specific structural qualities. Each negative applies exactly
one degradation strategy so CatBoost can learn individual feature
associations cleanly.

Six strategies target different feature groups:
  - shuffle_sections: breaks energy arc
  - flat_energy: removes dynamic range
  - vocals_in_bookends: inverts vocal placement
  - off_grid_boundaries: breaks groove coherence
  - wrong_peak_placement: misplaces climax
  - no_contrast: eliminates tension/release
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, replace
from typing import Callable

from musicmixer.models import RemixPlan, Section


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

StrategyFn = Callable[[RemixPlan, random.Random], RemixPlan]

_STRATEGIES: dict[str, StrategyFn] = {}


def _register(name: str):
    """Decorator to register a degradation strategy by name."""
    def decorator(fn: StrategyFn) -> StrategyFn:
        _STRATEGIES[name] = fn
        return fn
    return decorator


def get_strategy_names() -> list[str]:
    """Return all registered strategy names in insertion order."""
    return list(_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _deep_copy_plan(plan: RemixPlan) -> RemixPlan:
    """Deep-copy a RemixPlan, including all Section objects and their dicts."""
    new_sections = []
    for s in plan.sections:
        new_sections.append(Section(
            label=s.label,
            start_beat=s.start_beat,
            end_beat=s.end_beat,
            stem_gains=dict(s.stem_gains),
            transition_in=s.transition_in,
            transition_beats=s.transition_beats,
        ))
    return RemixPlan(
        vocal_source=plan.vocal_source,
        start_time_vocal=plan.start_time_vocal,
        end_time_vocal=plan.end_time_vocal,
        start_time_instrumental=plan.start_time_instrumental,
        end_time_instrumental=plan.end_time_instrumental,
        sections=new_sections,
        tempo_source=plan.tempo_source,
        key_source=plan.key_source,
        explanation=plan.explanation,
        warnings=list(plan.warnings),
        used_fallback=plan.used_fallback,
    )


def _section_energy(section: Section) -> float:
    """Compute total energy of a section as the sum of its stem gains."""
    return sum(section.stem_gains.values())


def _vocal_gain(section: Section) -> float:
    """Return the vocal gain for a section (sum of all vocal-related stems)."""
    total = 0.0
    for stem_name, gain in section.stem_gains.items():
        if "vocal" in stem_name.lower():
            total += gain
    return total


@_register("shuffle_sections")
def shuffle_sections(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Randomly reorder sections, recomputing start_beat/end_beat.

    Preserves each section's duration and internal properties but
    destroys the energy arc by putting sections in a random order.
    """
    degraded = _deep_copy_plan(plan)
    sections = list(degraded.sections)

    if len(sections) <= 1:
        return degraded

    # Compute durations before shuffling
    durations = [s.end_beat - s.start_beat for s in sections]

    # Shuffle until the order actually changes (avoid no-op)
    original_labels = [s.label for s in sections]
    combined = list(zip(sections, durations))
    for _ in range(20):
        rng.shuffle(combined)
        if [s.label for s, _ in combined] != original_labels:
            break

    # Recompute contiguous beat boundaries
    current_beat = 0
    new_sections = []
    for section, duration in combined:
        section.start_beat = current_beat
        section.end_beat = current_beat + duration
        current_beat += duration
        new_sections.append(section)

    degraded.sections = new_sections
    degraded.explanation = "NEGATIVE [shuffle_sections]: Sections randomly reordered, breaking energy arc"
    degraded.used_fallback = True
    return degraded


@_register("flat_energy")
def flat_energy(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Set all stem_gains to a uniform value across all sections.

    Removes dynamic contrast by making every section sound the same.
    The uniform value is 0.5 for all stems present in each section.
    """
    degraded = _deep_copy_plan(plan)

    for section in degraded.sections:
        for stem_name in section.stem_gains:
            section.stem_gains[stem_name] = 0.5

    degraded.explanation = "NEGATIVE [flat_energy]: All stem gains set to uniform 0.5, removing dynamic range"
    degraded.used_fallback = True
    return degraded


@_register("vocals_in_bookends")
def vocals_in_bookends(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Move vocal-heavy sections to intro/outro, instrumentals to middle.

    This inverts the typical good arrangement where vocals are in the
    body and bookends are instrumental. Sections are reordered by vocal
    gain: highest-vocal sections go to the edges, lowest-vocal to center.
    Start/end beats are recomputed to maintain contiguous layout.
    """
    degraded = _deep_copy_plan(plan)
    sections = list(degraded.sections)

    if len(sections) <= 2:
        return degraded

    # Sort by vocal gain descending
    durations = [s.end_beat - s.start_beat for s in sections]
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda x: _vocal_gain(x[1]), reverse=True)

    # Place highest-vocal sections at edges (first and last positions),
    # lowest-vocal in the center. Build a new ordering where vocal-heavy
    # sections alternate between front and back.
    n = len(sections)
    new_order: list[Section | None] = [None] * n
    front = 0
    back = n - 1
    use_front = True

    reordered_sections = []
    reordered_durations = []
    for orig_idx, section in indexed:
        reordered_sections.append(section)
        reordered_durations.append(durations[orig_idx])

    # Place them alternating front/back
    placed: list[tuple[Section, int]] = [None] * n  # type: ignore[assignment]
    front_idx = 0
    back_idx = n - 1
    for i, (sec, dur) in enumerate(zip(reordered_sections, reordered_durations)):
        if i % 2 == 0:
            placed[front_idx] = (sec, dur)
            front_idx += 1
        else:
            placed[back_idx] = (sec, dur)
            back_idx -= 1

    # Recompute beat boundaries
    current_beat = 0
    new_sections = []
    for section, duration in placed:
        section.start_beat = current_beat
        section.end_beat = current_beat + duration
        current_beat += duration
        new_sections.append(section)

    # Verify the reorder actually changed something
    original_labels = [s.label for s in plan.sections]
    new_labels = [s.label for s in new_sections]
    if original_labels == new_labels:
        # Force a swap if the order didn't change
        if len(new_sections) >= 3:
            # Swap first and second sections
            s0_dur = new_sections[0].end_beat - new_sections[0].start_beat
            s1_dur = new_sections[1].end_beat - new_sections[1].start_beat
            new_sections[0], new_sections[1] = new_sections[1], new_sections[0]
            # Recompute beats
            current_beat = 0
            for sec in new_sections:
                dur = s1_dur if sec is new_sections[0] else s0_dur if sec is new_sections[1] else (sec.end_beat - sec.start_beat)
                sec.start_beat = current_beat
                sec.end_beat = current_beat + (sec.end_beat - sec.start_beat) if sec not in (new_sections[0], new_sections[1]) else current_beat + dur
                current_beat = sec.end_beat

    degraded.sections = new_sections
    degraded.explanation = "NEGATIVE [vocals_in_bookends]: Vocal-heavy sections moved to intro/outro positions"
    degraded.used_fallback = True
    return degraded


@_register("off_grid_boundaries")
def off_grid_boundaries(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Shift section boundaries by non-4-beat amounts.

    Breaks phrase alignment by adding +1 or +3 beat offsets to section
    boundaries. The first section always starts at 0 and sections
    remain contiguous, but internal boundaries land off the 4-beat grid.
    """
    degraded = _deep_copy_plan(plan)
    sections = degraded.sections

    if len(sections) <= 1:
        return degraded

    # For each internal boundary (between sections), shift by +1 or +3 beats.
    # This keeps sections contiguous but breaks the 4-beat phrase grid.
    offsets = [1, 3]
    current_beat = 0
    for i, section in enumerate(sections):
        duration = plan.sections[i].end_beat - plan.sections[i].start_beat
        section.start_beat = current_beat
        if i < len(sections) - 1:
            # Apply a non-4-aligned offset to this boundary
            offset = rng.choice(offsets)
            adjusted_duration = duration + offset
            # Ensure minimum section duration of 2 beats
            if adjusted_duration < 2:
                adjusted_duration = duration
            section.end_beat = current_beat + adjusted_duration
        else:
            # Last section: compute remaining to keep total consistent
            section.end_beat = current_beat + duration
        current_beat = section.end_beat

    degraded.explanation = "NEGATIVE [off_grid_boundaries]: Section boundaries shifted off 4-beat grid"
    degraded.used_fallback = True
    return degraded


@_register("wrong_peak_placement")
def wrong_peak_placement(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Move the highest-energy section to an early position.

    In a good arrangement, the peak energy section is near the middle or
    later (climax position). This strategy moves it to position 0 or 1,
    destroying the build-up and climax arc.
    """
    degraded = _deep_copy_plan(plan)
    sections = list(degraded.sections)

    if len(sections) <= 2:
        return degraded

    # Find the peak-energy section
    energies = [_section_energy(s) for s in sections]
    peak_idx = energies.index(max(energies))

    # If peak is already at position 0 or 1, nothing to degrade -- pick position 0 anyway
    target_pos = rng.choice([0, 1]) if len(sections) > 2 else 0

    if peak_idx == target_pos:
        # Peak is already early. Try the other early position.
        target_pos = 1 - target_pos if len(sections) > 2 else 0
        if peak_idx == target_pos:
            # Already in an early position -- still produce a valid negative
            # by swapping with the last non-outro section
            for j in range(len(sections) - 1, -1, -1):
                if j != peak_idx:
                    target_pos = j
                    break

    # Swap peak section with the target position
    durations = [s.end_beat - s.start_beat for s in sections]
    sections[peak_idx], sections[target_pos] = sections[target_pos], sections[peak_idx]
    durations[peak_idx], durations[target_pos] = durations[target_pos], durations[peak_idx]

    # Recompute contiguous beat boundaries
    current_beat = 0
    for section, duration in zip(sections, durations):
        section.start_beat = current_beat
        section.end_beat = current_beat + duration
        current_beat += duration

    degraded.sections = sections
    degraded.explanation = (
        f"NEGATIVE [wrong_peak_placement]: Peak energy section moved from "
        f"position {peak_idx} to position {target_pos}"
    )
    degraded.used_fallback = True
    return degraded


@_register("no_contrast")
def no_contrast(plan: RemixPlan, rng: random.Random) -> RemixPlan:
    """Average all stem_gains across sections.

    Every section gets the same gain profile (the mean of all sections),
    removing all tension/release dynamics.
    """
    degraded = _deep_copy_plan(plan)
    sections = degraded.sections

    if not sections:
        return degraded

    # Collect all stem names across all sections
    all_stems: set[str] = set()
    for section in sections:
        all_stems.update(section.stem_gains.keys())

    # Compute average gain per stem across all sections
    avg_gains: dict[str, float] = {}
    for stem_name in all_stems:
        values = [s.stem_gains.get(stem_name, 0.0) for s in sections]
        avg_gains[stem_name] = sum(values) / len(sections)

    # Apply the averaged gains to every section
    for section in sections:
        section.stem_gains = dict(avg_gains)

    degraded.explanation = "NEGATIVE [no_contrast]: All stem gains averaged across sections, removing tension/release"
    degraded.used_fallback = True
    return degraded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_negatives(
    plan: RemixPlan,
    n: int = 3,
    rng: random.Random | None = None,
) -> list[RemixPlan]:
    """Generate n degraded versions of a positive plan.

    For each negative, a distinct degradation strategy is randomly selected
    from the 6 available strategies. Each negative applies exactly ONE
    strategy (no stacking) to keep the training signal clean.

    Args:
        plan: A positive RemixPlan to degrade.
        n: Number of negatives to generate (default 3). Must be <= 6.
        rng: Optional Random instance for reproducibility.

    Returns:
        List of n degraded RemixPlan objects, each with used_fallback=True
        and explanation describing the applied degradation.

    Raises:
        ValueError: If n > number of available strategies (6).
    """
    if rng is None:
        rng = random.Random()

    strategy_names = get_strategy_names()
    if n > len(strategy_names):
        raise ValueError(
            f"Cannot generate {n} negatives: only {len(strategy_names)} "
            f"strategies available ({strategy_names})"
        )

    # Select n distinct strategies
    selected = rng.sample(strategy_names, k=n)

    negatives = []
    for name in selected:
        strategy_fn = _STRATEGIES[name]
        degraded = strategy_fn(plan, rng)
        negatives.append(degraded)

    return negatives
