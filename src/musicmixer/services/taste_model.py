"""Heuristic taste scorer and CatBoost model stub for remix plan ranking.

Scores candidate RemixPlan objects using a weighted heuristic rubric across
7 dimensions (arrangement, energy arc, vocal intelligibility, harmonic fit,
transition quality, groove coherence, loudness/fatigue). The heuristic scorer
works standalone without any ML dependencies -- CatBoost is a future
enhancement loaded optionally.

Phase 0-1: Heuristic scorer is the primary scoring mechanism.
Phase 2+:  CatBoost pairwise ranker takes over, heuristic becomes fallback.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from musicmixer.models import AudioMetadata, RemixPlan, Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dimension weights from plan section 3.3
DIMENSION_WEIGHTS: dict[str, float] = {
    "arrangement_quality": 0.15,
    "energy_arc": 0.15,
    "vocal_intelligibility": 0.15,
    "harmonic_fit": 0.15,
    "transition_quality": 0.15,
    "groove_coherence": 0.15,
    "loudness_fatigue": 0.10,
}

# Tie-breaking threshold -- when top-2 margin is below this, prefer
# the more conservative candidate (less stretch, fewer transitions, lower gains).
TIE_THRESHOLD = 0.05

# Ideal section count range
IDEAL_SECTION_COUNT_MIN = 4
IDEAL_SECTION_COUNT_MAX = 7

# Minimum section length in beats
MIN_SECTION_BEATS = 4
STANDARD_MIN_SECTION_BEATS = 8

# Peak energy should land between 55-80% of the timeline
PEAK_POSITION_MIN = 0.55
PEAK_POSITION_MAX = 0.80

# Labels that typically have lower energy
LOW_ENERGY_LABELS = {"intro", "outro", "breakdown"}
HIGH_ENERGY_LABELS = {"main", "build", "peak", "drop", "chorus"}

# All known stems
ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]


# ---------------------------------------------------------------------------
# Scored candidate dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """A candidate plan with its score breakdown."""

    plan: RemixPlan
    total_score: float  # 0.0 to 1.0
    dimension_scores: dict[str, float]  # dimension_name -> score
    rank: int = 0  # Set after sorting


# ---------------------------------------------------------------------------
# Individual dimension scorers
# ---------------------------------------------------------------------------

def _score_arrangement_quality(plan: RemixPlan) -> float:
    """Score arrangement quality: section pacing, phrasing, stem density.

    Higher score for:
    - Section count in 4-7 range
    - Standard section proportions (no tiny or enormous sections)
    - Phrase alignment (boundaries on 4-beat multiples)
    - Reasonable stem density (not all maxed, not all muted)
    """
    sections = plan.sections
    if not sections:
        return 0.0

    score = 0.0
    n_sections = len(sections)

    # Section count: 4-7 is ideal (0.3 weight within dimension)
    if IDEAL_SECTION_COUNT_MIN <= n_sections <= IDEAL_SECTION_COUNT_MAX:
        count_score = 1.0
    elif n_sections == 3 or n_sections == 8:
        count_score = 0.6
    elif n_sections == 2 or n_sections == 9:
        count_score = 0.3
    else:
        count_score = 0.1
    score += count_score * 0.3

    # Section proportions: penalize very short or very long sections (0.3 weight)
    total_beats = sections[-1].end_beat - sections[0].start_beat
    if total_beats <= 0:
        return score

    proportion_scores = []
    for sec in sections:
        sec_beats = sec.end_beat - sec.start_beat
        if sec_beats < MIN_SECTION_BEATS:
            proportion_scores.append(0.2)
        elif sec_beats < STANDARD_MIN_SECTION_BEATS:
            proportion_scores.append(0.6)
        else:
            # Check proportion -- no single section > 50% or < 5% of total
            proportion = sec_beats / total_beats
            if 0.05 <= proportion <= 0.50:
                proportion_scores.append(1.0)
            elif proportion > 0.50:
                proportion_scores.append(0.4)
            else:
                proportion_scores.append(0.5)

    score += (sum(proportion_scores) / len(proportion_scores)) * 0.3

    # Phrase alignment: boundaries on 4-beat multiples (0.2 weight)
    aligned = 0
    total_boundaries = 0
    for sec in sections:
        total_boundaries += 1
        if sec.start_beat % 4 == 0:
            aligned += 1
    if total_boundaries > 0:
        score += (aligned / total_boundaries) * 0.2

    # Stem density variety: not all maxed, not all muted (0.2 weight)
    density_scores = []
    for sec in sections:
        gains = list(sec.stem_gains.values())
        if not gains:
            density_scores.append(0.3)
            continue
        active_count = sum(1 for g in gains if g > 0.1)
        total_count = len(gains)
        # Best density: 2-5 active stems out of 6
        if total_count > 0 and 2 <= active_count <= 5:
            density_scores.append(1.0)
        elif active_count == 1 or active_count == total_count:
            density_scores.append(0.5)
        else:
            density_scores.append(0.3)
    if density_scores:
        score += (sum(density_scores) / len(density_scores)) * 0.2

    return min(max(score, 0.0), 1.0)


def _score_energy_arc(plan: RemixPlan) -> float:
    """Score energy arc: builds/releases feel intentional.

    Higher score when:
    - Clear build-peak-release shape
    - Peak in 55-80% range of timeline
    - Not flat (some variation) and not chaotic (no wild jumps)
    """
    sections = plan.sections
    if not sections:
        return 0.0

    total_beats = sections[-1].end_beat - sections[0].start_beat
    if total_beats <= 0:
        return 0.0

    score = 0.0

    # Compute per-section energy proxy from stem gains
    energies = []
    for sec in sections:
        gains = list(sec.stem_gains.values())
        # Energy = sum of squared gains (louder and more stems = more energy)
        energy = sum(g ** 2 for g in gains) / max(len(gains), 1)
        midpoint = (sec.start_beat + sec.end_beat) / 2
        position = midpoint / total_beats
        energies.append((position, energy))

    if not energies:
        return 0.0

    # Find peak energy position (0.3 weight)
    max_energy_idx = max(range(len(energies)), key=lambda i: energies[i][1])
    peak_position = energies[max_energy_idx][0]
    if PEAK_POSITION_MIN <= peak_position <= PEAK_POSITION_MAX:
        score += 1.0 * 0.3
    elif 0.40 <= peak_position <= 0.90:
        score += 0.5 * 0.3
    else:
        score += 0.1 * 0.3

    # Check for build-peak-release shape (0.4 weight)
    # Energy should generally increase before peak and decrease after
    pre_peak_rising = 0
    post_peak_falling = 0
    pre_count = 0
    post_count = 0

    for i in range(1, len(energies)):
        if i <= max_energy_idx:
            pre_count += 1
            if energies[i][1] >= energies[i - 1][1] - 0.05:
                pre_peak_rising += 1
        else:
            post_count += 1
            if energies[i][1] <= energies[i - 1][1] + 0.05:
                post_peak_falling += 1

    shape_score = 0.0
    if pre_count > 0:
        shape_score += (pre_peak_rising / pre_count) * 0.5
    else:
        shape_score += 0.25  # Peak is first section, partial credit
    if post_count > 0:
        shape_score += (post_peak_falling / post_count) * 0.5
    else:
        shape_score += 0.25  # Peak is last section, partial credit
    score += shape_score * 0.4

    # Check for variation (not flat) (0.3 weight)
    energy_values = [e[1] for e in energies]
    if len(energy_values) >= 2:
        energy_range = max(energy_values) - min(energy_values)
        if energy_range > 0.3:
            score += 1.0 * 0.3  # Good dynamic range
        elif energy_range > 0.15:
            score += 0.7 * 0.3
        elif energy_range > 0.05:
            score += 0.4 * 0.3
        else:
            score += 0.1 * 0.3  # Very flat
    else:
        score += 0.1 * 0.3

    return min(max(score, 0.0), 1.0)


def _score_vocal_intelligibility(plan: RemixPlan) -> float:
    """Score vocal intelligibility: vocal placement fit.

    Higher score when:
    - Vocals are muted in intro/outro
    - Vocals active in middle sections
    - Not competing with too many simultaneous instruments
    """
    sections = plan.sections
    if not sections:
        return 0.0

    score = 0.0

    # Check vocal muting in intro/outro (0.4 weight)
    intro_outro_scores = []
    for sec in sections:
        vocal_gain = sec.stem_gains.get("vocals", 0.0)
        if sec.label in ("intro", "outro"):
            # Vocals should be low/muted in intro/outro
            if vocal_gain <= 0.1:
                intro_outro_scores.append(1.0)
            elif vocal_gain <= 0.3:
                intro_outro_scores.append(0.7)
            elif vocal_gain <= 0.5:
                intro_outro_scores.append(0.4)
            else:
                intro_outro_scores.append(0.1)

    if intro_outro_scores:
        score += (sum(intro_outro_scores) / len(intro_outro_scores)) * 0.4
    else:
        # No intro/outro sections -- partial credit
        score += 0.5 * 0.4

    # Check vocals active in middle sections (0.3 weight)
    middle_scores = []
    for sec in sections:
        vocal_gain = sec.stem_gains.get("vocals", 0.0)
        if sec.label not in ("intro", "outro"):
            if vocal_gain >= 0.7:
                middle_scores.append(1.0)
            elif vocal_gain >= 0.4:
                middle_scores.append(0.7)
            elif vocal_gain > 0.1:
                middle_scores.append(0.4)
            else:
                middle_scores.append(0.2)

    if middle_scores:
        score += (sum(middle_scores) / len(middle_scores)) * 0.3
    else:
        score += 0.3 * 0.3

    # Check vocal masking risk from competing stems (0.3 weight)
    masking_scores = []
    for sec in sections:
        vocal_gain = sec.stem_gains.get("vocals", 0.0)
        if vocal_gain > 0.1:
            # Count competing stems with significant gain
            competing = sum(
                1 for stem, gain in sec.stem_gains.items()
                if stem != "vocals" and gain > 0.5
            )
            if competing <= 2:
                masking_scores.append(1.0)
            elif competing <= 3:
                masking_scores.append(0.7)
            elif competing <= 4:
                masking_scores.append(0.4)
            else:
                masking_scores.append(0.2)

    if masking_scores:
        score += (sum(masking_scores) / len(masking_scores)) * 0.3
    else:
        score += 0.5 * 0.3

    return min(max(score, 0.0), 1.0)


def _score_harmonic_fit(
    plan: RemixPlan,
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
) -> float:
    """Score harmonic fit: key distance, pitch shift amount.

    Higher score for key_source != 'none'. When audio metadata with key info
    is available, score based on Camelot distance. Falls back to plan-only
    scoring when metadata is unavailable.
    """
    score = 0.0

    # Key source decision quality (0.4 weight)
    if plan.key_source != "none":
        score += 1.0 * 0.4
    else:
        score += 0.3 * 0.4

    # If we have key metadata, score Camelot distance (0.6 weight)
    if meta_a and meta_b and meta_a.key and meta_b.key:
        distance = _camelot_distance(meta_a.key, meta_a.scale, meta_b.key, meta_b.scale)
        if distance <= 1:
            score += 1.0 * 0.6  # Same or adjacent key
        elif distance <= 2:
            score += 0.7 * 0.6
        elif distance <= 3:
            score += 0.4 * 0.6
        else:
            score += 0.1 * 0.6  # Very distant keys
    else:
        # No key data -- give moderate credit for having a key source
        if plan.key_source != "none":
            score += 0.7 * 0.6
        else:
            score += 0.4 * 0.6

    return min(max(score, 0.0), 1.0)


# Camelot wheel mapping: key -> (number, letter)
# Standard Camelot notation: 1A-12A (minor), 1B-12B (major)
_CAMELOT_MAP: dict[str, tuple[int, str]] = {
    # Major keys -> B column
    "C": (8, "B"), "Db": (3, "B"), "D": (10, "B"), "Eb": (5, "B"),
    "E": (12, "B"), "F": (7, "B"), "F#": (2, "B"), "Gb": (2, "B"),
    "G": (9, "B"), "Ab": (4, "B"), "A": (11, "B"), "Bb": (6, "B"),
    "B": (1, "B"),
    # Minor keys -> A column
    "Cm": (5, "A"), "C#m": (12, "A"), "Dbm": (12, "A"), "Dm": (7, "A"),
    "D#m": (2, "A"), "Ebm": (2, "A"), "Em": (9, "A"), "Fm": (4, "A"),
    "F#m": (11, "A"), "Gbm": (11, "A"), "Gm": (6, "A"),
    "G#m": (1, "A"), "Abm": (1, "A"), "Am": (8, "A"),
    "A#m": (3, "A"), "Bbm": (3, "A"), "Bm": (10, "A"),
}


def _camelot_distance(
    key_a: str,
    scale_a: str | None,
    key_b: str,
    scale_b: str | None,
) -> int:
    """Compute Camelot wheel distance between two keys.

    Returns the minimum number of steps on the Camelot wheel.
    Lower distance = better harmonic compatibility.
    """
    # Build full key name (e.g. "Am", "C", "Ebm")
    name_a = key_a + ("m" if scale_a and "min" in scale_a.lower() else "")
    name_b = key_b + ("m" if scale_b and "min" in scale_b.lower() else "")

    cam_a = _CAMELOT_MAP.get(name_a)
    cam_b = _CAMELOT_MAP.get(name_b)

    if cam_a is None or cam_b is None:
        return 4  # Unknown key -- assume poor fit

    num_a, letter_a = cam_a
    num_b, letter_b = cam_b

    # Distance on the number wheel (circular, 1-12)
    num_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))

    # Same column (A-A or B-B): distance is just the number distance
    if letter_a == letter_b:
        return num_dist

    # Different columns at same number: distance is 1 (relative major/minor)
    if num_a == num_b:
        return 1

    # Different columns and different numbers: num_dist + 1
    return num_dist + 1


def _score_transition_quality(plan: RemixPlan) -> float:
    """Score transition quality: variety, appropriate lengths.

    Higher score for:
    - Variety in transition types (not all the same)
    - Reasonable transition lengths (not too long)
    - At least some crossfades (smoother)
    """
    sections = plan.sections
    if not sections:
        return 0.0

    score = 0.0

    # Collect transition types and lengths
    transition_types = [sec.transition_in for sec in sections]
    transition_lengths = [sec.transition_beats for sec in sections]

    # Transition type variety (0.5 weight)
    unique_types = len(set(transition_types))
    total_types = len(transition_types)
    if total_types <= 1:
        variety_score = 0.5  # Single section, can't judge variety
    elif unique_types >= 3:
        variety_score = 1.0
    elif unique_types == 2:
        variety_score = 0.7
    else:
        variety_score = 0.3  # All same type
    score += variety_score * 0.5

    # Transition length appropriateness (0.3 weight)
    length_scores = []
    for i, sec in enumerate(sections):
        sec_beats = sec.end_beat - sec.start_beat
        trans_beats = sec.transition_beats
        if sec_beats <= 0:
            length_scores.append(0.3)
            continue

        # Transition should not exceed half the section length
        if trans_beats > sec_beats / 2:
            length_scores.append(0.1)
        elif 2 <= trans_beats <= 8:
            length_scores.append(1.0)
        elif trans_beats == 0:
            # Zero-length transition (hard cut) -- acceptable but not ideal
            length_scores.append(0.6)
        else:
            length_scores.append(0.5)

    if length_scores:
        score += (sum(length_scores) / len(length_scores)) * 0.3
    else:
        score += 0.5 * 0.3

    # Crossfade presence (0.2 weight) -- crossfades are generally smoother
    has_crossfade = any(t == "crossfade" for t in transition_types)
    has_fade = any(t == "fade" for t in transition_types)
    if has_crossfade:
        score += 1.0 * 0.2
    elif has_fade:
        score += 0.7 * 0.2
    else:
        score += 0.3 * 0.2

    return min(max(score, 0.0), 1.0)


def _score_groove_coherence(plan: RemixPlan) -> float:
    """Score groove coherence: beat alignment, grid compliance.

    Higher score for:
    - Section boundaries on 4-beat multiples
    - Consistent beat grid alignment
    - Section lengths that are multiples of 4
    """
    sections = plan.sections
    if not sections:
        return 0.0

    score = 0.0

    # Beat grid alignment: boundaries on 4-beat multiples (0.5 weight)
    aligned = 0
    total_boundaries = 0
    for sec in sections:
        if sec.start_beat % 4 == 0:
            aligned += 1
        total_boundaries += 1
        if sec.end_beat % 4 == 0:
            aligned += 1
        total_boundaries += 1

    if total_boundaries > 0:
        score += (aligned / total_boundaries) * 0.5

    # Section lengths as multiples of 4 (0.3 weight)
    length_scores = []
    for sec in sections:
        sec_beats = sec.end_beat - sec.start_beat
        if sec_beats <= 0:
            length_scores.append(0.0)
        elif sec_beats % 4 == 0:
            length_scores.append(1.0)
        elif sec_beats % 2 == 0:
            length_scores.append(0.6)
        else:
            length_scores.append(0.2)  # Odd beat count -- poor groove

    if length_scores:
        score += (sum(length_scores) / len(length_scores)) * 0.3

    # Transition beat alignment (0.2 weight)
    trans_scores = []
    for sec in sections:
        if sec.transition_beats % 2 == 0:
            trans_scores.append(1.0)
        else:
            trans_scores.append(0.4)

    if trans_scores:
        score += (sum(trans_scores) / len(trans_scores)) * 0.2

    return min(max(score, 0.0), 1.0)


def _score_loudness_fatigue(plan: RemixPlan) -> float:
    """Score loudness/fatigue: gain headroom, not all maxed.

    Higher score when:
    - Gains are moderate (not all 1.0)
    - Some dynamic variation across sections
    - Total gain headroom is reasonable
    """
    sections = plan.sections
    if not sections:
        return 0.0

    score = 0.0

    # Per-section gain moderation (0.5 weight)
    moderation_scores = []
    for sec in sections:
        gains = list(sec.stem_gains.values())
        if not gains:
            moderation_scores.append(0.5)
            continue

        avg_gain = sum(gains) / len(gains)
        max_gain = max(gains)
        active_gains = [g for g in gains if g > 0.1]

        # Penalize when everything is maxed
        if max_gain >= 1.0 and avg_gain > 0.8:
            moderation_scores.append(0.3)
        elif avg_gain > 0.9:
            moderation_scores.append(0.4)
        elif 0.3 <= avg_gain <= 0.8:
            moderation_scores.append(1.0)
        elif avg_gain < 0.2:
            moderation_scores.append(0.4)  # Too quiet
        else:
            moderation_scores.append(0.7)

    if moderation_scores:
        score += (sum(moderation_scores) / len(moderation_scores)) * 0.5

    # Dynamic variation: gain range across sections (0.3 weight)
    all_section_energies = []
    for sec in sections:
        gains = list(sec.stem_gains.values())
        if gains:
            all_section_energies.append(sum(gains) / len(gains))

    if len(all_section_energies) >= 2:
        energy_range = max(all_section_energies) - min(all_section_energies)
        if energy_range > 0.2:
            score += 1.0 * 0.3
        elif energy_range > 0.1:
            score += 0.7 * 0.3
        elif energy_range > 0.05:
            score += 0.4 * 0.3
        else:
            score += 0.1 * 0.3  # Very flat dynamics
    else:
        score += 0.3 * 0.3

    # Mute ratio: at least some stems muted somewhere (headroom) (0.2 weight)
    has_muted = False
    for sec in sections:
        if any(g <= 0.1 for g in sec.stem_gains.values()):
            has_muted = True
            break
    score += (1.0 if has_muted else 0.3) * 0.2

    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Main scoring functions
# ---------------------------------------------------------------------------

def score_candidate(
    plan: RemixPlan,
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
) -> ScoredCandidate:
    """Score a single candidate plan using the heuristic rubric.

    Computes scores across 7 dimensions and produces a weighted total.
    Works without audio metadata (plan-only scoring) when meta_a/meta_b
    are None.
    """
    dimension_scores: dict[str, float] = {
        "arrangement_quality": _score_arrangement_quality(plan),
        "energy_arc": _score_energy_arc(plan),
        "vocal_intelligibility": _score_vocal_intelligibility(plan),
        "harmonic_fit": _score_harmonic_fit(plan, meta_a, meta_b),
        "transition_quality": _score_transition_quality(plan),
        "groove_coherence": _score_groove_coherence(plan),
        "loudness_fatigue": _score_loudness_fatigue(plan),
    }

    total_score = sum(
        dimension_scores[dim] * weight
        for dim, weight in DIMENSION_WEIGHTS.items()
    )

    return ScoredCandidate(
        plan=plan,
        total_score=min(max(total_score, 0.0), 1.0),
        dimension_scores=dimension_scores,
    )


def _conservatism_score(plan: RemixPlan) -> float:
    """Compute a conservatism metric for tie-breaking.

    Lower score = more conservative plan. Prefers:
    - Less tempo stretch
    - Fewer transitions
    - More moderate gains
    """
    risk = 0.0

    # Number of sections (more = more transitions = more risk)
    risk += len(plan.sections) * 0.1

    # Average gain level (higher = less headroom)
    all_gains = []
    for sec in plan.sections:
        all_gains.extend(sec.stem_gains.values())
    if all_gains:
        risk += (sum(all_gains) / len(all_gains)) * 0.5

    # Total transition beats
    total_trans = sum(sec.transition_beats for sec in plan.sections)
    risk += total_trans * 0.01

    return risk


def select_best(
    candidates: list[RemixPlan],
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
) -> tuple[RemixPlan, list[ScoredCandidate]]:
    """Score all candidates, return (best_plan, all_scored_sorted).

    Selection policy: pick top-1. If margin between top-1 and top-2
    is less than TIE_THRESHOLD (0.05), choose the candidate with
    fewer arrangement risks (lower stretch amount, fewer transitions,
    more conservative gains).

    Raises ValueError if candidates list is empty.
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidates list")

    # Score all candidates
    scored = [
        score_candidate(plan, meta_a, meta_b)
        for plan in candidates
    ]

    # Sort by total_score descending
    scored.sort(key=lambda sc: sc.total_score, reverse=True)

    # Assign ranks (1-based)
    for i, sc in enumerate(scored):
        sc.rank = i + 1

    # Single candidate -- return it
    if len(scored) == 1:
        return scored[0].plan, scored

    # Check tie-breaking
    top1 = scored[0]
    top2 = scored[1]
    margin = top1.total_score - top2.total_score

    if margin < TIE_THRESHOLD:
        # Tie-break: prefer the more conservative candidate
        risk1 = _conservatism_score(top1.plan)
        risk2 = _conservatism_score(top2.plan)
        if risk2 < risk1:
            # top2 is more conservative -- swap
            logger.info(
                "Tie-break: margin %.3f < %.3f, preferring more conservative "
                "candidate (risk %.2f vs %.2f)",
                margin, TIE_THRESHOLD, risk2, risk1,
            )
            scored[0], scored[1] = scored[1], scored[0]
            scored[0].rank = 1
            scored[1].rank = 2

    best = scored[0]
    logger.info(
        "Selected candidate rank=%d score=%.3f (dimensions: %s)",
        best.rank,
        best.total_score,
        {k: f"{v:.2f}" for k, v in best.dimension_scores.items()},
    )

    return best.plan, scored


# ---------------------------------------------------------------------------
# CatBoost model stub
# ---------------------------------------------------------------------------

_catboost_model: object | None = None


def load_model(model_path: str | None = None) -> bool:
    """Attempt to load a CatBoost model. Returns True if successful.

    If catboost is not installed or model_path is None, returns False.
    The heuristic scorer remains the fallback.
    """
    global _catboost_model

    if model_path is None:
        logger.debug("No model path provided, using heuristic scorer")
        return False

    try:
        import catboost  # type: ignore[import-untyped]
        model = catboost.CatBoost()
        model.load_model(model_path)
        _catboost_model = model
        logger.info("Loaded CatBoost model from %s", model_path)
        return True
    except ImportError:
        logger.debug("catboost not installed, using heuristic scorer")
        return False
    except Exception:
        logger.warning("Failed to load CatBoost model from %s", model_path, exc_info=True)
        return False


def score_with_model(
    feature_vectors: list[dict[str, float]],
) -> list[float] | None:
    """Score candidates with CatBoost if model is loaded.

    Returns None if no model is loaded (caller should use heuristic).
    """
    if _catboost_model is None:
        return None

    try:
        import catboost  # type: ignore[import-untyped]
        import numpy as np

        # Convert feature dicts to matrix
        if not feature_vectors:
            return []

        feature_names = sorted(feature_vectors[0].keys())
        matrix = np.array([
            [fv[name] for name in feature_names]
            for fv in feature_vectors
        ])

        pool = catboost.Pool(data=matrix, feature_names=feature_names)
        predictions = _catboost_model.predict(pool)
        return list(predictions)
    except Exception:
        logger.warning("CatBoost prediction failed, falling back to heuristic", exc_info=True)
        return None
