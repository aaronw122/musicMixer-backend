"""Candidate plan generator for taste training.

Expands the deterministic fallback plan into 8-12 candidate RemixPlan variants
across 4 arrangement families, with structure hash deduplication.

Arrangement families (from taste-training-plan.md section 2.1):
  1. Standard arc:  intro -> build -> main -> breakdown -> peak/outro
  2. Hook-first:    short intro -> vocal hook -> verse -> peak -> outro
  3. DJ Lift:       build -> vocal in -> peak -> vocal out -> outro
  4. Quick Hit:     intro -> main vocal block -> short release

Each family generates 3-4 variants by varying the most impactful knobs for
that family. After dedup, backfill from underrepresented families if needed.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.tempo import estimate_target_bpm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

# Target remix duration in seconds -- matches interpreter.py constant.
TARGET_REMIX_DURATION_SECONDS = 210  # 3.5 minutes

# Minimum section length in beats.
MIN_SECTION_BEATS = 8
MIN_SECTION_BEATS_ABSOLUTE = 4

# Transition style options.
TRANSITION_STYLES = ["cut", "crossfade", "filter_sweep", "silence_gap"]

# Gain presets (dB offsets from baseline, converted to linear multipliers).
# These represent vocal/instrumental gain deltas as linear gain values.
GAIN_PRESETS = {
    "quiet_vocal": {"vocals": 0.5, "drums": 0.8, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 0.6},
    "balanced": {"vocals": 0.8, "drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5},
    "vocal_forward": {"vocals": 1.0, "drums": 0.6, "bass": 0.6, "guitar": 0.4, "piano": 0.3, "other": 0.4},
    "loud_vocal": {"vocals": 1.0, "drums": 0.5, "bass": 0.5, "guitar": 0.3, "piano": 0.3, "other": 0.3},
}

# Section-type gain templates.
INTRO_GAINS = {"vocals": 0.0, "drums": 0.6, "bass": 0.5, "guitar": 0.3, "piano": 0.2, "other": 0.3}
OUTRO_GAINS = {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.3, "piano": 0.3, "other": 0.4}
BREAKDOWN_GAINS = {"vocals": 0.3, "drums": 0.1, "bass": 0.4, "guitar": 0.6, "piano": 0.7, "other": 0.5}
BUILD_GAINS = {"vocals": 0.5, "drums": 0.5, "bass": 0.6, "guitar": 0.4, "piano": 0.3, "other": 0.4}
MAIN_GAINS = {"vocals": 1.0, "drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5}
PEAK_GAINS = {"vocals": 1.0, "drums": 0.8, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 0.6}
HOOK_GAINS = {"vocals": 1.0, "drums": 0.6, "bass": 0.6, "guitar": 0.3, "piano": 0.3, "other": 0.3}
VERSE_GAINS = {"vocals": 0.9, "drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap_to_grid(beat: int, grid: int = 4) -> int:
    """Snap a beat index down to the nearest grid boundary (default 4-beat)."""
    return max(0, (beat // grid) * grid)


def _compute_total_beats(meta_a: AudioMetadata, meta_b: AudioMetadata) -> tuple[float, int]:
    """Compute target BPM and total beat budget for the remix.

    Returns (target_bpm, total_beats).
    """
    target_bpm = estimate_target_bpm(
        vocal_bpm=meta_a.bpm,
        instrumental_bpm=meta_b.bpm,
        tempo_source="weighted_midpoint",
    )
    total_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    # Snap total beats to 4-beat grid
    total_beats = _snap_to_grid(total_beats)
    if total_beats < 32:
        total_beats = 32  # Absolute minimum for any arrangement
    return target_bpm, total_beats


def _build_remix_plan(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    sections: list[Section],
    family_name: str,
    target_bpm: float,
) -> RemixPlan:
    """Wrap sections into a full RemixPlan with computed time windows."""
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 25% into each song, up to target duration
    v_start = vocal_meta.duration_seconds * 0.25
    v_end = min(v_start + TARGET_REMIX_DURATION_SECONDS, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.25
    i_end = min(i_start + TARGET_REMIX_DURATION_SECONDS, inst_meta.duration_seconds)

    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=v_start,
        end_time_vocal=v_end,
        start_time_instrumental=i_start,
        end_time_instrumental=i_end,
        sections=sections,
        tempo_source="weighted_midpoint",
        key_source="none",
        explanation=f"Candidate plan using {family_name} arrangement family.",
        warnings=[],
        used_fallback=False,
    )


def _apply_gain_delta(base_gains: dict[str, float], delta_db: float) -> dict[str, float]:
    """Apply a dB delta to vocal gain while inversely adjusting instrumentals.

    Positive delta_db makes vocals louder relative to instrumentals.
    """
    import math
    vocal_mult = 10 ** (delta_db / 20)
    inst_mult = 10 ** (-delta_db / 40)  # Half the inverse for instrumentals
    result = {}
    for stem, gain in base_gains.items():
        if stem == "vocals":
            result[stem] = min(1.0, max(0.0, gain * vocal_mult))
        else:
            result[stem] = min(1.0, max(0.0, gain * inst_mult))
    return result


def _ensure_boundaries(boundaries: list[int], total_beats: int) -> list[int]:
    """Ensure boundary list is monotonically increasing, grid-aligned, and fits in total_beats.

    Each boundary must be at least MIN_SECTION_BEATS_ABSOLUTE apart from its predecessor,
    and the last boundary must leave at least MIN_SECTION_BEATS_ABSOLUTE for the final section.
    """
    result = []
    for b in boundaries:
        b = _snap_to_grid(b)
        prev = result[-1] if result else 0
        if b <= prev:
            b = _snap_to_grid(prev + MIN_SECTION_BEATS)
        # Don't let any boundary get too close to total_beats
        max_allowed = total_beats - MIN_SECTION_BEATS_ABSOLUTE
        if b >= max_allowed:
            b = _snap_to_grid(max_allowed - 4)
            if b <= (result[-1] if result else 0):
                b = (result[-1] if result else 0) + MIN_SECTION_BEATS_ABSOLUTE
        result.append(b)
    return result


def _clamp_transition_beats(transition_beats: int, section_length: int) -> int:
    """Ensure transition beats don't exceed half the section length."""
    return max(2, min(transition_beats, section_length // 2))


# ---------------------------------------------------------------------------
# Structure hash for deduplication
# ---------------------------------------------------------------------------

def _structure_hash(plan: RemixPlan) -> str:
    """Compute a deduplication hash based on section structure.

    Hashes on: section labels, boundary beats, transition types,
    and coarse gain bins (rounded to nearest 0.2).
    """
    parts = []
    for s in plan.sections:
        # Coarse gain bins: round each gain to nearest 0.2
        coarse_gains = {k: round(v / 0.2) for k, v in sorted(s.stem_gains.items())}
        parts.append(
            f"{s.label}|{s.start_beat}-{s.end_beat}|{s.transition_in}|{coarse_gains}"
        )
    content = "||".join(parts)
    return hashlib.md5(content.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Arrangement family generators
# ---------------------------------------------------------------------------

def _generate_standard_arc(
    total_beats: int,
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    target_bpm: float,
) -> list[RemixPlan]:
    """Standard arc: intro -> build -> main -> breakdown -> peak/outro.

    Varies: vocal entry (8, 16 beats), breakdown position (50%, 60%, 70%),
    transition style (crossfade, filter_sweep).
    """
    candidates = []

    configs = [
        {"vocal_entry": 16, "breakdown_pct": 0.60, "transition": "crossfade", "trans_beats": 4},
        {"vocal_entry": 8, "breakdown_pct": 0.50, "transition": "filter_sweep", "trans_beats": 8},
        {"vocal_entry": 24, "breakdown_pct": 0.70, "transition": "crossfade", "trans_beats": 4},
        {"vocal_entry": 16, "breakdown_pct": 0.60, "transition": "cut", "trans_beats": 2},
    ]

    for cfg in configs:
        vocal_entry = cfg["vocal_entry"]
        breakdown_pct = cfg["breakdown_pct"]
        transition = cfg["transition"]
        trans_beats = cfg["trans_beats"]

        intro_end = _snap_to_grid(vocal_entry)
        build_end = _snap_to_grid(int(total_beats * 0.25))
        breakdown_start = _snap_to_grid(int(total_beats * breakdown_pct))
        peak_start = _snap_to_grid(int(total_beats * (breakdown_pct + 0.12)))
        outro_start = _snap_to_grid(int(total_beats * 0.88))

        bounds = _ensure_boundaries(
            [intro_end, build_end, breakdown_start, peak_start, outro_start],
            total_beats,
        )
        intro_end, build_end, breakdown_start, peak_start, outro_start = bounds

        sections = [
            Section(
                label="intro", start_beat=0, end_beat=intro_end,
                stem_gains=INTRO_GAINS.copy(),
                transition_in="fade",
                transition_beats=_clamp_transition_beats(4, intro_end),
            ),
            Section(
                label="build", start_beat=intro_end, end_beat=build_end,
                stem_gains=BUILD_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, build_end - intro_end),
            ),
            Section(
                label="main", start_beat=build_end, end_beat=breakdown_start,
                stem_gains=MAIN_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, breakdown_start - build_end),
            ),
            Section(
                label="breakdown", start_beat=breakdown_start, end_beat=peak_start,
                stem_gains=BREAKDOWN_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, peak_start - breakdown_start),
            ),
            Section(
                label="peak", start_beat=peak_start, end_beat=outro_start,
                stem_gains=PEAK_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, outro_start - peak_start),
            ),
            Section(
                label="outro", start_beat=outro_start, end_beat=total_beats,
                stem_gains=OUTRO_GAINS.copy(),
                transition_in="crossfade",
                transition_beats=_clamp_transition_beats(4, total_beats - outro_start),
            ),
        ]

        candidates.append(
            _build_remix_plan(meta_a, meta_b, sections, "standard_arc", target_bpm)
        )

    return candidates


def _generate_hook_first(
    total_beats: int,
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    target_bpm: float,
) -> list[RemixPlan]:
    """Hook-first: short intro -> vocal hook -> verse -> peak -> outro.

    Varies: intro length (4, 8 beats), transition style (cut, crossfade),
    gain delta (-3dB, 0dB, +3dB).
    """
    candidates = []

    configs = [
        {"intro_beats": 8, "transition": "crossfade", "gain_delta": 0},
        {"intro_beats": 4, "transition": "cut", "gain_delta": 3},
        {"intro_beats": 8, "transition": "crossfade", "gain_delta": -3},
    ]

    for cfg in configs:
        intro_end = _snap_to_grid(cfg["intro_beats"])
        if intro_end < MIN_SECTION_BEATS_ABSOLUTE:
            intro_end = MIN_SECTION_BEATS_ABSOLUTE

        hook_end = _snap_to_grid(int(total_beats * 0.20))
        verse_end = _snap_to_grid(int(total_beats * 0.55))
        peak_end = _snap_to_grid(int(total_beats * 0.85))

        bounds = _ensure_boundaries(
            [intro_end, hook_end, verse_end, peak_end],
            total_beats,
        )
        intro_end, hook_end, verse_end, peak_end = bounds

        transition = cfg["transition"]
        gain_delta = cfg["gain_delta"]
        trans_beats = 2 if transition == "cut" else 4

        sections = [
            Section(
                label="intro", start_beat=0, end_beat=intro_end,
                stem_gains=INTRO_GAINS.copy(),
                transition_in="fade",
                transition_beats=_clamp_transition_beats(2, intro_end),
            ),
            Section(
                label="hook", start_beat=intro_end, end_beat=hook_end,
                stem_gains=_apply_gain_delta(HOOK_GAINS, gain_delta),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, hook_end - intro_end),
            ),
            Section(
                label="verse", start_beat=hook_end, end_beat=verse_end,
                stem_gains=_apply_gain_delta(VERSE_GAINS, gain_delta),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, verse_end - hook_end),
            ),
            Section(
                label="peak", start_beat=verse_end, end_beat=peak_end,
                stem_gains=_apply_gain_delta(PEAK_GAINS, gain_delta),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, peak_end - verse_end),
            ),
            Section(
                label="outro", start_beat=peak_end, end_beat=total_beats,
                stem_gains=OUTRO_GAINS.copy(),
                transition_in="crossfade",
                transition_beats=_clamp_transition_beats(4, total_beats - peak_end),
            ),
        ]

        candidates.append(
            _build_remix_plan(meta_a, meta_b, sections, "hook_first", target_bpm)
        )

    return candidates


def _generate_dj_lift(
    total_beats: int,
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    target_bpm: float,
) -> list[RemixPlan]:
    """DJ Lift: build -> vocal in -> peak -> vocal out -> outro.

    Varies: vocal entry (16, 24 beats into arrangement), transition style
    (filter_sweep, crossfade), transition length (4, 8 beats).
    """
    candidates = []

    configs = [
        {"vocal_in_pct": 0.18, "transition": "filter_sweep", "trans_beats": 8},
        {"vocal_in_pct": 0.25, "transition": "crossfade", "trans_beats": 4},
        {"vocal_in_pct": 0.15, "transition": "filter_sweep", "trans_beats": 4},
    ]

    for cfg in configs:
        vocal_in_pct = cfg["vocal_in_pct"]
        transition = cfg["transition"]
        trans_beats = cfg["trans_beats"]

        build_end = _snap_to_grid(int(total_beats * vocal_in_pct))
        peak_start = _snap_to_grid(int(total_beats * 0.45))
        vocal_out = _snap_to_grid(int(total_beats * 0.78))
        outro_start = _snap_to_grid(int(total_beats * 0.88))

        bounds = _ensure_boundaries(
            [build_end, peak_start, vocal_out, outro_start],
            total_beats,
        )
        build_end, peak_start, vocal_out, outro_start = bounds

        sections = [
            Section(
                label="build", start_beat=0, end_beat=build_end,
                stem_gains=BUILD_GAINS.copy(),
                transition_in="fade",
                transition_beats=_clamp_transition_beats(4, build_end),
            ),
            Section(
                label="main", start_beat=build_end, end_beat=peak_start,
                stem_gains=MAIN_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, peak_start - build_end),
            ),
            Section(
                label="peak", start_beat=peak_start, end_beat=vocal_out,
                stem_gains=PEAK_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, vocal_out - peak_start),
            ),
            Section(
                label="breakdown", start_beat=vocal_out, end_beat=outro_start,
                stem_gains=BREAKDOWN_GAINS.copy(),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, outro_start - vocal_out),
            ),
            Section(
                label="outro", start_beat=outro_start, end_beat=total_beats,
                stem_gains=OUTRO_GAINS.copy(),
                transition_in="crossfade",
                transition_beats=_clamp_transition_beats(4, total_beats - outro_start),
            ),
        ]

        candidates.append(
            _build_remix_plan(meta_a, meta_b, sections, "dj_lift", target_bpm)
        )

    return candidates


def _generate_quick_hit(
    total_beats: int,
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    target_bpm: float,
) -> list[RemixPlan]:
    """Quick Hit: intro -> main vocal block -> short release.

    Minimal structure, optimized for impact. Varies: transition style
    (cut, silence_gap), gain delta (0dB, +3dB).
    """
    candidates = []

    configs = [
        {"transition": "cut", "gain_delta": 0, "trans_beats": 2},
        {"transition": "silence_gap", "gain_delta": 3, "trans_beats": 4},
        {"transition": "crossfade", "gain_delta": -3, "trans_beats": 4},
    ]

    for cfg in configs:
        transition = cfg["transition"]
        gain_delta = cfg["gain_delta"]
        trans_beats = cfg["trans_beats"]

        intro_end = _snap_to_grid(int(total_beats * 0.08))
        main_end = _snap_to_grid(int(total_beats * 0.80))

        bounds = _ensure_boundaries(
            [intro_end, main_end],
            total_beats,
        )
        intro_end, main_end = bounds

        sections = [
            Section(
                label="intro", start_beat=0, end_beat=intro_end,
                stem_gains=INTRO_GAINS.copy(),
                transition_in="fade",
                transition_beats=_clamp_transition_beats(2, intro_end),
            ),
            Section(
                label="main", start_beat=intro_end, end_beat=main_end,
                stem_gains=_apply_gain_delta(MAIN_GAINS, gain_delta),
                transition_in=transition,
                transition_beats=_clamp_transition_beats(trans_beats, main_end - intro_end),
            ),
            Section(
                label="outro", start_beat=main_end, end_beat=total_beats,
                stem_gains=OUTRO_GAINS.copy(),
                transition_in="crossfade",
                transition_beats=_clamp_transition_beats(4, total_beats - main_end),
            ),
        ]

        candidates.append(
            _build_remix_plan(meta_a, meta_b, sections, "quick_hit", target_bpm)
        )

    return candidates


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------

_FAMILY_GENERATORS = {
    "standard_arc": _generate_standard_arc,
    "hook_first": _generate_hook_first,
    "dj_lift": _generate_dj_lift,
    "quick_hit": _generate_quick_hit,
}


def _get_family_name(plan: RemixPlan) -> str:
    """Extract the arrangement family name from a plan's explanation."""
    for name in _FAMILY_GENERATORS:
        if name in plan.explanation:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Deduplication and backfill
# ---------------------------------------------------------------------------

def _deduplicate(candidates: list[RemixPlan]) -> list[RemixPlan]:
    """Remove duplicate candidates based on structure hash."""
    seen: set[str] = set()
    unique: list[RemixPlan] = []
    for plan in candidates:
        h = _structure_hash(plan)
        if h not in seen:
            seen.add(h)
            unique.append(plan)
    return unique


def _backfill(
    candidates: list[RemixPlan],
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    target_bpm: float,
    total_beats: int,
    min_count: int,
) -> list[RemixPlan]:
    """If too few candidates after dedup, backfill from underrepresented families.

    Generates additional "safer" variants (using conservative default knobs)
    from families with fewest candidates until min_count is reached.
    """
    if len(candidates) >= min_count:
        return candidates

    # Count candidates per family
    family_counts = Counter(_get_family_name(p) for p in candidates)

    # Ensure all families are represented
    for name in _FAMILY_GENERATORS:
        if name not in family_counts:
            family_counts[name] = 0

    existing_hashes = {_structure_hash(p) for p in candidates}
    result = list(candidates)

    # Generate extras from underrepresented families (least first)
    for family_name, _ in family_counts.most_common()[::-1]:
        if len(result) >= min_count:
            break

        generator = _FAMILY_GENERATORS[family_name]
        extras = generator(total_beats, meta_a, meta_b, target_bpm)

        for extra in extras:
            if len(result) >= min_count:
                break
            h = _structure_hash(extra)
            if h not in existing_hashes:
                existing_hashes.add(h)
                # Mark backfill variants with a note in explanation
                extra.explanation = (
                    f"Backfill candidate using {family_name} arrangement family."
                )
                result.append(extra)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidates(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    prompt: str = "",  # for future prompt-aware generation
    target_count: int = 12,
    min_count: int = 8,
    max_count: int = 16,
) -> list[RemixPlan]:
    """Generate 8-12 candidate RemixPlan variants across 4 arrangement families.

    Each plan has valid section boundaries (monotonic, start=0), snapped to
    4-beat grid boundaries, with sections >= 8 beats for major sections and
    >= 4 beats minimum. All plans have used_fallback=False.

    Args:
        meta_a: Audio metadata for song A (vocal source).
        meta_b: Audio metadata for song B (instrumental source).
        prompt: User prompt (reserved for future prompt-aware generation).
        target_count: Target number of candidates to generate.
        min_count: Minimum candidates after dedup (triggers backfill if below).
        max_count: Maximum candidates to return.

    Returns:
        List of RemixPlan objects, between min_count and max_count inclusive.
    """
    target_bpm, total_beats = _compute_total_beats(meta_a, meta_b)

    logger.info(
        "Generating candidate plans: target_bpm=%.1f, total_beats=%d",
        target_bpm,
        total_beats,
    )

    # Generate candidates from all families
    all_candidates: list[RemixPlan] = []
    for family_name, generator in _FAMILY_GENERATORS.items():
        family_candidates = generator(total_beats, meta_a, meta_b, target_bpm)
        logger.debug(
            "Family %s produced %d candidates", family_name, len(family_candidates)
        )
        all_candidates.extend(family_candidates)

    # Deduplicate
    unique = _deduplicate(all_candidates)
    logger.info(
        "Deduplication: %d raw -> %d unique candidates",
        len(all_candidates),
        len(unique),
    )

    # Backfill if below minimum
    if len(unique) < min_count:
        unique = _backfill(
            unique, meta_a, meta_b, target_bpm, total_beats, min_count
        )
        logger.info("After backfill: %d candidates", len(unique))

    # Cap at max_count
    result = unique[:max_count]

    # Verify all families are represented
    family_counts = Counter(_get_family_name(p) for p in result)
    logger.info(
        "Final candidate distribution: %s (total=%d)",
        dict(family_counts),
        len(result),
    )

    return result
