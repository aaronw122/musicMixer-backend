"""Key convergence algorithm for remix pitch alignment.

Determines how to shift Song A (vocals) and Song B (instrumentals) so they
converge to a compatible key. Uses the chromatic circle to find the shortest
path, then allocates semitone shifts with an instrumental-heavy bias (the
instrumental track absorbs most of the shift to protect vocal quality).

The algorithm handles:
- Same-mode keys (both major or both minor): direct chromatic distance
- Different-mode keys: converts via relative key (+3/-3 semitones), picks
  the path with smallest total shift
- Rap vocals: skips key convergence entirely (spoken word is pitch-agnostic)
- Low-confidence or missing key data: skips gracefully
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class KeyPlan:
    """Result of key convergence analysis."""
    action: str          # "skip" | "shift" | "warning" | "incompatible"
    shift_a: float       # Semitones to shift Song A (vocals) audio (signed)
    shift_b: float       # Semitones to shift Song B (instrumentals) audio (signed)
    target_key: str      # The target key both songs converge toward
    target_scale: str    # "major" or "minor"
    reason: str          # Human-readable explanation
    distance: int        # Original chromatic distance (after mode conversion)


# Map note names to semitone offsets from C (0-11).
_NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

# Semitone value -> canonical note name (prefer sharps for output)
_SEMITONE_TO_NOTE = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
}

# Shift allocation table: distance -> (instrumental_shift, vocal_shift)
# Instrumental (Song B) absorbs most of the shift to protect vocal quality.
_SHIFT_ALLOCATION = {
    1: (1, 0),
    2: (2, 0),
    3: (3, 0),
    4: (4, 0),
    5: (4, 1),
    6: (4, 2),
}

# Confidence threshold below which we skip key convergence.
_MIN_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def note_to_semitone(key: str) -> int:
    """Map a note name to its semitone value (0-11).

    Supports sharps (#) and flats (b), including enharmonics E#=5, B#=0.

    Raises:
        ValueError: If the key is not recognized.
    """
    semi = _NOTE_TO_SEMITONE.get(key)
    if semi is None:
        raise ValueError(f"Unrecognized key: {key!r}")
    return semi


def chromatic_distance(semi_a: int, semi_b: int) -> int:
    """Minimum distance between two semitone values on the chromatic circle.

    Returns a value in range 0-6.
    """
    diff = abs(semi_a - semi_b) % 12
    return min(diff, 12 - diff)


def signed_shift(from_semi: int, to_semi: int) -> int:
    """Shortest signed shift from one semitone to another (-6 to +6).

    Positive = shift up, negative = shift down.
    """
    diff = (to_semi - from_semi) % 12
    if diff > 6:
        return diff - 12
    return diff


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def compute_key_plan(
    key_a: str | None,
    scale_a: str | None,
    conf_a: float | None,
    mod_a: bool | None,
    key_b: str | None,
    scale_b: str | None,
    conf_b: float | None,
    mod_b: bool | None,
    rap_vocals: bool = False,
) -> KeyPlan:
    """Compute the key convergence plan for two songs.

    Args:
        key_a: Detected key of Song A (vocals), e.g. "C#"
        scale_a: "major" or "minor" for Song A
        conf_a: Key detection confidence for Song A (0.0-1.0)
        mod_a: Whether modulation was detected in Song A
        key_b: Detected key of Song B (instrumentals)
        scale_b: "major" or "minor" for Song B
        conf_b: Key detection confidence for Song B (0.0-1.0)
        mod_b: Whether modulation was detected in Song B
        rap_vocals: If True, skip key convergence (rap is pitch-agnostic)

    Returns:
        KeyPlan with the recommended action and shift amounts.
    """
    skip = lambda reason: KeyPlan(  # noqa: E731
        action="skip", shift_a=0, shift_b=0,
        target_key="", target_scale="", reason=reason, distance=0,
    )

    # Gate: rap vocals
    if rap_vocals:
        return skip("rap vocals — key convergence not applicable")

    # Gate: missing key data
    if key_a is None or key_b is None or scale_a is None or scale_b is None:
        return skip("missing key data — cannot compute convergence")

    # Gate: low confidence
    if conf_a is not None and conf_a < _MIN_CONFIDENCE:
        return skip(f"low confidence on Song A ({conf_a:.2f})")
    if conf_b is not None and conf_b < _MIN_CONFIDENCE:
        return skip(f"low confidence on Song B ({conf_b:.2f})")

    # Resolve semitones
    try:
        semi_a = note_to_semitone(key_a)
        semi_b = note_to_semitone(key_b)
    except ValueError as e:
        return skip(str(e))

    # Normalize scales
    scale_a_norm = scale_a.lower().strip()
    scale_b_norm = scale_b.lower().strip()

    if scale_a_norm == scale_b_norm:
        # Same mode: direct chromatic distance
        return _plan_same_mode(semi_a, key_a, semi_b, key_b, scale_a_norm)
    else:
        # Different mode: try relative key conversion both ways, pick best
        return _plan_different_mode(
            semi_a, key_a, scale_a_norm,
            semi_b, key_b, scale_b_norm,
        )


def _plan_same_mode(
    semi_a: int, key_a: str,
    semi_b: int, key_b: str,
    scale: str,
) -> KeyPlan:
    """Build a KeyPlan when both songs are in the same mode."""
    dist = chromatic_distance(semi_a, semi_b)
    return _build_plan(semi_a, semi_b, dist, scale)


def _plan_different_mode(
    semi_a: int, key_a: str, scale_a: str,
    semi_b: int, key_b: str, scale_b: str,
) -> KeyPlan:
    """Build a KeyPlan when songs are in different modes.

    Converts via relative key: minor->major = +3, major->minor = -3.
    Tries both conversion directions and picks the one with smallest distance.
    """
    # Path 1: convert A to B's mode
    if scale_a == "minor" and scale_b == "major":
        conv_a_1 = (semi_a + 3) % 12  # A's relative major
    else:  # scale_a == "major" and scale_b == "minor"
        conv_a_1 = (semi_a - 3) % 12  # A's relative minor
    dist_1 = chromatic_distance(conv_a_1, semi_b)

    # Path 2: convert B to A's mode
    if scale_b == "minor" and scale_a == "major":
        conv_b_2 = (semi_b + 3) % 12  # B's relative major
    else:  # scale_b == "major" and scale_a == "minor"
        conv_b_2 = (semi_b - 3) % 12  # B's relative minor
    dist_2 = chromatic_distance(semi_a, conv_b_2)

    if dist_1 <= dist_2:
        # Use path 1: compare in B's mode space
        return _build_plan(conv_a_1, semi_b, dist_1, scale_b)
    else:
        # Use path 2: compare in A's mode space
        return _build_plan(semi_a, conv_b_2, dist_2, scale_a)


def _build_plan(
    effective_semi_a: int,
    effective_semi_b: int,
    dist: int,
    target_scale: str,
) -> KeyPlan:
    """Build a KeyPlan from effective semitone positions and distance.

    effective_semi_a/b are the semitone positions AFTER any mode conversion.
    The shift allocation table determines how many semitones each song shifts.
    Signs are computed so both songs move TOWARD each other.
    """
    if dist == 0:
        # Same key (possibly after relative key conversion)
        target_note = _SEMITONE_TO_NOTE[effective_semi_a % 12]
        return KeyPlan(
            action="skip",
            shift_a=0, shift_b=0,
            target_key=target_note,
            target_scale=target_scale,
            reason="keys are compatible (same key or relative key)",
            distance=0,
        )

    if dist > 6:
        # Should not happen given chromatic_distance returns 0-6,
        # but guard anyway.
        return KeyPlan(
            action="incompatible",
            shift_a=0, shift_b=0,
            target_key="",
            target_scale=target_scale,
            reason=f"chromatic distance {dist} — keys are incompatible",
            distance=dist,
        )

    # Get allocation from table
    inst_mag, vocal_mag = _SHIFT_ALLOCATION[dist]

    # Determine direction: B (instrumental) shifts TOWARD A's position
    b_direction = signed_shift(effective_semi_b, effective_semi_a)
    # Normalize direction to unit sign
    b_sign = 1 if b_direction > 0 else -1

    # A (vocal) shifts TOWARD B's position (opposite direction)
    a_direction = signed_shift(effective_semi_a, effective_semi_b)
    a_sign = 1 if a_direction > 0 else -1

    shift_b = b_sign * inst_mag
    shift_a = a_sign * vocal_mag

    # Compute target key: where B ends up after shifting
    target_semi = (effective_semi_b + shift_b) % 12
    target_note = _SEMITONE_TO_NOTE[target_semi]

    action = "warning" if dist == 6 else "shift"
    reason_prefix = "tritone — proceed with caution" if dist == 6 else f"distance {dist}"

    return KeyPlan(
        action=action,
        shift_a=float(shift_a),
        shift_b=float(shift_b),
        target_key=target_note,
        target_scale=target_scale,
        reason=(
            f"{reason_prefix}: shift vocal {shift_a:+.0f}, "
            f"instrumental {shift_b:+.0f} → {target_note} {target_scale}"
        ),
        distance=dist,
    )
