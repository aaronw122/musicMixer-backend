"""Shared BPM estimation logic: single source of truth for target BPM computation.

All remix subsystems (processor, interpreter, analysis) must use these functions
instead of inlining their own BPM formulas. This eliminates divergence between
the 4+ sites that previously answered "what BPM will this remix render at?"
with different results.

Zero external dependencies (stdlib only).
"""

from __future__ import annotations


def estimate_target_bpm(
    vocal_bpm: float,
    instrumental_bpm: float,
    tempo_source: str = "weighted_midpoint",
) -> float:
    """Estimate the target BPM for a remix given vocal and instrumental BPMs.

    Replicates the exact tiered logic from compute_tempo_plan() in processor.py.

    Tiers (for weighted_midpoint / average / default):
      - gap <= 4%:  target = instrumental_bpm  (DJ-transparent)
      - gap <= 10%: target = instrumental_bpm * 0.65 + vocal_bpm * 0.35
      - gap > 10%:  target = instrumental_bpm * 0.70 + vocal_bpm * 0.30
      - If gap > 20% AND vocal stretch > 12%: clamp to vocal_bpm * 1.12 or * 0.88

    Args:
        vocal_bpm: BPM of the vocal source song.
        instrumental_bpm: BPM of the instrumental source song.
        tempo_source: One of "song_a", "song_b", "weighted_midpoint", "average".
                      Any unrecognized value returns instrumental_bpm directly.

    Returns:
        The estimated target BPM for the remix.
    """
    # Guard: if either BPM is invalid, return the best we have
    if vocal_bpm <= 0 and instrumental_bpm <= 0:
        return 1.0
    if vocal_bpm <= 0:
        return instrumental_bpm
    if instrumental_bpm <= 0:
        return vocal_bpm

    if tempo_source == "song_a":
        return vocal_bpm
    if tempo_source == "song_b":
        return instrumental_bpm
    if tempo_source not in ("weighted_midpoint", "average"):
        return instrumental_bpm

    # weighted_midpoint or average
    gap_pct = abs(vocal_bpm - instrumental_bpm) / max(vocal_bpm, instrumental_bpm)

    if gap_pct <= 0.04:
        # Within DJ-transparent range -- match to instrumental
        target_bpm = instrumental_bpm
    elif gap_pct <= 0.10:
        # Safe mashup range -- 65/35 instrumental bias
        target_bpm = instrumental_bpm * 0.65 + vocal_bpm * 0.35
    elif gap_pct <= 0.20:
        # Extended range -- 70/30 bias
        target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
    else:
        # Beyond practical range -- 70/30 bias with 12% vocal clamp
        target_bpm = instrumental_bpm * 0.70 + vocal_bpm * 0.30
        # Clamp: if vocal stretch > 12%, pull target toward vocal
        vocal_stretch = abs(vocal_bpm - target_bpm) / vocal_bpm
        if vocal_stretch > 0.12:
            if vocal_bpm > target_bpm:
                target_bpm = vocal_bpm * 0.88  # Max 12% slowdown
            else:
                target_bpm = vocal_bpm * 1.12  # Max 12% speedup

    return target_bpm


def compute_stretch_pct(
    vocal_bpm: float,
    instrumental_bpm: float,
    tempo_source: str = "weighted_midpoint",
) -> float:
    """Compute the maximum stretch percentage either song undergoes.

    Uses estimate_target_bpm() to find the target, then returns the larger
    of the two stretch percentages (vocal and instrumental).

    Returns:
        The maximum stretch percentage as a positive float (e.g. 12.5 for 12.5%).
    """
    if vocal_bpm <= 0 or instrumental_bpm <= 0:
        return 0.0

    target = estimate_target_bpm(vocal_bpm, instrumental_bpm, tempo_source)

    vocal_stretch = abs(target - vocal_bpm) / vocal_bpm * 100
    inst_stretch = abs(target - instrumental_bpm) / instrumental_bpm * 100
    return max(vocal_stretch, inst_stretch)
