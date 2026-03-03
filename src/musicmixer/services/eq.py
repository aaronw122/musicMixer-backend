"""Per-stem corrective EQ with preset profiles.

Gentle corrective EQ applied per stem type before mixing. All boosts capped
at +0.75 dB to avoid stripping character from source material. Cuts are
slightly more aggressive since removing problems is lower-risk.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pedalboard import (
    HighpassFilter,
    HighShelfFilter,
    LowpassFilter,
    Pedalboard,
    PeakFilter,
)

from musicmixer.services.audio_utils import process_with_pedalboard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Q values: cuts are wider (gentler correction), boosts are narrower
# (more surgical enhancement).
Q_CUT = 1.5
Q_BOOST = 2.0

# ---------------------------------------------------------------------------
# Preset EQ Profiles
# ---------------------------------------------------------------------------

# Each profile is a list of (PluginClass, kwargs) tuples.
# Filters are applied in order.

_EQ_PRESETS: dict[str, list[tuple[type, dict[str, Any]]]] = {
    "vocals": [
        (HighpassFilter, {"cutoff_frequency_hz": 80.0}),
        (PeakFilter, {"cutoff_frequency_hz": 250.0, "gain_db": -1.5, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 800.0, "gain_db": -1.5, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 3000.0, "gain_db": 0.75, "q": Q_BOOST}),
        (LowpassFilter, {"cutoff_frequency_hz": 16000.0}),
    ],
    "drums": [
        (HighpassFilter, {"cutoff_frequency_hz": 30.0}),
        (PeakFilter, {"cutoff_frequency_hz": 400.0, "gain_db": -1.5, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 800.0, "gain_db": -2.0, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 5000.0, "gain_db": 0.75, "q": Q_BOOST}),
        (HighShelfFilter, {"cutoff_frequency_hz": 12000.0, "gain_db": -1.0}),
    ],
    "bass": [
        (HighpassFilter, {"cutoff_frequency_hz": 30.0}),
        (PeakFilter, {"cutoff_frequency_hz": 250.0, "gain_db": -2.0, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 800.0, "gain_db": -2.0, "q": Q_CUT}),
        (LowpassFilter, {"cutoff_frequency_hz": 8000.0}),
    ],
    "guitar": [
        (HighpassFilter, {"cutoff_frequency_hz": 80.0}),
        (PeakFilter, {"cutoff_frequency_hz": 200.0, "gain_db": -2.0, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 1200.0, "gain_db": -1.5, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 3500.0, "gain_db": 0.75, "q": Q_BOOST}),
        (LowpassFilter, {"cutoff_frequency_hz": 14000.0}),
    ],
    "piano": [
        (HighpassFilter, {"cutoff_frequency_hz": 60.0}),
        (PeakFilter, {"cutoff_frequency_hz": 300.0, "gain_db": -1.5, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 2500.0, "gain_db": 0.75, "q": Q_BOOST}),
        (LowpassFilter, {"cutoff_frequency_hz": 16000.0}),
    ],
    "other": [
        (HighpassFilter, {"cutoff_frequency_hz": 80.0}),
        (PeakFilter, {"cutoff_frequency_hz": 400.0, "gain_db": -2.0, "q": Q_CUT}),
        (PeakFilter, {"cutoff_frequency_hz": 2500.0, "gain_db": 0.5, "q": Q_BOOST}),
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_corrective_eq(
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    apply_preset: bool = True,
    adaptive_corrections: list[tuple[float, float, float]] | None = None,
    **kwargs,
) -> np.ndarray:
    """Apply corrective EQ to a stem.

    Applies broad preset EQ profiles per stem type, optionally followed by
    adaptive corrections from spectral analysis. All filters run in a single
    pedalboard pass — no extra audio processing cost.

    Args:
        audio: Input audio, shape (samples, channels) or (samples,).
        sr: Sample rate in Hz.
        stem_type: One of vocals, drums, bass, guitar, piano, other.
            Unknown types fall back to "other".
        apply_preset: Whether to apply the preset EQ profile.
        adaptive_corrections: Optional list of (frequency_hz, gain_db, q_factor)
            tuples. Each becomes a PeakFilter appended to the preset board.
            Clamping is handled upstream (spectral.py); this function applies
            whatever values it receives.
        **kwargs: Accepted for backward compatibility.

    Returns:
        Processed audio with same shape and dtype as input.
    """
    # Fall back to "other" for unknown stem types
    if stem_type not in _EQ_PRESETS:
        logger.warning("Unknown stem type '%s', falling back to 'other'", stem_type)
        stem_type = "other"

    has_preset = apply_preset
    has_adaptive = bool(adaptive_corrections)

    # Build a single board with preset + adaptive filters for one pass
    if has_preset or has_adaptive:
        board = Pedalboard()

        if has_preset:
            preset = _EQ_PRESETS.get(stem_type, _EQ_PRESETS["other"])
            for plugin_cls, plugin_kwargs in preset:
                board.append(plugin_cls(**plugin_kwargs))

        if has_adaptive:
            for freq_hz, gain_db, q_factor in adaptive_corrections:
                board.append(
                    PeakFilter(
                        cutoff_frequency_hz=freq_hz,
                        gain_db=gain_db,
                        q=q_factor,
                    )
                )
            logger.debug(
                "Added %d adaptive corrections for stem_type='%s'",
                len(adaptive_corrections),
                stem_type,
            )

        result = process_with_pedalboard(audio, board, sr)
        logger.debug("Applied EQ for stem_type='%s' (preset=%s, adaptive=%s)",
                      stem_type, has_preset, has_adaptive)
        return result

    return audio
