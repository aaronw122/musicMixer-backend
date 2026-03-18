"""Per-stem corrective EQ: adaptive-primary, preset-fallback.

When adaptive EQ corrections are available (from spectral analysis), only
adaptive filters are applied. When adaptive is unavailable (None), preset
profiles provide a safety-net fallback. All boosts capped at +0.75 dB to
avoid stripping character from source material. Cuts are slightly more
aggressive since removing problems is lower-risk.
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

    Adaptive-primary, preset-fallback: when adaptive corrections are provided,
    only those are applied. When adaptive is None, preset EQ profiles provide
    a safety-net fallback. Both paths run through a single pedalboard pass.

    Args:
        audio: Input audio, shape (samples, channels) or (samples,).
        sr: Sample rate in Hz.
        stem_type: One of vocals, drums, bass, guitar, piano, other.
            Unknown types fall back to "other".
        apply_preset: Whether to apply the preset EQ profile. Callers
            should set this to ``adaptive_corrections is None`` so preset
            only fires as a fallback.
        adaptive_corrections: Optional list of (frequency_hz, gain_db, q_factor)
            tuples. Each becomes a PeakFilter. Clamping is handled upstream
            (spectral.py); this function applies whatever values it receives.
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
        _path = "adaptive" if has_adaptive else "preset-fallback"
        logger.debug("Applied EQ for stem_type='%s' (path=%s, preset=%s, adaptive=%s)",
                      stem_type, _path, has_preset, has_adaptive)
        return result

    return audio
