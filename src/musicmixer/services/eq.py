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

# Each profile is a list of (PluginClass, kwargs) tuples. The `_gain_db` key
# in kwargs is used to identify HF boosts for the halve_hf_boosts feature.
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

# HF boost threshold: frequencies at or above this are considered "high frequency"
# for the halve_hf_boosts feature (lossy YouTube sources).
_HF_THRESHOLD_HZ = 2000.0


def _build_preset_board(stem_type: str, halve_hf_boosts: bool = False) -> Pedalboard:
    """Build a Pedalboard from the preset profile for a given stem type.

    Args:
        stem_type: One of vocals, drums, bass, guitar, piano, other.
        halve_hf_boosts: When True, multiply all HF boost gains by 0.5
            (for lossy YouTube sources to avoid amplifying codec artifacts).

    Returns:
        Configured Pedalboard instance.
    """
    preset = _EQ_PRESETS.get(stem_type, _EQ_PRESETS["other"])
    plugins = []

    for plugin_cls, kwargs in preset:
        kw = dict(kwargs)  # copy to avoid mutating the preset

        # Apply halve_hf_boosts: halve positive gain_db on HF filters
        if halve_hf_boosts and "gain_db" in kw and kw["gain_db"] > 0:
            freq = kw.get("cutoff_frequency_hz", 0)
            if freq >= _HF_THRESHOLD_HZ:
                kw["gain_db"] = kw["gain_db"] * 0.5

        plugins.append(plugin_cls(**kw))

    return Pedalboard(plugins)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_corrective_eq(
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    apply_preset: bool = True,
    halve_hf_boosts: bool = False,
    **kwargs,
) -> np.ndarray:
    """Apply corrective EQ to a stem.

    Applies broad preset EQ profiles per stem type. Safe to use before
    tempo stretching.

    Args:
        audio: Input audio, shape (samples, channels) or (samples,).
        sr: Sample rate in Hz.
        stem_type: One of vocals, drums, bass, guitar, piano, other.
            Unknown types fall back to "other".
        apply_preset: Whether to apply the preset EQ profile.
        halve_hf_boosts: When True, multiply all HF boost gains by 0.5
            (for lossy YouTube sources).
        **kwargs: Accepted for backward compatibility (e.g. apply_resonance_cuts).

    Returns:
        Processed audio with same shape and dtype as input.
    """
    # Fall back to "other" for unknown stem types
    if stem_type not in _EQ_PRESETS:
        logger.warning("Unknown stem type '%s', falling back to 'other'", stem_type)
        stem_type = "other"

    result = audio

    # Preset EQ (broad cuts/boosts)
    if apply_preset:
        board = _build_preset_board(stem_type, halve_hf_boosts=halve_hf_boosts)
        result = process_with_pedalboard(result, board, sr)
        logger.debug("Applied preset EQ for stem_type='%s'", stem_type)

    return result
