"""Shared DSP utilities for pedalboard-based audio processing.

Used by eq.py, multiband.py, and mastering.py.
"""

from __future__ import annotations

import logging

import numpy as np
from pedalboard import Pedalboard

logger = logging.getLogger(__name__)


def process_with_pedalboard(audio: np.ndarray, board: Pedalboard, sr: int) -> np.ndarray:
    """Run audio through a pedalboard plugin chain, handling channel convention.

    Our pipeline uses (samples, channels) — pedalboard expects (channels, samples).
    This helper transposes before/after processing.

    Args:
        audio: Input audio as float32 (N, 2) stereo or (N,) mono.
        board: A pedalboard.Pedalboard instance with plugins configured.
        sr: Sample rate in Hz.

    Returns:
        Processed audio as float32 in the original shape convention (samples, channels).
    """
    mono = audio.ndim == 1
    if mono:
        # Pedalboard expects (channels, samples) — wrap mono as (1, N)
        pb_input = audio[np.newaxis, :].astype(np.float32)
    else:
        # (N, 2) -> (2, N)
        pb_input = audio.T.astype(np.float32)

    try:
        pb_output = board(pb_input, sr)
    except Exception:
        plugin_names = [type(p).__name__ for p in board]
        logger.error(
            "Pedalboard processing failed (plugins=%s, sr=%d, shape=%s), "
            "returning unprocessed audio",
            plugin_names,
            sr,
            audio.shape,
            exc_info=True,
        )
        return audio

    if mono:
        return pb_output[0].astype(np.float32)
    else:
        # (2, N) -> (N, 2)
        return pb_output.T.astype(np.float32)
