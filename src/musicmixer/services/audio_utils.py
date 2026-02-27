"""Shared DSP utilities for pedalboard-based audio processing.

Used by eq.py, multiband.py, and mastering.py.
"""

from __future__ import annotations

import numpy as np
from pedalboard import Pedalboard


def process_with_pedalboard(
    audio: np.ndarray, board: Pedalboard, sr: int
) -> np.ndarray:
    """Run audio through a pedalboard, handling channel convention conversion.

    Our pipeline uses (samples, channels) — pedalboard expects (channels, samples).
    This helper transposes in and out so callers don't need to worry about it.

    Args:
        audio: Float32 array, shape (N,) for mono or (N, 2) for stereo.
        board: A pedalboard.Pedalboard instance with plugins to apply.
        sr: Sample rate in Hz.

    Returns:
        Processed float32 array with same shape as input.
    """
    mono = audio.ndim == 1

    if mono:
        # Pedalboard expects (channels, samples) — mono is (1, N)
        pb_input = audio[np.newaxis, :].astype(np.float32)
    else:
        # (N, 2) -> (2, N)
        pb_input = audio.T.astype(np.float32)

    pb_output = board(pb_input, sr)

    if mono:
        return pb_output[0].astype(np.float32)
    else:
        # (2, N) -> (N, 2)
        return pb_output.T.astype(np.float32)
