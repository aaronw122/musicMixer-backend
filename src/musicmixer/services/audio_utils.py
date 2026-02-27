"""Shared DSP utilities for the sound quality enhancement chain.

Used by eq.py, multiband.py, and mastering.py.
"""

from __future__ import annotations

import numpy as np
from pedalboard import Pedalboard


def process_with_pedalboard(audio: np.ndarray, board: Pedalboard, sr: int) -> np.ndarray:
    """Run audio through a pedalboard, handling channel convention conversion.

    Our pipeline uses (samples, channels) — pedalboard expects (channels, samples).
    This helper transposes in and out, preserving float32 dtype.

    Args:
        audio: Input audio array, shape (samples, channels) or (samples,) for mono.
        board: Pedalboard instance with plugins to apply.
        sr: Sample rate in Hz.

    Returns:
        Processed audio in (samples, channels) format, float32.
    """
    if audio.ndim == 1:
        # Mono: pedalboard expects (1, samples)
        pb_input = audio[np.newaxis, :].astype(np.float32)
        pb_output = board(pb_input, sr)
        return pb_output[0, :].astype(np.float32)

    # Stereo / multi-channel: (samples, channels) -> (channels, samples)
    pb_input = audio.T.astype(np.float32)
    pb_output = board(pb_input, sr)
    return pb_output.T.astype(np.float32)
