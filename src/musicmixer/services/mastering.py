"""Static mastering chain: constrained LUFS normalization + true-peak limiter.

Replaces the standard limiter chain (steps 14/15/15.5) when
ab_static_mastering_v1 is enabled. Applies operations in standard
mastering order:

1. Optional low-pass filter (for lossy sources)
2. Constrained LUFS normalization at -12 LUFS (with +3 dB headroom)
3. True-peak limiter at -1.0 dBTP
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.signal import butter, sosfiltfilt

from musicmixer.services.processor import lufs_normalize_constrained, true_peak_limit

logger = logging.getLogger(__name__)


def master_static(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = -12.0,
    ceiling_dbtp: float = -1.0,
    lossy_lpf_hz: float | None = None,
) -> np.ndarray:
    """Static mastering chain: normalize loudness then limit peaks.

    Operations in sequence:
    1. Optional low-pass filter at lossy_lpf_hz (rolls off codec artifacts)
    2. Constrained LUFS normalization to target_lufs with +3 dB headroom
       (the limiter immediately follows, so peaks can overshoot the ceiling
       by up to 3 dB and still be caught)
    3. True-peak limiter at ceiling_dbtp

    Args:
        audio: Input audio as float32 (N, 2) stereo or (N,) mono.
        sr: Sample rate in Hz.
        target_lufs: Target integrated loudness in LUFS. Default -12.0.
        ceiling_dbtp: True-peak ceiling in dBTP. Default -1.0.
        lossy_lpf_hz: If set, apply a gentle low-pass filter at this
            frequency before mastering. Used for lossy sources (e.g. 16kHz
            for Opus 128kbps) to roll off codec artifacts.

    Returns:
        Mastered audio as float32 in the same shape as input.
    """
    t_chain = time.monotonic()

    # 1. Optional low-pass filter for lossy sources
    if lossy_lpf_hz is not None:
        t0 = time.monotonic()
        audio = _gentle_lowpass(audio, sr, lossy_lpf_hz)
        logger.info("master_static: LPF at %.0f Hz took %.2fs", lossy_lpf_hz, time.monotonic() - t0)

    # 2. Constrained LUFS normalization with +3 dB headroom.
    #    The +3 dB headroom is intentional: the limiter immediately follows
    #    and catches peaks that overshoot the ceiling. This lets the normalizer
    #    push the signal closer to the target LUFS without being overly
    #    constrained by peak headroom.
    t0 = time.monotonic()
    audio = lufs_normalize_constrained(
        audio, sr,
        target_lufs=target_lufs,
        ceiling_dbtp=ceiling_dbtp,
        headroom_db=3.0,
    )
    logger.info("master_static: LUFS normalize took %.2fs", time.monotonic() - t0)

    # 3. True-peak limiter at the ceiling
    t0 = time.monotonic()
    audio = true_peak_limit(audio, sr, ceiling_dbtp=ceiling_dbtp)
    logger.info("master_static: true-peak limiter took %.2fs", time.monotonic() - t0)

    logger.info("master_static: chain complete in %.2fs (target=%.1f LUFS, ceiling=%.1f dBTP)",
                time.monotonic() - t_chain, target_lufs, ceiling_dbtp)

    return audio


def _gentle_lowpass(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 2,
) -> np.ndarray:
    """Apply a gentle Butterworth low-pass filter (zero-phase).

    Used to roll off high-frequency codec artifacts from lossy sources
    before mastering. Order 2 gives a gentle -12 dB/oct slope.
    """
    sos = butter(order, cutoff_hz, btype='low', fs=sr, output='sos')
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)
