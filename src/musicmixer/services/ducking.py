"""Spectral ducking: carve a mid-range pocket in the instrumental for vocals.

Reduces instrumental energy in the 300Hz-3kHz range when vocals are active,
so vocals sit *inside* the instrumental rather than competing on top. This is
the highest-ROI mixing improvement -- the plain bus sum (vocal + instrumental)
has zero frequency-aware interaction without it.

Detection band (300-3500 Hz) is intentionally wider than ducking band
(300-3000 Hz). The 500 Hz gap captures upper harmonics for better vocal
activity detection without ducking content above the vocal formant range.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import butter, lfilter, sosfiltfilt

logger = logging.getLogger(__name__)


def spectral_duck(
    instrumental: np.ndarray,
    vocal: np.ndarray,
    sr: int,
    cut_db: float = -3.5,
    lo: float = 300,
    hi: float = 3000,
) -> np.ndarray:
    """Duck instrumental mid-range energy where vocals are active.

    Uses zero-phase (sosfiltfilt) filtering for mid-band extraction to avoid
    comb-filter artifacts. Causal sosfilt introduces frequency-dependent phase
    delay; subtracting a phase-shifted mid-band from the original creates
    metallic/hollow coloration.

    CRITICAL: Returns a NEW array -- does NOT mutate the input instrumental.
    The caller must use a separate variable (``ducked_instrumental``) so the
    auto-leveler continues to see the un-ducked ``instrumental_bus``.

    CRITICAL: Preserves the full-length instrumental array. Ducking is only
    applied to the overlapping region ``[:min_len]``; beyond that the
    instrumental passes through unchanged.

    Args:
        instrumental: Stereo instrumental audio (N, 2).
        vocal: Stereo vocal audio (M, 2). May differ in length from
            instrumental.
        sr: Sample rate (44100).
        cut_db: Gain reduction in dB when vocals are active. Default -3.5 dB
            is conservative -- creates space without making instrumental hollow.
        lo: Low cutoff for the ducking mid-band (Hz). Default 300.
        hi: High cutoff for the ducking mid-band (Hz). Default 3000.

    Returns:
        Ducked instrumental audio, same shape as input ``instrumental``.
        Full-length is preserved -- no truncation.
    """
    if instrumental.ndim != 2:
        raise ValueError(
            f"Expected stereo instrumental (N, 2), got ndim={instrumental.ndim}"
        )

    # Work on a copy -- never mutate the input
    result = instrumental.copy()

    min_len = min(len(instrumental), len(vocal))
    if min_len == 0:
        return result

    # Slice the overlapping region for processing (views, not copies yet)
    inst_overlap = instrumental[:min_len]
    vocal_overlap = vocal[:min_len]

    # ── 1. Detect vocal activity via RMS envelope (50ms frames) ──────────
    vocal_mono = (
        vocal_overlap.mean(axis=1)
        if vocal_overlap.ndim == 2
        else vocal_overlap
    )
    frame_len = int(0.05 * sr)  # 50ms = 2205 samples at 44.1kHz
    if frame_len == 0:
        return result

    # Bandpass vocal to 300-3500 Hz for energy detection.
    # Detection band is intentionally wider than ducking band (300-3000 Hz) --
    # the 500 Hz gap captures harmonics for better vocal activity detection.
    sos_bp = butter(4, [300, 3500], btype="band", fs=sr, output="sos")
    vocal_filtered = sosfiltfilt(sos_bp, vocal_mono)

    # Compute per-frame RMS energy
    n_frames = (len(vocal_filtered) - frame_len) // frame_len + 1
    if n_frames <= 0:
        return result

    # Reshape into (n_frames, frame_len) matrix for vectorized RMS.
    # Truncate to n_frames * frame_len to avoid leftover samples.
    frames = vocal_filtered[: n_frames * frame_len].reshape(n_frames, frame_len)
    vocal_energy = np.sqrt(np.mean(frames**2, axis=1))

    # Noise-floor-relative threshold with hysteresis.
    # The absolute floor (1e-5) prevents perpetual ducking with very clean
    # BS-RoFormer separations where the 10th percentile is near zero.
    noise_floor = np.percentile(vocal_energy, 10)
    onset_threshold = max(noise_floor * 4.0, 1e-5)
    offset_threshold = max(noise_floor * 2.0, 1e-5 / 2.0)

    # Hysteresis state machine: active when > onset, stays active until < offset
    vocal_active = np.zeros(n_frames, dtype=np.float64)
    active = False
    for i in range(n_frames):
        if active:
            active = vocal_energy[i] >= offset_threshold
        else:
            active = vocal_energy[i] > onset_threshold
        vocal_active[i] = float(active)

    # ── 2. Upsample mask to sample rate + smooth with attack/release ─────
    # Build mask at min_len. np.repeat may produce fewer than min_len samples
    # if n_frames * frame_len < min_len, so pad to min_len first.
    raw_mask = np.repeat(vocal_active, frame_len)
    if len(raw_mask) < min_len:
        mask_overlap = np.zeros(min_len, dtype=raw_mask.dtype)
        mask_overlap[: len(raw_mask)] = raw_mask
    else:
        mask_overlap = raw_mask[:min_len]

    # Hold: after vocal activity drops, keep mask active for hold_ms before
    # starting release. Prevents rapid on/off cycling on intermittent vocals
    # (e.g. 21 Savage pausing between bars).
    hold_ms = 200.0
    hold_samples = int(hold_ms / 1000.0 * sr)
    if hold_samples > 0:
        # Max filter extends each active region by hold_samples.
        # Positive origin shifts the window RIGHT (forward in time), creating
        # a trailing hold that extends ~200ms after vocal drops. O(n) via scipy.
        from scipy.ndimage import maximum_filter1d
        mask_overlap = maximum_filter1d(
            mask_overlap, size=hold_samples, origin=+(hold_samples // 2 - 1)
        )

    # Exponential IIR smoothing: 30ms attack, 400ms release
    # 400ms release gives a full quarter-note recovery at 85 BPM
    attack_alpha = 1 - math.exp(-1.0 / (0.03 * sr))
    release_alpha = 1 - math.exp(-1.0 / (0.40 * sr))

    # Two-pass lfilter approximation (runs in <0.05s vs 5-15s for naive loop)
    attack_smoothed = lfilter(
        [attack_alpha], [1, -(1 - attack_alpha)], mask_overlap
    )
    release_smoothed = lfilter(
        [release_alpha], [1, -(1 - release_alpha)], mask_overlap[::-1]
    )[::-1]
    mask_overlap = np.maximum(attack_smoothed, release_smoothed)

    # Zero-pad mask to full instrumental length.
    # Zeros beyond min_len = no ducking = instrumental passes through cleanly
    # after vocals end.
    full_len = len(instrumental)
    mask = np.zeros(full_len, dtype=mask_overlap.dtype)
    mask[:min_len] = mask_overlap

    # ── 3. Extract mid-band from instrumental (ZERO-PHASE -- critical) ───
    # sosfiltfilt eliminates phase shift; causal filtering would create
    # comb-filter artifacts when subtracted from the original.
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    mid_band = sosfiltfilt(sos, result, axis=0)

    # ── 4. Reduce mid-band energy where vocals are active (stereo-safe) ──
    gain = 10 ** (cut_db / 20)  # -3.5 dB -> ~0.668
    reduction = mid_band * (1 - gain) * mask[:, np.newaxis]
    result = result - reduction

    # Log summary
    active_pct = float(np.mean(mask_overlap > 0.5)) * 100
    logger.info(
        "spectral_duck: cut=%.1f dB, band=%d-%d Hz, vocal_active=%.0f%%, "
        "onset_threshold=%.2e, full_len=%d, overlap_len=%d",
        cut_db,
        lo,
        hi,
        active_pct,
        onset_threshold,
        full_len,
        min_len,
    )

    return result.astype(np.float32)
