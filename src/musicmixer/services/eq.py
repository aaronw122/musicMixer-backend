"""Per-stem corrective EQ with preset profiles and resonance detection.

Gentle corrective EQ applied per stem type before mixing. All boosts capped
at +0.75 dB to avoid stripping character from source material. Cuts are
slightly more aggressive since removing problems is lower-risk.

Resonance detection is a rescue feature for bad sources, gated behind
ab_resonance_detection_v1 (off by default). Only applied to non-pitched
stems (vocals, drums, other) to avoid stripping harmonic content from
pitched instruments.
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

# Stems eligible for resonance detection. Pitched instruments (bass, guitar,
# piano) are excluded because the algorithm can't distinguish intentional
# harmonic content from problematic resonances.
RESONANCE_ELIGIBLE_STEMS = {"vocals", "drums", "other"}

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
# Resonance Detection
# ---------------------------------------------------------------------------


def detect_resonances(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = 10.0,
    max_resonances: int = 3,
    freq_ranges: tuple[tuple[float, float], ...] = ((200, 600), (2000, 4000)),
) -> list[tuple[float, float]]:
    """Detect resonant peaks in audio via FFT analysis.

    Averages the magnitude spectrum over multiple windows, computes a smoothed
    baseline, and finds peaks exceeding the baseline by threshold_db within
    the specified frequency ranges.

    Args:
        audio: Input audio, shape (samples, channels) or (samples,).
        sr: Sample rate in Hz.
        threshold_db: Minimum dB above baseline for a peak to be flagged.
        max_resonances: Maximum number of resonances to return.
        freq_ranges: Frequency ranges to scan for resonances.

    Returns:
        List of (frequency_hz, magnitude_above_baseline_db) tuples, sorted
        by magnitude descending. At most max_resonances entries.
    """
    # Convert to mono for analysis
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio

    # Use windowed FFT averaging for robust spectrum estimate
    window_size = 4096
    hop = window_size // 2
    n_windows = max(1, (len(mono) - window_size) // hop)

    # Accumulate magnitude spectra
    magnitude_sum = np.zeros(window_size // 2 + 1, dtype=np.float64)
    hann = np.hanning(window_size)

    for i in range(n_windows):
        start = i * hop
        segment = mono[start : start + window_size]
        if len(segment) < window_size:
            break
        windowed = segment * hann
        spectrum = np.abs(np.fft.rfft(windowed))
        magnitude_sum += spectrum

    if n_windows == 0:
        return []

    avg_magnitude = magnitude_sum / n_windows

    # Convert to dB
    mag_db = 20 * np.log10(avg_magnitude + 1e-20)

    # Compute frequency axis
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sr)

    # Smoothed baseline via moving average (wide window for broad shape)
    baseline_window = max(1, int(len(mag_db) * 0.05))  # 5% of spectrum width
    kernel = np.ones(baseline_window) / baseline_window
    padded = np.pad(mag_db, baseline_window // 2, mode="edge")
    baseline_db = np.convolve(padded, kernel, mode="valid")[: len(mag_db)]

    # Find peaks above baseline within specified frequency ranges
    prominence = mag_db - baseline_db
    candidates: list[tuple[float, float]] = []

    for low_hz, high_hz in freq_ranges:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        range_prominence = prominence[indices]
        range_freqs = freqs[indices]

        # Find local maxima in prominence
        for j in range(1, len(range_prominence) - 1):
            if (
                range_prominence[j] > threshold_db
                and range_prominence[j] > range_prominence[j - 1]
                and range_prominence[j] > range_prominence[j + 1]
            ):
                candidates.append((float(range_freqs[j]), float(range_prominence[j])))

    # Sort by prominence descending, take top max_resonances
    candidates.sort(key=lambda x: x[1], reverse=True)
    result = candidates[:max_resonances]

    if result:
        logger.info(
            "Detected %d resonance(s): %s",
            len(result),
            [(f"{f:.0f}Hz", f"{db:.1f}dB") for f, db in result],
        )

    return result


def _build_resonance_board(resonances: list[tuple[float, float]]) -> Pedalboard | None:
    """Build a Pedalboard with narrow notch cuts at detected resonance frequencies.

    Args:
        resonances: List of (frequency_hz, magnitude_db) from detect_resonances.

    Returns:
        Pedalboard with notch filters, or None if no resonances.
    """
    if not resonances:
        return None

    plugins = []
    for freq_hz, mag_db in resonances:
        # Q between 4-6: scale with magnitude (stronger resonance -> narrower notch)
        q = min(6.0, max(4.0, mag_db / 5.0))
        # Depth: proportional to magnitude, capped at -3dB
        depth_db = max(-3.0, -(mag_db * 0.3))
        depth_db = max(depth_db, -3.0)  # ensure cap

        plugins.append(PeakFilter(cutoff_frequency_hz=freq_hz, gain_db=depth_db, q=q))
        logger.debug(
            "Resonance notch: %.0f Hz, Q=%.1f, depth=%.1f dB", freq_hz, q, depth_db
        )

    return Pedalboard(plugins)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_corrective_eq(
    audio: np.ndarray,
    sr: int,
    stem_type: str,
    apply_preset: bool = True,
    apply_resonance_cuts: bool = True,
    halve_hf_boosts: bool = False,
) -> np.ndarray:
    """Apply corrective EQ to a stem.

    Designed for two-pass usage in the pipeline:
    1. Before tempo stretch: apply_preset=True, apply_resonance_cuts=False
       (broad preset EQ is safe before stretching)
    2. After tempo stretch: apply_preset=False, apply_resonance_cuts=True
       (narrow notch cuts would cause ringing if applied before stretch)

    Args:
        audio: Input audio, shape (samples, channels) or (samples,).
        sr: Sample rate in Hz.
        stem_type: One of vocals, drums, bass, guitar, piano, other.
            Unknown types fall back to "other".
        apply_preset: Whether to apply the preset EQ profile.
        apply_resonance_cuts: Whether to detect and cut resonances.
        halve_hf_boosts: When True, multiply all HF boost gains by 0.5
            (for lossy YouTube sources).

    Returns:
        Processed audio with same shape and dtype as input.
    """
    # Fall back to "other" for unknown stem types
    if stem_type not in _EQ_PRESETS:
        logger.warning("Unknown stem type '%s', falling back to 'other'", stem_type)
        stem_type = "other"

    result = audio

    # Pass 1: Preset EQ (broad cuts/boosts, safe before stretch)
    if apply_preset:
        board = _build_preset_board(stem_type, halve_hf_boosts=halve_hf_boosts)
        result = process_with_pedalboard(result, board, sr)
        logger.debug("Applied preset EQ for stem_type='%s'", stem_type)

    # Pass 2: Resonance detection + notch cuts (narrow, must be after stretch)
    if apply_resonance_cuts and stem_type in RESONANCE_ELIGIBLE_STEMS:
        resonances = detect_resonances(result, sr)
        if resonances:
            res_board = _build_resonance_board(resonances)
            if res_board is not None:
                result = process_with_pedalboard(result, res_board, sr)
                logger.debug(
                    "Applied %d resonance notch cut(s) for stem_type='%s'",
                    len(resonances),
                    stem_type,
                )

    return result
