"""Spectral analysis for adaptive EQ.

Computes 1/3-octave spectral profiles per stem, detects cross-stem frequency
conflicts, and generates adaptive correction parameters (frequency, gain, Q).

All corrections are cuts only (no boosts).  Per-stem anomaly threshold is
+6 dB relative deviation from a flat reference.  Cross-stem conflict cuts are
capped at -3 dB; per-stem anomaly cuts are capped at -4 dB.  Maximum 4
adaptive filters per stem.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import find_peaks, welch

from musicmixer.models import FrequencyConflict, SpectralProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ISO 266 standard 1/3-octave band center frequencies (31 bands, 20 Hz–20 kHz)
ISO_BAND_CENTERS: np.ndarray = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
    20000,
], dtype=np.float64)

# 1/3-octave band edges: each band spans [center / 2^(1/6), center * 2^(1/6)]
_OCTAVE_SIXTH = 2.0 ** (1.0 / 6.0)
_BAND_LOWER = ISO_BAND_CENTERS / _OCTAVE_SIXTH
_BAND_UPPER = ISO_BAND_CENTERS * _OCTAVE_SIXTH

# Thresholds
ANOMALY_THRESHOLD_DB = 6.0    # per-stem: deviation from flat reference to trigger cut
MAX_PER_STEM_CUT_DB = -4.0    # most aggressive per-stem cut
MAX_CROSS_STEM_CUT_DB = -3.0  # most aggressive cross-stem cut
MAX_CORRECTIONS_PER_STEM = 4  # cap on adaptive filters per stem

# Stem priority for conflict resolution (higher = more important, never cut)
# Vocals always win in conflicts; lower-priority stem gets cut.
STEM_PRIORITY: dict[str, int] = {
    "vocals": 100,
    "bass": 80,
    "drums": 60,
    "guitar": 40,
    "piano": 40,
    "other": 20,
}

# Vocal presence range: 2-5 kHz.  In this range, always cut instrumental.
VOCAL_PRESENCE_LOW_HZ = 2000.0
VOCAL_PRESENCE_HIGH_HZ = 5000.0


# ---------------------------------------------------------------------------
# Spectral profile computation
# ---------------------------------------------------------------------------

def compute_spectral_profile(
    audio: np.ndarray,
    sr: int,
    stem_type: str = "other",
) -> SpectralProfile:
    """Compute a 1/3-octave spectral profile for a stem.

    Args:
        audio: Audio signal, shape ``(samples,)`` or ``(samples, channels)``.
            Stereo is downmixed to mono before analysis.
        sr: Sample rate in Hz.
        stem_type: Stem type label (vocals, drums, bass, guitar, piano, other).

    Returns:
        SpectralProfile with 31-band energy and detected peaks.
    """
    # Ensure mono
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1)
    else:
        mono = audio
    mono = mono.astype(np.float64)

    # Handle silence / very short audio
    if len(mono) < 4096:
        # Pad to minimum length for Welch
        mono = np.pad(mono, (0, 4096 - len(mono)), mode="constant")

    # Welch PSD: nperseg=4096, 50% overlap
    nperseg = min(4096, len(mono))
    freqs, psd = welch(mono, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    # Convert to dB (floor at -120 dB to avoid log(0))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))

    # Map to 1/3-octave bands
    band_energies_raw = _map_to_bands(freqs, psd_db)

    # Normalize to relative deviation from flat reference (mean = 0 dB).
    # Only include non-silent bands in the mean so silence floor doesn't
    # pull the reference down.
    active_mask = band_energies_raw > -119.0  # above silence floor
    if np.any(active_mask):
        reference_db = np.mean(band_energies_raw[active_mask])
    else:
        reference_db = -120.0
    band_energies_db = band_energies_raw - reference_db

    # Peak detection on smoothed band energies
    peak_indices, properties = find_peaks(
        band_energies_db,
        prominence=3.0,
    )
    peak_frequencies_hz = ISO_BAND_CENTERS[peak_indices]
    peak_magnitudes_db = band_energies_db[peak_indices]

    return SpectralProfile(
        stem_type=stem_type,
        band_centers_hz=ISO_BAND_CENTERS.copy(),
        band_energies_db=band_energies_db,
        peak_frequencies_hz=peak_frequencies_hz,
        peak_magnitudes_db=peak_magnitudes_db,
    )


def _map_to_bands(freqs: np.ndarray, psd_db: np.ndarray) -> np.ndarray:
    """Map a linear-frequency PSD (in dB) to 31 ISO 1/3-octave bands.

    For each band, compute the mean energy (in linear power) of all PSD bins
    whose center frequency falls within the band edges, then convert back to
    dB.  Bands with no bins get -120 dB (silence floor).
    """
    band_energies = np.full(len(ISO_BAND_CENTERS), -120.0, dtype=np.float64)
    for i in range(len(ISO_BAND_CENTERS)):
        mask = (freqs >= _BAND_LOWER[i]) & (freqs < _BAND_UPPER[i])
        if np.any(mask):
            # Average in linear power domain, then back to dB
            linear_mean = np.mean(10.0 ** (psd_db[mask] / 10.0))
            band_energies[i] = 10.0 * np.log10(max(linear_mean, 1e-12))
    return band_energies


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(
    vocal_profiles: list[SpectralProfile],
    inst_profiles: list[SpectralProfile],
) -> list[FrequencyConflict]:
    """Detect masking conflicts between vocal-source and instrumental-source stems.

    For each pair of (vocal-source stem, instrumental-source stem), checks
    every 1/3-octave band.  A conflict is flagged when both stems exceed
    +6 dB relative deviation (from a flat 0 dB reference) in the same band.

    In the vocal presence range (2-5 kHz), the instrumental stem is always
    the one that gets cut, regardless of priority.

    Args:
        vocal_profiles: SpectralProfiles from the vocal-source song stems.
        inst_profiles: SpectralProfiles from the instrumental-source song stems.

    Returns:
        List of FrequencyConflict instances, sorted by severity (descending).
    """
    conflicts: list[FrequencyConflict] = []

    for vp in vocal_profiles:
        for ip in inst_profiles:
            for band_idx in range(len(ISO_BAND_CENTERS)):
                v_dev = vp.band_energies_db[band_idx]
                i_dev = ip.band_energies_db[band_idx]

                # Both must exceed anomaly threshold in the same band
                if v_dev <= ANOMALY_THRESHOLD_DB or i_dev <= ANOMALY_THRESHOLD_DB:
                    continue

                center_hz = float(ISO_BAND_CENTERS[band_idx])
                severity = min(v_dev, i_dev)

                # Determine which stem to cut
                cut_stem = _resolve_cut_stem(
                    vp.stem_type, ip.stem_type, center_hz
                )

                # Cut proportional to severity, capped
                cut_db = max(MAX_CROSS_STEM_CUT_DB, -(severity - ANOMALY_THRESHOLD_DB) * 0.5)
                cut_db = min(cut_db, 0.0)  # ensure no boost

                q = _compute_q(severity)

                conflicts.append(FrequencyConflict(
                    stem_a=vp.stem_type,
                    stem_b=ip.stem_type,
                    center_hz=center_hz,
                    severity_db=float(severity),
                    recommended_cut_stem=cut_stem,
                    recommended_cut_db=float(cut_db),
                    recommended_q=float(q),
                ))

    # Sort by severity descending
    conflicts.sort(key=lambda c: c.severity_db, reverse=True)
    return conflicts


def _resolve_cut_stem(stem_a: str, stem_b: str, center_hz: float) -> str:
    """Determine which stem should be cut in a conflict.

    - In the vocal presence range (2-5 kHz), always cut the instrumental stem
      (stem_b), never any vocal-source stem.
    - Otherwise, cut the lower-priority stem.
    """
    in_presence_range = (
        VOCAL_PRESENCE_LOW_HZ <= center_hz <= VOCAL_PRESENCE_HIGH_HZ
    )
    if in_presence_range:
        return stem_b  # instrumental-source always loses in the vocal presence range

    prio_a = STEM_PRIORITY.get(stem_a, 20)
    prio_b = STEM_PRIORITY.get(stem_b, 20)

    # Cut the lower-priority stem; stem_b (instrumental) is cut on ties
    if prio_a >= prio_b:
        return stem_b
    return stem_a


def _compute_q(deviation_db: float) -> float:
    """Compute filter Q from deviation magnitude.

    Wider Q for broad problems, narrower for localized peaks.
    Clamped to [1.5, 3.0].
    """
    return max(1.5, min(3.0, deviation_db / 4.0))


# ---------------------------------------------------------------------------
# Adaptive correction generation
# ---------------------------------------------------------------------------

def compute_adaptive_corrections(
    conflicts: list[FrequencyConflict],
    vocal_profiles: list[SpectralProfile],
    inst_profiles: list[SpectralProfile],
) -> tuple[dict[str, list[tuple[float, float, float]]], dict[str, list[tuple[float, float, float]]]]:
    """Generate per-stem adaptive EQ corrections.

    Merges two sources of corrections:
    1. **Per-stem anomaly corrections** — bands where a single stem deviates
       more than +6 dB from flat reference, corrected at half the excess,
       capped at -4 dB.
    2. **Cross-stem conflict corrections** — from ``detect_conflicts()``
       output, capped at -3 dB.

    Each correction is a ``(frequency_hz, gain_db, q)`` tuple.  Gain is
    always negative (cuts only), clamped to [-4, 0] dB.  Maximum 4
    corrections per stem; the most severe are kept.

    Args:
        conflicts: Output of ``detect_conflicts()``.
        vocal_profiles: Vocal-source stem profiles.
        inst_profiles: Instrumental-source stem profiles.

    Returns:
        ``(vocal_corrections, inst_corrections)`` — each a dict mapping
        stem_type to a list of ``(freq_hz, gain_db, q)`` tuples.
    """
    vocal_corrections: dict[str, list[tuple[float, float, float]]] = {}
    inst_corrections: dict[str, list[tuple[float, float, float]]] = {}

    # 1. Per-stem anomaly corrections
    for profile in vocal_profiles:
        corrections = _anomaly_corrections(profile)
        if corrections:
            vocal_corrections.setdefault(profile.stem_type, []).extend(corrections)

    for profile in inst_profiles:
        corrections = _anomaly_corrections(profile)
        if corrections:
            inst_corrections.setdefault(profile.stem_type, []).extend(corrections)

    # 2. Cross-stem conflict corrections
    # Route based on positional convention: detect_conflicts() always sets
    # stem_a = vocal-source, stem_b = instrumental-source.  Using set
    # membership on stem type strings would misroute stems that share type
    # names across sources (e.g., both songs have an "other" stem).
    for conflict in conflicts:
        cut_stem = conflict.recommended_cut_stem
        correction = (
            conflict.center_hz,
            conflict.recommended_cut_db,
            conflict.recommended_q,
        )
        if cut_stem == conflict.stem_a:
            vocal_corrections.setdefault(cut_stem, []).append(correction)
        else:
            inst_corrections.setdefault(cut_stem, []).append(correction)

    # 3. Cap at MAX_CORRECTIONS_PER_STEM, keeping most severe
    _cap_corrections(vocal_corrections)
    _cap_corrections(inst_corrections)

    return vocal_corrections, inst_corrections


def _anomaly_corrections(
    profile: SpectralProfile,
) -> list[tuple[float, float, float]]:
    """Generate per-stem anomaly corrections for bands exceeding threshold.

    Reference envelope is flat (0 dB) for v1, so deviation = band_energy.
    """
    corrections: list[tuple[float, float, float]] = []

    for i, energy_db in enumerate(profile.band_energies_db):
        deviation = energy_db  # flat reference: deviation = absolute value
        if deviation <= ANOMALY_THRESHOLD_DB:
            continue

        # Correct half the excess, floored at max cut
        gain_db = max(MAX_PER_STEM_CUT_DB, -(deviation - ANOMALY_THRESHOLD_DB) * 0.5)
        gain_db = min(gain_db, 0.0)  # ensure no boost
        q = _compute_q(deviation)
        freq = float(ISO_BAND_CENTERS[i])

        corrections.append((freq, gain_db, q))

    return corrections


def _cap_corrections(
    corrections_dict: dict[str, list[tuple[float, float, float]]],
) -> None:
    """Cap each stem to MAX_CORRECTIONS_PER_STEM, keeping the most severe.

    Deduplicates corrections at the same frequency band (keeps the stronger cut).
    Clamps all gains to [-4, 0] dB.
    """
    for stem_type in list(corrections_dict.keys()):
        corrections = corrections_dict[stem_type]

        # Deduplicate by frequency: group by nearest band center, keep strongest cut
        by_freq: dict[float, tuple[float, float, float]] = {}
        for freq, gain, q in corrections:
            if freq in by_freq:
                existing_gain = by_freq[freq][1]
                if gain < existing_gain:  # more negative = stronger cut
                    by_freq[freq] = (freq, gain, q)
            else:
                by_freq[freq] = (freq, gain, q)

        # Sort by gain (most negative first = most severe)
        deduped = sorted(by_freq.values(), key=lambda c: c[1])

        # Cap count
        deduped = deduped[:MAX_CORRECTIONS_PER_STEM]

        # Clamp gains to [-4, 0]
        clamped = [
            (freq, max(-4.0, min(0.0, gain)), q)
            for freq, gain, q in deduped
        ]

        corrections_dict[stem_type] = clamped
