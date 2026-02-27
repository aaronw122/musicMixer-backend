"""4-band multiband compressor with LR4 (Linkwitz-Riley 4th-order) crossovers.

Splits audio into 4 frequency bands using a cascaded binary tree of
complementary LR4 crossover pairs, compresses each band independently
with pedalboard.Compressor, then recombines. Internal LUFS-based gain
staging ensures level-neutral output.

Crossover frequencies: 150 Hz / 600 Hz / 3000 Hz
Band layout:
    Low      0-150 Hz
    Low-Mid  150-600 Hz
    Mid      600-3000 Hz
    High     3000-20000 Hz
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pyloudnorm
from pedalboard import Compressor, Pedalboard
from scipy.signal import butter, sosfilt

from musicmixer.services.audio_utils import process_with_pedalboard

logger = logging.getLogger(__name__)

LUFS_FLOOR = -40.0  # Below this, audio is effectively silence


@dataclass
class BandSettings:
    """Compression settings for a single frequency band."""

    name: str
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float = 0.0


# Default per-band compression settings
DEFAULT_BAND_SETTINGS: dict[str, BandSettings] = {
    "low": BandSettings(
        name="low",
        threshold_db=-14.0,
        ratio=4.0,
        attack_ms=20.0,
        release_ms=200.0,
    ),
    "low_mid": BandSettings(
        name="low_mid",
        threshold_db=-20.0,
        ratio=3.0,
        attack_ms=10.0,
        release_ms=120.0,
    ),
    "mid": BandSettings(
        name="mid",
        threshold_db=-20.0,
        ratio=2.5,
        attack_ms=5.0,
        release_ms=100.0,
    ),
    "high": BandSettings(
        name="high",
        threshold_db=-20.0,
        ratio=2.0,
        attack_ms=5.0,
        release_ms=120.0,
    ),
}


def lr4_split(
    x: np.ndarray, fc: float, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split signal into low and high bands using LR4 crossover at fc Hz.

    LR4 = two cascaded Butterworth-2 filters. At the crossover frequency,
    each output is at -6 dB and they sum to unity (allpass).

    Uses sosfilt (causal, NOT zero-phase) to preserve the complementary
    magnitude relationship needed for unity reconstruction.

    Args:
        x: Input signal, shape (N,) or (N, channels).
        fc: Crossover frequency in Hz.
        sr: Sample rate in Hz.

    Returns:
        Tuple of (low_band, high_band), same shape as input.
    """
    sos_lp = butter(2, fc, btype="low", fs=sr, output="sos")
    sos_hp = butter(2, fc, btype="high", fs=sr, output="sos")

    # LR4 = two cascaded Butterworth-2 filters
    # Process along axis=0 (samples axis) for both mono and stereo
    y_low = sosfilt(sos_lp, sosfilt(sos_lp, x, axis=0), axis=0)
    y_high = sosfilt(sos_hp, sosfilt(sos_hp, x, axis=0), axis=0)

    return y_low.astype(np.float32), y_high.astype(np.float32)


def split_4_bands(
    audio: np.ndarray, sr: int, crossovers: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    """Split audio into 4 frequency bands using cascaded LR4 crossover tree.

    Topology:
                        Input
                          |
                    split @ crossovers[1] (600 Hz)
                     /         \\
               low-half       high-half
                 |                |
           split @ crossovers[0]  split @ crossovers[2]
           (150 Hz)               (3000 Hz)
            /       \\             /        \\
         Low    Low-Mid        Mid       High

    Args:
        audio: Input signal, shape (N,) or (N, channels).
        sr: Sample rate in Hz.
        crossovers: Three crossover frequencies (low, mid, high).

    Returns:
        Dict mapping band name to band audio array.
    """
    fc_low, fc_mid, fc_high = crossovers

    # First split at the middle crossover
    low_half, high_half = lr4_split(audio, fc_mid, sr)

    # Split low half into Low and Low-Mid
    low_band, low_mid_band = lr4_split(low_half, fc_low, sr)

    # Split high half into Mid and High
    mid_band, high_band = lr4_split(high_half, fc_high, sr)

    return {
        "low": low_band,
        "low_mid": low_mid_band,
        "mid": mid_band,
        "high": high_band,
    }


def _compress_band(
    band_audio: np.ndarray, sr: int, band_settings: BandSettings
) -> np.ndarray:
    """Compress a single frequency band using pedalboard.Compressor.

    Applies makeup gain manually after compression since pedalboard's
    Compressor has no makeup_gain parameter.

    Logs per-band gain reduction for diagnostics.
    """
    # Measure pre-compression peak for gain reduction logging
    pre_peak = float(np.max(np.abs(band_audio))) if len(band_audio) > 0 else 0.0

    board = Pedalboard([
        Compressor(
            threshold_db=band_settings.threshold_db,
            ratio=band_settings.ratio,
            attack_ms=band_settings.attack_ms,
            release_ms=band_settings.release_ms,
        )
    ])

    compressed = process_with_pedalboard(band_audio, board, sr)

    # Apply makeup gain manually
    if band_settings.makeup_db != 0.0:
        compressed = compressed * (10 ** (band_settings.makeup_db / 20.0))

    # Log gain reduction
    post_peak = float(np.max(np.abs(compressed))) if len(compressed) > 0 else 0.0
    if pre_peak > 1e-12 and post_peak > 1e-12:
        gr = 20.0 * np.log10(post_peak / pre_peak)
        logger.debug("band=%s gain_reduction_dB=%.1f", band_settings.name, gr)
    else:
        logger.debug("band=%s gain_reduction_dB=0.0 (silent)", band_settings.name)

    return compressed


def multiband_compress(
    audio: np.ndarray,
    sr: int,
    crossovers: tuple[float, float, float] = (150, 600, 3000),
    settings: dict[str, BandSettings] | None = None,
) -> np.ndarray:
    """Apply 4-band multiband compression with LR4 crossovers.

    1. Measure integrated LUFS of input
    2. Split into 4 bands using cascaded LR4 crossover tree
    3. Compress each band independently
    4. Recombine bands by summation
    5. Restore pre-compression integrated LUFS (internal gain staging)

    Args:
        audio: Input signal, float32, shape (N,) or (N, 2).
        sr: Sample rate in Hz.
        crossovers: Three crossover frequencies. Default (150, 600, 3000).
        settings: Optional dict of band name -> BandSettings overrides.
            Missing bands use DEFAULT_BAND_SETTINGS.

    Returns:
        Compressed float32 array with same shape as input.
    """
    if len(audio) == 0:
        return audio

    band_settings = dict(DEFAULT_BAND_SETTINGS)
    if settings is not None:
        band_settings.update(settings)

    # Step 1: Measure integrated LUFS of input before band splitting
    meter = pyloudnorm.Meter(sr)
    # pyloudnorm requires stereo (N, 2) or mono (N,) — handle both
    if audio.ndim == 1:
        lufs_input_audio = np.column_stack([audio, audio])
    else:
        lufs_input_audio = audio
    input_lufs = meter.integrated_loudness(lufs_input_audio)

    # Step 2: Split into 4 bands
    bands = split_4_bands(audio, sr, crossovers)

    # Step 3: Compress each band
    compressed_bands: dict[str, np.ndarray] = {}
    for band_name, band_audio in bands.items():
        bs = band_settings.get(band_name, DEFAULT_BAND_SETTINGS.get(band_name))
        if bs is None:
            compressed_bands[band_name] = band_audio
            continue
        compressed_bands[band_name] = _compress_band(band_audio, sr, bs)

    # Step 4: Recombine by summation
    output = sum(compressed_bands.values())

    # NaN guard: if compression produced NaN, fall back to uncompressed input
    if np.any(np.isnan(output)):
        logger.error("NaN detected in multiband output, falling back to uncompressed input")
        return audio

    # Step 5: Restore pre-compression integrated LUFS
    if not math.isinf(input_lufs) and input_lufs > LUFS_FLOOR:
        if output.ndim == 1:
            lufs_output_audio = np.column_stack([output, output])
        else:
            lufs_output_audio = output
        output_lufs = meter.integrated_loudness(lufs_output_audio)

        if not math.isinf(output_lufs) and output_lufs > LUFS_FLOOR:
            gain_db = input_lufs - output_lufs
            gain_db = float(np.clip(gain_db, -12.0, 12.0))
            output = output * (10 ** (gain_db / 20.0))
            logger.info(
                "Multiband compress: input=%.1f LUFS, post-compress=%.1f LUFS, "
                "makeup=%.1f dB",
                input_lufs,
                output_lufs,
                gain_db,
            )
        else:
            logger.warning(
                "Multiband compress: output near-silent (%.1f LUFS), "
                "skipping LUFS restoration",
                output_lufs,
            )
    else:
        logger.warning(
            "Multiband compress: input near-silent (%.1f LUFS), "
            "skipping LUFS restoration",
            input_lufs,
        )

    return output.astype(np.float32)
