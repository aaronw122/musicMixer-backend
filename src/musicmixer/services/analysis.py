"""Audio analysis: BPM detection, song structure analysis, and cross-song reconciliation.

Step 2 of Day 2 pipeline + Step 3.5 of Day 3 pipeline. Provides:
- analyze_audio(): BPM, beat positions, duration, confidence for a single track
- reconcile_bpm(): Cross-song BPM reconciliation using expanded interpretation matrix
- analyze_stems(): Full song structure analysis (energy, vocals, sections)
- detect_key(): Key detection with essentia primary / librosa fallback
- detect_sections(): 3-stage section boundary detection, labeling, and merging
- compute_relationships(): Cross-song comparison metrics
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from musicmixer.models import (
    VOCAL_SOURCE,
    AudioMetadata,
    CrossSongRelationships,
    EnergyBuckets,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
)
from musicmixer.services.tempo import compute_stretch_pct

logger = logging.getLogger(__name__)

# Try to import essentia; fall back to librosa-only key detection if unavailable.
try:
    import essentia.standard as es
    _HAS_ESSENTIA = True
except ImportError:
    _HAS_ESSENTIA = False

# ---------------------------------------------------------------------------
# Constants (spec section 3)
# ---------------------------------------------------------------------------
SMOOTHING_WINDOW: int = 4                # bars
BOUNDARY_THRESHOLD_MULT: float = 3.0     # x median
BOUNDARY_THRESHOLD_FLOOR: float = 0.05
PHRASE_GRID: int = 4                      # bars
MIN_SECTION_BARS: int = 4
BUILD_RATIO: float = 1.5
BUILD_EPSILON: float = 0.01
DROP_RATIO: float = 1.5
BUCKET_NOISE_FLOOR: float = 0.02         # for bucket classification
PRE_NORM_NOISE_FLOOR: float = 0.001      # -60 dBFS pre-normalization filter
VOCAL_ONSET_RATIO: float = 0.15          # 15% of stem peak
VOCAL_SUSTAIN_RATIO: float = 0.08        # 8% of stem peak
VOCAL_MIN_DURATION: int = 2              # bars
ANALYSIS_SR: int = 22050                  # sample rate for stem loading
KEY_DETECTION_SR: int = 44100             # sample rate for essentia key detection
STEM_NAMES: list[str] = ["drums", "bass", "guitar", "piano", "vocals", "other"]


def analyze_audio(audio_path: Path) -> AudioMetadata:
    """Analyze audio file for BPM, beat positions, duration, and mean RMS.

    Loads at 22050 Hz (sufficient for BPM detection, saves memory).
    Beat positions are stored in frame units.
    mean_rms is computed from the original mix audio (NOT summed stems).
    """
    y, sr = librosa.load(str(audio_path), sr=22050)

    # BPM detection
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    # librosa may return tempo as array in newer versions
    bpm = float(np.atleast_1d(tempo)[0])

    duration = float(librosa.get_duration(y=y, sr=sr))
    total_beats = round(bpm * duration / 60 / 4) * 4  # Round to nearest bar

    # BPM confidence via tempogram peak sharpness
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    # Peak sharpness: ratio of max to mean in the tempo range
    tempo_profile = np.mean(tempogram, axis=1)
    if np.max(tempo_profile) > 0:
        bpm_confidence = float(np.max(tempo_profile) / np.mean(tempo_profile))
        bpm_confidence = min(bpm_confidence / 10.0, 1.0)  # Normalize to 0-1 range
    else:
        bpm_confidence = 0.0

    # Mean RMS from original mix audio (spec section 8)
    mean_rms = float(np.sqrt(np.mean(y ** 2)))

    logger.info(
        "Audio analysis complete: path=%s bpm=%.1f confidence=%.2f duration=%.1fs beats=%d rms=%.4f",
        audio_path.name,
        bpm,
        bpm_confidence,
        duration,
        total_beats,
        mean_rms,
    )

    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=max(total_beats, 4),  # At least 1 bar
        mean_rms=mean_rms,
    )


def analyze_audio_full(audio_path: Path) -> AudioMetadata:
    """Single-load audio analysis: BPM, key, modulation in one pass.

    Loads the audio file ONCE at 44100 Hz (needed for essentia key detection),
    runs key detection and modulation detection on the high-rate data, then
    downsamples to 22050 Hz in-memory for BPM / beat detection.

    This replaces three separate calls (``analyze_audio``, ``detect_key``,
    ``detect_modulation``) that each loaded the file independently.

    Returns an :class:`AudioMetadata` with key/scale/confidence/has_modulation
    already populated.
    """
    # --- 1. Load audio once at 44100 Hz (mono, float32) ---
    y_44k, _ = librosa.load(str(audio_path), sr=KEY_DETECTION_SR, mono=True)

    # --- 2. Key detection (uses 44100 Hz data) ---
    key: str | None = None
    scale: str | None = None
    key_confidence: float | None = None
    try:
        key, scale, key_confidence = detect_key(
            audio_path, audio_44k=y_44k,
        )
    except Exception as exc:
        logger.warning("Key detection failed for %s: %s", audio_path.name, exc)

    # --- 3. Modulation detection (uses 44100 Hz data) ---
    has_modulation = detect_modulation(audio_path, audio_44k=y_44k)

    # --- 4. Downsample to 22050 Hz in-memory for BPM detection ---
    y_22k = librosa.resample(y_44k, orig_sr=KEY_DETECTION_SR, target_sr=ANALYSIS_SR)

    # --- 5. BPM / beat detection (identical logic to analyze_audio) ---
    sr = ANALYSIS_SR
    tempo, beat_frames = librosa.beat.beat_track(y=y_22k, sr=sr, units="frames")
    bpm = float(np.atleast_1d(tempo)[0])

    duration = float(librosa.get_duration(y=y_22k, sr=sr))
    total_beats = round(bpm * duration / 60 / 4) * 4  # Round to nearest bar

    # BPM confidence via tempogram peak sharpness
    tempogram = librosa.feature.tempogram(y=y_22k, sr=sr)
    tempo_profile = np.mean(tempogram, axis=1)
    if np.max(tempo_profile) > 0:
        bpm_confidence = float(np.max(tempo_profile) / np.mean(tempo_profile))
        bpm_confidence = min(bpm_confidence / 10.0, 1.0)
    else:
        bpm_confidence = 0.0

    # Mean RMS from original mix audio (spec section 8)
    mean_rms = float(np.sqrt(np.mean(y_22k ** 2)))

    logger.info(
        "Full audio analysis complete: path=%s bpm=%.1f confidence=%.2f "
        "duration=%.1fs beats=%d rms=%.4f key=%s %s (key_conf=%.2f, modulation=%s)",
        audio_path.name,
        bpm,
        bpm_confidence,
        duration,
        total_beats,
        mean_rms,
        key,
        scale,
        key_confidence if key_confidence is not None else 0.0,
        has_modulation,
    )

    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=max(total_beats, 4),
        mean_rms=mean_rms,
        key=key,
        scale=scale,
        key_confidence=key_confidence,
        has_modulation=has_modulation,
    )


def reconcile_bpm(
    meta_a: AudioMetadata, meta_b: AudioMetadata
) -> tuple[AudioMetadata, AudioMetadata]:
    """Cross-song BPM reconciliation using expanded interpretation matrix.

    For each song, generates plausible BPM interpretations (original, halved,
    doubled, 3/2, 2/3), filters to 70-180 BPM range, and selects the pair
    with the smallest combined score (percentage gap + transformation penalties).

    Returns COPIES of metadata with reconciled BPMs -- originals are not mutated.
    """

    def interpretations(bpm: float) -> dict[str, tuple[float, float]]:
        candidates = {
            "original": bpm,
            "halved": bpm / 2,
            "doubled": bpm * 2,
            "3/2": bpm * 3 / 2,
            "2/3": bpm * 2 / 3,
        }
        penalties = {
            "original": 0.0,
            "halved": 0.05,
            "doubled": 0.05,
            "3/2": 0.15,
            "2/3": 0.15,
        }
        return {
            k: (v, penalties[k]) for k, v in candidates.items() if 70 <= v <= 180
        }

    interps_a = interpretations(meta_a.bpm)
    interps_b = interpretations(meta_b.bpm)

    best_score = float("inf")
    best_pair = (meta_a.bpm, meta_b.bpm)
    best_labels = ("original", "original")

    for label_a, (bpm_a, pen_a) in interps_a.items():
        for label_b, (bpm_b, pen_b) in interps_b.items():
            gap = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)
            score = gap + pen_a + pen_b
            if score < best_score:
                best_score = score
                best_pair = (bpm_a, bpm_b)
                best_labels = (label_a, label_b)

    logger.info(
        "BPM reconciliation: A=%.1f->%.1f (%s), B=%.1f->%.1f (%s), score=%.4f",
        meta_a.bpm,
        best_pair[0],
        best_labels[0],
        meta_b.bpm,
        best_pair[1],
        best_labels[1],
        best_score,
    )

    new_a = replace(
        meta_a,
        bpm=best_pair[0],
        beat_frames=_transform_beat_frames(meta_a.beat_frames, best_labels[0]),
        total_beats=_transform_total_beats(meta_a.total_beats, best_labels[0]),
    )
    new_b = replace(
        meta_b,
        bpm=best_pair[1],
        beat_frames=_transform_beat_frames(meta_b.beat_frames, best_labels[1]),
        total_beats=_transform_total_beats(meta_b.total_beats, best_labels[1]),
    )
    return new_a, new_b


def _transform_beat_frames(
    beat_frames: np.ndarray, interpretation: str
) -> np.ndarray:
    """Transform beat_frames to match a reconciled BPM interpretation.

    When BPM is halved, the beat grid has twice as many entries as the
    reconciled BPM implies -- take every other beat.
    When BPM is doubled, interpolate midpoints between consecutive beats.
    For 3/2 and 2/3 multipliers, leave as-is (re-detected post-stretch).
    """
    if interpretation == "original" or len(beat_frames) < 2:
        return beat_frames

    if interpretation == "halved":
        # BPM halved -> half as many beats -> take every other frame
        return beat_frames[::2]

    if interpretation == "doubled":
        # BPM doubled -> twice as many beats -> interpolate midpoints
        midpoints = (beat_frames[:-1] + beat_frames[1:]) // 2
        # Interleave: original[0], mid[0], original[1], mid[1], ...
        interleaved = np.empty(len(beat_frames) + len(midpoints), dtype=beat_frames.dtype)
        interleaved[0::2] = beat_frames
        interleaved[1::2] = midpoints
        return interleaved

    # "3/2" or "2/3": leave beat_frames as-is (re-detected post-stretch)
    return beat_frames


def _transform_total_beats(total_beats: int, interpretation: str) -> int:
    """Adjust total_beats count to match the reconciled BPM interpretation."""
    if interpretation == "halved":
        return max(total_beats // 2, 4)  # At least 1 bar
    if interpretation == "doubled":
        return total_beats * 2
    # "original", "3/2", "2/3": unchanged
    return total_beats


# ===================================================================
# Song Structure Analysis (Step 3.5)
# ===================================================================


# ---------------------------------------------------------------------------
# 2.0 Stem loading & bar grid
# ---------------------------------------------------------------------------

def _compute_bar_boundaries(beat_frames: np.ndarray, audio_length: int) -> np.ndarray:
    """Compute bar boundaries from beat frames (1 bar = 4 beats).

    Returns frame indices marking the start of each bar. Partial final bar
    is discarded if <4 beats remain, else last bar extends to audio end.
    """
    if len(beat_frames) < 4:
        # Too few beats for even one bar; return the whole audio as one bar
        return np.array([0, audio_length], dtype=np.intp)

    # Take every 4th beat as a bar boundary
    bar_starts = beat_frames[::4]

    # Handle partial final bar
    remaining_beats = len(beat_frames) - (len(bar_starts) - 1) * 4
    # Remaining beats after last bar start (those past last included bar start)
    beats_after_last = len(beat_frames) % 4
    if beats_after_last == 0:
        # Perfect alignment: extend last bar to audio end
        bar_boundaries = np.append(bar_starts, audio_length)
    elif beats_after_last >= 4:
        # Shouldn't happen with mod 4, but safety net
        bar_boundaries = np.append(bar_starts, audio_length)
    else:
        # Partial final bar (<4 beats): discard it, last bar extends to audio end
        bar_boundaries = np.append(bar_starts, audio_length)

    return bar_boundaries


def _compute_bar_rms(audio: np.ndarray, bar_boundaries: np.ndarray) -> np.ndarray:
    """Compute RMS energy for each bar defined by bar_boundaries.

    Args:
        audio: 1D audio signal.
        bar_boundaries: Frame indices marking bar edges (n_bars + 1 entries).

    Returns:
        Array of RMS values, one per bar.
    """
    n_bars = len(bar_boundaries) - 1
    if n_bars <= 0:
        return np.array([], dtype=np.float64)

    rms = np.empty(n_bars, dtype=np.float64)
    for i in range(n_bars):
        start = int(bar_boundaries[i])
        end = int(bar_boundaries[i + 1])
        segment = audio[start:end]
        if len(segment) == 0:
            rms[i] = 0.0
        else:
            rms[i] = float(np.sqrt(np.mean(segment ** 2)))
    return rms


def _load_stems(
    stem_paths: dict[str, Path],
    sr: int = ANALYSIS_SR,
) -> dict[str, np.ndarray]:
    """Load stem audio files at the given sample rate.

    Always returns 6 stems (STEM_NAMES). Missing stems are filled with zeros
    using the length of the first available stem.
    """
    loaded: dict[str, np.ndarray] = {}
    ref_length: int = 0

    for name in STEM_NAMES:
        path = stem_paths.get(name)
        if path is not None and path.exists():
            y, _ = librosa.load(str(path), sr=sr, mono=True)
            loaded[name] = y
            if ref_length == 0:
                ref_length = len(y)
        else:
            loaded[name] = None  # type: ignore[assignment]

    # Fill missing stems with zeros of reference length
    if ref_length == 0:
        ref_length = 1  # safety: at least 1 sample
    for name in STEM_NAMES:
        if loaded[name] is None:
            loaded[name] = np.zeros(ref_length, dtype=np.float32)

    return loaded


# ---------------------------------------------------------------------------
# 2.1 Adaptive percentile bucketing (normalization pipeline)
# ---------------------------------------------------------------------------

def compute_adaptive_buckets(
    bar_rms_per_stem: dict[str, np.ndarray],
) -> tuple[np.ndarray, EnergyBuckets]:
    """Compute normalized combined energy and adaptive bucket thresholds.

    1. Apply pre-normalization noise floor filter (0.001 / -60 dBFS).
    2. Compute combined energy as equal-weighted sum of all stems.
    3. Normalize to p99 = 1.0.
    4. Compute bucket thresholds from active bars (above 0.02 noise floor).

    Returns:
        (combined_energy, bucket_thresholds)
    """
    # Stack all stem RMS values (n_stems x n_bars)
    stem_names = list(bar_rms_per_stem.keys())
    if not stem_names:
        empty = np.array([], dtype=np.float64)
        return empty, EnergyBuckets(noise_floor=BUCKET_NOISE_FLOOR, p10=0.0, p50=0.0, p85=0.0)

    n_bars = len(bar_rms_per_stem[stem_names[0]])
    if n_bars == 0:
        empty = np.array([], dtype=np.float64)
        return empty, EnergyBuckets(noise_floor=BUCKET_NOISE_FLOOR, p10=0.0, p50=0.0, p85=0.0)

    # Step 1: Pre-normalization noise floor filter (0.001 / -60 dBFS)
    # Zero out bars below the pre-normalization noise floor in each stem
    filtered_rms: dict[str, np.ndarray] = {}
    for name in stem_names:
        rms = bar_rms_per_stem[name].copy()
        rms[rms < PRE_NORM_NOISE_FLOOR] = 0.0
        filtered_rms[name] = rms

    # Step 2: Combined energy = equal-weighted sum of all stems
    combined = np.zeros(n_bars, dtype=np.float64)
    for name in stem_names:
        combined += filtered_rms[name]

    # Step 3: Normalize to p99 = 1.0
    if len(combined) > 0 and np.any(combined > 0):
        p99 = float(np.percentile(combined[combined > 0], 99))
        if p99 > 0:
            combined = combined / p99
        # Clip to prevent extreme outliers
        combined = np.clip(combined, 0.0, None)

    # Step 4: Compute adaptive bucket thresholds on active bars (above bucket noise floor)
    active_mask = combined >= BUCKET_NOISE_FLOOR
    active_bars = combined[active_mask]

    if len(active_bars) == 0:
        return combined, EnergyBuckets(
            noise_floor=BUCKET_NOISE_FLOOR, p10=0.0, p50=0.0, p85=0.0,
        )

    p10 = float(np.percentile(active_bars, 10))
    p50 = float(np.percentile(active_bars, 50))
    p85 = float(np.percentile(active_bars, 85))

    return combined, EnergyBuckets(
        noise_floor=BUCKET_NOISE_FLOOR, p10=p10, p50=p50, p85=p85,
    )


def classify_energy(value: float, buckets: EnergyBuckets) -> str:
    """Classify an energy value into a bucket label.

    silent=<noise_floor, low=<p10, medium=p10-p50, high=p50-p85, peak=>p85.
    """
    if value < buckets.noise_floor:
        return "silent"
    if value < buckets.p10:
        return "low"
    if value < buckets.p50:
        return "medium"
    if value < buckets.p85:
        return "high"
    return "peak"


# ---------------------------------------------------------------------------
# 2.2 Dual-threshold vocal hysteresis
# ---------------------------------------------------------------------------

def detect_vocal_activity(vocal_rms: np.ndarray) -> np.ndarray:
    """Detect vocal activity using dual-threshold hysteresis.

    Onset: bar RMS > 15% of stem peak -> start active region.
    Sustain: stay active while bar RMS > 8% of stem peak.
    Min duration: 2 bars (discard shorter active regions).

    Returns:
        Boolean array of per-bar vocal activity.
    """
    n_bars = len(vocal_rms)
    if n_bars == 0:
        return np.array([], dtype=bool)

    stem_peak = float(np.max(vocal_rms))
    if stem_peak <= 0:
        return np.zeros(n_bars, dtype=bool)

    onset_threshold = stem_peak * VOCAL_ONSET_RATIO
    sustain_threshold = stem_peak * VOCAL_SUSTAIN_RATIO

    # Hysteresis pass
    active = np.zeros(n_bars, dtype=bool)
    in_region = False

    for i in range(n_bars):
        if not in_region:
            if vocal_rms[i] > onset_threshold:
                in_region = True
                active[i] = True
        else:
            if vocal_rms[i] > sustain_threshold:
                active[i] = True
            else:
                in_region = False

    # Enforce minimum duration: discard active regions shorter than VOCAL_MIN_DURATION
    result = np.zeros(n_bars, dtype=bool)
    region_start: Optional[int] = None

    for i in range(n_bars):
        if active[i]:
            if region_start is None:
                region_start = i
        else:
            if region_start is not None:
                region_len = i - region_start
                if region_len >= VOCAL_MIN_DURATION:
                    result[region_start:i] = True
                region_start = None

    # Handle region that extends to end
    if region_start is not None:
        region_len = n_bars - region_start
        if region_len >= VOCAL_MIN_DURATION:
            result[region_start:n_bars] = True

    return result


# ---------------------------------------------------------------------------
# 2.3 Vocal gap detection
# ---------------------------------------------------------------------------

def detect_vocal_gaps(vocal_active: np.ndarray) -> list[VocalGap]:
    """Find contiguous runs of 2+ bars where vocal_active == False.

    Returns list of VocalGap objects.
    """
    n_bars = len(vocal_active)
    if n_bars == 0:
        return []

    gaps: list[VocalGap] = []
    gap_start: Optional[int] = None

    for i in range(n_bars):
        if not vocal_active[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len >= 2:
                    gaps.append(VocalGap(
                        start_bar=gap_start,
                        end_bar=i - 1,
                        length_bars=gap_len,
                    ))
                gap_start = None

    # Handle gap that extends to end
    if gap_start is not None:
        gap_len = n_bars - gap_start
        if gap_len >= 2:
            gaps.append(VocalGap(
                start_bar=gap_start,
                end_bar=n_bars - 1,
                length_bars=gap_len,
            ))

    return gaps


# ---------------------------------------------------------------------------
# 2.5 Key detection + modulation
# ---------------------------------------------------------------------------

def _detect_key_essentia(
    audio_path: Path,
    audio_44k: np.ndarray | None = None,
) -> tuple[str, str, float]:
    """Detect key using essentia KeyExtractor at 44.1 kHz.

    If *audio_44k* is provided (mono float32 at 44100 Hz), it is used directly
    instead of loading from disk -- this avoids a redundant file read when the
    caller has already loaded the audio.

    Returns (key, scale, confidence).
    Raises ImportError or RuntimeError if essentia is unavailable or fails.
    """
    if not _HAS_ESSENTIA:
        raise ImportError("essentia not available")

    if audio_44k is not None:
        audio = audio_44k
    else:
        loader = es.MonoLoader(filename=str(audio_path), sampleRate=KEY_DETECTION_SR)
        audio = loader()
    key_extractor = es.KeyExtractor()
    key, scale, confidence = key_extractor(audio)
    return str(key), str(scale), float(confidence)


def _detect_key_librosa(
    audio_path: Path,
    audio_22k: np.ndarray | None = None,
) -> tuple[str, str, float]:
    """Detect key using librosa chroma_cqt (fallback).

    If *audio_22k* is provided (mono float32 at 22050 Hz), it is used directly
    instead of loading from disk.

    Returns (key, scale, confidence). Confidence is derived from
    the sharpness of the chroma profile peak.
    """
    if audio_22k is not None:
        y = audio_22k
        sr = 22050
    else:
        y, sr = librosa.load(str(audio_path), sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Average chroma profile across time
    chroma_profile = np.mean(chroma, axis=1)

    # Key names (C, C#, D, ...)
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Find the dominant pitch class
    dominant_idx = int(np.argmax(chroma_profile))
    key = key_names[dominant_idx]

    # Determine major/minor by comparing major and minor profiles
    # Krumhansl-Kessler profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    # Correlate with rotated profiles
    best_corr_major = -1.0
    best_corr_minor = -1.0
    for shift in range(12):
        rotated_major = np.roll(major_profile, shift)
        rotated_minor = np.roll(minor_profile, shift)
        corr_major = float(np.corrcoef(chroma_profile, rotated_major)[0, 1])
        corr_minor = float(np.corrcoef(chroma_profile, rotated_minor)[0, 1])
        if corr_major > best_corr_major:
            best_corr_major = corr_major
        if corr_minor > best_corr_minor:
            best_corr_minor = corr_minor

    if best_corr_minor > best_corr_major:
        scale = "minor"
    else:
        scale = "major"

    # Confidence: ratio of peak to mean in chroma profile
    if np.mean(chroma_profile) > 0:
        confidence = float(np.max(chroma_profile) / np.mean(chroma_profile)) / 2.0
        confidence = min(confidence, 1.0)
    else:
        confidence = 0.0

    return key, scale, confidence


def detect_key(
    audio_path: Path,
    audio_44k: np.ndarray | None = None,
    audio_22k: np.ndarray | None = None,
) -> tuple[str, str, float]:
    """Detect key of an audio file.

    Primary: essentia KeyExtractor at 44.1 kHz.
    Fallback: librosa chroma_cqt.
    If essentia fails, falls back to librosa silently.

    Optional pre-loaded audio arrays (*audio_44k* at 44100 Hz for essentia,
    *audio_22k* at 22050 Hz for the librosa fallback) avoid redundant disk I/O
    when the caller has already loaded the file.

    Returns (key, scale, confidence).
    """
    try:
        return _detect_key_essentia(audio_path, audio_44k=audio_44k)
    except Exception:
        logger.debug("Essentia key detection failed, falling back to librosa")
        return _detect_key_librosa(audio_path, audio_22k=audio_22k)


def detect_modulation(
    audio_path: Path,
    audio_44k: np.ndarray | None = None,
    audio_22k: np.ndarray | None = None,
) -> bool:
    """Detect key modulation by comparing first 60% and last 40% of the audio.

    Optional pre-loaded audio arrays (*audio_44k* at 44100 Hz for essentia,
    *audio_22k* at 22050 Hz for the librosa fallback) avoid redundant disk I/O.

    Returns True if the keys differ between the two segments.
    """
    try:
        if _HAS_ESSENTIA:
            if audio_44k is not None:
                audio = audio_44k
            else:
                loader = es.MonoLoader(filename=str(audio_path), sampleRate=KEY_DETECTION_SR)
                audio = loader()
            split_point = int(len(audio) * 0.6)

            key_extractor = es.KeyExtractor()
            key_first, scale_first, _ = key_extractor(audio[:split_point])
            key_last, scale_last, _ = key_extractor(audio[split_point:])

            return key_first != key_last or scale_first != scale_last
        else:
            # Librosa fallback for modulation detection
            if audio_22k is not None:
                y = audio_22k
                sr = 22050
            else:
                y, sr = librosa.load(str(audio_path), sr=22050)
            split_point = int(len(y) * 0.6)

            # Compare chroma profiles of first 60% and last 40%
            chroma_first = np.mean(librosa.feature.chroma_cqt(y=y[:split_point], sr=sr), axis=1)
            chroma_last = np.mean(librosa.feature.chroma_cqt(y=y[split_point:], sr=sr), axis=1)

            key_first = int(np.argmax(chroma_first))
            key_last = int(np.argmax(chroma_last))

            return key_first != key_last
    except Exception:
        logger.debug("Modulation detection failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# 2.4 Section detection (3-stage)
# ---------------------------------------------------------------------------

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average with the given window size.

    Uses 'same' mode (output same length as input) via numpy convolution.
    """
    if len(arr) == 0 or window <= 0:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def detect_boundaries(
    bar_rms_per_stem: dict[str, np.ndarray],
    combined_energy: np.ndarray,
) -> np.ndarray:
    """Stage 1: Derivative boundary detection with per-stem max pool.

    1. Smooth combined energy (4-bar window).
    2. Compute absolute derivative of smoothed combined energy.
    3. For each stem, smooth and take absolute derivative.
    4. Take element-wise max across all per-stem derivatives.
    5. change_signal = max(combined_deriv, max_stem_deriv).
    6. Threshold = max(median(change_signal) * 3.0, 0.05).
    7. Find peaks above threshold, minimum 4 bars apart.

    Returns array of bar indices where boundaries occur.
    """
    n_bars = len(combined_energy)
    if n_bars < 2:
        return np.array([], dtype=np.intp)

    # Smooth combined energy
    smoothed = _moving_average(combined_energy, SMOOTHING_WINDOW)

    # Absolute derivative of combined energy
    deriv = np.abs(np.diff(smoothed))

    # Per-stem derivatives with max pool
    stem_derivs: list[np.ndarray] = []
    for name in bar_rms_per_stem:
        stem_smoothed = _moving_average(bar_rms_per_stem[name], SMOOTHING_WINDOW)
        stem_deriv = np.abs(np.diff(stem_smoothed))
        stem_derivs.append(stem_deriv)

    if stem_derivs:
        stacked = np.stack(stem_derivs, axis=0)
        max_stem_deriv = np.max(stacked, axis=0)
    else:
        max_stem_deriv = np.zeros_like(deriv)

    # Change signal: element-wise max of combined deriv and max stem deriv
    change_signal = np.maximum(deriv, max_stem_deriv)

    if len(change_signal) == 0:
        return np.array([], dtype=np.intp)

    # Threshold
    median_val = float(np.median(change_signal))
    threshold = max(median_val * BOUNDARY_THRESHOLD_MULT, BOUNDARY_THRESHOLD_FLOOR)

    # Find peaks above threshold, minimum 4 bars apart
    peaks, _ = find_peaks(change_signal, height=threshold, distance=MIN_SECTION_BARS)

    # Offset by 1 since diff shifts indices (boundary is at bar after the derivative)
    boundaries = peaks + 1

    # Filter out any boundaries beyond valid range
    boundaries = boundaries[boundaries < n_bars]

    return boundaries.astype(np.intp)


def quantize_to_phrases(
    boundaries: np.ndarray,
    total_bars: int,
) -> np.ndarray:
    """Stage 1b: Snap boundaries to nearest 4-bar grid, deduplicate, remove <4 bar segments.

    Args:
        boundaries: Raw boundary bar indices.
        total_bars: Total number of bars in the song.

    Returns:
        Sorted, deduplicated array of quantized boundary positions.
    """
    if len(boundaries) == 0:
        return np.array([], dtype=np.intp)

    # Snap to nearest 4-bar grid
    quantized = np.round(boundaries / PHRASE_GRID) * PHRASE_GRID
    quantized = quantized.astype(np.intp)

    # Clamp to valid range
    quantized = np.clip(quantized, 0, total_bars)

    # Deduplicate and sort
    quantized = np.unique(quantized)

    # Remove boundary at 0 (that's always the start) and at total_bars (that's the end)
    quantized = quantized[(quantized > 0) & (quantized < total_bars)]

    # Remove boundaries that create segments < 4 bars
    if len(quantized) == 0:
        return quantized

    # Build full segment list: [0, b1, b2, ..., total_bars]
    all_points = np.concatenate([[0], quantized, [total_bars]])
    segment_lengths = np.diff(all_points)

    # Filter: keep boundaries that don't create too-short segments
    # Iteratively remove boundaries that create short segments
    valid = list(quantized)
    changed = True
    while changed:
        changed = False
        points = [0] + valid + [total_bars]
        new_valid: list[int] = []
        for i, b in enumerate(valid):
            idx = i + 1  # position in points list
            left_len = points[idx] - points[idx - 1]
            right_len = points[idx + 1] - points[idx]
            if left_len >= MIN_SECTION_BARS and right_len >= MIN_SECTION_BARS:
                new_valid.append(b)
            else:
                changed = True
        valid = new_valid

    return np.array(valid, dtype=np.intp)


def _segment_vocal_status(
    vocal_active: np.ndarray,
    start_bar: int,
    end_bar: int,
) -> str:
    """Determine vocal status for a segment.

    Returns "vox:yes", "vox:no", or "vox:fading".
    """
    segment = vocal_active[start_bar:end_bar]
    if len(segment) == 0:
        return "vox:no"

    active_ratio = float(np.mean(segment))
    if active_ratio > 0.5:
        # Check for fading: is the last third less active than the first third?
        third = max(1, len(segment) // 3)
        first_third = float(np.mean(segment[:third]))
        last_third = float(np.mean(segment[-third:]))
        if first_third > 0.5 and last_third < 0.5:
            return "vox:fading"
        return "vox:yes"
    return "vox:no"


def _compute_density(
    bar_rms_per_stem: dict[str, np.ndarray],
    start_bar: int,
    end_bar: int,
) -> str:
    """Compute arrangement density for a segment.

    Stem "active" = segment mean > p25 of that stem's own bars.
    sparse=0-2 active, mid=3-4, full=5-6 (none at peak),
    full+extra=5-6 with at least one stem's segment-max > p90 of own bars.
    """
    active_count = 0
    has_peak_stem = False

    for name, rms in bar_rms_per_stem.items():
        segment_rms = rms[start_bar:end_bar]
        if len(segment_rms) == 0:
            continue

        # p25 of the stem's own bars (all bars, not just segment)
        all_bars = rms[rms > 0]
        if len(all_bars) == 0:
            continue

        p25 = float(np.percentile(all_bars, 25))
        p90 = float(np.percentile(all_bars, 90))

        segment_mean = float(np.mean(segment_rms))
        segment_max = float(np.max(segment_rms))

        if segment_mean > p25:
            active_count += 1

        if segment_max > p90:
            has_peak_stem = True

    if active_count >= 5:
        if has_peak_stem:
            return "full+extra"
        return "full"
    if active_count >= 3:
        return "mid"
    return "sparse"


def _energy_trajectory(
    combined_energy: np.ndarray,
    start_bar: int,
    end_bar: int,
    buckets: EnergyBuckets,
) -> str:
    """Compute energy trajectory string for a section.

    Split section into thirds, bucket each third, format as "low->medium->high",
    deduplicate adjacent same.
    """
    segment = combined_energy[start_bar:end_bar]
    if len(segment) == 0:
        return "silent"

    n = len(segment)
    third = max(1, n // 3)

    # Split into thirds (handle uneven division)
    parts = [
        segment[:third],
        segment[third:2 * third],
        segment[2 * third:],
    ]

    labels: list[str] = []
    for part in parts:
        if len(part) == 0:
            continue
        mean_val = float(np.mean(part))
        labels.append(classify_energy(mean_val, buckets))

    if not labels:
        return "silent"

    # Deduplicate adjacent same labels
    deduped: list[str] = [labels[0]]
    for lbl in labels[1:]:
        if lbl != deduped[-1]:
            deduped.append(lbl)

    return "->".join(deduped)


def label_sections(
    boundaries: np.ndarray,
    total_bars: int,
    combined_energy: np.ndarray,
    vocal_active: np.ndarray,
    bar_rms_per_stem: dict[str, np.ndarray],
    buckets: EnergyBuckets,
    bpm: float,
    bar_boundaries_frames: np.ndarray,
    sr: int = ANALYSIS_SR,
) -> list[SectionInfo]:
    """Stage 2: Label segments using percentile-based decision tree.

    Applies the full decision tree for section labels, density, vocal status,
    energy trajectory, GOOD INSTRUMENTAL SOURCE annotation, and build/drop detection.

    Args:
        boundaries: Quantized boundary positions (bar indices).
        total_bars: Total bar count.
        combined_energy: Per-bar normalized combined energy.
        vocal_active: Per-bar boolean vocal activity.
        bar_rms_per_stem: Raw per-stem bar RMS.
        buckets: Adaptive energy thresholds.
        bpm: Song BPM (for time computation).
        bar_boundaries_frames: Frame indices of bar boundaries.
        sr: Sample rate.

    Returns:
        List of SectionInfo objects.
    """
    # Build segment list: [0, b1, b2, ..., total_bars]
    all_points = np.concatenate([[0], boundaries, [total_bars]])
    n_segments = len(all_points) - 1

    if n_segments == 0:
        return []

    # Seconds per bar
    seconds_per_bar = 4 * 60.0 / bpm if bpm > 0 else 2.0

    # Pre-compute vocal stem peak for GOOD INSTRUMENTAL SOURCE annotation
    vocal_rms = bar_rms_per_stem.get("vocals", np.zeros(total_bars))
    vocal_peak = float(np.max(vocal_rms)) if len(vocal_rms) > 0 else 0.0
    sustain_thresh = vocal_peak * VOCAL_SUSTAIN_RATIO

    sections: list[SectionInfo] = []

    for seg_idx in range(n_segments):
        start_bar = int(all_points[seg_idx])
        end_bar = int(all_points[seg_idx + 1])
        bar_count = end_bar - start_bar

        if bar_count <= 0:
            continue

        # Time computation from bar boundaries
        if start_bar < len(bar_boundaries_frames):
            start_time = float(bar_boundaries_frames[start_bar]) / sr
        else:
            start_time = start_bar * seconds_per_bar

        if end_bar < len(bar_boundaries_frames):
            end_time = float(bar_boundaries_frames[end_bar]) / sr
        else:
            end_time = end_bar * seconds_per_bar

        # Segment energy stats
        seg_energy = combined_energy[start_bar:end_bar]
        seg_mean_energy = float(np.mean(seg_energy)) if len(seg_energy) > 0 else 0.0
        energy_level = classify_energy(seg_mean_energy, buckets)

        # Vocal info
        seg_vocal = vocal_active[start_bar:end_bar]
        has_vocals = bool(np.any(seg_vocal)) if len(seg_vocal) > 0 else False
        vocal_ratio = float(np.mean(seg_vocal)) if len(seg_vocal) > 0 else 0.0
        vocal_status = _segment_vocal_status(vocal_active, start_bar, end_bar)

        # Drums energy for breakdown detection
        drums_rms = bar_rms_per_stem.get("drums", np.zeros(total_bars))
        seg_drums = drums_rms[start_bar:end_bar]
        if len(seg_drums) > 0 and len(drums_rms[drums_rms > 0]) > 0:
            drums_p25 = float(np.percentile(drums_rms[drums_rms > 0], 25))
            drums_low = float(np.mean(seg_drums)) < drums_p25
        else:
            drums_low = True

        # Position info
        is_first = seg_idx == 0
        is_last = seg_idx == n_segments - 1
        position_ratio = start_bar / total_bars if total_bars > 0 else 0.0

        # Build detection: steady rise from first to last quarter
        is_build = False
        if bar_count >= 4:
            quarter = max(1, bar_count // 4)
            first_q = float(np.mean(seg_energy[:quarter]))
            last_q = float(np.mean(seg_energy[-quarter:]))
            first_q_level = classify_energy(first_q, buckets)
            if first_q_level not in ("high", "peak") and first_q > BUILD_EPSILON:
                if last_q > first_q * BUILD_RATIO:
                    is_build = True

        # Decision tree (spec 2.4 Stage 2)
        if is_first and (energy_level in ("low", "silent") or (energy_level == "medium" and not has_vocals)):
            label = "intro"
        elif is_last and (energy_level in ("low", "silent") or position_ratio > 0.85):
            label = "outro"
        elif is_build:
            label = "build"
        elif energy_level in ("high", "peak") and has_vocals:
            label = "chorus"
        elif energy_level in ("high", "peak") and not has_vocals:
            label = "instrumental"
        elif drums_low and has_vocals:
            label = "breakdown"
        elif energy_level == "medium" and has_vocals:
            label = "verse"
        elif energy_level in ("low", "silent") and not has_vocals:
            label = "instrumental"
        elif energy_level == "medium" and not has_vocals:
            label = "instrumental"
        else:
            label = "verse"

        # Density
        density = _compute_density(bar_rms_per_stem, start_bar, end_bar)

        # Energy trajectory
        trajectory = _energy_trajectory(combined_energy, start_bar, end_bar, buckets)

        # Per-section vocal prominence
        seg_bar_rms = {k: v[start_bar:end_bar] for k, v in bar_rms_per_stem.items()}
        seg_vocal_active = vocal_active[start_bar:end_bar]

        # Guard: clip arrays to same length to prevent IndexError
        min_len = min(len(seg_vocal_active), *(len(v) for v in seg_bar_rms.values())) if seg_bar_rms else 0
        if min_len > 0:
            seg_bar_rms = {k: v[:min_len] for k, v in seg_bar_rms.items()}
            seg_vocal_active = seg_vocal_active[:min_len]

        # Require minimum 3 vocal-active bars for stable prominence
        active_count = int(np.sum(seg_vocal_active.astype(bool))) if min_len > 0 else 0
        if active_count >= 3:
            sec_prominence = compute_vocal_prominence(seg_bar_rms, seg_vocal_active)
        else:
            sec_prominence = None

        # Annotations
        annotations: list[str] = []

        # GOOD INSTRUMENTAL SOURCE: vocal stem below sustain threshold for entire section
        seg_vocal_rms = vocal_rms[start_bar:end_bar]
        if len(seg_vocal_rms) > 0 and vocal_peak > 0:
            if np.all(seg_vocal_rms < sustain_thresh):
                annotations.append("GOOD INSTRUMENTAL SOURCE")

        # Build annotation
        if is_build:
            annotations.append("BUILD")

        sections.append(SectionInfo(
            start_bar=start_bar,
            end_bar=end_bar,
            bar_count=bar_count,
            start_time=round(start_time, 2),
            end_time=round(end_time, 2),
            label=label,
            energy_level=energy_level,
            energy_trajectory=trajectory,
            density=density,
            vocal_status=vocal_status,
            vocal_prominence_db=round(sec_prominence, 1) if sec_prominence is not None else None,
            annotations=annotations,
        ))

    # Drop detection (compare adjacent sections)
    for i in range(1, len(sections)):
        prev = sections[i - 1]
        curr = sections[i]
        # Previous section's last bar energy
        prev_last = combined_energy[prev.end_bar - 1] if prev.end_bar > 0 else 0.0
        # Current section's first bar energy
        curr_first = combined_energy[curr.start_bar] if curr.start_bar < len(combined_energy) else 0.0
        if prev_last > BUILD_EPSILON and curr_first > prev_last * DROP_RATIO:
            if "DROP" not in curr.annotations:
                curr.annotations.append("DROP")

    return sections


def merge_sections(sections: list[SectionInfo]) -> list[SectionInfo]:
    """Stage 3: Merge adjacent same-label sections and absorb <4 bar sections.

    1. Merge adjacent sections with the same label.
    2. Absorb sections <4 bars into their louder neighbor.
    """
    if len(sections) <= 1:
        return sections

    # Pass 1: Merge adjacent same-label
    merged: list[SectionInfo] = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        if sec.label == prev.label:
            # Merge: combine bars, times, recalculate stats
            merged[-1] = replace(
                prev,
                end_bar=sec.end_bar,
                bar_count=sec.end_bar - prev.start_bar,
                end_time=sec.end_time,
                energy_level=_merge_energy_level(prev.energy_level, sec.energy_level),
                energy_trajectory=prev.energy_trajectory + "->" + sec.energy_trajectory
                    if prev.energy_trajectory != sec.energy_trajectory
                    else prev.energy_trajectory,
                density=_merge_density(prev.density, sec.density),
                vocal_status=_merge_vocal_status(prev.vocal_status, sec.vocal_status),
                vocal_prominence_db=None,
                annotations=list(set(prev.annotations + sec.annotations)),
            )
        else:
            merged.append(sec)

    # Pass 2: Absorb sections <4 bars into louder neighbor
    changed = True
    while changed and len(merged) > 1:
        changed = False
        new_merged: list[SectionInfo] = []
        i = 0
        while i < len(merged):
            sec = merged[i]
            if sec.bar_count < MIN_SECTION_BARS and len(merged) > 1:
                # Find louder neighbor
                left_energy = _energy_rank(merged[i - 1].energy_level) if i > 0 else -1
                right_energy = _energy_rank(merged[i + 1].energy_level) if i < len(merged) - 1 else -1

                if left_energy >= right_energy and i > 0:
                    # Absorb into left
                    prev = new_merged[-1]
                    new_merged[-1] = replace(
                        prev,
                        end_bar=sec.end_bar,
                        bar_count=sec.end_bar - prev.start_bar,
                        end_time=sec.end_time,
                        energy_level=_merge_energy_level(prev.energy_level, sec.energy_level),
                        density=_merge_density(prev.density, sec.density),
                        vocal_status=_merge_vocal_status(prev.vocal_status, sec.vocal_status),
                        vocal_prominence_db=None,
                        annotations=list(set(prev.annotations + sec.annotations)),
                    )
                    changed = True
                elif right_energy >= 0 and i < len(merged) - 1:
                    # Absorb into right
                    next_sec = merged[i + 1]
                    new_sec = replace(
                        next_sec,
                        start_bar=sec.start_bar,
                        bar_count=next_sec.end_bar - sec.start_bar,
                        start_time=sec.start_time,
                        energy_level=_merge_energy_level(sec.energy_level, next_sec.energy_level),
                        density=_merge_density(sec.density, next_sec.density),
                        vocal_status=_merge_vocal_status(sec.vocal_status, next_sec.vocal_status),
                        vocal_prominence_db=None,
                        annotations=list(set(sec.annotations + next_sec.annotations)),
                    )
                    new_merged.append(new_sec)
                    i += 2  # Skip next since it was absorbed
                    changed = True
                    continue
                else:
                    new_merged.append(sec)
            else:
                new_merged.append(sec)
            i += 1
        merged = new_merged

    return merged


def _energy_rank(level: str) -> int:
    """Convert energy level string to numeric rank for comparison."""
    ranks = {"silent": 0, "low": 1, "medium": 2, "high": 3, "peak": 4}
    return ranks.get(level, 0)


def _merge_energy_level(a: str, b: str) -> str:
    """Pick the higher energy level when merging two sections."""
    if _energy_rank(a) >= _energy_rank(b):
        return a
    return b


def _merge_density(a: str, b: str) -> str:
    """Pick the higher density when merging."""
    density_rank = {"sparse": 0, "mid": 1, "full": 2, "full+extra": 3}
    if density_rank.get(a, 0) >= density_rank.get(b, 0):
        return a
    return b


def _merge_vocal_status(a: str, b: str) -> str:
    """Merge vocal status: if either has vocals, result has vocals."""
    if a == "vox:yes" or b == "vox:yes":
        return "vox:yes"
    if a == "vox:fading" or b == "vox:fading":
        return "vox:fading"
    return "vox:no"


# ---------------------------------------------------------------------------
# Full section detection pipeline
# ---------------------------------------------------------------------------

def detect_sections(
    bar_rms_per_stem: dict[str, np.ndarray],
    combined_energy: np.ndarray,
    vocal_active: np.ndarray,
    buckets: EnergyBuckets,
    total_bars: int,
    bpm: float,
    bar_boundaries_frames: np.ndarray,
    sr: int = ANALYSIS_SR,
) -> list[SectionInfo]:
    """Run the full 3-stage section detection pipeline.

    Stage 1: Detect boundaries via derivative analysis.
    Stage 1b: Quantize to phrase grid.
    Stage 2: Label sections.
    Stage 3: Merge adjacent same-label and absorb short sections.
    """
    # Stage 1: Boundary detection
    raw_boundaries = detect_boundaries(bar_rms_per_stem, combined_energy)

    # Stage 1b: Phrase quantization
    quantized = quantize_to_phrases(raw_boundaries, total_bars)

    logger.info(
        "Section detection: %d raw boundaries -> %d quantized, %d total bars",
        len(raw_boundaries),
        len(quantized),
        total_bars,
    )

    # Stage 2: Label segments
    sections = label_sections(
        boundaries=quantized,
        total_bars=total_bars,
        combined_energy=combined_energy,
        vocal_active=vocal_active,
        bar_rms_per_stem=bar_rms_per_stem,
        buckets=buckets,
        bpm=bpm,
        bar_boundaries_frames=bar_boundaries_frames,
        sr=sr,
    )

    # Stage 3: Merge and cleanup
    sections = merge_sections(sections)

    # Post-merge: recompute vocal_prominence_db for merged sections that lost it
    for i, sec in enumerate(sections):
        if sec.vocal_prominence_db is None and sec.vocal_status != "vox:no":
            seg_bar_rms = {k: v[sec.start_bar:sec.end_bar] for k, v in bar_rms_per_stem.items()}
            seg_vocal_active = vocal_active[sec.start_bar:sec.end_bar]
            min_len = min(len(seg_vocal_active), *(len(v) for v in seg_bar_rms.values())) if seg_bar_rms else 0
            if min_len > 0:
                seg_bar_rms = {k: v[:min_len] for k, v in seg_bar_rms.items()}
                seg_vocal_active = seg_vocal_active[:min_len]
            active_count = int(np.sum(seg_vocal_active.astype(bool))) if min_len > 0 else 0
            if active_count >= 3:
                prom = compute_vocal_prominence(seg_bar_rms, seg_vocal_active)
                sections[i] = replace(
                    sec,
                    vocal_prominence_db=round(prom, 1) if prom is not None else None,
                )

    logger.info("Section detection complete: %d sections", len(sections))
    return sections


# ---------------------------------------------------------------------------
# 2.6 Cross-song RMS loudness
# ---------------------------------------------------------------------------

def compute_loudness_diff(mean_rms_a: float, mean_rms_b: float) -> Optional[float]:
    """Compute cross-song loudness difference in dB.

    Returns 20*log10(rms_a/rms_b), positive means A is louder.
    Returns None if either RMS is below 0.001.
    """
    if mean_rms_a < 0.001 or mean_rms_b < 0.001:
        return None
    return 20.0 * np.log10(max(mean_rms_a, 1e-10) / max(mean_rms_b, 1e-10))


# ---------------------------------------------------------------------------
# 2.7 Vocal prominence
# ---------------------------------------------------------------------------

def compute_vocal_prominence(
    bar_rms_per_stem: dict[str, np.ndarray],
    vocal_active: np.ndarray,
) -> Optional[float]:
    """Compute vocal prominence in dB above accompaniment.

    Both means computed over vocal-active bars only.
    mean_vocal_rms = vocal stem mean across bars where vocal_active == True.
    mean_non_vocal_rms = sum of non-vocal stems across those same bars.

    Returns prominence_db or None if insufficient data.
    """
    vocal_rms = bar_rms_per_stem.get("vocals")
    if vocal_rms is None or len(vocal_rms) == 0:
        return None

    active_mask = vocal_active.astype(bool)
    if not np.any(active_mask):
        return None

    mean_vocal = float(np.mean(vocal_rms[active_mask]))

    # Sum of non-vocal stems over the same active bars
    non_vocal_sum = np.zeros(int(np.sum(active_mask)), dtype=np.float64)
    for name, rms in bar_rms_per_stem.items():
        if name == "vocals":
            continue
        non_vocal_sum += rms[active_mask]
    mean_non_vocal = float(np.mean(non_vocal_sum))

    if mean_non_vocal < 1e-10 or mean_vocal < 1e-10:
        return None

    return 20.0 * np.log10(mean_vocal / mean_non_vocal)


def _classify_energy_profile(combined_energy: np.ndarray) -> str:
    """Classify the overall energy profile of a song.

    Returns one of: "consistent high", "consistent low", "wide dynamic range",
    "moderate dynamic range".
    """
    if len(combined_energy) == 0:
        return "consistent low"

    active = combined_energy[combined_energy > BUCKET_NOISE_FLOOR]
    if len(active) == 0:
        return "consistent low"

    std = float(np.std(active))
    mean = float(np.mean(active))

    # Coefficient of variation
    cv = std / mean if mean > 0 else 0.0

    if cv < 0.15:
        if mean > 0.6:
            return "consistent high"
        return "consistent low"
    if cv > 0.35:
        return "wide dynamic range"
    return "moderate dynamic range"


# ---------------------------------------------------------------------------
# ML section conversion and enrichment
# ---------------------------------------------------------------------------

def _segments_to_sections(
    ml_segments: list[dict],
    bar_boundaries: np.ndarray,
    sr: int,
    audio_length: int,
) -> list[SectionInfo]:
    """Convert raw ML segment dicts into SectionInfo objects.

    Maps start/end times (seconds) to bar indices using searchsorted on
    bar boundary times. Uses librosa.frames_to_time for frame-to-time
    conversion (bar_boundaries are in hop-length frames, not samples).

    Args:
        ml_segments: List of {"label": str, "start": float, "end": float}.
        bar_boundaries: Frame indices of bar boundaries (n_bars + 1 entries).
        sr: Sample rate.
        audio_length: Total audio length in samples.

    Returns:
        List of SectionInfo with section_source="ml".
    """
    if len(bar_boundaries) < 2:
        return []

    # Convert bar boundaries (hop-length frames) to times via librosa
    # The last entry may be audio_length in samples -- convert it separately
    frame_boundaries = bar_boundaries[:-1]
    bar_times = librosa.frames_to_time(frame_boundaries, sr=sr)
    # Append the final boundary as audio_length / sr
    final_time = audio_length / sr
    bar_times = np.append(bar_times, final_time)

    sections: list[SectionInfo] = []
    for seg in ml_segments:
        label = seg.get("label", "unknown")
        start_sec = float(seg.get("start", 0.0))
        end_sec = float(seg.get("end", 0.0))

        # Map times to bar indices
        start_bar = int(np.searchsorted(bar_times, start_sec, side="right") - 1)
        end_bar = int(np.searchsorted(bar_times, end_sec, side="right") - 1)

        # Clamp to valid range
        n_bars = len(bar_boundaries) - 1
        start_bar = max(0, min(start_bar, n_bars - 1))
        end_bar = max(start_bar + 1, min(end_bar, n_bars))

        bar_count = end_bar - start_bar
        if bar_count <= 0:
            continue

        sections.append(replace(
            SectionInfo(
                start_bar=start_bar,
                end_bar=end_bar,
                bar_count=bar_count,
                start_time=round(start_sec, 2),
                end_time=round(end_sec, 2),
                label=label,
                energy_level="medium",
                energy_trajectory="medium",
                density="mid",
                vocal_status="vox:no",
            ),
            section_source="ml",
        ))

    return sections


def _enrich_sections(
    sections: list[SectionInfo],
    audio: np.ndarray,
    sr: int,
    bar_boundaries: np.ndarray,
    bar_rms_per_stem: dict[str, np.ndarray],
    combined_energy: np.ndarray,
    vocal_active: np.ndarray,
    buckets: EnergyBuckets,
) -> list[SectionInfo]:
    """Enrich ML-detected sections with audio features.

    Computes RMS energy, energy level classification, vocal prominence,
    vocal status, density, and energy trajectory for each section using
    the same helpers the heuristic path uses.

    Args:
        sections: ML-detected SectionInfo objects (section_source="ml").
        audio: Reference audio signal (1D, for RMS if needed).
        sr: Sample rate.
        bar_boundaries: Frame indices of bar boundaries.
        bar_rms_per_stem: Per-stem bar RMS arrays.
        combined_energy: Normalized combined energy per bar.
        vocal_active: Per-bar boolean vocal activity.
        buckets: Adaptive energy bucket thresholds.

    Returns:
        Enriched SectionInfo list with section_source="enriched".
    """
    total_bars = len(bar_boundaries) - 1
    enriched: list[SectionInfo] = []

    for sec in sections:
        start_bar = sec.start_bar
        end_bar = sec.end_bar

        # Energy level from combined energy
        seg_energy = combined_energy[start_bar:end_bar]
        seg_mean_energy = float(np.mean(seg_energy)) if len(seg_energy) > 0 else 0.0
        energy_level = classify_energy(seg_mean_energy, buckets)

        # Energy trajectory
        trajectory = _energy_trajectory(combined_energy, start_bar, end_bar, buckets)

        # Vocal status
        vocal_status = _segment_vocal_status(vocal_active, start_bar, end_bar)

        # Density
        density = _compute_density(bar_rms_per_stem, start_bar, end_bar)

        # Vocal prominence
        seg_bar_rms = {k: v[start_bar:end_bar] for k, v in bar_rms_per_stem.items()}
        seg_vocal_active = vocal_active[start_bar:end_bar]
        min_len = min(len(seg_vocal_active), *(len(v) for v in seg_bar_rms.values())) if seg_bar_rms else 0
        if min_len > 0:
            seg_bar_rms = {k: v[:min_len] for k, v in seg_bar_rms.items()}
            seg_vocal_active = seg_vocal_active[:min_len]
        active_count = int(np.sum(seg_vocal_active.astype(bool))) if min_len > 0 else 0
        vocal_prominence = None
        if active_count >= 3:
            vocal_prominence = compute_vocal_prominence(seg_bar_rms, seg_vocal_active)

        enriched.append(replace(
            sec,
            energy_level=energy_level,
            energy_trajectory=trajectory,
            density=density,
            vocal_status=vocal_status,
            vocal_prominence_db=round(vocal_prominence, 1) if vocal_prominence is not None else None,
            section_source="enriched",
        ))

    return enriched


# ---------------------------------------------------------------------------
# Main entry points: analyze_stems() and compute_relationships()
# ---------------------------------------------------------------------------

def analyze_stems(
    stem_paths: dict[str, Path],
    beat_frames: np.ndarray,
    bpm: float,
    audio_path: Optional[Path] = None,
    ml_segments: list[dict] | None = None,
) -> tuple[StemAnalysis, SongStructure]:
    """Orchestrate the full song structure analysis pipeline.

    1. Load stems at 22050 Hz.
    2. Compute bar grid from beat_frames[::4].
    3. Compute per-bar RMS for each stem.
    4. Run normalization pipeline (combined energy, adaptive buckets).
    5. Detect vocal activity.
    6. Detect vocal gaps.
    7. Run section detection (ML-based if ml_segments provided, else heuristic).

    Args:
        stem_paths: Dict mapping stem name to WAV file path.
        beat_frames: Reconciled beat frame positions.
        bpm: Reconciled BPM.
        audio_path: Path to original mix audio (for BPM re-detection if needed).
        ml_segments: Optional ML-detected segments from structure_ml. If provided
            and non-empty, these are converted to SectionInfo and enriched with
            audio features, skipping the heuristic detection path.

    Returns:
        (StemAnalysis, SongStructure)
    """
    # BPM outside 70-170: re-run beat_track on drum stem
    if (bpm < 70 or bpm > 170) and "drums" in stem_paths and stem_paths["drums"].exists():
        logger.info("BPM %.1f outside 70-170 range, re-detecting from drum stem", bpm)
        y_drums, sr_drums = librosa.load(str(stem_paths["drums"]), sr=ANALYSIS_SR)
        tempo_d, beat_frames_d = librosa.beat.beat_track(y=y_drums, sr=sr_drums, units="frames")
        new_bpm = float(np.atleast_1d(tempo_d)[0])
        if 70 <= new_bpm <= 170:
            bpm = new_bpm
            beat_frames = beat_frames_d
            logger.info("Re-detected BPM from drums: %.1f", bpm)

    # Load all stems
    stems = _load_stems(stem_paths, sr=ANALYSIS_SR)

    # Find reference audio length (use first non-zero stem)
    audio_length = 0
    for name in STEM_NAMES:
        if len(stems[name]) > audio_length:
            audio_length = len(stems[name])

    if audio_length == 0:
        # Empty stems: return minimal result
        empty_rms: dict[str, np.ndarray] = {n: np.array([]) for n in STEM_NAMES}
        stem_analysis = StemAnalysis(
            bar_rms=empty_rms,
            combined_energy=np.array([]),
            vocal_active=np.array([], dtype=bool),
            vocal_gaps=[],
            bucket_thresholds=EnergyBuckets(
                noise_floor=BUCKET_NOISE_FLOOR, p10=0.0, p50=0.0, p85=0.0,
            ),
        )
        song_structure = SongStructure(sections=[], vocal_gaps=[], total_bars=0)
        return stem_analysis, song_structure

    # Compute bar boundaries from beat frames
    bar_boundaries = _compute_bar_boundaries(beat_frames, audio_length)
    total_bars = len(bar_boundaries) - 1

    if total_bars <= 0:
        empty_rms = {n: np.array([]) for n in STEM_NAMES}
        stem_analysis = StemAnalysis(
            bar_rms=empty_rms,
            combined_energy=np.array([]),
            vocal_active=np.array([], dtype=bool),
            vocal_gaps=[],
            bucket_thresholds=EnergyBuckets(
                noise_floor=BUCKET_NOISE_FLOOR, p10=0.0, p50=0.0, p85=0.0,
            ),
        )
        song_structure = SongStructure(sections=[], vocal_gaps=[], total_bars=0)
        return stem_analysis, song_structure

    # Compute per-bar RMS for each stem (raw values)
    bar_rms: dict[str, np.ndarray] = {}
    for name in STEM_NAMES:
        bar_rms[name] = _compute_bar_rms(stems[name], bar_boundaries)

    # Normalization pipeline: combined energy + adaptive buckets
    combined_energy, buckets = compute_adaptive_buckets(bar_rms)

    # Vocal activity detection
    vocal_rms = bar_rms.get("vocals", np.zeros(total_bars))
    vocal_active = detect_vocal_activity(vocal_rms)

    # Vocal gap detection
    vocal_gaps = detect_vocal_gaps(vocal_active)

    # Section detection: ML path or heuristic path
    if ml_segments:
        logger.info("Using ML sections (%d segments provided)", len(ml_segments))
        sections = _segments_to_sections(
            ml_segments=ml_segments,
            bar_boundaries=bar_boundaries,
            sr=ANALYSIS_SR,
            audio_length=audio_length,
        )
        # Find a reference audio array for enrichment (use first non-empty stem)
        ref_audio = np.zeros(1, dtype=np.float32)
        for name in STEM_NAMES:
            if len(stems[name]) > 0:
                ref_audio = stems[name]
                break
        sections = _enrich_sections(
            sections=sections,
            audio=ref_audio,
            sr=ANALYSIS_SR,
            bar_boundaries=bar_boundaries,
            bar_rms_per_stem=bar_rms,
            combined_energy=combined_energy,
            vocal_active=vocal_active,
            buckets=buckets,
        )
    else:
        logger.info("Using heuristic sections")
        sections = detect_sections(
            bar_rms_per_stem=bar_rms,
            combined_energy=combined_energy,
            vocal_active=vocal_active,
            buckets=buckets,
            total_bars=total_bars,
            bpm=bpm,
            bar_boundaries_frames=bar_boundaries,
            sr=ANALYSIS_SR,
        )

    stem_analysis = StemAnalysis(
        bar_rms=bar_rms,
        combined_energy=combined_energy,
        vocal_active=vocal_active,
        vocal_gaps=vocal_gaps,
        bucket_thresholds=buckets,
    )

    song_structure = SongStructure(
        sections=sections,
        vocal_gaps=vocal_gaps,
        total_bars=total_bars,
    )

    logger.info(
        "Stem analysis complete: %d bars, %d sections, %d vocal gaps",
        total_bars,
        len(sections),
        len(vocal_gaps),
    )

    return stem_analysis, song_structure


def compute_relationships(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
) -> CrossSongRelationships:
    """Compute cross-song relationships for remix planning.

    Analyzes loudness difference, energy profiles, vocal prominence, and
    identifies the best vocal and instrumental sources.

    Args:
        meta_a: Audio metadata for song A (with stem_analysis populated).
        meta_b: Audio metadata for song B (with stem_analysis populated).

    Returns:
        CrossSongRelationships with all comparison metrics.
    """
    # Loudness difference
    rms_a = meta_a.mean_rms or 0.0
    rms_b = meta_b.mean_rms or 0.0
    loudness_diff = compute_loudness_diff(rms_a, rms_b)
    if loudness_diff is None:
        loudness_diff = 0.0

    # Energy profiles
    energy_a = meta_a.stem_analysis.combined_energy if meta_a.stem_analysis else np.array([])
    energy_b = meta_b.stem_analysis.combined_energy if meta_b.stem_analysis else np.array([])
    profile_a = _classify_energy_profile(energy_a)
    profile_b = _classify_energy_profile(energy_b)

    # Vocal prominence for both songs
    prom_a: Optional[float] = None
    prom_b: Optional[float] = None

    if meta_a.stem_analysis:
        prom_a = compute_vocal_prominence(
            meta_a.stem_analysis.bar_rms,
            meta_a.stem_analysis.vocal_active,
        )
    if meta_b.stem_analysis:
        prom_b = compute_vocal_prominence(
            meta_b.stem_analysis.bar_rms,
            meta_b.stem_analysis.vocal_active,
        )

    prom_a_db = prom_a if prom_a is not None else 0.0
    prom_b_db = prom_b if prom_b is not None else 0.0

    # Fixed convention: Song A always provides vocals, Song B always provides instrumentals.
    vocal_source = VOCAL_SOURCE

    # Instrumental sections from Song B (fixed instrumental source)
    instrumental_source_meta = meta_b
    instrumental_sections: list[str] = []
    if instrumental_source_meta.song_structure:
        for sec in instrumental_source_meta.song_structure.sections:
            if "GOOD INSTRUMENTAL SOURCE" in sec.annotations:
                instrumental_sections.append(f"bars {sec.start_bar}-{sec.end_bar}")

    # Frequency conflict detection (basic heuristic)
    frequency_conflicts = ""
    if meta_a.stem_analysis and meta_b.stem_analysis:
        # Check if both songs have significant "other" stem energy
        other_a = meta_a.stem_analysis.bar_rms.get("other", np.array([]))
        other_b = meta_b.stem_analysis.bar_rms.get("other", np.array([]))
        guitar_b = meta_b.stem_analysis.bar_rms.get("guitar", np.array([]))

        mean_other_a = float(np.mean(other_a)) if len(other_a) > 0 else 0.0
        mean_other_b = float(np.mean(other_b)) if len(other_b) > 0 else 0.0
        mean_guitar_b = float(np.mean(guitar_b)) if len(guitar_b) > 0 else 0.0

        if mean_other_a > 0.01 and mean_guitar_b > 0.01:
            frequency_conflicts = (
                "Song A 'other' stem may mask Song B guitar at 1-4 kHz"
            )
        elif mean_other_a > 0.01 and mean_other_b > 0.01:
            frequency_conflicts = (
                "Both songs have significant 'other' stem energy; "
                "may cause masking in mid-frequencies"
            )

    # Stretch percentage -- Song A is always vocal, Song B is always instrumental.
    stretch_pct = compute_stretch_pct(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)

    return CrossSongRelationships(
        loudness_diff_db=round(loudness_diff, 1),
        energy_profile_a=profile_a,
        energy_profile_b=profile_b,
        vocal_source=vocal_source,
        vocal_prominence_a_db=round(prom_a_db, 1),
        vocal_prominence_b_db=round(prom_b_db, 1),
        instrumental_sections=instrumental_sections,
        frequency_conflicts=frequency_conflicts,
        stretch_pct=round(stretch_pct, 1),
    )
