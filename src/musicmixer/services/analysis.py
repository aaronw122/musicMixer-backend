"""Audio analysis: BPM detection and cross-song reconciliation.

Step 2 of Day 2 pipeline. Provides:
- analyze_audio(): BPM, beat positions, duration, confidence for a single track
- reconcile_bpm(): Cross-song BPM reconciliation using expanded interpretation matrix
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import librosa
import numpy as np

from musicmixer.models import AudioMetadata

logger = logging.getLogger(__name__)


def analyze_audio(audio_path: Path) -> AudioMetadata:
    """Analyze audio file for BPM, beat positions, and duration.

    Loads at 22050 Hz (sufficient for BPM detection, saves memory).
    Beat positions are stored in frame units.
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

    logger.info(
        "Audio analysis complete: path=%s bpm=%.1f confidence=%.2f duration=%.1fs beats=%d",
        audio_path.name,
        bpm,
        bpm_confidence,
        duration,
        total_beats,
    )

    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=bpm_confidence,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=max(total_beats, 4),  # At least 1 bar
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
