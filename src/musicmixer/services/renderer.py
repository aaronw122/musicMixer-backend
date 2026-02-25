"""Section-based arrangement renderer.

Converts a RemixPlan's sections into audio by applying per-stem gains
with smooth interpolation at section boundaries. Outputs two separate buses
(vocal and instrumental) to enable spectral ducking on Day 4.

The renderer builds a continuous gain curve per stem across the full track,
interpolating over transition_beats at each section boundary. This avoids
the "fade from silence" problem that occurs when contiguous sections use
crossfade envelopes without overlap.
"""

from __future__ import annotations

import logging

import numpy as np

from musicmixer.models import Section

logger = logging.getLogger(__name__)


def beats_to_samples(
    beat_index: int,
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int = 512,
    analysis_sr: int = 22050,
) -> int:
    """Convert a beat index to a sample position using the actual beat grid.

    Uses beat_frames from librosa (NOT constant-BPM math, which drifts).
    If beat_index exceeds the detected beats, extrapolate using the average
    interval of the last 8 beats.

    Args:
        beat_index: Index into beat_frames (or beyond for extrapolation).
        beat_frames: Frame indices from librosa beat detection.
        sr: Target sample rate of the stem audio (e.g. 44100).
        hop_length: Hop length used during beat detection (default 512).
        analysis_sr: Sample rate used during beat analysis (default 22050).
            Beat frames are in units of this rate. When sr != analysis_sr,
            positions are scaled accordingly.
    """
    sr_scale = sr / analysis_sr  # e.g. 44100 / 22050 = 2.0

    if len(beat_frames) < 2:
        # Degenerate case: no usable beat grid
        return beat_index * sr

    if beat_index >= len(beat_frames):
        # Extrapolate beyond last detected beat
        last_n = beat_frames[-8:] if len(beat_frames) >= 8 else beat_frames
        avg_beat_len = float(np.mean(np.diff(last_n))) * hop_length * sr_scale
        overshoot = beat_index - len(beat_frames) + 1
        return int(beat_frames[-1] * hop_length * sr_scale + overshoot * avg_beat_len)

    return int(beat_frames[beat_index] * hop_length * sr_scale)


def _build_gain_curves(
    sections: list[Section],
    all_stem_names: list[str],
    total_samples: int,
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int = 512,
    last_beat: int = 0,
) -> dict[str, np.ndarray]:
    """Build continuous per-stem gain curves across the full track.

    For each stem, creates a gain curve that:
    1. Holds the section's gain value throughout each section's body
    2. Smoothly interpolates (cosine) between adjacent sections' gains
       over the transition_beats region at each boundary

    This eliminates the "fade from silence" bug where crossfades between
    contiguous (non-overlapping) sections caused volume drops to zero.

    Returns:
        Dict mapping stem name to float32 array of shape (total_samples,).
    """
    gain_curves: dict[str, np.ndarray] = {}

    for stem_name in all_stem_names:
        curve = np.zeros(total_samples, dtype=np.float32)

        # First pass: fill each section with its flat gain
        for section in sections:
            start = beats_to_samples(section.start_beat, beat_frames, sr, hop_length)
            end = beats_to_samples(section.end_beat, beat_frames, sr, hop_length)
            start = max(0, min(start, total_samples))
            end = max(start, min(end, total_samples))
            gain = section.stem_gains.get(stem_name, 0.0)
            curve[start:end] = gain

        # Second pass: smooth transitions at section boundaries
        for i in range(1, len(sections)):
            section = sections[i]
            prev_section = sections[i - 1]

            # Clamp transition to a short, fixed duration (4 beats)
            # to prevent transitions from consuming entire short sections.
            # The arrangement's transition_beats is treated as a hint but
            # capped here to avoid compounding energy dips.
            trans_beats = min(section.transition_beats, 4)
            if trans_beats <= 0:
                continue

            # Split: 1 beat before boundary, rest after (quick out, smooth in)
            before_beats = 1
            after_beats = trans_beats - before_beats
            trans_start = beats_to_samples(
                max(0, section.start_beat - before_beats), beat_frames, sr, hop_length
            )
            trans_end = beats_to_samples(
                min(last_beat, section.start_beat + after_beats), beat_frames, sr, hop_length
            )
            trans_start = max(0, min(trans_start, total_samples))
            trans_end = max(trans_start, min(trans_end, total_samples))
            trans_len = trans_end - trans_start

            if trans_len <= 0:
                continue

            prev_gain = prev_section.stem_gains.get(stem_name, 0.0)
            curr_gain = section.stem_gains.get(stem_name, 0.0)

            if abs(prev_gain - curr_gain) < 0.001:
                continue  # No change, skip interpolation

            # Cosine interpolation: smooth S-curve from prev_gain to curr_gain
            t = np.linspace(0, 1, trans_len, dtype=np.float32)
            interp = prev_gain + (curr_gain - prev_gain) * (1 - np.cos(t * np.pi)) / 2
            curve[trans_start:trans_end] = interp

        gain_curves[stem_name] = curve

    return gain_curves


def render_arrangement(
    sections: list[Section],
    vocal_stems: dict[str, np.ndarray],
    instrumental_stems: dict[str, np.ndarray],
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Render sections into a vocal bus + instrumental bus.

    Uses continuous gain curves (no per-section envelopes) to avoid volume
    drops at section boundaries. Each stem's gain smoothly interpolates
    between sections using cosine curves.

    Args:
        sections: Beat-aligned arrangement sections with per-stem gains.
        vocal_stems: Dict of vocal stem arrays (typically just {"vocals": array}).
        instrumental_stems: Dict of instrumental stem arrays
            (e.g. {"drums": ..., "bass": ..., "guitar": ..., "piano": ..., "other": ...}).
        beat_frames: Post-stretch beat grid (frame indices from librosa).
        sr: Sample rate of all stem audio (must be uniform, typically 44100).
        hop_length: Hop length used for beat_frames (default 512).

    Returns:
        (vocal_bus, instrumental_bus) -- both shape (total_samples, 2), float32.
        For Day 2, the caller sums these; Day 4 adds spectral ducking between them.
    """
    if not sections:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty

    # Compute total output length from last section's end_beat
    last_beat = max(s.end_beat for s in sections)
    total_samples = beats_to_samples(last_beat, beat_frames, sr, hop_length)

    if total_samples <= 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty

    # Collect all stem names referenced in any section
    all_stem_names = sorted(
        {name for section in sections for name in section.stem_gains}
    )

    # Build continuous gain curves
    gain_curves = _build_gain_curves(
        sections, all_stem_names, total_samples, beat_frames, sr, hop_length,
        last_beat=last_beat,
    )

    # Allocate output buses
    vocal_bus = np.zeros((total_samples, 2), dtype=np.float32)
    instrumental_bus = np.zeros((total_samples, 2), dtype=np.float32)

    # Apply gain curves to each stem and add to appropriate bus
    for stem_name in all_stem_names:
        curve = gain_curves[stem_name]

        # Get stem audio
        if stem_name == "vocals":
            stem_audio = vocal_stems.get("vocals")
        else:
            stem_audio = instrumental_stems.get(stem_name)

        if stem_audio is None:
            continue

        # Trim or pad stem to match total_samples
        stem_len = len(stem_audio)
        if stem_len == 0:
            continue

        usable_len = min(stem_len, total_samples)

        # Apply gain curve (vectorized multiply)
        gained = stem_audio[:usable_len] * curve[:usable_len, np.newaxis]

        if stem_name == "vocals":
            vocal_bus[:usable_len] += gained
        else:
            instrumental_bus[:usable_len] += gained

    # Apply fade-in to the very start (intro)
    if sections[0].transition_in == "fade":
        fade_in_beats = sections[0].transition_beats
        fade_in_samples = beats_to_samples(
            fade_in_beats, beat_frames, sr, hop_length
        )
        fade_in_samples = max(0, min(fade_in_samples, total_samples // 4))
        if fade_in_samples > 0:
            fade_in_curve = np.sin(
                np.linspace(0, np.pi / 2, fade_in_samples)
            ).astype(np.float32)
            vocal_bus[:fade_in_samples] *= fade_in_curve[:, np.newaxis]
            instrumental_bus[:fade_in_samples] *= fade_in_curve[:, np.newaxis]

    return vocal_bus, instrumental_bus


def snap_to_bar(
    sample_position: int,
    beat_positions: np.ndarray,
    beats_per_bar: int = 4,
) -> int:
    """Snap a sample position to the nearest bar boundary (downbeat).

    Args:
        sample_position: Sample index to snap.
        beat_positions: Array of beat sample positions.
        beats_per_bar: Number of beats per bar (default 4 for 4/4 time).

    Returns:
        The sample position of the nearest bar boundary.
    """
    if len(beat_positions) == 0:
        return sample_position

    bar_positions = beat_positions[::beats_per_bar]
    if len(bar_positions) == 0:
        return sample_position

    idx = np.argmin(np.abs(bar_positions - sample_position))
    return int(bar_positions[idx])
