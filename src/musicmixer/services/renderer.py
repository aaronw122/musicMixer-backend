"""Section-based arrangement renderer.

Converts a RemixPlan's sections into audio by applying per-stem gains,
transition envelopes, and crossfades. Outputs two separate buses (vocal
and instrumental) to enable spectral ducking on Day 4.
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
) -> int:
    """Convert a beat index to a sample position using the actual beat grid.

    Uses beat_frames from librosa (NOT constant-BPM math, which drifts).
    If beat_index exceeds the detected beats, extrapolate using the average
    interval of the last 8 beats.
    """
    if len(beat_frames) < 2:
        # Degenerate case: no usable beat grid
        return beat_index * sr

    if beat_index >= len(beat_frames):
        # Extrapolate beyond last detected beat
        last_n = beat_frames[-8:] if len(beat_frames) >= 8 else beat_frames
        avg_beat_len = float(np.mean(np.diff(last_n))) * hop_length
        overshoot = beat_index - len(beat_frames) + 1
        return int(beat_frames[-1] * hop_length + overshoot * avg_beat_len)

    return int(beat_frames[beat_index] * hop_length)


def make_transition_envelope(
    n_samples: int,
    transition_type: str,
    sr: int = 44100,
) -> np.ndarray:
    """Generate a transition-in envelope for a section boundary.

    Args:
        n_samples: Length of the transition region in samples.
        transition_type: One of "fade", "crossfade", or "cut".
        sr: Sample rate (used to compute micro-crossfade length for "cut").

    Returns:
        1-D float32 array of length n_samples, values in [0, 1].
    """
    if n_samples <= 0:
        return np.array([], dtype=np.float32)

    if transition_type in ("fade", "crossfade"):
        # Cosine-squared fade-in: 0 -> 1
        return (np.cos(np.linspace(np.pi / 2, 0, n_samples)) ** 2).astype(np.float32)

    if transition_type == "cut":
        # Micro-crossfade (~2 ms = 88 samples at 44.1 kHz) to prevent clicks
        micro_len = min(int(0.002 * sr), n_samples)
        env = np.ones(n_samples, dtype=np.float32)
        if micro_len > 0:
            env[:micro_len] = np.linspace(0, 1, micro_len, dtype=np.float32)
        return env

    # Unknown type -- no envelope
    return np.ones(n_samples, dtype=np.float32)


def render_arrangement(
    sections: list[Section],
    vocal_stems: dict[str, np.ndarray],
    instrumental_stems: dict[str, np.ndarray],
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Render sections into a vocal bus + instrumental bus.

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

    vocal_bus = np.zeros((total_samples, 2), dtype=np.float32)
    instrumental_bus = np.zeros((total_samples, 2), dtype=np.float32)

    for i, section in enumerate(sections):
        start_sample = beats_to_samples(section.start_beat, beat_frames, sr, hop_length)
        end_sample = beats_to_samples(section.end_beat, beat_frames, sr, hop_length)
        section_len = end_sample - start_sample

        if section_len <= 0:
            continue

        # Compute transition envelope length in samples
        trans_samples = (
            beats_to_samples(
                section.start_beat + section.transition_beats,
                beat_frames,
                sr,
                hop_length,
            )
            - start_sample
        )
        trans_samples = max(0, min(trans_samples, section_len // 2))

        # For crossfade: fade out the previous section's tail in the overlap region
        if section.transition_in == "crossfade" and i > 0 and trans_samples > 0:
            fade_out = (np.cos(np.linspace(0, np.pi / 2, trans_samples)) ** 2).astype(
                np.float32
            )
            vocal_bus[start_sample : start_sample + trans_samples] *= fade_out[
                :, np.newaxis
            ]
            instrumental_bus[start_sample : start_sample + trans_samples] *= fade_out[
                :, np.newaxis
            ]

        # Build transition-in envelope for this section
        in_env = make_transition_envelope(trans_samples, section.transition_in, sr)
        full_env = np.ones(section_len, dtype=np.float32)
        if trans_samples > 0 and len(in_env) > 0:
            full_env[:trans_samples] = in_env

        # Apply per-stem gains and add to appropriate bus
        for stem_name, gain in section.stem_gains.items():
            if gain < 0.001:
                continue  # Skip effectively silent stems

            # Route: vocals go to vocal_bus, everything else to instrumental_bus
            if stem_name == "vocals":
                stem_audio = vocal_stems.get("vocals")
            else:
                stem_audio = instrumental_stems.get(stem_name)

            if stem_audio is None:
                continue  # Missing stem (e.g., guitar/piano in 4-stem fallback)

            # Extract section range from stem (with bounds checking)
            stem_section = stem_audio[start_sample:end_sample]
            actual_len = len(stem_section)

            if actual_len == 0:
                continue

            if actual_len < section_len:
                # Pad with silence if stem is shorter than section
                pad = np.zeros(
                    (section_len - actual_len, 2),
                    dtype=np.float32,
                )
                stem_section = np.concatenate([stem_section, pad])

            # Apply gain and transition envelope
            stem_section = (
                stem_section[:section_len] * gain * full_env[:, np.newaxis]
            )

            if stem_name == "vocals":
                vocal_bus[start_sample:end_sample] += stem_section
            else:
                instrumental_bus[start_sample:end_sample] += stem_section

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
