"""Deterministic fallback remix plan generator.

Produces a RemixPlan without any LLM involvement. This is the remix plan
for ALL of Day 2. Day 3 adds LLM-powered interpretation that replaces this
for prompt-driven remixes.
"""

from __future__ import annotations

import logging

from musicmixer.models import AudioMetadata, RemixPlan, Section

logger = logging.getLogger(__name__)


def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
    """Generate a deterministic remix plan from audio analysis metadata.

    Defaults: vocals from song_a, instrumentals from song_b.
    Uses the region starting at 15% into each song, capped at 90 seconds.
    Tempo target is the instrumental song's BPM.
    """
    # Vocal source defaults to song_a (Day 2 -- no vocal_prominence_db yet)
    vocal_src = "song_a"
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 15% into each song, up to 90 seconds
    v_start = vocal_meta.duration_seconds * 0.15
    v_end = min(v_start + 90.0, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.15
    i_end = min(i_start + 90.0, inst_meta.duration_seconds)

    tempo_src = "weighted_midpoint"  # Split stretch burden between both songs
    total_beats = int(inst_meta.bpm * 90 / 60)  # Beats in 90 seconds

    logger.info(
        "Generated fallback plan: vocals=%s, tempo=%s, total_beats=%d",
        vocal_src,
        tempo_src,
        total_beats,
    )

    return RemixPlan(
        vocal_source=vocal_src,
        start_time_vocal=v_start,
        end_time_vocal=v_end,
        start_time_instrumental=i_start,
        end_time_instrumental=i_end,
        sections=default_arrangement(total_beats),
        tempo_source=tempo_src,
        key_source="none",
        explanation=(
            "We created a remix using the strongest sections of each song. "
            "Vocals from Song A layered over Song B's instrumentals."
        ),
        warnings=["Using automatic remix layout (no prompt interpretation yet)."],
        used_fallback=True,
    )


def default_arrangement(total_beats: int) -> list[Section]:
    """Build a 5-section fallback arrangement: intro -> build -> main -> breakdown -> outro.

    Beat boundaries are proportional to total_beats:
      intro:     [0, 1/8)
      build:     [1/8, 1/4)
      main:      [1/4, 3/4)
      breakdown: [3/4, 7/8)
      outro:     [7/8, total)
    """
    eighth = total_beats // 8
    quarter = total_beats // 4
    three_quarter = total_beats * 3 // 4
    seven_eighth = total_beats * 7 // 8

    return [
        Section(
            label="intro",
            start_beat=0,
            end_beat=eighth,
            stem_gains={
                "vocals": 0.0,
                "drums": 0.8,
                "bass": 0.7,
                "guitar": 0.6,
                "piano": 0.5,
                "other": 0.8,
            },
            transition_in="fade",
            transition_beats=8,  # Long fade-in for smooth intro
        ),
        Section(
            label="build",
            start_beat=eighth,
            end_beat=quarter,
            stem_gains={
                "vocals": 0.6,
                "drums": 0.75,
                "bass": 0.8,
                "guitar": 0.5,
                "piano": 0.4,
                "other": 0.5,
            },
            transition_in="crossfade",
            transition_beats=16,  # Vocals entering -- needs long smooth transition
        ),
        Section(
            label="main",
            start_beat=quarter,
            end_beat=three_quarter,
            stem_gains={
                "vocals": 1.0,
                "drums": 0.7,
                "bass": 0.8,
                "guitar": 0.5,
                "piano": 0.4,
                "other": 0.5,
            },
            transition_in="crossfade",
            transition_beats=8,  # Decisive energy increase, not abrupt
        ),
        Section(
            label="breakdown",
            start_beat=three_quarter,
            end_beat=seven_eighth,
            stem_gains={
                "vocals": 0.9,
                "drums": 0.2,  # Reduced but NOT silent -- avoids "song stopped" feel
                "bass": 0.5,
                "guitar": 0.7,
                "piano": 0.8,
                "other": 0.6,
            },
            transition_in="crossfade",
            transition_beats=16,  # Gradual energy reduction
        ),
        Section(
            label="outro",
            start_beat=seven_eighth,
            end_beat=total_beats,
            stem_gains={
                "vocals": 0.3,  # Faint vocal echo, not a silent cut
                "drums": 0.6,
                "bass": 0.5,
                "guitar": 0.5,
                "piano": 0.5,
                "other": 0.7,
            },
            transition_in="crossfade",
            transition_beats=8,  # Smooth transition to final fade
        ),
    ]
