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
    Uses the region starting at 25% into each song, capped at 90 seconds.
    Tempo target is the instrumental song's BPM.
    """
    # Vocal source defaults to song_a (Day 2 -- no vocal_prominence_db yet)
    vocal_src = "song_a"
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 25% into each song, up to 90 seconds
    v_start = vocal_meta.duration_seconds * 0.25
    v_end = min(v_start + 90.0, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.25
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
    """Build a 5- or 6-section fallback arrangement.

    6-section (>= 96 beats): intro -> build -> main -> breakdown -> drop -> outro
    5-section (< 96 beats):  intro -> build -> main -> breakdown -> outro

    Beat boundaries are snapped to 4-bar (16-beat) phrase boundaries for
    musically coherent transitions.
    """
    MIN_SECTION_BEATS = 8  # Absolute minimum -- shorter sections are musically meaningless

    def snap_to_phrase(beat: int, phrase_beats: int = 16) -> int:
        """Snap a beat index DOWN to the nearest phrase boundary (default 4 bars).
        Uses floor (not round) for predictable section proportions."""
        return max(0, (beat // phrase_beats) * phrase_beats)

    # For very short arrangements, skip phrase snapping
    if total_beats < 48:
        eighth = total_beats // 8
        quarter = total_beats // 4
        three_quarter = total_beats * 3 // 4
        seven_eighth = total_beats * 7 // 8
    else:
        eighth = snap_to_phrase(total_beats // 8)
        quarter = snap_to_phrase(total_beats // 4)
        three_quarter = snap_to_phrase(total_beats * 3 // 4)
        seven_eighth = snap_to_phrase(total_beats * 7 // 8)

        # Guard: ensure monotonically increasing boundaries with minimum section size.
        if eighth < MIN_SECTION_BEATS:
            eighth = min(16, total_beats // 6)
        if quarter <= eighth:
            quarter = min(eighth + 16, total_beats - 3 * MIN_SECTION_BEATS)
        if three_quarter <= quarter:
            three_quarter = min(quarter + 16, total_beats - 2 * MIN_SECTION_BEATS)
        if seven_eighth <= three_quarter:
            seven_eighth = min(three_quarter + 16, total_beats - MIN_SECTION_BEATS)

        # Final validation: all boundaries within [0, total_beats]
        eighth = max(MIN_SECTION_BEATS, min(eighth, total_beats - 4 * MIN_SECTION_BEATS))
        quarter = max(eighth + MIN_SECTION_BEATS, min(quarter, total_beats - 3 * MIN_SECTION_BEATS))
        three_quarter = max(quarter + MIN_SECTION_BEATS, min(three_quarter, total_beats - 2 * MIN_SECTION_BEATS))
        seven_eighth = max(three_quarter + MIN_SECTION_BEATS, min(seven_eighth, total_beats - MIN_SECTION_BEATS))

    # Build and main MUST share identical instrumental gains to prevent volume dips
    # at the build->main transition. The auto-leveler's detector_audio uses the
    # instrumental bus -- different gains would change detected energy at boundaries,
    # re-triggering the volume dip bug fixed in the 2026-02-25 investigation.
    inst_body =      {"drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5}
    inst_intro =     {"drums": 0.6, "bass": 0.5, "guitar": 0.3, "piano": 0.2, "other": 0.3}
    inst_breakdown = {"drums": 0.1, "bass": 0.4, "guitar": 0.6, "piano": 0.7, "other": 0.5}
    inst_outro =     {"drums": 0.5, "bass": 0.5, "guitar": 0.3, "piano": 0.3, "other": 0.4}

    # 6-section: if enough beats, add a "drop" section between breakdown and outro
    if total_beats >= 96:
        # Recalculate boundaries for 6 sections
        # main ends earlier to make room; breakdown and drop each get 1/8
        five_eighth = snap_to_phrase(total_beats * 5 // 8)
        six_eighth = snap_to_phrase(total_beats * 6 // 8)
        drop_start = six_eighth
        drop_end = snap_to_phrase(total_beats * 7 // 8)
        outro_start = drop_end

        # Validate drop section has minimum beats
        if drop_end - drop_start < MIN_SECTION_BEATS:
            drop_end = min(drop_start + 16, total_beats - MIN_SECTION_BEATS)
            outro_start = drop_end

        return [
            Section(label="intro", start_beat=0, end_beat=eighth,
                    stem_gains={"vocals": 0.0, **inst_intro},
                    transition_in="fade", transition_beats=4),
            Section(label="build", start_beat=eighth, end_beat=quarter,
                    stem_gains={"vocals": 0.5, **inst_body},
                    transition_in="crossfade", transition_beats=min(8, (quarter - eighth) // 3)),
            Section(label="main", start_beat=quarter, end_beat=five_eighth,
                    stem_gains={"vocals": 1.0, **inst_body},
                    transition_in="crossfade", transition_beats=4),
            Section(label="breakdown", start_beat=five_eighth, end_beat=drop_start,
                    stem_gains={"vocals": 0.4, **inst_breakdown},
                    transition_in="crossfade", transition_beats=min(8, (drop_start - five_eighth) // 3)),
            Section(label="drop", start_beat=drop_start, end_beat=drop_end,
                    stem_gains={"vocals": 0.8, **inst_body},
                    transition_in="crossfade", transition_beats=2),
            Section(label="outro", start_beat=outro_start, end_beat=total_beats,
                    stem_gains={"vocals": 0.0, **inst_outro},
                    transition_in="crossfade", transition_beats=min(8, (total_beats - outro_start) // 2)),
        ]

    # 5-section fallback (not enough beats for drop section)
    return [
        Section(label="intro", start_beat=0, end_beat=eighth,
                stem_gains={"vocals": 0.0, **inst_intro},
                transition_in="fade", transition_beats=4),
        Section(label="build", start_beat=eighth, end_beat=quarter,
                stem_gains={"vocals": 0.5, **inst_body},
                transition_in="crossfade", transition_beats=4),
        Section(label="main", start_beat=quarter, end_beat=three_quarter,
                stem_gains={"vocals": 1.0, **inst_body},
                transition_in="crossfade", transition_beats=4),
        Section(label="breakdown", start_beat=three_quarter, end_beat=seven_eighth,
                stem_gains={"vocals": 0.4, **inst_breakdown},
                transition_in="crossfade", transition_beats=4),
        Section(label="outro", start_beat=seven_eighth, end_beat=total_beats,
                stem_gains={"vocals": 0.0, **inst_outro},
                transition_in="crossfade", transition_beats=4),
    ]
