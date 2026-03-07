"""Hard constraint validation for candidate RemixPlan objects.

Validates candidates against 14 non-negotiable rules derived from mixTaste.md
and DJ conventions. Any candidate violating these constraints is rejected
before scoring by the taste model.

Constraints are split into two groups:
- Plan-level (1-7, 11-13): Checkable from the RemixPlan alone.
- Audio-dependent (8-10, 14): Require AudioMetadata or other analysis data.
  These gracefully skip when the relevant data is None.

Each constraint is a separate function for independent testability.
validate_candidate() runs ALL applicable constraints and collects every
violation (no short-circuiting).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.tempo import estimate_target_bpm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure codes
# ---------------------------------------------------------------------------

class ConstraintCode(str, Enum):
    """Failure reason codes for hard constraints."""

    SECTION_OVERLAP = "section_overlap"
    SECTION_TOO_SHORT = "section_too_short"
    BEAT_GRID_MISALIGN = "beat_grid_misalign"
    MVP_SOURCE_SPLIT = "mvp_source_split"
    TEMPO_STRETCH_UNSAFE = "tempo_stretch_unsafe"
    PITCH_SHIFT_UNSAFE = "pitch_shift_unsafe"
    TRANSITION_TOO_LONG = "transition_too_long"
    TRUE_PEAK_EXCEEDED = "true_peak_exceeded"
    LUFS_OUT_OF_RANGE = "lufs_out_of_range"
    LRA_TOO_LOW = "lra_too_low"
    NO_CONTRAST = "no_contrast"
    DUAL_LEAD_VOCALS = "dual_lead_vocals"
    OUTRO_QUALITY = "outro_quality"
    STEM_QUALITY_GATE = "stem_quality_gate"


@dataclass
class ConstraintViolation:
    """A single constraint violation with context for logging."""

    code: ConstraintCode
    message: str
    section_index: int | None = None


# ---------------------------------------------------------------------------
# Plan-level constraints (1-7, 11-13)
# ---------------------------------------------------------------------------

def check_contiguous_sections(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 1: Sections must be contiguous and non-overlapping.

    Rules:
    - First section starts at beat 0.
    - Boundaries are monotonically increasing.
    - Each section's end_beat == next section's start_beat (no gaps/overlaps).
    """
    violations: list[ConstraintViolation] = []
    sections = plan.sections

    if not sections:
        violations.append(ConstraintViolation(
            code=ConstraintCode.SECTION_OVERLAP,
            message="Plan has no sections",
        ))
        return violations

    if sections[0].start_beat != 0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.SECTION_OVERLAP,
            message=f"First section starts at beat {sections[0].start_beat}, expected 0",
            section_index=0,
        ))

    for i, sec in enumerate(sections):
        if sec.end_beat <= sec.start_beat:
            violations.append(ConstraintViolation(
                code=ConstraintCode.SECTION_OVERLAP,
                message=(
                    f"Section {i} ({sec.label}): end_beat {sec.end_beat} "
                    f"<= start_beat {sec.start_beat}"
                ),
                section_index=i,
            ))

    for i in range(len(sections) - 1):
        cur = sections[i]
        nxt = sections[i + 1]
        if cur.end_beat != nxt.start_beat:
            violations.append(ConstraintViolation(
                code=ConstraintCode.SECTION_OVERLAP,
                message=(
                    f"Gap/overlap between section {i} ({cur.label}) end_beat "
                    f"{cur.end_beat} and section {i + 1} ({nxt.label}) "
                    f"start_beat {nxt.start_beat}"
                ),
                section_index=i,
            ))

    return violations


# Major section labels that require the longer minimum (8 beats).
_MAJOR_LABELS = {"intro", "main", "build", "breakdown", "outro", "chorus", "verse"}


def check_section_min_length(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 2: Section minimum length.

    - Major sections (intro, main, build, breakdown, outro, chorus, verse)
      must be >= 8 beats.
    - All sections must be >= 4 beats.
    """
    violations: list[ConstraintViolation] = []

    for i, sec in enumerate(plan.sections):
        length = sec.end_beat - sec.start_beat
        label_lower = sec.label.lower()

        if label_lower in _MAJOR_LABELS and length < 8:
            violations.append(ConstraintViolation(
                code=ConstraintCode.SECTION_TOO_SHORT,
                message=(
                    f"Section {i} ({sec.label}): {length} beats, "
                    f"major section requires >= 8"
                ),
                section_index=i,
            ))
        elif length < 4:
            violations.append(ConstraintViolation(
                code=ConstraintCode.SECTION_TOO_SHORT,
                message=(
                    f"Section {i} ({sec.label}): {length} beats, minimum is 4"
                ),
                section_index=i,
            ))

    return violations


def check_beat_grid_alignment(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 3: Beat grid alignment.

    All section boundaries (start_beat and end_beat) must fall on 4-beat
    multiples.
    """
    violations: list[ConstraintViolation] = []

    for i, sec in enumerate(plan.sections):
        if sec.start_beat % 4 != 0:
            violations.append(ConstraintViolation(
                code=ConstraintCode.BEAT_GRID_MISALIGN,
                message=(
                    f"Section {i} ({sec.label}): start_beat {sec.start_beat} "
                    f"not on 4-beat grid"
                ),
                section_index=i,
            ))
        if sec.end_beat % 4 != 0:
            violations.append(ConstraintViolation(
                code=ConstraintCode.BEAT_GRID_MISALIGN,
                message=(
                    f"Section {i} ({sec.label}): end_beat {sec.end_beat} "
                    f"not on 4-beat grid"
                ),
                section_index=i,
            ))

    return violations


def check_mvp_source_split(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 4: MVP source split.

    One song provides vocals, the other provides instrumentals. The plan's
    vocal_source must be either "song_a" or "song_b".
    """
    violations: list[ConstraintViolation] = []

    if plan.vocal_source not in ("song_a", "song_b"):
        violations.append(ConstraintViolation(
            code=ConstraintCode.MVP_SOURCE_SPLIT,
            message=(
                f"vocal_source is '{plan.vocal_source}', "
                f"expected 'song_a' or 'song_b'"
            ),
        ))

    return violations


def check_tempo_stretch_safety(
    plan: RemixPlan,
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
) -> list[ConstraintViolation]:
    """Constraint 5: Tempo stretch safety.

    Maximum allowed stretch percentages per stem type:
    - Drums: <= 12%
    - Vocals: <= 35%
    - Other stems: <= 40%

    Requires both AudioMetadata objects to compute the stretch percentage
    between the two songs. Skips if either is None.
    """
    violations: list[ConstraintViolation] = []

    if meta_a is None or meta_b is None:
        return violations

    # Determine target BPM using the canonical algorithm.
    target_bpm = estimate_target_bpm(meta_a.bpm, meta_b.bpm, plan.tempo_source)

    # The vocal source song stretches its vocals to the target.
    # The instrumental source stretches drums/bass/other to the target.
    vocal_meta = meta_a if plan.vocal_source == "song_a" else meta_b
    inst_meta = meta_b if plan.vocal_source == "song_a" else meta_a

    vocal_stretch = abs(target_bpm - vocal_meta.bpm) / vocal_meta.bpm * 100
    inst_stretch = abs(target_bpm - inst_meta.bpm) / inst_meta.bpm * 100

    # Thresholds per stem type
    if vocal_stretch > 35.0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.TEMPO_STRETCH_UNSAFE,
            message=(
                f"Vocal stretch {vocal_stretch:.1f}% exceeds 35% limit "
                f"(source {vocal_meta.bpm:.1f} BPM -> target {target_bpm:.1f} BPM)"
            ),
        ))

    if inst_stretch > 12.0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.TEMPO_STRETCH_UNSAFE,
            message=(
                f"Drum stretch {inst_stretch:.1f}% exceeds 12% limit "
                f"(source {inst_meta.bpm:.1f} BPM -> target {target_bpm:.1f} BPM)"
            ),
        ))

    # "Other" stems also come from the instrumental source
    if inst_stretch > 40.0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.TEMPO_STRETCH_UNSAFE,
            message=(
                f"Other-stem stretch {inst_stretch:.1f}% exceeds 40% limit "
                f"(source {inst_meta.bpm:.1f} BPM -> target {target_bpm:.1f} BPM)"
            ),
        ))

    return violations


def check_transition_bounds(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 7: Transition bounds.

    transition_beats must be <= half the section length for every section.
    """
    violations: list[ConstraintViolation] = []

    for i, sec in enumerate(plan.sections):
        section_length = sec.end_beat - sec.start_beat
        max_transition = section_length // 2

        if sec.transition_beats > max_transition:
            violations.append(ConstraintViolation(
                code=ConstraintCode.TRANSITION_TOO_LONG,
                message=(
                    f"Section {i} ({sec.label}): transition_beats "
                    f"{sec.transition_beats} > half section length "
                    f"{max_transition} ({section_length} beats / 2)"
                ),
                section_index=i,
            ))

    return violations


def check_contrast_requirement(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 11: Contrast requirement.

    There must be at least one contrast event before the peak section.
    A contrast event is defined as a section boundary where:
    - Stem count drops by >= 2, OR
    - Energy drops by >= 20% (approximated by gain sum drop).

    The peak section is the one with the highest total stem gain. If there
    are no sections or only one section, this is vacuously satisfied.
    """
    violations: list[ConstraintViolation] = []
    sections = plan.sections

    if len(sections) < 2:
        return violations

    # Find peak section (highest total gain).
    def _total_gain(sec: Section) -> float:
        return sum(sec.stem_gains.values())

    peak_idx = max(range(len(sections)), key=lambda i: _total_gain(sections[i]))

    if peak_idx == 0:
        # Peak is the first section — no room for contrast before it.
        violations.append(ConstraintViolation(
            code=ConstraintCode.NO_CONTRAST,
            message="Peak section is the first section; no contrast event possible before it",
        ))
        return violations

    # Check for contrast events in sections before the peak.
    found_contrast = False
    for i in range(1, peak_idx + 1):
        prev = sections[i - 1]
        cur = sections[i]

        # Stem count drop: count stems with gain > 0.
        prev_active = sum(1 for g in prev.stem_gains.values() if g > 0)
        cur_active = sum(1 for g in cur.stem_gains.values() if g > 0)
        stem_drop = prev_active - cur_active

        # Energy drop: compare total gains.
        prev_energy = _total_gain(prev)
        cur_energy = _total_gain(cur)
        energy_drop_pct = 0.0
        if prev_energy > 0:
            energy_drop_pct = (prev_energy - cur_energy) / prev_energy * 100

        if stem_drop >= 2 or energy_drop_pct >= 20.0:
            found_contrast = True
            break

    if not found_contrast:
        violations.append(ConstraintViolation(
            code=ConstraintCode.NO_CONTRAST,
            message=(
                f"No contrast event found before peak section {peak_idx} "
                f"({sections[peak_idx].label}): need stem count drop >= 2 "
                f"or energy drop >= 20%"
            ),
        ))

    return violations


def check_no_dual_lead_vocals(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 12: No dual lead vocals.

    Only one lead vocal at a time. In the MVP, vocals come from a single
    source (vocal_source), so this checks that no section has more than one
    vocal-type stem active (gain > 0). The relevant stem name is "vocals".

    Since MVP uses a single vocal source, dual lead vocals would require
    both song_a and song_b vocals active. We check that no section has
    stem_gains entries for multiple vocal-like stems with non-zero gain.
    """
    violations: list[ConstraintViolation] = []

    # In the current stem schema, only "vocals" is a vocal stem.
    # This constraint is forward-looking: if we ever add "vocals_a" and
    # "vocals_b" as separate stems, this would catch overlaps.
    vocal_stems = {"vocals", "vocals_a", "vocals_b", "lead_vocals", "backing_vocals"}

    for i, sec in enumerate(plan.sections):
        active_vocal_count = sum(
            1 for stem, gain in sec.stem_gains.items()
            if stem in vocal_stems and gain > 0
        )
        if active_vocal_count > 1:
            violations.append(ConstraintViolation(
                code=ConstraintCode.DUAL_LEAD_VOCALS,
                message=(
                    f"Section {i} ({sec.label}): {active_vocal_count} "
                    f"vocal stems active simultaneously"
                ),
                section_index=i,
            ))

    return violations


def check_outro_quality(plan: RemixPlan) -> list[ConstraintViolation]:
    """Constraint 13: Outro quality.

    - Final section must be labeled "outro".
    - Must be >= 8 beats long.
    - Energy (total stem gain) must be below the plan's peak energy.
    """
    violations: list[ConstraintViolation] = []
    sections = plan.sections

    if not sections:
        violations.append(ConstraintViolation(
            code=ConstraintCode.OUTRO_QUALITY,
            message="Plan has no sections; cannot validate outro",
        ))
        return violations

    final = sections[-1]

    if final.label.lower() != "outro":
        violations.append(ConstraintViolation(
            code=ConstraintCode.OUTRO_QUALITY,
            message=(
                f"Final section labeled '{final.label}', expected 'outro'"
            ),
            section_index=len(sections) - 1,
        ))

    final_length = final.end_beat - final.start_beat
    if final_length < 8:
        violations.append(ConstraintViolation(
            code=ConstraintCode.OUTRO_QUALITY,
            message=(
                f"Outro is {final_length} beats, minimum is 8"
            ),
            section_index=len(sections) - 1,
        ))

    # Energy below peak: outro total gain must be less than the max total gain.
    def _total_gain(sec: Section) -> float:
        return sum(sec.stem_gains.values())

    peak_energy = max(_total_gain(s) for s in sections)
    outro_energy = _total_gain(final)

    if len(sections) > 1 and outro_energy >= peak_energy:
        violations.append(ConstraintViolation(
            code=ConstraintCode.OUTRO_QUALITY,
            message=(
                f"Outro energy ({outro_energy:.2f}) is not below peak "
                f"({peak_energy:.2f})"
            ),
            section_index=len(sections) - 1,
        ))

    return violations


# ---------------------------------------------------------------------------
# Audio-dependent constraints (8, 9, 10, 14)
# ---------------------------------------------------------------------------

def check_true_peak_ceiling(
    true_peak_dbtp: float | None,
) -> list[ConstraintViolation]:
    """Constraint 8: True peak ceiling <= -1.0 dBTP.

    Skips if true_peak_dbtp is None (data not yet available).
    """
    violations: list[ConstraintViolation] = []

    if true_peak_dbtp is None:
        return violations

    if true_peak_dbtp > -1.0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.TRUE_PEAK_EXCEEDED,
            message=(
                f"True peak {true_peak_dbtp:.1f} dBTP exceeds "
                f"-1.0 dBTP ceiling"
            ),
        ))

    return violations


def check_lufs_window(
    lufs: float | None,
    genre: str | None = None,
) -> list[ConstraintViolation]:
    """Constraint 9: LUFS window (genre-conditional).

    - lo-fi: -16 to -12 LUFS
    - EDM: -12 to -9 LUFS
    - default: within 2 dB of -12 LUFS (i.e. -14 to -10)

    Skips if lufs is None.
    """
    violations: list[ConstraintViolation] = []

    if lufs is None:
        return violations

    genre_lower = (genre or "").lower().strip()

    if genre_lower in ("lo-fi", "lofi", "lo fi"):
        low, high = -16.0, -12.0
    elif genre_lower in ("edm", "electronic", "house", "techno", "trance", "dubstep"):
        low, high = -12.0, -9.0
    else:
        low, high = -14.0, -10.0  # default: -12 +/- 2 dB

    if lufs < low or lufs > high:
        violations.append(ConstraintViolation(
            code=ConstraintCode.LUFS_OUT_OF_RANGE,
            message=(
                f"Integrated loudness {lufs:.1f} LUFS outside "
                f"[{low:.0f}, {high:.0f}] range "
                f"(genre: {genre_lower or 'default'})"
            ),
        ))

    return violations


def check_lra_floor(
    lra: float | None,
) -> list[ConstraintViolation]:
    """Constraint 10: Loudness Range (LRA) >= 4 dB.

    Prevents wall-of-sound mastering with no dynamic variation.
    Skips if lra is None.
    """
    violations: list[ConstraintViolation] = []

    if lra is None:
        return violations

    if lra < 4.0:
        violations.append(ConstraintViolation(
            code=ConstraintCode.LRA_TOO_LOW,
            message=(
                f"Loudness Range {lra:.1f} dB below 4.0 dB floor "
                f"(wall-of-sound risk)"
            ),
        ))

    return violations


def check_stem_quality_gate(
    plan: RemixPlan,
    stem_quality: float | None = None,
) -> list[ConstraintViolation]:
    """Constraint 14: Stem quality gate.

    No solo vocal sections when stem separation quality is below threshold
    (cross-bleed ratio < 0.7). A solo vocal section is one where only
    the "vocals" stem has gain > 0.

    Skips if stem_quality is None.
    """
    violations: list[ConstraintViolation] = []

    if stem_quality is None:
        return violations

    if stem_quality >= 0.7:
        return violations

    for i, sec in enumerate(plan.sections):
        active_stems = [stem for stem, gain in sec.stem_gains.items() if gain > 0]
        if active_stems == ["vocals"]:
            violations.append(ConstraintViolation(
                code=ConstraintCode.STEM_QUALITY_GATE,
                message=(
                    f"Section {i} ({sec.label}): solo vocal section with "
                    f"stem quality {stem_quality:.2f} < 0.7 threshold"
                ),
                section_index=i,
            ))

    return violations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_candidate(
    plan: RemixPlan,
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
    genre: str | None = None,
    stem_quality: float | None = None,
    true_peak_dbtp: float | None = None,
    lufs: float | None = None,
    lra: float | None = None,
) -> tuple[bool, list[ConstraintViolation]]:
    """Validate a candidate plan against all hard constraints.

    Returns (passed, violations). passed=True means no violations.
    Audio-dependent constraints are skipped when relevant data is None.

    Args:
        plan: The candidate RemixPlan to validate.
        meta_a: AudioMetadata for song A (optional, for tempo/pitch checks).
        meta_b: AudioMetadata for song B (optional, for tempo/pitch checks).
        genre: Genre string for LUFS window selection (optional).
        stem_quality: Cross-bleed ratio from stem separation (optional).
        true_peak_dbtp: Measured true peak in dBTP (optional).
        lufs: Measured integrated loudness in LUFS (optional).
        lra: Measured Loudness Range in dB (optional).

    Returns:
        Tuple of (passed: bool, violations: list[ConstraintViolation]).
    """
    violations: list[ConstraintViolation] = []

    # Plan-level constraints (always run)
    violations.extend(check_contiguous_sections(plan))
    violations.extend(check_section_min_length(plan))
    violations.extend(check_beat_grid_alignment(plan))
    violations.extend(check_mvp_source_split(plan))
    violations.extend(check_tempo_stretch_safety(plan, meta_a, meta_b))
    violations.extend(check_transition_bounds(plan))
    violations.extend(check_contrast_requirement(plan))
    violations.extend(check_no_dual_lead_vocals(plan))
    violations.extend(check_outro_quality(plan))

    # Audio-dependent constraints (skip when data is None)
    violations.extend(check_true_peak_ceiling(true_peak_dbtp))
    violations.extend(check_lufs_window(lufs, genre))
    violations.extend(check_lra_floor(lra))
    violations.extend(check_stem_quality_gate(plan, stem_quality))

    passed = len(violations) == 0

    if not passed:
        logger.debug(
            "Candidate failed %d constraint(s): %s",
            len(violations),
            ", ".join(v.code.value for v in violations),
        )

    return passed, violations
