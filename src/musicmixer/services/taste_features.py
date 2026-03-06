"""Tier 1 feature extraction for the taste training pipeline.

Extracts 20-30 metadata-only features from candidate RemixPlan objects.
All features are computable from the plan + cached song analysis (AudioMetadata)
without audio rendering.

Feature groups (Tier 1 only):
  Group 1: Structure (10-12 features)
  Group 2: Energy Arc (8-10 features)
  Group 4: Harmonic/Tempo Risk (6-8 features)
  Group 7: Prompt Fit (5-8 features)

Every extraction produces a versioned FeatureManifest so training data
is always tagged with the exact feature set used.
"""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_constraints import _key_semitone_distance
from musicmixer.services.tempo import estimate_target_bpm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Computation version -- bump when feature semantics change (even if names
# stay the same). This ensures the manifest hash changes on logic updates.
# ---------------------------------------------------------------------------
_COMPUTATION_VERSION = "2"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard stems that contribute to "active stem count" when gain > threshold.
_ALL_STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
_ACTIVE_GAIN_THRESHOLD = 0.05  # Below this gain, stem is considered inactive.

# Energy arc templates: normalized expected energy per section position (0-1).
# Each template maps a fractional position to an expected relative energy level.
_ENERGY_TEMPLATES: dict[str, list[float]] = {
    "classic": [0.3, 0.5, 0.7, 0.9, 1.0, 0.8, 0.5, 0.3],
    "edm": [0.3, 0.5, 0.8, 0.4, 0.9, 1.0, 0.7, 0.3],
    "hiphop": [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 0.7, 0.4],
    "dj_lift": [0.3, 0.6, 0.9, 1.0, 0.9, 0.7, 0.4, 0.3],
}

# Camelot wheel for key distance computation.
# Maps (key_name, scale) -> camelot_number (1-12) and letter (A=minor, B=major).
_CAMELOT_WHEEL: dict[tuple[str, str], tuple[int, str]] = {
    ("Ab", "minor"): (1, "A"), ("B", "major"): (1, "B"),
    ("Eb", "minor"): (2, "A"), ("F#", "major"): (2, "B"),
    ("Bb", "minor"): (3, "A"), ("Db", "major"): (3, "B"),
    ("F", "minor"): (4, "A"), ("Ab", "major"): (4, "B"),
    ("C", "minor"): (5, "A"), ("Eb", "major"): (5, "B"),
    ("G", "minor"): (6, "A"), ("Bb", "major"): (6, "B"),
    ("D", "minor"): (7, "A"), ("F", "major"): (7, "B"),
    ("A", "minor"): (8, "A"), ("C", "major"): (8, "B"),
    ("E", "minor"): (9, "A"), ("G", "major"): (9, "B"),
    ("B", "minor"): (10, "A"), ("D", "major"): (10, "B"),
    ("F#", "minor"): (11, "A"), ("A", "major"): (11, "B"),
    ("Db", "minor"): (12, "A"), ("E", "major"): (12, "B"),
    # Enharmonic aliases
    ("G#", "minor"): (1, "A"), ("Cb", "major"): (1, "B"),
    ("D#", "minor"): (2, "A"), ("Gb", "major"): (2, "B"),
    ("A#", "minor"): (3, "A"), ("C#", "major"): (3, "B"),
    ("C#", "minor"): (12, "A"), ("Fb", "major"): (12, "B"),
}

# Arrangement family templates: expected section label sequences.
_ARRANGEMENT_FAMILIES: dict[str, list[str]] = {
    "standard_arc": ["intro", "build", "main", "breakdown", "main", "outro"],
    "hook_first": ["intro", "main", "build", "main", "outro"],
    "dj_lift": ["build", "main", "breakdown", "main", "outro"],
    "quick_hit": ["intro", "main", "outro"],
}


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FeatureManifest:
    """Versioned manifest of feature names and computation methods."""
    version: str  # hash of feature names + computation version
    feature_names: list[str]
    tier: int = 1


@dataclass
class FeatureVector:
    """Extracted features for a single candidate plan."""
    features: dict[str, float]  # feature_name -> value
    manifest_version: str


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _compute_manifest_version(feature_names: list[str]) -> str:
    """Compute a stable hash from sorted feature names + computation version."""
    payload = "\n".join(sorted(feature_names)) + "\n__version__=" + _COMPUTATION_VERSION
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def get_manifest() -> FeatureManifest:
    """Return the current Tier 1 feature manifest."""
    names = sorted(_ALL_FEATURE_NAMES)
    version = _compute_manifest_version(names)
    return FeatureManifest(version=version, feature_names=names, tier=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_beat_length(section: Section) -> int:
    """Number of beats in a section."""
    return section.end_beat - section.start_beat


def _total_beats(plan: RemixPlan) -> int:
    """Total beat length of the remix plan."""
    if not plan.sections:
        return 0
    return plan.sections[-1].end_beat - plan.sections[0].start_beat


def _active_stem_count(section: Section) -> int:
    """Count stems with gain above threshold in a section."""
    return sum(
        1 for stem in _ALL_STEMS
        if section.stem_gains.get(stem, 0.0) > _ACTIVE_GAIN_THRESHOLD
    )


def _section_energy_proxy(section: Section) -> float:
    """Estimate section energy from stem gains (sum of all gains)."""
    return sum(section.stem_gains.get(s, 0.0) for s in _ALL_STEMS)


def _camelot_distance(key_a: str, scale_a: str, key_b: str, scale_b: str) -> int:
    """Compute Camelot wheel distance between two keys.

    Returns minimum distance on the Camelot wheel (0-6).
    Returns 6 (max penalty) if either key is unknown.
    """
    cam_a = _CAMELOT_WHEEL.get((key_a, scale_a))
    cam_b = _CAMELOT_WHEEL.get((key_b, scale_b))

    if cam_a is None or cam_b is None:
        return 6  # Unknown key -> max penalty

    num_a, letter_a = cam_a
    num_b, letter_b = cam_b

    # Same position (same number and letter) = 0
    # Adjacent number same letter = 1
    # Same number different letter (relative major/minor) = 1
    # Otherwise circular distance on the 12-position wheel

    if letter_a == letter_b:
        # Same mode: circular distance on the wheel
        raw = abs(num_a - num_b)
        return min(raw, 12 - raw)
    else:
        # Cross-mode: same number = 1 step, otherwise number distance + 1
        raw = abs(num_a - num_b)
        circ = min(raw, 12 - raw)
        return circ + 1


def _estimate_pitch_shift_semitones(
    plan: RemixPlan,
    meta_a: AudioMetadata | None,
    meta_b: AudioMetadata | None,
) -> float:
    """Estimate the pitch shift in semitones implied by the plan.

    If key_source is set to a song and both songs have key data, compute the
    semitone difference. Otherwise return 0.0 (no shift).
    """
    if plan.key_source == "none" or meta_a is None or meta_b is None:
        return 0.0
    if meta_a.key is None or meta_b.key is None:
        return 0.0
    if meta_a.scale is None or meta_b.scale is None:
        return 0.0

    # Compute actual semitone distance on the chromatic circle.
    semitones = _key_semitone_distance(meta_a.key, meta_b.key)
    if semitones is None:
        return 0.0  # Unknown key -> no shift estimate
    return float(semitones)


def _estimate_tempo_stretch(
    plan: RemixPlan,
    meta_a: AudioMetadata | None,
    meta_b: AudioMetadata | None,
) -> tuple[float, float]:
    """Estimate tempo stretch percentages for vocal and instrumental sources.

    Returns (vocal_stretch_pct, instrumental_stretch_pct) as positive floats.
    E.g. 12.0 means 12% stretch.
    """
    if meta_a is None or meta_b is None:
        return (0.0, 0.0)

    vocal_bpm = meta_a.bpm if plan.vocal_source == "song_a" else meta_b.bpm
    inst_bpm = meta_b.bpm if plan.vocal_source == "song_a" else meta_a.bpm

    if vocal_bpm <= 0 or inst_bpm <= 0:
        return (0.0, 0.0)

    # Use the canonical algorithm for target BPM.
    target = estimate_target_bpm(vocal_bpm, inst_bpm, plan.tempo_source)

    vocal_stretch = abs(target - vocal_bpm) / vocal_bpm * 100
    inst_stretch = abs(target - inst_bpm) / inst_bpm * 100
    return (vocal_stretch, inst_stretch)


# ---------------------------------------------------------------------------
# Group 1: Structure features (10 features)
# ---------------------------------------------------------------------------

def _extract_structure_features(plan: RemixPlan) -> dict[str, float]:
    """Extract structure-related features from the plan."""
    features: dict[str, float] = {}
    sections = plan.sections

    # 1. Section count
    features["struct_section_count"] = float(len(sections))

    if not sections:
        # Graceful degradation: return neutral defaults for remaining features
        features["struct_mean_section_duration_beats"] = 0.0
        features["struct_std_section_duration_beats"] = 0.0
        features["struct_min_section_length_beats"] = 0.0
        features["struct_max_section_length_beats"] = 0.0
        features["struct_phrase_boundary_hit_rate"] = 0.0
        features["struct_section_validity_ratio"] = 0.0
        features["struct_vocal_placement_fit"] = 0.5
        features["struct_arrangement_family_match"] = 0.0
        features["struct_total_beats"] = 0.0
        features["struct_section_count_norm"] = 0.0
        return features

    durations = [_section_beat_length(s) for s in sections]
    total = _total_beats(plan)

    # 2. Mean section duration (beats)
    features["struct_mean_section_duration_beats"] = float(statistics.mean(durations))

    # 3. Std section duration (beats)
    features["struct_std_section_duration_beats"] = (
        float(statistics.stdev(durations)) if len(durations) >= 2 else 0.0
    )

    # 4. Min section length (beats)
    features["struct_min_section_length_beats"] = float(min(durations))

    # 5. Max section length (beats)
    features["struct_max_section_length_beats"] = float(max(durations))

    # 6. Phrase boundary hit rate (% of boundaries on 16-beat multiples)
    boundaries = [s.start_beat for s in sections] + [sections[-1].end_beat]
    on_phrase = sum(1 for b in boundaries if b % 16 == 0)
    features["struct_phrase_boundary_hit_rate"] = on_phrase / len(boundaries) if boundaries else 0.0

    # 7. Section validity ratio (% of sections >= 8 beats with monotonic bounds)
    valid_count = 0
    for i, s in enumerate(sections):
        length = _section_beat_length(s)
        monotonic = s.start_beat < s.end_beat
        if i > 0:
            monotonic = monotonic and s.start_beat >= sections[i - 1].end_beat
        if length >= 8 and monotonic:
            valid_count += 1
    features["struct_section_validity_ratio"] = valid_count / len(sections)

    # 8. Vocal placement fit (intro/outro vocal mute compliance)
    # Score 1.0 if intro has vocals muted/low and outro has vocals muted/low.
    # Score 0.5 for partial compliance, 0.0 for neither.
    score = 0.0
    intro_sections = [s for s in sections if s.label == "intro"]
    outro_sections = [s for s in sections if s.label == "outro"]

    if intro_sections:
        intro_vocal_gain = intro_sections[0].stem_gains.get("vocals", 0.0)
        if intro_vocal_gain <= 0.2:
            score += 0.5
    else:
        # No intro section: neutral (neither reward nor penalize)
        score += 0.25

    if outro_sections:
        outro_vocal_gain = outro_sections[-1].stem_gains.get("vocals", 0.0)
        if outro_vocal_gain <= 0.2:
            score += 0.5
    else:
        score += 0.25

    features["struct_vocal_placement_fit"] = score

    # 9. Arrangement family match score
    # Find the best match across all family templates.
    plan_labels = [s.label for s in sections]
    best_match = 0.0
    for family_labels in _ARRANGEMENT_FAMILIES.values():
        # Use longest common subsequence ratio as match metric.
        match_score = _lcs_ratio(plan_labels, family_labels)
        best_match = max(best_match, match_score)
    features["struct_arrangement_family_match"] = best_match

    # 10. Total beats (useful as a scaling feature)
    features["struct_total_beats"] = float(total)

    # 11. Normalized section count (section count / total beats * 100)
    features["struct_section_count_norm"] = (
        len(sections) / total * 100 if total > 0 else 0.0
    )

    return features


def _lcs_ratio(seq_a: list[str], seq_b: list[str]) -> float:
    """Longest common subsequence length / max(len(a), len(b)).

    Returns 0.0 if either sequence is empty.
    """
    if not seq_a or not seq_b:
        return 0.0

    m, n = len(seq_a), len(seq_b)
    # DP table (space-optimized: two rows)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    return lcs_len / max(m, n)


# ---------------------------------------------------------------------------
# Group 2: Energy Arc features (8 features)
# ---------------------------------------------------------------------------

def _extract_energy_features(plan: RemixPlan) -> dict[str, float]:
    """Extract energy-arc features from the plan."""
    features: dict[str, float] = {}
    sections = plan.sections

    if not sections:
        features["energy_template_corr_classic"] = 0.0
        features["energy_template_corr_edm"] = 0.0
        features["energy_template_corr_hiphop"] = 0.0
        features["energy_template_corr_dj_lift"] = 0.0
        features["energy_peak_timing_score"] = 0.0
        features["energy_rise_fall_sanity"] = 0.5
        features["energy_contrast_index"] = 0.0
        features["energy_density_contour_smoothness"] = 0.0
        return features

    # Build per-section energy profile
    section_energies = [_section_energy_proxy(s) for s in sections]
    max_energy = max(section_energies) if section_energies else 1.0
    if max_energy <= 0:
        max_energy = 1.0
    # Normalize to 0-1
    norm_energies = [e / max_energy for e in section_energies]

    # 1-4. Template correlation (one feature per template)
    for template_name, template_values in _ENERGY_TEMPLATES.items():
        corr = _resample_and_correlate(norm_energies, template_values)
        features[f"energy_template_corr_{template_name}"] = corr

    # 5. Peak timing score: max energy section at 55-80% of timeline -> score 0-1
    total = _total_beats(plan)
    if total > 0:
        peak_idx = section_energies.index(max(section_energies))
        peak_section = sections[peak_idx]
        peak_midpoint = (peak_section.start_beat + peak_section.end_beat) / 2
        peak_position = peak_midpoint / total  # 0-1
        # Score 1.0 if peak is at 55-80%, decays linearly outside
        if 0.55 <= peak_position <= 0.80:
            features["energy_peak_timing_score"] = 1.0
        elif peak_position < 0.55:
            features["energy_peak_timing_score"] = max(0.0, peak_position / 0.55)
        else:
            features["energy_peak_timing_score"] = max(0.0, 1.0 - (peak_position - 0.80) / 0.20)
    else:
        features["energy_peak_timing_score"] = 0.0

    # 6. Rise/fall sanity: check verse->chorus gain delta is in +2 to +6 dB range
    # Use gain delta as proxy. Find first verse and first chorus-like section.
    verse_energy = None
    chorus_energy = None
    for s in sections:
        if s.label in ("build", "verse") and verse_energy is None:
            verse_energy = _section_energy_proxy(s)
        if s.label in ("main", "chorus") and chorus_energy is None:
            chorus_energy = _section_energy_proxy(s)
    if verse_energy is not None and chorus_energy is not None and verse_energy > 0:
        # Convert gain ratio to dB-like value
        gain_ratio = chorus_energy / verse_energy
        if gain_ratio > 0:
            db_delta = 20 * math.log10(gain_ratio)
        else:
            db_delta = 0.0
        # Score: 1.0 if delta in [2, 6], linear decay outside
        if 2.0 <= db_delta <= 6.0:
            features["energy_rise_fall_sanity"] = 1.0
        elif db_delta < 2.0:
            features["energy_rise_fall_sanity"] = max(0.0, db_delta / 2.0)
        else:
            features["energy_rise_fall_sanity"] = max(0.0, 1.0 - (db_delta - 6.0) / 6.0)
    else:
        features["energy_rise_fall_sanity"] = 0.5  # Neutral when labels not found

    # 7. Contrast index: normalized variance of stem activity across sections
    stem_counts = [_active_stem_count(s) for s in sections]
    if len(stem_counts) >= 2:
        mean_sc = statistics.mean(stem_counts)
        var_sc = statistics.variance(stem_counts)
        # Normalize by max possible variance (6 stems -> max variance ~ 9)
        features["energy_contrast_index"] = min(1.0, var_sc / 9.0)
    else:
        features["energy_contrast_index"] = 0.0

    # 8. Density contour smoothness: mean absolute change in active-stem count
    if len(stem_counts) >= 2:
        abs_changes = [abs(stem_counts[i] - stem_counts[i - 1]) for i in range(1, len(stem_counts))]
        # Lower is smoother. Normalize: 0 changes = 1.0, 6 changes per step = 0.0
        mean_change = statistics.mean(abs_changes)
        features["energy_density_contour_smoothness"] = max(0.0, 1.0 - mean_change / 6.0)
    else:
        features["energy_density_contour_smoothness"] = 1.0  # Single section = perfectly smooth

    return features


def _resample_and_correlate(observed: list[float], template: list[float]) -> float:
    """Resample observed and template to same length, return Pearson correlation.

    Returns 0.0 on degenerate inputs (all-constant, empty).
    """
    if len(observed) < 2 or len(template) < 2:
        return 0.0

    # Resample both to a common length
    n = max(len(observed), len(template))
    obs_resampled = _resample(observed, n)
    tmpl_resampled = _resample(template, n)

    # Pearson correlation
    mean_o = statistics.mean(obs_resampled)
    mean_t = statistics.mean(tmpl_resampled)

    cov = sum((o - mean_o) * (t - mean_t) for o, t in zip(obs_resampled, tmpl_resampled))
    var_o = sum((o - mean_o) ** 2 for o in obs_resampled)
    var_t = sum((t - mean_t) ** 2 for t in tmpl_resampled)

    denom = math.sqrt(var_o * var_t)
    if denom < 1e-10:
        return 0.0

    return cov / denom


def _resample(values: list[float], target_len: int) -> list[float]:
    """Linearly resample a list of floats to target_len."""
    if target_len <= 0:
        return []
    if len(values) == 0:
        return [0.0] * target_len
    if len(values) == 1 or target_len == 1:
        return [values[0]] * target_len

    result = []
    for i in range(target_len):
        # Map i in [0, target_len-1] to source index in [0, len-1]
        src = i * (len(values) - 1) / (target_len - 1)
        lo = int(src)
        hi = min(lo + 1, len(values) - 1)
        frac = src - lo
        result.append(values[lo] * (1 - frac) + values[hi] * frac)
    return result


# ---------------------------------------------------------------------------
# Group 4: Harmonic/Tempo Risk features (6 features)
# ---------------------------------------------------------------------------

def _extract_harmonic_tempo_features(
    plan: RemixPlan,
    meta_a: AudioMetadata | None,
    meta_b: AudioMetadata | None,
) -> dict[str, float]:
    """Extract harmonic and tempo risk features."""
    features: dict[str, float] = {}

    # 1. Camelot distance after pitch decision (0 = best, >3 penalized heavily)
    if (
        meta_a is not None and meta_b is not None
        and meta_a.key is not None and meta_b.key is not None
        and meta_a.scale is not None and meta_b.scale is not None
    ):
        dist = _camelot_distance(meta_a.key, meta_a.scale, meta_b.key, meta_b.scale)
        # If a key_source is specified, the pitch shift would close the distance.
        # For MVP, report raw distance; the model learns the penalty.
        features["harmonic_camelot_distance"] = float(dist)
    else:
        features["harmonic_camelot_distance"] = 3.0  # Neutral unknown

    # 2. Absolute pitch shift semitones
    features["harmonic_pitch_shift_semitones"] = _estimate_pitch_shift_semitones(
        plan, meta_a, meta_b
    )

    # 3-4. Tempo stretch per source type
    vocal_stretch, inst_stretch = _estimate_tempo_stretch(plan, meta_a, meta_b)
    features["tempo_vocal_stretch_pct"] = vocal_stretch
    features["tempo_instrumental_stretch_pct"] = inst_stretch

    # 5. Max stretch (combined risk indicator)
    features["tempo_max_stretch_pct"] = max(vocal_stretch, inst_stretch)

    # 6. Stretch direction penalty: slow-down penalized more than speed-up
    # Positive value = speed-up, negative = slow-down. Penalty is asymmetric.
    if meta_a is not None and meta_b is not None:
        vocal_bpm = meta_a.bpm if plan.vocal_source == "song_a" else meta_b.bpm
        inst_bpm = meta_b.bpm if plan.vocal_source == "song_a" else meta_a.bpm

        if vocal_bpm > 0 and inst_bpm > 0:
            target = estimate_target_bpm(vocal_bpm, inst_bpm, plan.tempo_source)

            # Vocal direction: positive = speeding up vocals, negative = slowing down
            vocal_direction = (target - vocal_bpm) / vocal_bpm * 100
            # Penalty: slowdown gets 1.5x weight
            if vocal_direction < 0:
                features["tempo_stretch_direction_penalty"] = abs(vocal_direction) * 1.5
            else:
                features["tempo_stretch_direction_penalty"] = abs(vocal_direction)
        else:
            features["tempo_stretch_direction_penalty"] = 0.0
    else:
        features["tempo_stretch_direction_penalty"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Group 7: Prompt Fit features (4 features)
# ---------------------------------------------------------------------------

def _extract_prompt_fit_features(plan: RemixPlan) -> dict[str, float]:
    """Extract prompt-fit proxy features.

    For MVP, these are computed from plan metadata rather than actual prompt
    parsing (which is deferred to Phase 2+ with NLU).
    """
    features: dict[str, float] = {}
    sections = plan.sections

    # 1. Energy level from gain profile (proxy for prompt energy intent)
    # Average total gain across all sections, normalized by max possible (6 stems * 1.0)
    if sections:
        avg_energy = statistics.mean(_section_energy_proxy(s) for s in sections)
        features["prompt_energy_level"] = min(1.0, avg_energy / 6.0)
    else:
        features["prompt_energy_level"] = 0.0

    # 2. Structural complexity (section count / duration proxy)
    # Higher = more complex arrangement
    total = _total_beats(plan)
    if total > 0 and sections:
        # Normalize: 5 sections per 200 beats ~ 1.0
        features["prompt_structural_complexity"] = len(sections) / total * 200 / 5
    else:
        features["prompt_structural_complexity"] = 0.0

    # 3. Vocal prominence (vocal duty cycle -- fraction of remix with vocals active)
    if sections and total > 0:
        vocal_beats = sum(
            _section_beat_length(s)
            for s in sections
            if s.stem_gains.get("vocals", 0.0) > _ACTIVE_GAIN_THRESHOLD
        )
        features["prompt_vocal_prominence"] = vocal_beats / total
    else:
        features["prompt_vocal_prominence"] = 0.0

    # 4. Genre compatibility score (placeholder: 1.0 for now)
    features["prompt_genre_compatibility"] = 1.0

    return features


# ---------------------------------------------------------------------------
# Feature name registry (defines the canonical set)
# ---------------------------------------------------------------------------

# Collect all feature names by running extraction on a dummy plan to discover names.
# This is safer than maintaining a manual list that can drift from the code.
def _discover_feature_names() -> list[str]:
    """Run extraction logic to discover all feature names."""
    dummy_section = Section(
        label="main",
        start_beat=0,
        end_beat=32,
        stem_gains={"vocals": 1.0, "drums": 0.7, "bass": 0.8},
        transition_in="crossfade",
        transition_beats=4,
    )
    dummy_plan = RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=60.0,
        start_time_instrumental=0.0,
        end_time_instrumental=60.0,
        sections=[dummy_section],
        tempo_source="weighted_midpoint",
        key_source="none",
        explanation="dummy",
    )
    features: dict[str, float] = {}
    features.update(_extract_structure_features(dummy_plan))
    features.update(_extract_energy_features(dummy_plan))
    features.update(_extract_harmonic_tempo_features(dummy_plan, None, None))
    features.update(_extract_prompt_fit_features(dummy_plan))
    return sorted(features.keys())


_ALL_FEATURE_NAMES: list[str] = _discover_feature_names()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(
    plan: RemixPlan,
    meta_a: AudioMetadata | None = None,
    meta_b: AudioMetadata | None = None,
) -> FeatureVector:
    """Extract Tier 1 features from a candidate plan.

    Features that require audio metadata gracefully degrade when
    meta_a/meta_b are None (use default/neutral values).

    Args:
        plan: The candidate RemixPlan to evaluate.
        meta_a: Audio metadata for song A (optional).
        meta_b: Audio metadata for song B (optional).

    Returns:
        FeatureVector with all Tier 1 features and the current manifest version.
    """
    manifest = get_manifest()

    features: dict[str, float] = {}
    features.update(_extract_structure_features(plan))
    features.update(_extract_energy_features(plan))
    features.update(_extract_harmonic_tempo_features(plan, meta_a, meta_b))
    features.update(_extract_prompt_fit_features(plan))

    # Verify all features are present and are floats
    for name in manifest.feature_names:
        if name not in features:
            logger.warning("Missing feature %s, defaulting to 0.0", name)
            features[name] = 0.0
        elif not isinstance(features[name], (int, float)):
            logger.warning(
                "Feature %s has non-numeric value %r, defaulting to 0.0",
                name,
                features[name],
            )
            features[name] = 0.0

    return FeatureVector(features=features, manifest_version=manifest.version)
