"""Tests for musicmixer.services.taste_features -- Tier 1 feature extraction.

Covers:
  - Feature count in 20-30 range
  - All features return float values
  - Works with minimal input (plan only, no metadata)
  - Works with full input (plan + metadata)
  - Manifest version stability and change detection
  - Structure feature correctness
  - Energy arc feature correctness
  - Harmonic/tempo feature correctness
  - Prompt fit feature correctness
  - Edge cases: empty sections, single section, minimal plan
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.taste_features import (
    FeatureManifest,
    FeatureVector,
    _active_stem_count,
    _camelot_distance,
    _extract_energy_features,
    _extract_harmonic_tempo_features,
    _extract_prompt_fit_features,
    _extract_structure_features,
    _lcs_ratio,
    _resample_and_correlate,
    _section_energy_proxy,
    extract_features,
    get_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_section(
    label: str = "main",
    start_beat: int = 0,
    end_beat: int = 32,
    vocals: float = 1.0,
    drums: float = 0.7,
    bass: float = 0.8,
    guitar: float = 0.0,
    piano: float = 0.0,
    other: float = 0.0,
    transition_in: str = "crossfade",
    transition_beats: int = 4,
) -> Section:
    """Create a Section with convenient defaults."""
    return Section(
        label=label,
        start_beat=start_beat,
        end_beat=end_beat,
        stem_gains={
            "vocals": vocals,
            "drums": drums,
            "bass": bass,
            "guitar": guitar,
            "piano": piano,
            "other": other,
        },
        transition_in=transition_in,
        transition_beats=transition_beats,
    )


def _make_plan(
    sections: list[Section] | None = None,
    vocal_source: str = "song_a",
    tempo_source: str = "weighted_midpoint",
) -> RemixPlan:
    """Create a RemixPlan with convenient defaults."""
    if sections is None:
        sections = [
            _make_section("intro", 0, 16, vocals=0.0, drums=0.3, bass=0.0),
            _make_section("build", 16, 48, vocals=0.5, drums=0.5, bass=0.5),
            _make_section("main", 48, 112, vocals=1.0, drums=0.8, bass=0.8, guitar=0.5),
            _make_section("breakdown", 112, 144, vocals=0.3, drums=0.3, bass=0.5),
            _make_section("main", 144, 208, vocals=1.0, drums=0.9, bass=0.9, guitar=0.6, piano=0.3),
            _make_section("outro", 208, 240, vocals=0.0, drums=0.4, bass=0.3),
        ]
    return RemixPlan(
        vocal_source=vocal_source,
        start_time_vocal=0.0,
        end_time_vocal=120.0,
        start_time_instrumental=0.0,
        end_time_instrumental=120.0,
        sections=sections,
        tempo_source=tempo_source,
        explanation="test plan",
    )


def _make_metadata(
    bpm: float = 120.0,
    duration: float = 240.0,
    key: str | None = None,
    scale: str | None = None,
) -> AudioMetadata:
    """Create a minimal AudioMetadata for testing."""
    total_beats = round(bpm * duration / 60 / 4) * 4
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=np.arange(0, 100, dtype=np.intp),
        duration_seconds=duration,
        total_beats=max(total_beats, 4),
        key=key,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_returns_correct_type(self) -> None:
        manifest = get_manifest()
        assert isinstance(manifest, FeatureManifest)
        assert isinstance(manifest.version, str)
        assert isinstance(manifest.feature_names, list)
        assert manifest.tier == 1

    def test_feature_count_in_range(self) -> None:
        """Tier 1 should have 20-30 features."""
        manifest = get_manifest()
        assert 20 <= len(manifest.feature_names) <= 30, (
            f"Expected 20-30 features, got {len(manifest.feature_names)}"
        )

    def test_manifest_version_stable(self) -> None:
        """Same inputs produce the same manifest hash."""
        m1 = get_manifest()
        m2 = get_manifest()
        assert m1.version == m2.version
        assert m1.feature_names == m2.feature_names

    def test_manifest_version_is_hex(self) -> None:
        """Version should be a hex string (from sha256)."""
        manifest = get_manifest()
        assert len(manifest.version) == 12
        int(manifest.version, 16)  # Should not raise

    def test_all_feature_names_are_strings(self) -> None:
        manifest = get_manifest()
        assert all(isinstance(name, str) for name in manifest.feature_names)

    def test_feature_names_sorted(self) -> None:
        manifest = get_manifest()
        assert manifest.feature_names == sorted(manifest.feature_names)

    def test_feature_names_have_group_prefix(self) -> None:
        """Every feature should have a known group prefix."""
        valid_prefixes = ("struct_", "energy_", "harmonic_", "tempo_", "prompt_")
        manifest = get_manifest()
        for name in manifest.feature_names:
            assert any(name.startswith(p) for p in valid_prefixes), (
                f"Feature {name!r} has no valid group prefix"
            )


# ---------------------------------------------------------------------------
# Extract features: basic tests
# ---------------------------------------------------------------------------

class TestExtractFeaturesBasic:
    def test_returns_feature_vector(self) -> None:
        plan = _make_plan()
        result = extract_features(plan)
        assert isinstance(result, FeatureVector)
        assert isinstance(result.features, dict)
        assert isinstance(result.manifest_version, str)

    def test_all_features_are_floats(self) -> None:
        """Every feature value must be a float (CatBoost-compatible)."""
        plan = _make_plan()
        result = extract_features(plan)
        for name, value in result.features.items():
            assert isinstance(value, float), (
                f"Feature {name!r} has type {type(value).__name__}, expected float"
            )

    def test_feature_count_matches_manifest(self) -> None:
        manifest = get_manifest()
        plan = _make_plan()
        result = extract_features(plan)
        assert len(result.features) == len(manifest.feature_names)

    def test_manifest_version_matches(self) -> None:
        manifest = get_manifest()
        plan = _make_plan()
        result = extract_features(plan)
        assert result.manifest_version == manifest.version

    def test_all_manifest_names_present(self) -> None:
        manifest = get_manifest()
        plan = _make_plan()
        result = extract_features(plan)
        for name in manifest.feature_names:
            assert name in result.features, f"Missing feature: {name!r}"

    def test_works_without_metadata(self) -> None:
        """Features should work with plan only, no AudioMetadata."""
        plan = _make_plan()
        result = extract_features(plan, meta_a=None, meta_b=None)
        assert len(result.features) == len(get_manifest().feature_names)
        # All values should be real numbers (not NaN or inf)
        for name, value in result.features.items():
            assert math.isfinite(value), f"Feature {name!r} is not finite: {value}"

    def test_works_with_full_metadata(self) -> None:
        """Features should work with plan + both AudioMetadata objects."""
        plan = _make_plan()
        meta_a = _make_metadata(bpm=120.0, key="C", scale="major")
        meta_b = _make_metadata(bpm=128.0, key="G", scale="major")
        result = extract_features(plan, meta_a=meta_a, meta_b=meta_b)
        assert len(result.features) == len(get_manifest().feature_names)
        for name, value in result.features.items():
            assert math.isfinite(value), f"Feature {name!r} is not finite: {value}"


# ---------------------------------------------------------------------------
# Structure features
# ---------------------------------------------------------------------------

class TestStructureFeatures:
    def test_section_count(self) -> None:
        plan = _make_plan()
        features = _extract_structure_features(plan)
        assert features["struct_section_count"] == 6.0

    def test_mean_section_duration(self) -> None:
        sections = [
            _make_section("intro", 0, 16),
            _make_section("main", 16, 48),
            _make_section("outro", 48, 64),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        # Durations: 16, 32, 16 -> mean = 64/3 ~ 21.33
        expected_mean = (16 + 32 + 16) / 3
        assert abs(features["struct_mean_section_duration_beats"] - expected_mean) < 0.01

    def test_std_section_duration(self) -> None:
        sections = [
            _make_section("intro", 0, 16),
            _make_section("main", 16, 48),
            _make_section("outro", 48, 64),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        # stdev of [16, 32, 16]
        import statistics
        expected_std = statistics.stdev([16, 32, 16])
        assert abs(features["struct_std_section_duration_beats"] - expected_std) < 0.01

    def test_min_section_length(self) -> None:
        sections = [
            _make_section("intro", 0, 8),
            _make_section("main", 8, 72),
            _make_section("outro", 72, 80),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_min_section_length_beats"] == 8.0

    def test_max_section_length(self) -> None:
        sections = [
            _make_section("intro", 0, 8),
            _make_section("main", 8, 72),
            _make_section("outro", 72, 80),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_max_section_length_beats"] == 64.0

    def test_phrase_boundary_hit_rate_perfect(self) -> None:
        """All boundaries on 16-beat multiples -> hit rate = 1.0."""
        sections = [
            _make_section("intro", 0, 16),
            _make_section("main", 16, 48),
            _make_section("outro", 48, 64),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_phrase_boundary_hit_rate"] == 1.0

    def test_phrase_boundary_hit_rate_partial(self) -> None:
        """Some boundaries not on 16-beat multiples."""
        sections = [
            _make_section("intro", 0, 12),   # 12 is NOT on 16-beat multiple
            _make_section("main", 12, 48),
            _make_section("outro", 48, 64),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        # Boundaries: 0, 12, 48, 64. On 16-beat: 0, 48, 64 = 3/4
        assert abs(features["struct_phrase_boundary_hit_rate"] - 0.75) < 0.01

    def test_section_validity_ratio(self) -> None:
        """Mix of valid and invalid sections."""
        sections = [
            _make_section("intro", 0, 4),     # Too short (< 8 beats)
            _make_section("main", 4, 36),      # Valid (32 beats)
            _make_section("outro", 36, 52),    # Valid (16 beats)
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        # 2 valid out of 3
        assert abs(features["struct_section_validity_ratio"] - 2 / 3) < 0.01

    def test_vocal_placement_fit_good(self) -> None:
        """Intro has vocals muted, outro has vocals muted -> 1.0."""
        sections = [
            _make_section("intro", 0, 16, vocals=0.0),
            _make_section("main", 16, 48, vocals=1.0),
            _make_section("outro", 48, 64, vocals=0.1),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_vocal_placement_fit"] == 1.0

    def test_vocal_placement_fit_bad(self) -> None:
        """Intro has loud vocals, outro has loud vocals -> 0.0."""
        sections = [
            _make_section("intro", 0, 16, vocals=0.8),
            _make_section("main", 16, 48, vocals=1.0),
            _make_section("outro", 48, 64, vocals=0.9),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_vocal_placement_fit"] == 0.0

    def test_arrangement_family_match_standard_arc(self) -> None:
        """A plan closely matching standard arc should have high match score."""
        sections = [
            _make_section("intro", 0, 16),
            _make_section("build", 16, 48),
            _make_section("main", 48, 112),
            _make_section("breakdown", 112, 144),
            _make_section("main", 144, 208),
            _make_section("outro", 208, 240),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        # Should have a reasonable match (exact depends on LCS calculation)
        assert features["struct_arrangement_family_match"] > 0.5

    def test_total_beats(self) -> None:
        sections = [
            _make_section("intro", 0, 16),
            _make_section("main", 16, 80),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_structure_features(plan)
        assert features["struct_total_beats"] == 80.0


# ---------------------------------------------------------------------------
# Energy arc features
# ---------------------------------------------------------------------------

class TestEnergyFeatures:
    def test_template_correlations_are_bounded(self) -> None:
        """Template correlations should be in [-1, 1]."""
        plan = _make_plan()
        features = _extract_energy_features(plan)
        for template_name in ("classic", "edm", "hiphop", "dj_lift"):
            key = f"energy_template_corr_{template_name}"
            assert -1.0 <= features[key] <= 1.0 + 1e-9, (
                f"{key} = {features[key]} is out of bounds"
            )

    def test_peak_timing_score_center(self) -> None:
        """Peak at ~65% of timeline should score 1.0."""
        # Build plan where peak is at position 5 of 8 (62.5%)
        sections = [
            _make_section("intro", 0, 16, vocals=0.0, drums=0.3),
            _make_section("build", 16, 32, vocals=0.3, drums=0.5),
            _make_section("main", 32, 48, vocals=0.8, drums=0.7, bass=0.8),
            _make_section("build", 48, 64, vocals=0.5, drums=0.6),
            _make_section("main", 64, 80, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
            _make_section("breakdown", 80, 96, vocals=0.3, drums=0.3),
            _make_section("main", 96, 112, vocals=0.8, drums=0.7),
            _make_section("outro", 112, 128, vocals=0.0, drums=0.2),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Peak is section at 64-80 beats, midpoint 72/128 = 0.5625
        # That's right at the boundary -- score should be ~1.0
        assert features["energy_peak_timing_score"] >= 0.9

    def test_peak_timing_score_early(self) -> None:
        """Peak at the very start should have low score."""
        sections = [
            _make_section("main", 0, 32, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
            _make_section("build", 32, 64, vocals=0.3, drums=0.3),
            _make_section("outro", 64, 128, vocals=0.1, drums=0.1),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Peak at 0-32, midpoint 16/128 = 0.125 -> low score
        assert features["energy_peak_timing_score"] < 0.5

    def test_rise_fall_sanity_good(self) -> None:
        """Verse->chorus with +3dB delta should score well."""
        # verse energy ~ 2.0, chorus energy ~ 2.0 * 10^(3/20) ~ 2.83
        sections = [
            _make_section("intro", 0, 16, vocals=0.0, drums=0.2),
            _make_section("build", 16, 48, vocals=0.4, drums=0.4, bass=0.4),
            _make_section("main", 48, 80, vocals=1.0, drums=0.8, bass=0.8),
            _make_section("outro", 80, 96, vocals=0.0, drums=0.2),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Build energy = 1.2, Main energy = 2.6
        # dB delta = 20*log10(2.6/1.2) ~ 6.7 -> slightly outside 2-6
        # But should still have a decent score
        assert features["energy_rise_fall_sanity"] > 0.3

    def test_contrast_index_uniform(self) -> None:
        """All sections with same stem count -> zero contrast."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.5, drums=0.5, bass=0.5),
            _make_section("main", 32, 64, vocals=0.8, drums=0.8, bass=0.8),
            _make_section("outro", 64, 96, vocals=0.6, drums=0.6, bass=0.6),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # All have 3 active stems -> variance = 0
        assert features["energy_contrast_index"] == 0.0

    def test_contrast_index_varied(self) -> None:
        """Sections with very different stem counts -> high contrast."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.0, drums=0.5),
            _make_section("main", 32, 64, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
            _make_section("outro", 64, 96, vocals=0.0, drums=0.3),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Stem counts: 1, 6, 1 -> high variance
        assert features["energy_contrast_index"] > 0.5

    def test_density_contour_smoothness_smooth(self) -> None:
        """Gradually changing stem counts -> high smoothness."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.0, drums=0.5),
            _make_section("build", 32, 64, vocals=0.5, drums=0.5, bass=0.5),
            _make_section("main", 64, 96, vocals=1.0, drums=0.8, bass=0.8, guitar=0.5),
            _make_section("outro", 96, 128, vocals=0.5, drums=0.5, bass=0.5),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Stem counts: 1, 3, 4, 3 -> changes: 2, 1, 1 -> mean 1.33 -> smoothness ~ 0.78
        assert features["energy_density_contour_smoothness"] > 0.7

    def test_density_contour_smoothness_jagged(self) -> None:
        """Wildly changing stem counts -> low smoothness."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.0, drums=0.5, bass=0.0),
            _make_section("main", 32, 64, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
            _make_section("breakdown", 64, 96, vocals=0.0, drums=0.3, bass=0.0),
            _make_section("main", 96, 128, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_energy_features(plan)
        # Stem counts: 1, 6, 1, 6 -> changes: 5, 5, 5 -> mean 5.0 -> smoothness ~ 0.17
        assert features["energy_density_contour_smoothness"] < 0.3


# ---------------------------------------------------------------------------
# Harmonic/Tempo features
# ---------------------------------------------------------------------------

class TestHarmonicTempoFeatures:
    def test_camelot_distance_same_key(self) -> None:
        """Same key -> distance 0."""
        assert _camelot_distance("C", "major", "C", "major") == 0

    def test_camelot_distance_adjacent(self) -> None:
        """Adjacent keys on Camelot wheel -> distance 1."""
        assert _camelot_distance("C", "major", "G", "major") == 1

    def test_camelot_distance_relative_minor(self) -> None:
        """Relative minor (same number, different letter) -> distance 1."""
        assert _camelot_distance("C", "major", "A", "minor") == 1

    def test_camelot_distance_unknown_key(self) -> None:
        """Unknown key -> max penalty 6."""
        assert _camelot_distance("X", "major", "C", "major") == 6

    def test_harmonic_features_no_metadata(self) -> None:
        """Without metadata, harmonic features should use neutral defaults."""
        plan = _make_plan()
        features = _extract_harmonic_tempo_features(plan, None, None)
        assert features["harmonic_camelot_distance"] == 3.0  # neutral
        assert features["harmonic_pitch_shift_semitones"] == 0.0
        assert features["tempo_vocal_stretch_pct"] == 0.0
        assert features["tempo_instrumental_stretch_pct"] == 0.0

    def test_harmonic_features_with_metadata(self) -> None:
        """With metadata, should compute real values."""
        plan = _make_plan()
        meta_a = _make_metadata(bpm=120.0, key="C", scale="major")
        meta_b = _make_metadata(bpm=130.0, key="G", scale="major")
        features = _extract_harmonic_tempo_features(plan, meta_a, meta_b)
        # C major (8B) to G major (9B) = Camelot distance 1
        assert features["harmonic_camelot_distance"] == 1.0
        # Pitch shift is always 0.0 now (key matching handled by pipeline)
        assert features["harmonic_pitch_shift_semitones"] == 0.0
        # Tempo stretch should be nonzero
        assert features["tempo_vocal_stretch_pct"] > 0.0

    def test_tempo_stretch_same_bpm(self) -> None:
        """Same BPM -> zero stretch."""
        plan = _make_plan()
        meta_a = _make_metadata(bpm=120.0)
        meta_b = _make_metadata(bpm=120.0)
        features = _extract_harmonic_tempo_features(plan, meta_a, meta_b)
        assert features["tempo_vocal_stretch_pct"] == 0.0
        assert features["tempo_instrumental_stretch_pct"] == 0.0
        assert features["tempo_max_stretch_pct"] == 0.0
        assert features["tempo_stretch_direction_penalty"] == 0.0

    def test_stretch_direction_penalty_slowdown(self) -> None:
        """Slowing down vocals should be penalized more than speeding up."""
        # Plan where vocal is faster than target (slowing down)
        plan = _make_plan(tempo_source="song_b")
        meta_a = _make_metadata(bpm=140.0)  # vocal source, faster
        meta_b = _make_metadata(bpm=100.0)  # instrumental source, slower -> target
        features_slow = _extract_harmonic_tempo_features(plan, meta_a, meta_b)

        # Plan where vocal is slower than target (speeding up)
        plan2 = _make_plan(tempo_source="song_b")
        meta_a2 = _make_metadata(bpm=100.0)  # vocal source, slower
        meta_b2 = _make_metadata(bpm=140.0)  # instrumental source, faster -> target
        features_fast = _extract_harmonic_tempo_features(plan2, meta_a2, meta_b2)

        # Both have same absolute stretch, but slowdown should have higher penalty
        # (Slowdown penalty is 1.5x)
        assert features_slow["tempo_stretch_direction_penalty"] > features_fast["tempo_stretch_direction_penalty"]


# ---------------------------------------------------------------------------
# Prompt fit features
# ---------------------------------------------------------------------------

class TestPromptFitFeatures:
    def test_energy_level_bounded(self) -> None:
        """Energy level should be in [0, 1]."""
        plan = _make_plan()
        features = _extract_prompt_fit_features(plan)
        assert 0.0 <= features["prompt_energy_level"] <= 1.0

    def test_structural_complexity_increases_with_sections(self) -> None:
        """More sections per beat -> higher complexity."""
        few = _make_plan(sections=[
            _make_section("intro", 0, 64),
            _make_section("main", 64, 192),
            _make_section("outro", 192, 256),
        ])
        many = _make_plan(sections=[
            _make_section("intro", 0, 32),
            _make_section("build", 32, 64),
            _make_section("main", 64, 96),
            _make_section("breakdown", 96, 128),
            _make_section("main", 128, 160),
            _make_section("build", 160, 192),
            _make_section("main", 192, 224),
            _make_section("outro", 224, 256),
        ])
        few_feat = _extract_prompt_fit_features(few)
        many_feat = _extract_prompt_fit_features(many)
        assert many_feat["prompt_structural_complexity"] > few_feat["prompt_structural_complexity"]

    def test_vocal_prominence_full_vocals(self) -> None:
        """All sections have vocals active -> prominence = 1.0."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.8),
            _make_section("main", 32, 64, vocals=1.0),
            _make_section("outro", 64, 96, vocals=0.5),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_prompt_fit_features(plan)
        assert features["prompt_vocal_prominence"] == 1.0

    def test_vocal_prominence_no_vocals(self) -> None:
        """No vocals anywhere -> prominence = 0.0."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.0),
            _make_section("main", 32, 64, vocals=0.0),
            _make_section("outro", 64, 96, vocals=0.0),
        ]
        plan = _make_plan(sections=sections)
        features = _extract_prompt_fit_features(plan)
        assert features["prompt_vocal_prominence"] == 0.0

    def test_genre_compatibility_placeholder(self) -> None:
        """Genre compatibility should be 1.0 (placeholder)."""
        plan = _make_plan()
        features = _extract_prompt_fit_features(plan)
        assert features["prompt_genre_compatibility"] == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_sections(self) -> None:
        """Plan with no sections should return neutral defaults."""
        plan = _make_plan(sections=[])
        result = extract_features(plan)
        assert len(result.features) == len(get_manifest().feature_names)
        for name, value in result.features.items():
            assert math.isfinite(value), f"Feature {name!r} is not finite: {value}"
        assert result.features["struct_section_count"] == 0.0

    def test_single_section(self) -> None:
        """Plan with exactly one section."""
        plan = _make_plan(sections=[
            _make_section("main", 0, 64, vocals=1.0, drums=0.8, bass=0.7),
        ])
        result = extract_features(plan)
        assert len(result.features) == len(get_manifest().feature_names)
        assert result.features["struct_section_count"] == 1.0
        assert result.features["struct_std_section_duration_beats"] == 0.0
        # Single section -> density smoothness = 1.0 (no changes)
        assert result.features["energy_density_contour_smoothness"] == 1.0

    def test_very_short_plan(self) -> None:
        """Plan with a single 4-beat section."""
        plan = _make_plan(sections=[
            _make_section("intro", 0, 4, vocals=0.0, drums=0.5),
        ])
        result = extract_features(plan)
        assert result.features["struct_total_beats"] == 4.0
        assert result.features["struct_min_section_length_beats"] == 4.0

    def test_zero_gain_sections(self) -> None:
        """All gains are zero (silent plan)."""
        sections = [
            _make_section("intro", 0, 32, vocals=0.0, drums=0.0, bass=0.0),
            _make_section("main", 32, 64, vocals=0.0, drums=0.0, bass=0.0),
        ]
        plan = _make_plan(sections=sections)
        result = extract_features(plan)
        assert result.features["prompt_energy_level"] == 0.0
        assert result.features["prompt_vocal_prominence"] == 0.0
        assert result.features["energy_contrast_index"] == 0.0

    def test_plan_with_all_stems_active(self) -> None:
        """All 6 stems at full gain in every section."""
        sections = [
            _make_section("main", 0, 64, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
            _make_section("outro", 64, 128, vocals=1.0, drums=1.0, bass=1.0, guitar=1.0, piano=1.0, other=1.0),
        ]
        plan = _make_plan(sections=sections)
        result = extract_features(plan)
        assert result.features["prompt_energy_level"] == 1.0
        assert result.features["energy_contrast_index"] == 0.0

    def test_partial_metadata(self) -> None:
        """One metadata present, other missing."""
        plan = _make_plan()
        meta_a = _make_metadata(bpm=120.0)
        result = extract_features(plan, meta_a=meta_a, meta_b=None)
        assert len(result.features) == len(get_manifest().feature_names)
        for name, value in result.features.items():
            assert math.isfinite(value), f"Feature {name!r} is not finite: {value}"


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_lcs_ratio_identical(self) -> None:
        assert _lcs_ratio(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_lcs_ratio_empty(self) -> None:
        assert _lcs_ratio([], ["a", "b"]) == 0.0
        assert _lcs_ratio(["a"], []) == 0.0

    def test_lcs_ratio_partial(self) -> None:
        # LCS of ["a","b","c"] and ["a","c"] = ["a","c"] length 2
        # ratio = 2 / 3
        ratio = _lcs_ratio(["a", "b", "c"], ["a", "c"])
        assert abs(ratio - 2 / 3) < 0.01

    def test_active_stem_count(self) -> None:
        section = _make_section(vocals=1.0, drums=0.5, bass=0.0, guitar=0.0)
        assert _active_stem_count(section) == 2

    def test_section_energy_proxy(self) -> None:
        section = _make_section(vocals=1.0, drums=0.5, bass=0.3)
        expected = 1.0 + 0.5 + 0.3  # other stems are 0.0
        assert abs(_section_energy_proxy(section) - expected) < 0.01

    def test_resample_and_correlate_identical(self) -> None:
        """Identical lists should correlate at 1.0."""
        vals = [0.1, 0.5, 0.9, 0.3]
        corr = _resample_and_correlate(vals, vals)
        assert abs(corr - 1.0) < 0.01

    def test_resample_and_correlate_opposite(self) -> None:
        """Opposite trend should have negative correlation."""
        a = [0.0, 0.5, 1.0]
        b = [1.0, 0.5, 0.0]
        corr = _resample_and_correlate(a, b)
        assert corr < -0.9
