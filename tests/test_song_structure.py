"""Tests for song structure analysis engine.

Covers: adaptive bucketing, vocal detection, vocal gaps, section detection,
key detection fallback, cross-song relationships, and the analyze_stems entry point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from musicmixer.models import (
    AudioMetadata,
    CrossSongRelationships,
    EnergyBuckets,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
)
from musicmixer.services.analysis import (
    BUCKET_NOISE_FLOOR,
    MIN_SECTION_BARS,
    PHRASE_GRID,
    VOCAL_MIN_DURATION,
    VOCAL_ONSET_RATIO,
    VOCAL_SUSTAIN_RATIO,
    _compute_bar_boundaries,
    _compute_bar_rms,
    _energy_rank,
    _moving_average,
    classify_energy,
    compute_adaptive_buckets,
    compute_loudness_diff,
    compute_relationships,
    compute_vocal_prominence,
    detect_boundaries,
    detect_sections,
    detect_vocal_activity,
    detect_vocal_gaps,
    label_sections,
    merge_sections,
    quantize_to_phrases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stem_rms(
    n_bars: int = 32,
    base_level: float = 0.1,
    pattern: str = "flat",
) -> np.ndarray:
    """Generate synthetic per-bar RMS values with a given pattern."""
    if pattern == "flat":
        return np.full(n_bars, base_level, dtype=np.float64)
    if pattern == "rising":
        return np.linspace(base_level * 0.2, base_level * 2.0, n_bars)
    if pattern == "verse_chorus":
        # Low 16, high 16
        rms = np.empty(n_bars, dtype=np.float64)
        rms[:n_bars // 2] = base_level * 0.5
        rms[n_bars // 2:] = base_level * 1.5
        return rms
    return np.full(n_bars, base_level, dtype=np.float64)


def _make_6stem_rms(n_bars: int = 32, base: float = 0.1) -> dict[str, np.ndarray]:
    """Create a 6-stem bar_rms dict with uniform energy."""
    return {
        "drums": np.full(n_bars, base, dtype=np.float64),
        "bass": np.full(n_bars, base * 0.8, dtype=np.float64),
        "guitar": np.full(n_bars, base * 0.5, dtype=np.float64),
        "piano": np.full(n_bars, base * 0.3, dtype=np.float64),
        "vocals": np.full(n_bars, base * 0.6, dtype=np.float64),
        "other": np.full(n_bars, base * 0.2, dtype=np.float64),
    }


def _make_metadata_with_stems(
    bpm: float = 120.0,
    n_bars: int = 32,
    mean_rms: float = 0.1,
    vocal_level: float = 0.1,
    non_vocal_level: float = 0.05,
) -> AudioMetadata:
    """Create AudioMetadata with stem_analysis populated."""
    bar_rms = _make_6stem_rms(n_bars, base=non_vocal_level)
    bar_rms["vocals"] = np.full(n_bars, vocal_level, dtype=np.float64)

    combined_energy, buckets = compute_adaptive_buckets(bar_rms)
    vocal_active = detect_vocal_activity(bar_rms["vocals"])
    vocal_gaps = detect_vocal_gaps(vocal_active)

    stem_analysis = StemAnalysis(
        bar_rms=bar_rms,
        combined_energy=combined_energy,
        vocal_active=vocal_active,
        vocal_gaps=vocal_gaps,
        bucket_thresholds=buckets,
    )

    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.8,
        beat_frames=np.arange(0, n_bars * 4) * 1000,
        duration_seconds=n_bars * 4 * 60.0 / bpm,
        total_beats=n_bars * 4,
        mean_rms=mean_rms,
        stem_analysis=stem_analysis,
        song_structure=SongStructure(sections=[], vocal_gaps=vocal_gaps, total_bars=n_bars),
    )


# ---------------------------------------------------------------------------
# Bar boundaries & RMS
# ---------------------------------------------------------------------------

class TestBarBoundaries:
    def test_basic_boundaries(self) -> None:
        """Bar boundaries from evenly spaced beat frames."""
        # 16 beats = 4 bars
        beat_frames = np.arange(0, 16) * 1000
        audio_length = 16000
        boundaries = _compute_bar_boundaries(beat_frames, audio_length)
        # Should have bar starts at [0, 4000, 8000, 12000] + audio_length
        assert boundaries[0] == 0
        assert boundaries[-1] == audio_length
        assert len(boundaries) == 5  # 4 bars + 1 end

    def test_too_few_beats(self) -> None:
        """With <4 beats, return whole audio as one bar."""
        beat_frames = np.array([0, 100, 200])
        boundaries = _compute_bar_boundaries(beat_frames, 1000)
        assert len(boundaries) == 2
        assert boundaries[0] == 0
        assert boundaries[-1] == 1000

    def test_empty_beats(self) -> None:
        """Empty beat frames returns whole audio as one bar."""
        boundaries = _compute_bar_boundaries(np.array([]), 1000)
        assert len(boundaries) == 2


class TestBarRms:
    def test_uniform_signal(self) -> None:
        """RMS of a constant signal should be approximately equal to the value."""
        audio = np.full(4000, 0.5, dtype=np.float64)
        boundaries = np.array([0, 1000, 2000, 3000, 4000])
        rms = _compute_bar_rms(audio, boundaries)
        assert len(rms) == 4
        np.testing.assert_allclose(rms, 0.5, atol=1e-6)

    def test_silent_bar(self) -> None:
        """Silent bar should have zero RMS."""
        audio = np.zeros(2000, dtype=np.float64)
        audio[:1000] = 0.5  # First bar has signal
        boundaries = np.array([0, 1000, 2000])
        rms = _compute_bar_rms(audio, boundaries)
        assert rms[0] > 0
        assert rms[1] == 0.0

    def test_empty(self) -> None:
        """Empty boundaries return empty RMS."""
        rms = _compute_bar_rms(np.zeros(100), np.array([0]))
        assert len(rms) == 0


# ---------------------------------------------------------------------------
# Adaptive bucketing
# ---------------------------------------------------------------------------

class TestAdaptiveBuckets:
    def test_uniform_energy(self) -> None:
        """Uniform stems produce uniform combined energy."""
        bar_rms = _make_6stem_rms(n_bars=32, base=0.1)
        combined, buckets = compute_adaptive_buckets(bar_rms)
        assert len(combined) == 32
        assert buckets.p10 > 0
        assert buckets.p50 >= buckets.p10
        assert buckets.p85 >= buckets.p50

    def test_noise_floor_filtering(self) -> None:
        """Bars below pre-normalization noise floor (0.001) should be zeroed."""
        bar_rms = {
            "drums": np.array([0.0005, 0.1, 0.1, 0.1]),
            "bass": np.array([0.0005, 0.1, 0.1, 0.1]),
            "guitar": np.zeros(4),
            "piano": np.zeros(4),
            "vocals": np.zeros(4),
            "other": np.zeros(4),
        }
        combined, _ = compute_adaptive_buckets(bar_rms)
        # First bar should be zero after noise floor filtering
        assert combined[0] == 0.0
        # Other bars should be non-zero
        assert combined[1] > 0

    def test_p99_normalization(self) -> None:
        """Combined energy should be normalized so p99 ~ 1.0."""
        bar_rms = _make_6stem_rms(n_bars=100, base=0.1)
        combined, _ = compute_adaptive_buckets(bar_rms)
        active = combined[combined > BUCKET_NOISE_FLOOR]
        if len(active) > 0:
            p99 = np.percentile(active, 99)
            # After normalization, p99 should be approximately 1.0
            np.testing.assert_allclose(p99, 1.0, atol=0.1)

    def test_empty_input(self) -> None:
        """Empty input returns empty combined energy and zero thresholds."""
        combined, buckets = compute_adaptive_buckets({})
        assert len(combined) == 0
        assert buckets.p10 == 0.0


class TestEnergyClassification:
    def test_silent(self) -> None:
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.1, p50=0.3, p85=0.7)
        assert classify_energy(0.01, buckets) == "silent"

    def test_low(self) -> None:
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.1, p50=0.3, p85=0.7)
        assert classify_energy(0.05, buckets) == "low"

    def test_medium(self) -> None:
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.1, p50=0.3, p85=0.7)
        assert classify_energy(0.2, buckets) == "medium"

    def test_high(self) -> None:
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.1, p50=0.3, p85=0.7)
        assert classify_energy(0.5, buckets) == "high"

    def test_peak(self) -> None:
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.1, p50=0.3, p85=0.7)
        assert classify_energy(0.9, buckets) == "peak"


# ---------------------------------------------------------------------------
# Vocal detection
# ---------------------------------------------------------------------------

class TestVocalDetection:
    def test_strong_vocals(self) -> None:
        """Bars well above onset threshold should be active."""
        vocal_rms = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0])
        active = detect_vocal_activity(vocal_rms)
        # Peak = 0.5, onset = 0.075, sustain = 0.04
        # Bars 2-5 should be active (4 bars >= min duration 2)
        assert active[2] is np.True_
        assert active[5] is np.True_
        assert active[0] is np.False_

    def test_too_short(self) -> None:
        """Active region shorter than min duration (2 bars) should be discarded."""
        vocal_rms = np.array([0.0, 0.5, 0.0, 0.0, 0.0])
        active = detect_vocal_activity(vocal_rms)
        # Only 1 bar active, below min duration
        assert not np.any(active)

    def test_hysteresis(self) -> None:
        """Region starts at onset threshold and sustains at lower level."""
        # Peak = 1.0, onset = 0.15, sustain = 0.08
        vocal_rms = np.array([0.0, 0.2, 0.1, 0.09, 0.05, 0.0, 0.0, 1.0])
        active = detect_vocal_activity(vocal_rms)
        # Bars 1-3: onset at 0.2, sustain at 0.1 and 0.09 (above 0.08)
        # Bar 4: 0.05 < 0.08 sustain -> drops off
        assert active[1] is np.True_
        assert active[2] is np.True_
        assert active[3] is np.True_
        assert active[4] is np.False_

    def test_all_silent(self) -> None:
        """All-zero vocal RMS -> no active bars."""
        vocal_rms = np.zeros(16)
        active = detect_vocal_activity(vocal_rms)
        assert not np.any(active)

    def test_empty(self) -> None:
        """Empty input returns empty output."""
        active = detect_vocal_activity(np.array([]))
        assert len(active) == 0


# ---------------------------------------------------------------------------
# Vocal gap detection
# ---------------------------------------------------------------------------

class TestVocalGaps:
    def test_single_gap(self) -> None:
        """Single gap of 4 bars in the middle."""
        active = np.array([True, True, False, False, False, False, True, True])
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 1
        assert gaps[0].start_bar == 2
        assert gaps[0].end_bar == 5
        assert gaps[0].length_bars == 4

    def test_gap_at_start(self) -> None:
        """Gap at the beginning of the song."""
        active = np.array([False, False, False, True, True, True])
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 1
        assert gaps[0].start_bar == 0
        assert gaps[0].length_bars == 3

    def test_gap_at_end(self) -> None:
        """Gap at the end of the song."""
        active = np.array([True, True, True, False, False, False])
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 1
        assert gaps[0].end_bar == 5
        assert gaps[0].length_bars == 3

    def test_short_gap_ignored(self) -> None:
        """Gaps shorter than 2 bars are not reported."""
        active = np.array([True, False, True, True, True])
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 0

    def test_multiple_gaps(self) -> None:
        """Multiple gaps detected correctly."""
        active = np.array([
            False, False, False,  # gap 1: bars 0-2
            True, True, True,
            False, False,         # gap 2: bars 6-7
            True,
        ])
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 2
        assert gaps[0].length_bars == 3
        assert gaps[1].length_bars == 2

    def test_all_inactive(self) -> None:
        """Entire song with no vocals = one big gap."""
        active = np.zeros(16, dtype=bool)
        gaps = detect_vocal_gaps(active)
        assert len(gaps) == 1
        assert gaps[0].length_bars == 16

    def test_empty(self) -> None:
        """Empty input returns empty gaps."""
        gaps = detect_vocal_gaps(np.array([], dtype=bool))
        assert len(gaps) == 0


# ---------------------------------------------------------------------------
# Boundary detection & phrase quantization
# ---------------------------------------------------------------------------

class TestBoundaryDetection:
    def test_detects_energy_jump(self) -> None:
        """A large energy jump should produce a boundary."""
        n_bars = 32
        bar_rms = _make_6stem_rms(n_bars)
        # Create a sharp energy jump at bar 16
        for name in bar_rms:
            bar_rms[name][16:] *= 3.0

        combined, _ = compute_adaptive_buckets(bar_rms)
        boundaries = detect_boundaries(bar_rms, combined)

        # Should detect at least one boundary
        assert len(boundaries) > 0, "Expected at least one energy boundary from a 3x energy jump"
        # At least one boundary should be within 4 bars of bar 16
        assert any(abs(b - 16) <= 4 for b in boundaries), (
            f"Expected a boundary within 4 bars of bar 16, got: {boundaries}"
        )

    def test_flat_energy_no_boundaries(self) -> None:
        """Completely flat energy should produce no boundaries (or very few)."""
        bar_rms = _make_6stem_rms(n_bars=32, base=0.1)
        combined, _ = compute_adaptive_buckets(bar_rms)
        boundaries = detect_boundaries(bar_rms, combined)
        # Flat energy: derivative is ~0 everywhere, no peaks above threshold
        assert len(boundaries) <= 1

    def test_very_short_song(self) -> None:
        """Very short song (<2 bars) returns no boundaries."""
        bar_rms = {"drums": np.array([0.1])}
        combined = np.array([0.1])
        boundaries = detect_boundaries(bar_rms, combined)
        assert len(boundaries) == 0


class TestPhraseQuantization:
    def test_snaps_to_grid(self) -> None:
        """Boundaries should snap to 4-bar grid."""
        boundaries = np.array([5, 13, 23])
        quantized = quantize_to_phrases(boundaries, total_bars=32)
        # 5 -> 4, 13 -> 12, 23 -> 24
        for q in quantized:
            assert q % PHRASE_GRID == 0

    def test_deduplication(self) -> None:
        """Boundaries that snap to same position are deduplicated."""
        boundaries = np.array([3, 5])  # Both snap to 4
        quantized = quantize_to_phrases(boundaries, total_bars=32)
        assert len(quantized) == len(np.unique(quantized))

    def test_removes_short_segments(self) -> None:
        """Segments shorter than 4 bars are removed."""
        # Boundaries at 4, 6 (-> both snap to 4 after quant, deduplicated)
        boundaries = np.array([4, 8, 12])
        quantized = quantize_to_phrases(boundaries, total_bars=16)
        # All segments should be >= 4 bars
        points = np.concatenate([[0], quantized, [16]])
        lengths = np.diff(points)
        assert all(l >= MIN_SECTION_BARS for l in lengths)

    def test_empty_boundaries(self) -> None:
        """Empty input returns empty output."""
        quantized = quantize_to_phrases(np.array([]), total_bars=32)
        assert len(quantized) == 0


# ---------------------------------------------------------------------------
# Section labeling
# ---------------------------------------------------------------------------

class TestSectionLabeling:
    def _make_bar_boundaries(self, n_bars: int) -> np.ndarray:
        """Create evenly spaced bar boundaries."""
        return np.arange(n_bars + 1) * 1000

    def test_single_section_song(self) -> None:
        """Song with no internal boundaries -> one section."""
        n_bars = 16
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.ones(n_bars, dtype=bool)
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        assert len(sections) == 1
        assert sections[0].start_bar == 0
        assert sections[0].end_bar == n_bars

    def test_intro_label(self) -> None:
        """First segment with low energy and no vocals gets 'intro' label."""
        n_bars = 32
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        # First 8 bars are quiet
        for name in bar_rms:
            bar_rms[name][:8] *= 0.1

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        vocal_active[8:] = True
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([8], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        assert sections[0].label == "intro"

    def test_outro_label(self) -> None:
        """Last segment at position > 85% gets 'outro' label."""
        n_bars = 32
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        # Last 4 bars are quiet
        for name in bar_rms:
            bar_rms[name][28:] *= 0.1

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([28], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        # Last section should be outro (position 28/32 = 87.5% > 85%)
        assert sections[-1].label == "outro"

    def test_good_instrumental_source_annotation(self) -> None:
        """Sections with vocal stem below sustain threshold get annotated."""
        n_bars = 16
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        # Set vocals very low for first half
        bar_rms["vocals"][:8] = 0.001
        bar_rms["vocals"][8:] = 0.5  # High vocals in second half

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        vocal_active[8:] = True
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([8], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        # First section should have GOOD INSTRUMENTAL SOURCE annotation
        first_section = sections[0]
        assert "GOOD INSTRUMENTAL SOURCE" in first_section.annotations


# ---------------------------------------------------------------------------
# Section merge
# ---------------------------------------------------------------------------

class TestSectionMerge:
    def test_merge_adjacent_same_label(self) -> None:
        """Adjacent sections with same label should be merged."""
        sections = [
            SectionInfo(0, 8, 8, 0.0, 4.0, "verse", "medium", "medium",
                        "mid", "vox:yes", []),
            SectionInfo(8, 16, 8, 4.0, 8.0, "verse", "medium", "medium",
                        "mid", "vox:yes", []),
        ]
        merged = merge_sections(sections)
        assert len(merged) == 1
        assert merged[0].start_bar == 0
        assert merged[0].end_bar == 16
        assert merged[0].bar_count == 16

    def test_different_labels_not_merged(self) -> None:
        """Adjacent sections with different labels stay separate."""
        sections = [
            SectionInfo(0, 8, 8, 0.0, 4.0, "verse", "medium", "medium",
                        "mid", "vox:yes", []),
            SectionInfo(8, 16, 8, 4.0, 8.0, "chorus", "high", "high",
                        "full", "vox:yes", []),
        ]
        merged = merge_sections(sections)
        assert len(merged) == 2

    def test_absorb_short_section(self) -> None:
        """Section shorter than 4 bars should be absorbed into louder neighbor."""
        sections = [
            SectionInfo(0, 12, 12, 0.0, 6.0, "verse", "medium", "medium",
                        "mid", "vox:yes", []),
            SectionInfo(12, 14, 2, 6.0, 7.0, "instrumental", "low", "low",
                        "sparse", "vox:no", []),
            SectionInfo(14, 28, 14, 7.0, 14.0, "chorus", "high", "high",
                        "full", "vox:yes", []),
        ]
        merged = merge_sections(sections)
        # The 2-bar section should be absorbed
        assert all(s.bar_count >= MIN_SECTION_BARS for s in merged)

    def test_single_section_unchanged(self) -> None:
        """Single section should pass through unchanged."""
        sections = [
            SectionInfo(0, 16, 16, 0.0, 8.0, "verse", "medium", "medium",
                        "mid", "vox:yes", []),
        ]
        merged = merge_sections(sections)
        assert len(merged) == 1
        assert merged[0].bar_count == 16


# ---------------------------------------------------------------------------
# Cross-song analysis
# ---------------------------------------------------------------------------

class TestLoudnessDiff:
    def test_equal_rms(self) -> None:
        """Equal RMS should produce 0 dB difference."""
        diff = compute_loudness_diff(0.1, 0.1)
        assert diff is not None
        assert diff == pytest.approx(0.0)

    def test_a_louder(self) -> None:
        """A louder than B should produce positive dB."""
        diff = compute_loudness_diff(0.2, 0.1)
        assert diff is not None
        assert diff > 0
        # 20 * log10(2) ~ 6 dB
        assert diff == pytest.approx(6.02, abs=0.1)

    def test_b_louder(self) -> None:
        """B louder than A should produce negative dB."""
        diff = compute_loudness_diff(0.1, 0.2)
        assert diff is not None
        assert diff < 0

    def test_below_threshold(self) -> None:
        """RMS below 0.001 should return None."""
        assert compute_loudness_diff(0.0005, 0.1) is None
        assert compute_loudness_diff(0.1, 0.0005) is None


class TestVocalProminence:
    def test_prominent_vocals(self) -> None:
        """Strong vocals over weak accompaniment -> positive dB."""
        bar_rms = {
            "vocals": np.array([0.5, 0.5, 0.5, 0.5]),
            "drums": np.array([0.1, 0.1, 0.1, 0.1]),
            "bass": np.array([0.05, 0.05, 0.05, 0.05]),
            "guitar": np.array([0.02, 0.02, 0.02, 0.02]),
            "piano": np.array([0.01, 0.01, 0.01, 0.01]),
            "other": np.array([0.01, 0.01, 0.01, 0.01]),
        }
        vocal_active = np.array([True, True, True, True])
        prom = compute_vocal_prominence(bar_rms, vocal_active)
        assert prom is not None
        assert prom > 0  # Vocals louder than accompaniment

    def test_no_active_bars(self) -> None:
        """No vocal-active bars -> None."""
        bar_rms = {
            "vocals": np.array([0.1, 0.1]),
            "drums": np.array([0.1, 0.1]),
            "bass": np.zeros(2),
            "guitar": np.zeros(2),
            "piano": np.zeros(2),
            "other": np.zeros(2),
        }
        vocal_active = np.array([False, False])
        prom = compute_vocal_prominence(bar_rms, vocal_active)
        assert prom is None

    def test_computes_over_active_bars_only(self) -> None:
        """Prominence should only consider vocal-active bars."""
        bar_rms = {
            "vocals": np.array([0.0, 0.0, 0.5, 0.5]),
            "drums": np.array([0.2, 0.2, 0.1, 0.1]),
            "bass": np.zeros(4),
            "guitar": np.zeros(4),
            "piano": np.zeros(4),
            "other": np.zeros(4),
        }
        vocal_active = np.array([False, False, True, True])
        prom = compute_vocal_prominence(bar_rms, vocal_active)
        assert prom is not None
        # Mean vocal (bars 2-3) = 0.5, mean non-vocal (bars 2-3) = 0.1
        # 20 * log10(0.5 / 0.1) ~ 14 dB
        assert prom == pytest.approx(14.0, abs=1.0)


class TestComputeRelationships:
    def test_returns_all_fields(self) -> None:
        """compute_relationships returns a CrossSongRelationships with all fields."""
        meta_a = _make_metadata_with_stems(mean_rms=0.15, vocal_level=0.3, non_vocal_level=0.05)
        meta_b = _make_metadata_with_stems(mean_rms=0.08, vocal_level=0.1, non_vocal_level=0.08)

        rels = compute_relationships(meta_a, meta_b)

        assert isinstance(rels, CrossSongRelationships)
        assert rels.loudness_diff_db != 0.0  # A is louder
        assert rels.loudness_diff_db > 0  # A is louder than B
        assert rels.vocal_source in ("song_a", "song_b")
        assert rels.stretch_pct >= 0.0

    def test_vocal_source_selection(self) -> None:
        """Song with higher vocal prominence should be selected as vocal source."""
        # Song A: strong vocals, weak accompaniment
        meta_a = _make_metadata_with_stems(mean_rms=0.1, vocal_level=0.5, non_vocal_level=0.02)
        # Song B: weak vocals, strong accompaniment
        meta_b = _make_metadata_with_stems(mean_rms=0.1, vocal_level=0.05, non_vocal_level=0.1)

        rels = compute_relationships(meta_a, meta_b)
        assert rels.vocal_source == "song_a"


# ---------------------------------------------------------------------------
# Moving average helper
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_identity_window_1(self) -> None:
        """Window of 1 should return the input."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = _moving_average(arr, 1)
        np.testing.assert_allclose(result, arr)

    def test_smoothing(self) -> None:
        """Moving average should smooth spikes."""
        arr = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = _moving_average(arr, 3)
        # The spike at index 2 should be reduced
        assert result[2] < 1.0
        # Neighbors should gain some energy
        assert result[1] > 0.0
        assert result[3] > 0.0

    def test_empty(self) -> None:
        """Empty input returns empty output."""
        result = _moving_average(np.array([]), 4)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Key detection (librosa fallback only -- essentia may not be installed)
# ---------------------------------------------------------------------------

class TestKeyDetection:
    def test_librosa_fallback(self, tmp_path: Path) -> None:
        """librosa key detection should return valid key/scale/confidence."""
        from musicmixer.services.analysis import _detect_key_librosa

        # Generate a simple C major-ish tone
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # C4 (261.63 Hz) + E4 (329.63 Hz) + G4 (392 Hz) = C major triad
        signal = (
            0.3 * np.sin(2 * np.pi * 261.63 * t) +
            0.3 * np.sin(2 * np.pi * 329.63 * t) +
            0.3 * np.sin(2 * np.pi * 392.0 * t)
        ).astype(np.float32)

        wav_path = tmp_path / "c_major.wav"
        sf.write(str(wav_path), signal, sr)

        key, scale, confidence = _detect_key_librosa(wav_path)
        assert key in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        assert scale in ["major", "minor"]
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Energy rank helper
# ---------------------------------------------------------------------------

class TestEnergyRank:
    def test_ordering(self) -> None:
        """Energy levels should rank correctly."""
        assert _energy_rank("silent") < _energy_rank("low")
        assert _energy_rank("low") < _energy_rank("medium")
        assert _energy_rank("medium") < _energy_rank("high")
        assert _energy_rank("high") < _energy_rank("peak")

    def test_unknown(self) -> None:
        """Unknown energy level returns 0."""
        assert _energy_rank("unknown") == 0


# ---------------------------------------------------------------------------
# Integration: detect_sections pipeline
# ---------------------------------------------------------------------------

class TestDetectSections:
    def test_basic_pipeline(self) -> None:
        """Full section detection pipeline should produce valid sections."""
        n_bars = 48
        bar_rms = _make_6stem_rms(n_bars, base=0.1)

        # Create distinct regions: quiet intro, loud middle, quiet outro
        for name in bar_rms:
            bar_rms[name][:8] *= 0.2    # intro
            bar_rms[name][8:40] *= 1.0  # main body
            bar_rms[name][40:] *= 0.2   # outro

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        vocal_active[8:40] = True  # Vocals in main body
        bar_bounds = np.arange(n_bars + 1) * 1000

        sections = detect_sections(
            bar_rms_per_stem=bar_rms,
            combined_energy=combined,
            vocal_active=vocal_active,
            buckets=buckets,
            total_bars=n_bars,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        # Should have at least 1 section
        assert len(sections) > 0

        # All sections should cover the full song
        assert sections[0].start_bar == 0
        assert sections[-1].end_bar == n_bars

        # No gaps between sections
        for i in range(1, len(sections)):
            assert sections[i].start_bar == sections[i - 1].end_bar

        # All sections should have valid labels
        valid_labels = {"intro", "verse", "chorus", "instrumental", "breakdown", "build", "outro"}
        for sec in sections:
            assert sec.label in valid_labels

    def test_handles_very_short_song(self) -> None:
        """Song with <8 bars should still work."""
        n_bars = 4
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.ones(n_bars, dtype=bool)
        bar_bounds = np.arange(n_bars + 1) * 1000

        sections = detect_sections(
            bar_rms_per_stem=bar_rms,
            combined_energy=combined,
            vocal_active=vocal_active,
            buckets=buckets,
            total_bars=n_bars,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        assert len(sections) >= 1
        assert sections[0].start_bar == 0
        assert sections[-1].end_bar == n_bars


# ---------------------------------------------------------------------------
# Per-section vocal prominence
# ---------------------------------------------------------------------------

class TestPerSectionVocalProminence:
    def _make_bar_boundaries(self, n_bars: int) -> np.ndarray:
        """Create evenly spaced bar boundaries."""
        return np.arange(n_bars + 1) * 1000

    def test_label_sections_populates_vocal_prominence(self) -> None:
        """label_sections should populate vocal_prominence_db on sections with enough vocal bars."""
        n_bars = 32
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        # Strong vocals in second half
        bar_rms["vocals"][:16] = 0.001
        bar_rms["vocals"][16:] = 0.5

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        vocal_active[16:] = True  # 16 vocal-active bars in second half
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([16], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        # Second section (bars 16-32) has 16 vocal-active bars -> should have prominence
        vocal_section = [s for s in sections if s.start_bar == 16][0]
        assert vocal_section.vocal_prominence_db is not None
        assert isinstance(vocal_section.vocal_prominence_db, float)

    def test_short_vocal_section_gets_none(self) -> None:
        """Sections with fewer than 3 vocal-active bars get vocal_prominence_db = None."""
        n_bars = 16
        bar_rms = _make_6stem_rms(n_bars, base=0.1)
        bar_rms["vocals"] = np.full(n_bars, 0.5, dtype=np.float64)

        combined, buckets = compute_adaptive_buckets(bar_rms)
        # Only 2 vocal-active bars in the first section (bars 0-8)
        vocal_active = np.zeros(n_bars, dtype=bool)
        vocal_active[6:8] = True  # 2 bars active in first section
        vocal_active[8:] = True   # 8 bars active in second section
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([8], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        first_section = sections[0]
        # Only 2 active bars -> below threshold of 3
        assert first_section.vocal_prominence_db is None

    def test_no_vocal_section_gets_none(self) -> None:
        """Sections with no vocal activity get vocal_prominence_db = None."""
        n_bars = 16
        bar_rms = _make_6stem_rms(n_bars, base=0.1)

        combined, buckets = compute_adaptive_buckets(bar_rms)
        vocal_active = np.zeros(n_bars, dtype=bool)
        bar_bounds = self._make_bar_boundaries(n_bars)

        sections = label_sections(
            boundaries=np.array([], dtype=np.intp),
            total_bars=n_bars,
            combined_energy=combined,
            vocal_active=vocal_active,
            bar_rms_per_stem=bar_rms,
            buckets=buckets,
            bpm=120.0,
            bar_boundaries_frames=bar_bounds,
        )

        for sec in sections:
            assert sec.vocal_prominence_db is None

    def test_merge_sections_clears_prominence(self) -> None:
        """merge_sections sets vocal_prominence_db to None on merged sections."""
        sections = [
            SectionInfo(0, 8, 8, 0.0, 4.0, "verse", "medium", "medium",
                        "mid", "vox:yes", vocal_prominence_db=5.0),
            SectionInfo(8, 16, 8, 4.0, 8.0, "verse", "medium", "medium",
                        "mid", "vox:yes", vocal_prominence_db=7.0),
        ]
        merged = merge_sections(sections)
        assert len(merged) == 1
        # Merged section should have None prominence (lost raw data)
        assert merged[0].vocal_prominence_db is None


class TestSectionMapRendering:
    def test_db_rendering(self) -> None:
        """_build_section_map renders vox:+XdB format for sections with prominence."""
        from musicmixer.models import SongStructure, VocalGap
        from musicmixer.services.interpreter import _build_section_map

        sections = [
            SectionInfo(0, 8, 8, 0.0, 4.0, "intro", "low", "low",
                        "sparse", "vox:no", vocal_prominence_db=None),
            SectionInfo(8, 24, 16, 4.0, 12.0, "verse", "medium", "medium",
                        "mid", "vox:yes", vocal_prominence_db=6.3),
            SectionInfo(24, 32, 8, 12.0, 16.0, "outro", "low", "low->low",
                        "sparse", "vox:fading", vocal_prominence_db=3.0),
        ]
        structure = SongStructure(sections=sections, vocal_gaps=[], total_bars=32)
        result = _build_section_map("Song A", structure, 32)

        assert "vox:--" in result
        assert "vox:+6dB" in result
        assert "vox:+3dB(fading)" in result

    def test_no_prominence_fading_fallback(self) -> None:
        """Sections with vocal_status=fading but no prominence render as vox:fading."""
        from musicmixer.models import SongStructure
        from musicmixer.services.interpreter import _build_section_map

        sections = [
            SectionInfo(0, 8, 8, 0.0, 4.0, "outro", "low", "low",
                        "sparse", "vox:fading", vocal_prominence_db=None),
        ]
        structure = SongStructure(sections=sections, vocal_gaps=[], total_bars=8)
        result = _build_section_map("Song A", structure, 8)

        assert "vox:fading" in result
        assert "dB" not in result
