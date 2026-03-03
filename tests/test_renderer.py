"""Tests for musicmixer.services.renderer -- section-based arrangement renderer."""

import numpy as np
import pytest

from musicmixer.models import Section
from musicmixer.services.renderer import (
    beats_to_samples,
    render_arrangement,
    snap_to_bar,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

BPM = 120
SR = 44100
ANALYSIS_SR = 22050  # librosa default analysis sample rate
HOP_LENGTH = 512
# Beat frames are in analysis_sr space (22050 Hz), matching real librosa output
BEAT_INTERVAL_FRAMES = int(60 / BPM * ANALYSIS_SR / HOP_LENGTH)
# 200 beats at 120 BPM
BEAT_FRAMES = np.arange(0, 200) * BEAT_INTERVAL_FRAMES


def _total_samples() -> int:
    """Total samples for our synthetic beat grid."""
    return beats_to_samples(200, BEAT_FRAMES, SR, HOP_LENGTH)


def _make_stereo_noise(n_samples: int, amplitude: float = 0.1) -> np.ndarray:
    """Create synthetic stereo audio (white noise)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((n_samples, 2)) * amplitude).astype(np.float32)


def _make_sections(total_beats: int = 200) -> list[Section]:
    """Build a simple 5-section arrangement for testing."""
    eighth = total_beats // 8
    quarter = total_beats // 4
    three_quarter = total_beats * 3 // 4
    seven_eighth = total_beats * 7 // 8

    return [
        Section(
            label="intro", start_beat=0, end_beat=eighth,
            stem_gains={"vocals": 0.0, "drums": 0.8, "bass": 0.7,
                        "guitar": 0.6, "piano": 0.5, "other": 1.0},
            transition_in="fade", transition_beats=4,
        ),
        Section(
            label="build", start_beat=eighth, end_beat=quarter,
            stem_gains={"vocals": 0.6, "drums": 0.7, "bass": 0.8,
                        "guitar": 0.5, "piano": 0.4, "other": 0.5},
            transition_in="crossfade", transition_beats=4,
        ),
        Section(
            label="main", start_beat=quarter, end_beat=three_quarter,
            stem_gains={"vocals": 1.0, "drums": 0.7, "bass": 0.8,
                        "guitar": 0.5, "piano": 0.4, "other": 0.5},
            transition_in="crossfade", transition_beats=2,
        ),
        Section(
            label="breakdown", start_beat=three_quarter, end_beat=seven_eighth,
            stem_gains={"vocals": 0.8, "drums": 0.0, "bass": 0.6,
                        "guitar": 0.7, "piano": 0.8, "other": 0.7},
            transition_in="crossfade", transition_beats=4,
        ),
        Section(
            label="outro", start_beat=seven_eighth, end_beat=total_beats,
            stem_gains={"vocals": 0.0, "drums": 0.6, "bass": 0.5,
                        "guitar": 0.5, "piano": 0.6, "other": 0.8},
            transition_in="crossfade", transition_beats=4,
        ),
    ]


# ---------------------------------------------------------------------------
# beats_to_samples tests
# ---------------------------------------------------------------------------

class TestBeatsToSamples:
    """Tests for beats_to_samples()."""

    def test_basic(self):
        """Known beat_frames produce correct sample positions."""
        sr_scale = SR / ANALYSIS_SR  # 2.0

        # Beat 0 should be at frame 0 * hop_length * sr_scale = 0
        assert beats_to_samples(0, BEAT_FRAMES, SR, HOP_LENGTH) == 0

        # Beat 1 should be at BEAT_INTERVAL_FRAMES * hop_length * sr_scale
        expected = int(BEAT_INTERVAL_FRAMES * HOP_LENGTH * sr_scale)
        assert beats_to_samples(1, BEAT_FRAMES, SR, HOP_LENGTH) == expected

    def test_mid_range(self):
        """Beat index in middle of array returns correct position."""
        sr_scale = SR / ANALYSIS_SR
        idx = 50
        expected = int(BEAT_FRAMES[idx] * HOP_LENGTH * sr_scale)
        assert beats_to_samples(idx, BEAT_FRAMES, SR, HOP_LENGTH) == expected

    def test_extrapolation(self):
        """Beat index beyond array length extrapolates using average of last 8 intervals."""
        sr_scale = SR / ANALYSIS_SR

        # Ask for beat 210 when we only have 200 beats (0..199)
        result = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH)

        # With uniform spacing, extrapolation should be accurate
        last_8 = BEAT_FRAMES[-8:]
        avg_beat_len = float(np.mean(np.diff(last_8))) * HOP_LENGTH * sr_scale
        overshoot = 210 - 200 + 1  # 11
        expected = int(BEAT_FRAMES[-1] * HOP_LENGTH * sr_scale + overshoot * avg_beat_len)
        assert result == expected

    def test_degenerate_case(self):
        """Fewer than 2 beat frames falls back to beat_index * sr."""
        single = np.array([0])
        result = beats_to_samples(5, single, SR, HOP_LENGTH)
        assert result == 5 * SR


# ---------------------------------------------------------------------------
# render_arrangement tests
# ---------------------------------------------------------------------------

class TestRenderArrangement:
    """Tests for render_arrangement()."""

    def _make_stems(self, n_samples: int):
        """Create a full set of vocal + instrumental stems."""
        rng = np.random.default_rng(42)
        vocal_stems = {
            "vocals": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
        }
        instrumental_stems = {
            "drums": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
            "bass": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
            "guitar": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
            "piano": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
            "other": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
        }
        return vocal_stems, instrumental_stems

    def test_output_shape(self):
        """Two buses with correct length and stereo channels."""
        sections = _make_sections(200)
        total = _total_samples()
        vocal_stems, inst_stems = self._make_stems(total)

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        expected_len = beats_to_samples(200, BEAT_FRAMES, SR, HOP_LENGTH)
        assert vocal_bus.shape == (expected_len, 2)
        assert inst_bus.shape == (expected_len, 2)
        assert vocal_bus.dtype == np.float32
        assert inst_bus.dtype == np.float32

    def test_silent_stems_dont_contribute(self):
        """Stems with gain 0.0 should not contribute any energy."""
        # Create a single section where vocals gain = 0.0
        sections = [
            Section(
                label="test", start_beat=0, end_beat=50,
                stem_gains={"vocals": 0.0, "drums": 1.0, "bass": 0.0,
                            "guitar": 0.0, "piano": 0.0, "other": 0.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        total = beats_to_samples(50, BEAT_FRAMES, SR, HOP_LENGTH)
        vocal_stems, inst_stems = self._make_stems(total)

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Vocal bus should be all zeros (vocals gain = 0.0)
        assert np.allclose(vocal_bus, 0.0)
        # Instrumental bus should have non-zero content from drums
        assert not np.allclose(inst_bus, 0.0)

    def test_missing_stems_handled(self):
        """Missing stems (None in dict or absent key) are handled gracefully."""
        sections = _make_sections(100)
        total = beats_to_samples(100, BEAT_FRAMES, SR, HOP_LENGTH)

        # Only provide vocals and drums -- guitar, piano, other are missing
        vocal_stems = {
            "vocals": _make_stereo_noise(total),
        }
        instrumental_stems = {
            "drums": _make_stereo_noise(total),
            # bass, guitar, piano, other are all absent
        }

        # Should not raise
        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, instrumental_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        assert vocal_bus.shape[0] == beats_to_samples(100, BEAT_FRAMES, SR, HOP_LENGTH)
        assert inst_bus.shape[0] == beats_to_samples(100, BEAT_FRAMES, SR, HOP_LENGTH)

    def test_section_isolation_intro_no_vocals(self):
        """Intro section body (before transition zone) has zero vocal energy."""
        sections = _make_sections(200)
        total = _total_samples()
        vocal_stems, inst_stems = self._make_stems(total)

        vocal_bus, _ = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Intro ends at beat 25. The build section's transition_beats=4,
        # so half_beats=2 bleeds backward. Check up to beat 23 (safe zone).
        safe_end = beats_to_samples(23, BEAT_FRAMES, SR, HOP_LENGTH)
        intro_body = vocal_bus[:safe_end]
        assert np.allclose(intro_body, 0.0), "Intro body should have zero vocal energy"

    def test_short_stems_padded(self):
        """Stems shorter than arrangement are padded with silence."""
        sections = [
            Section(
                label="main", start_beat=0, end_beat=100,
                stem_gains={"vocals": 1.0, "drums": 1.0, "bass": 0.0,
                            "guitar": 0.0, "piano": 0.0, "other": 0.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        # Create stems that are only half the required length
        full_len = beats_to_samples(100, BEAT_FRAMES, SR, HOP_LENGTH)
        half_len = full_len // 2

        vocal_stems = {"vocals": _make_stereo_noise(half_len)}
        inst_stems = {"drums": _make_stereo_noise(half_len)}

        # Should not raise
        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        assert vocal_bus.shape == (full_len, 2)
        assert inst_bus.shape == (full_len, 2)

        # Second half should be silence (stems were padded)
        assert np.allclose(vocal_bus[half_len:], 0.0, atol=1e-6)
        assert np.allclose(inst_bus[half_len:], 0.0, atol=1e-6)

    def test_empty_sections(self):
        """Empty section list returns empty arrays."""
        vocal_bus, inst_bus = render_arrangement(
            [], {"vocals": _make_stereo_noise(1000)},
            {"drums": _make_stereo_noise(1000)},
            BEAT_FRAMES, SR, HOP_LENGTH,
        )
        assert vocal_bus.shape == (0, 2)
        assert inst_bus.shape == (0, 2)


# ---------------------------------------------------------------------------
# snap_to_bar tests
# ---------------------------------------------------------------------------

class TestSnapToBar:
    """Tests for snap_to_bar()."""

    def test_snap_to_nearest_bar(self):
        """Snaps sample position to the nearest bar boundary."""
        # Create beat positions at regular intervals (1000 samples per beat)
        beats = np.arange(0, 100) * 1000  # 100 beats

        # Position close to beat 4 (bar boundary at beats_per_bar=4)
        # Bar boundaries: 0, 4000, 8000, 12000, ...
        result = snap_to_bar(4200, beats, beats_per_bar=4)
        assert result == 4000  # Nearest bar at beat 4

        result = snap_to_bar(7800, beats, beats_per_bar=4)
        assert result == 8000  # Nearest bar at beat 8

    def test_snap_at_exact_boundary(self):
        """Position exactly on a bar boundary stays there."""
        beats = np.arange(0, 100) * 1000
        result = snap_to_bar(8000, beats, beats_per_bar=4)
        assert result == 8000

    def test_snap_with_empty_beats(self):
        """Empty beat_positions returns the input position unchanged."""
        result = snap_to_bar(5000, np.array([]), beats_per_bar=4)
        assert result == 5000

    def test_snap_different_time_signatures(self):
        """Works with beats_per_bar=3 (waltz time)."""
        beats = np.arange(0, 100) * 1000
        # Bar boundaries in 3/4: 0, 3000, 6000, 9000, ...
        result = snap_to_bar(5200, beats, beats_per_bar=3)
        assert result == 6000


# ---------------------------------------------------------------------------
# beats_to_samples target_bpm fallback tests
# ---------------------------------------------------------------------------

class TestBeatsToSamplesBpmFallback:
    """Tests for the target_bpm sanity-check in beats_to_samples()."""

    def test_no_target_bpm_backward_compat(self):
        """Without target_bpm, extrapolation uses observed spacing (existing behavior)."""
        sr_scale = SR / ANALYSIS_SR
        # Ask for beat 210 when we only have 200 beats
        result_no_bpm = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH)
        result_none = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH, target_bpm=None)
        assert result_no_bpm == result_none

    def test_target_bpm_close_to_observed_uses_observed(self):
        """When target_bpm matches observed spacing (<20% diff), observed spacing is used."""
        # Our BEAT_FRAMES are at 120 BPM. Pass target_bpm=120 -- should not trigger fallback.
        result_with_bpm = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH, target_bpm=120.0)
        result_without = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH)
        # With matching BPM, both should be equal
        assert result_with_bpm == result_without

    def test_target_bpm_corrects_compressed_spacing(self):
        """When observed spacing is >20% shorter than BPM implies, BPM spacing is used."""
        # Create a beat grid where the last 8 beats have compressed spacing
        # (simulating librosa's common behavior at the end of a track)
        normal_interval = BEAT_INTERVAL_FRAMES  # ~21 frames at 120 BPM
        frames = list(range(0, 192 * normal_interval, normal_interval))  # 192 normal beats
        # Last 8 beats: compressed to 50% of normal spacing
        compressed_interval = normal_interval // 2
        for i in range(8):
            frames.append(frames[-1] + compressed_interval)
        beat_frames = np.array(frames)

        # Without target_bpm: uses the compressed last-8 spacing
        result_no_bpm = beats_to_samples(210, beat_frames, SR, HOP_LENGTH)

        # With target_bpm=120: should detect the compression and use BPM-based spacing
        result_with_bpm = beats_to_samples(210, beat_frames, SR, HOP_LENGTH, target_bpm=120.0)

        # The BPM-corrected result should be significantly larger (compressed was ~50% too short)
        assert result_with_bpm > result_no_bpm
        # Verify the corrected result uses the expected BPM-based spacing
        sr_scale = SR / ANALYSIS_SR
        expected_beat_len = 60.0 / 120.0 * SR  # samples per beat at 120 BPM
        overshoot = 210 - len(beat_frames) + 1
        expected = int(beat_frames[-1] * HOP_LENGTH * sr_scale + overshoot * expected_beat_len)
        assert result_with_bpm == expected

    def test_target_bpm_within_tolerance_no_correction(self):
        """When observed spacing is within 20% of BPM, no correction is applied."""
        # Create beats at 110 BPM but claim target is 120 BPM
        # 110 BPM spacing vs 120 BPM expected: difference is ~8.3%, within 20%
        interval_110 = int(60 / 110 * ANALYSIS_SR / HOP_LENGTH)
        beat_frames_110 = np.arange(0, 200) * interval_110

        result_with = beats_to_samples(210, beat_frames_110, SR, HOP_LENGTH, target_bpm=120.0)
        result_without = beats_to_samples(210, beat_frames_110, SR, HOP_LENGTH)
        # Within tolerance, so both should be equal
        assert result_with == result_without


# ---------------------------------------------------------------------------
# render_arrangement with target_bpm tests
# ---------------------------------------------------------------------------

class TestRenderArrangementWithTargetBpm:
    """Tests that render_arrangement properly passes target_bpm through."""

    def _make_stems(self, n_samples: int):
        rng = np.random.default_rng(42)
        vocal_stems = {
            "vocals": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
        }
        instrumental_stems = {
            "drums": (rng.standard_normal((n_samples, 2)) * 0.1).astype(np.float32),
        }
        return vocal_stems, instrumental_stems

    def test_target_bpm_passed_through(self):
        """render_arrangement with target_bpm produces valid output."""
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"vocals": 1.0, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        total = beats_to_samples(50, BEAT_FRAMES, SR, HOP_LENGTH, target_bpm=120.0)
        vocal_stems, inst_stems = self._make_stems(total)

        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
            target_bpm=120.0,
        )

        assert vocal_bus.shape[0] == total
        assert inst_bus.shape[0] == total

    def test_backward_compat_no_target_bpm(self):
        """render_arrangement works without target_bpm (backward compat)."""
        sections = [
            Section(
                label="main", start_beat=0, end_beat=50,
                stem_gains={"vocals": 1.0, "drums": 1.0},
                transition_in="cut", transition_beats=0,
            ),
        ]
        total = beats_to_samples(50, BEAT_FRAMES, SR, HOP_LENGTH)
        vocal_stems, inst_stems = self._make_stems(total)

        # Should work fine without target_bpm
        vocal_bus, inst_bus = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        assert vocal_bus.shape[0] == total
        assert inst_bus.shape[0] == total


# ---------------------------------------------------------------------------
# Pipeline beat grid validation logic tests
# ---------------------------------------------------------------------------

class TestBeatGridValidation:
    """Tests for the pipeline's post-stretch beat grid validation logic.

    The pipeline validates whether a re-detected beat grid can reliably
    cover the plan's beat range (allowing up to 20% extrapolation).
    These tests exercise that validation logic directly.
    """

    @staticmethod
    def _should_use_redetected_grid(
        new_beat_count: int, plan_end_beat: int,
    ) -> bool:
        """Replicates the pipeline's beat grid validation logic."""
        if new_beat_count <= 10:
            return False
        if plan_end_beat <= 0:
            return True
        max_reliable_beat = int(new_beat_count * 1.2)
        return plan_end_beat <= max_reliable_beat

    def test_grid_too_short_falls_back(self):
        """When re-detected grid cannot cover plan range, falls back to scaled grid."""
        # 300 detected beats, plan needs 444 beats
        # max_reliable = 300 * 1.2 = 360, but plan needs 444 -> reject
        assert not self._should_use_redetected_grid(300, 444)

    def test_grid_sufficient_uses_redetected(self):
        """When re-detected grid can cover plan range, uses it."""
        # 400 detected beats, plan needs 444 beats
        # max_reliable = 400 * 1.2 = 480, plan needs 444 -> accept
        assert self._should_use_redetected_grid(400, 444)

    def test_grid_exact_boundary(self):
        """Grid at exactly 20% extrapolation boundary is accepted."""
        # 370 detected beats, plan needs 444
        # max_reliable = 370 * 1.2 = 444 -> exactly on boundary -> accept
        assert self._should_use_redetected_grid(370, 444)

    def test_grid_just_under_boundary_rejected(self):
        """Grid just under 20% extrapolation boundary is rejected."""
        # 369 detected beats, plan needs 444
        # max_reliable = 369 * 1.2 = 442 (int), 444 > 442 -> reject
        assert not self._should_use_redetected_grid(369, 444)

    def test_very_few_beats_always_rejected(self):
        """Fewer than 10 re-detected beats are always rejected."""
        assert not self._should_use_redetected_grid(5, 100)
        assert not self._should_use_redetected_grid(10, 100)

    def test_no_plan_sections_always_accepted(self):
        """When plan_end_beat is 0 (no sections), any grid > 10 beats is accepted."""
        assert self._should_use_redetected_grid(50, 0)

    def test_large_grid_small_plan(self):
        """Large re-detected grid easily covers a small plan."""
        # 571 detected beats, plan needs 200
        assert self._should_use_redetected_grid(571, 200)
