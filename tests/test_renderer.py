"""Tests for musicmixer.services.renderer -- section-based arrangement renderer."""

import numpy as np
import pytest

from musicmixer.models import Section
from musicmixer.services.renderer import (
    beats_to_samples,
    make_transition_envelope,
    render_arrangement,
    snap_to_bar,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

BPM = 120
SR = 44100
HOP_LENGTH = 512
BEAT_INTERVAL_FRAMES = int(60 / BPM * SR / HOP_LENGTH)
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
        # Beat 0 should be at frame 0 * hop_length = 0
        assert beats_to_samples(0, BEAT_FRAMES, SR, HOP_LENGTH) == 0

        # Beat 1 should be at BEAT_INTERVAL_FRAMES * hop_length
        expected = BEAT_INTERVAL_FRAMES * HOP_LENGTH
        assert beats_to_samples(1, BEAT_FRAMES, SR, HOP_LENGTH) == expected

    def test_mid_range(self):
        """Beat index in middle of array returns correct position."""
        idx = 50
        expected = int(BEAT_FRAMES[idx] * HOP_LENGTH)
        assert beats_to_samples(idx, BEAT_FRAMES, SR, HOP_LENGTH) == expected

    def test_extrapolation(self):
        """Beat index beyond array length extrapolates using average of last 8 intervals."""
        # Ask for beat 210 when we only have 200 beats (0..199)
        result = beats_to_samples(210, BEAT_FRAMES, SR, HOP_LENGTH)

        # With uniform spacing, extrapolation should be accurate
        last_8 = BEAT_FRAMES[-8:]
        avg_beat_len = float(np.mean(np.diff(last_8))) * HOP_LENGTH
        overshoot = 210 - 200 + 1  # 11
        expected = int(BEAT_FRAMES[-1] * HOP_LENGTH + overshoot * avg_beat_len)
        assert result == expected

    def test_degenerate_case(self):
        """Fewer than 2 beat frames falls back to beat_index * sr."""
        single = np.array([0])
        result = beats_to_samples(5, single, SR, HOP_LENGTH)
        assert result == 5 * SR


# ---------------------------------------------------------------------------
# make_transition_envelope tests
# ---------------------------------------------------------------------------

class TestMakeTransitionEnvelope:
    """Tests for make_transition_envelope()."""

    def test_fade_starts_at_zero_ends_at_one(self):
        """Fade envelope starts near 0 and ends near 1."""
        env = make_transition_envelope(1000, "fade", SR)
        assert len(env) == 1000
        assert env[0] == pytest.approx(0.0, abs=1e-5)
        assert env[-1] == pytest.approx(1.0, abs=1e-5)

    def test_crossfade_same_as_fade(self):
        """Crossfade in-curve is the same shape as fade."""
        fade = make_transition_envelope(500, "fade", SR)
        crossfade = make_transition_envelope(500, "crossfade", SR)
        np.testing.assert_array_almost_equal(fade, crossfade)

    def test_cut_has_micro_crossfade(self):
        """Cut transition has a short ramp at the start, then 1.0."""
        env = make_transition_envelope(1000, "cut", SR)
        assert len(env) == 1000
        # First sample should be 0 (or near 0)
        assert env[0] == pytest.approx(0.0, abs=0.02)
        # After micro-crossfade (~88 samples at 44100), should be 1.0
        assert env[100] == pytest.approx(1.0, abs=1e-5)
        # Last sample should be 1.0
        assert env[-1] == pytest.approx(1.0, abs=1e-5)

    def test_unknown_type_returns_ones(self):
        """Unknown transition type returns an envelope of all ones."""
        env = make_transition_envelope(500, "unknown_type", SR)
        np.testing.assert_array_equal(env, np.ones(500, dtype=np.float32))

    def test_zero_length(self):
        """Zero-length request returns empty array."""
        env = make_transition_envelope(0, "fade", SR)
        assert len(env) == 0


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
        """Intro section has vocal gain 0.0, so no vocal energy in that region."""
        sections = _make_sections(200)
        total = _total_samples()
        vocal_stems, inst_stems = self._make_stems(total)

        vocal_bus, _ = render_arrangement(
            sections, vocal_stems, inst_stems, BEAT_FRAMES, SR, HOP_LENGTH,
        )

        # Intro ends at beat 200 // 8 = 25
        intro_end = beats_to_samples(25, BEAT_FRAMES, SR, HOP_LENGTH)
        intro_region = vocal_bus[:intro_end]
        assert np.allclose(intro_region, 0.0), "Intro region should have zero vocal energy"

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
