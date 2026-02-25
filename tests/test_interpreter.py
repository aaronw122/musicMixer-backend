"""Tests for musicmixer.services.interpreter -- deterministic fallback plan."""

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, RemixPlan, Section
from musicmixer.services.interpreter import default_arrangement, generate_fallback_plan


def _make_metadata(bpm: float = 120.0, duration: float = 240.0) -> AudioMetadata:
    """Create a synthetic AudioMetadata for testing."""
    beat_interval_frames = int(60 / bpm * 44100 / 512)
    num_beats = int(bpm * duration / 60)
    beat_frames = np.arange(0, num_beats) * beat_interval_frames
    total_beats = round(bpm * duration / 60 / 4) * 4  # Round to nearest bar
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=total_beats,
    )


class TestGenerateFallbackPlan:
    """Tests for generate_fallback_plan()."""

    def test_fallback_plan_structure(self):
        """Verify RemixPlan has correct fields and types."""
        meta_a = _make_metadata(bpm=95.0, duration=210.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        assert isinstance(plan, RemixPlan)
        assert isinstance(plan.vocal_source, str)
        assert isinstance(plan.start_time_vocal, float)
        assert isinstance(plan.end_time_vocal, float)
        assert isinstance(plan.start_time_instrumental, float)
        assert isinstance(plan.end_time_instrumental, float)
        assert isinstance(plan.sections, list)
        assert all(isinstance(s, Section) for s in plan.sections)
        assert isinstance(plan.tempo_source, str)
        assert isinstance(plan.key_source, str)
        assert isinstance(plan.explanation, str)
        assert isinstance(plan.warnings, list)
        assert isinstance(plan.used_fallback, bool)
        assert plan.used_fallback is True

    def test_fallback_plan_sections_count(self):
        """Always produces exactly 5 sections."""
        meta_a = _make_metadata(bpm=95.0, duration=210.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        assert len(plan.sections) == 5

    def test_fallback_plan_sections_contiguous(self):
        """Each section's start_beat equals the previous section's end_beat."""
        meta_a = _make_metadata(bpm=100.0, duration=200.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        for i in range(1, len(plan.sections)):
            assert plan.sections[i].start_beat == plan.sections[i - 1].end_beat, (
                f"Section {i} ({plan.sections[i].label}) start_beat "
                f"{plan.sections[i].start_beat} != previous end_beat "
                f"{plan.sections[i - 1].end_beat}"
            )

    def test_fallback_plan_vocal_source(self):
        """Vocal source defaults to song_a for Day 2."""
        meta_a = _make_metadata(bpm=90.0, duration=180.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        assert plan.vocal_source == "song_a"
        assert plan.tempo_source == "song_b"
        assert plan.key_source == "none"

    def test_fallback_plan_time_ranges(self):
        """Start/end times are within each song's duration."""
        meta_a = _make_metadata(bpm=90.0, duration=180.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        # Vocal times within song A's duration
        assert 0.0 <= plan.start_time_vocal < meta_a.duration_seconds
        assert plan.start_time_vocal < plan.end_time_vocal <= meta_a.duration_seconds

        # Instrumental times within song B's duration
        assert 0.0 <= plan.start_time_instrumental < meta_b.duration_seconds
        assert (
            plan.start_time_instrumental
            < plan.end_time_instrumental
            <= meta_b.duration_seconds
        )

    def test_fallback_plan_time_ranges_short_song(self):
        """When a song is shorter than 90s, end time is capped at duration."""
        meta_a = _make_metadata(bpm=120.0, duration=60.0)
        meta_b = _make_metadata(bpm=120.0, duration=60.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        assert plan.end_time_vocal <= meta_a.duration_seconds
        assert plan.end_time_instrumental <= meta_b.duration_seconds


class TestDefaultArrangement:
    """Tests for default_arrangement()."""

    def test_default_arrangement_beat_range(self):
        """First section starts at 0, last ends at total_beats."""
        total_beats = 180
        sections = default_arrangement(total_beats)

        assert sections[0].start_beat == 0
        assert sections[-1].end_beat == total_beats

    def test_default_arrangement_five_sections(self):
        """Always returns 5 sections."""
        for total_beats in [100, 180, 200, 360]:
            sections = default_arrangement(total_beats)
            assert len(sections) == 5, f"Expected 5 sections for total_beats={total_beats}"

    def test_default_arrangement_labels(self):
        """Sections have the expected labels in order."""
        sections = default_arrangement(200)
        labels = [s.label for s in sections]
        assert labels == ["intro", "build", "main", "breakdown", "outro"]

    def test_default_arrangement_contiguous(self):
        """Sections are contiguous (no gaps or overlaps)."""
        sections = default_arrangement(200)
        for i in range(1, len(sections)):
            assert sections[i].start_beat == sections[i - 1].end_beat

    def test_default_arrangement_intro_no_vocals(self):
        """Intro section has vocal gain of 0.0."""
        sections = default_arrangement(200)
        intro = sections[0]
        assert intro.stem_gains["vocals"] == 0.0

    def test_default_arrangement_outro_no_vocals(self):
        """Outro section has vocal gain of 0.0."""
        sections = default_arrangement(200)
        outro = sections[-1]
        assert outro.stem_gains["vocals"] == 0.0

    def test_default_arrangement_breakdown_no_drums(self):
        """Breakdown section has drum gain of 0.0."""
        sections = default_arrangement(200)
        breakdown = [s for s in sections if s.label == "breakdown"][0]
        assert breakdown.stem_gains["drums"] == 0.0
