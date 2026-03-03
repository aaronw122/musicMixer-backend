"""Tests for musicmixer.services.interpreter -- deterministic fallback plan + prompt caching."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, EnergyBuckets, LyricLine, LyricsData, RemixPlan, Section, StemAnalysis
from musicmixer.services.interpreter import (
    _build_few_shot_messages,
    _build_system_prompt_blocks,
    _validate_remix_plan,
    default_arrangement,
    generate_fallback_plan,
    interpret_prompt,
)


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
        """Produces 5, 6, or 8 sections depending on total beat budget."""
        meta_a = _make_metadata(bpm=95.0, duration=210.0)
        meta_b = _make_metadata(bpm=120.0, duration=240.0)

        plan = generate_fallback_plan(meta_a, meta_b)

        assert len(plan.sections) in (5, 6, 8)

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
        assert plan.tempo_source == "average"
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
        """When a song is shorter than target duration, end time is capped at song duration."""
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

    def test_default_arrangement_section_count(self):
        """Returns 8 sections for >= 192 beats, 6 for 96-191, otherwise 5."""
        for total_beats in [40, 80, 100, 180, 200, 360]:
            sections = default_arrangement(total_beats)
            if total_beats >= 192:
                expected = 8
            elif total_beats >= 96:
                expected = 6
            else:
                expected = 5
            assert len(sections) == expected, (
                f"Expected {expected} sections for total_beats={total_beats}"
            )

    def test_default_arrangement_labels(self):
        """Sections have expected labels for 5-, 6-, or 8-section layouts."""
        # 6-section (96-191 beats)
        sections_6 = default_arrangement(150)
        labels_6 = [s.label for s in sections_6]
        assert labels_6 == ["intro", "build", "main", "breakdown", "drop", "outro"]

        # 8-section (>= 192 beats)
        sections_8 = default_arrangement(200)
        labels_8 = [s.label for s in sections_8]
        assert labels_8 == [
            "intro", "verse", "drop", "breakdown", "verse", "drop", "breakdown", "outro",
        ]

        # 5-section (< 96 beats)
        sections_5 = default_arrangement(80)
        labels_5 = [s.label for s in sections_5]
        assert labels_5 == ["intro", "build", "main", "breakdown", "outro"]

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

    def test_default_arrangement_outro_low_vocals(self):
        """Outro section keeps vocals non-negative and bounded."""
        sections = default_arrangement(200)
        outro = sections[-1]
        assert 0.0 <= outro.stem_gains["vocals"] <= 0.5

    def test_default_arrangement_breakdown_low_drums(self):
        """Breakdown section has reduced but non-zero drum gain (avoids 'song stopped' feel)."""
        sections = default_arrangement(200)
        breakdown = [s for s in sections if s.label == "breakdown"][0]
        assert 0.0 < breakdown.stem_gains["drums"] <= 0.3


# ---------------------------------------------------------------------------
# Helpers for prompt caching tests
# ---------------------------------------------------------------------------

def _make_lyrics(is_synced: bool = True, lines: list[LyricLine] | None = None) -> LyricsData:
    """Create synthetic lyrics data for testing."""
    if lines is None:
        lines = [
            LyricLine(text="Hello world", timestamp_seconds=5.0, bar_number=2),
            LyricLine(text="Second line", timestamp_seconds=10.0, bar_number=4),
        ]
    return LyricsData(
        artist="Test Artist",
        title="Test Song",
        source="lrclib",
        is_synced=is_synced,
        lines=lines,
        raw_text="Hello world\nSecond line",
    )


def _default_prompt_args(
    bpm_a: float = 120.0,
    bpm_b: float = 118.0,
    duration_a: float = 240.0,
    duration_b: float = 210.0,
    stretch_pct: float | None = None,
    lyrics_a: LyricsData | None = None,
    lyrics_b: LyricsData | None = None,
    vocal_stem_lufs: dict[str, float] | None = None,
    inst_stem_lufs: dict[str, float] | None = None,
) -> dict:
    """Build kwargs dict for _build_system_prompt_blocks."""
    meta_a = _make_metadata(bpm=bpm_a, duration=duration_a)
    meta_b = _make_metadata(bpm=bpm_b, duration=duration_b)
    from musicmixer.services.interpreter import _compute_key_guidance, estimate_target_bpm, TARGET_REMIX_DURATION_SECONDS
    _key_available, key_detail = _compute_key_guidance(meta_a, meta_b)
    target_bpm = estimate_target_bpm(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)
    total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    return dict(
        song_a_meta=meta_a,
        song_b_meta=meta_b,
        key_matching_available=_key_available,
        key_matching_detail=key_detail,
        total_available_beats=total_available_beats,
        stretch_pct=stretch_pct,
        lyrics_a=lyrics_a,
        lyrics_b=lyrics_b,
        vocal_stem_lufs=vocal_stem_lufs,
        inst_stem_lufs=inst_stem_lufs,
    )


# ---------------------------------------------------------------------------
# Phase 2 tests: _build_system_prompt_blocks
# ---------------------------------------------------------------------------


class TestBuildSystemPromptBlocks:
    """Tests for _build_system_prompt_blocks()."""

    def test_returns_two_blocks(self):
        """Returns exactly 2 content block dicts."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        assert isinstance(blocks, list)
        assert len(blocks) == 2

    def test_first_block_has_cache_control(self):
        """First block has cache_control: {type: ephemeral}."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        assert blocks[0]["type"] == "text"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_second_block_no_cache_control(self):
        """Second block does NOT have cache_control."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        assert blocks[1]["type"] == "text"
        assert "cache_control" not in blocks[1]

    def test_static_block_is_constant(self):
        """Different songs/BPMs produce identical block[0] text (cached block never changes)."""
        args1 = _default_prompt_args(bpm_a=90.0, bpm_b=140.0, duration_a=180.0, duration_b=300.0)
        args2 = _default_prompt_args(bpm_a=120.0, bpm_b=100.0, duration_a=240.0, duration_b=200.0)
        blocks1 = _build_system_prompt_blocks(**args1)
        blocks2 = _build_system_prompt_blocks(**args2)
        assert blocks1[0]["text"] == blocks2[0]["text"]

    def test_dynamic_block_varies_with_song_data(self):
        """Dynamic block changes when song metadata changes."""
        args1 = _default_prompt_args(bpm_a=90.0, bpm_b=140.0)
        args2 = _default_prompt_args(bpm_a=120.0, bpm_b=100.0)
        blocks1 = _build_system_prompt_blocks(**args1)
        blocks2 = _build_system_prompt_blocks(**args2)
        assert blocks1[1]["text"] != blocks2[1]["text"]

    def test_dynamic_block_contains_song_metadata(self):
        """Song names/BPMs appear in block[1] only, not block[0]."""
        args = _default_prompt_args(bpm_a=95.0, bpm_b=125.0)
        blocks = _build_system_prompt_blocks(**args)
        # BPM values appear in dynamic block (song data)
        assert "95 BPM" in blocks[1]["text"]
        assert "125 BPM" in blocks[1]["text"]
        # BPM values should NOT appear in static block
        assert "95 BPM" not in blocks[0]["text"]
        assert "125 BPM" not in blocks[0]["text"]

    def test_blocks_contain_all_expected_sections(self):
        """Combined blocks text contains all expected section markers.

        Verifies that every section of the system prompt is present across
        the two blocks (static + dynamic).
        """
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        combined = blocks[0]["text"] + "\n\n" + blocks[1]["text"]

        # Key strings that uniquely identify each section
        section_markers = [
            "You are a music remix planner",           # Section 1
            "CRITICAL MIXING RULES",                   # Section 2
            "MIXING ADVISORY",                         # Section 3
            "STEM LOUDNESS AWARENESS",                 # Section 3b
            "TRANSITIONS:",                            # Section 4
            "Template A (Standard Mashup)",            # Section 5
            "SECTION RULES:",                          # Section 6
            "GENRE GUIDANCE",                          # Section 7
            "TEMPO MATCHING:",                         # Section 8
            "HANDLING AMBIGUOUS PROMPTS",              # Section 9
            "STEM SEPARATION ARTIFACTS",               # Section 10
            "EXPLANATION: Write 2-3",                  # Section 11
            "SONG DATA:",                              # Section 12
            "1 bar = 4 beats",                         # Section 13
            "STEM GAIN REFERENCE",                     # Gain reference section
        ]
        for marker in section_markers:
            assert marker in combined, f"Marker missing from blocks: {marker}"

    @pytest.mark.parametrize("variant,kwargs", [
        ("no_lyrics_no_stretch", dict()),
        ("with_lyrics", dict(lyrics_a=_make_lyrics())),
        ("with_stretch_above_12", dict(stretch_pct=15.0)),
        ("with_key_detail", dict(bpm_a=100.0, bpm_b=105.0)),
    ])
    def test_blocks_contain_all_sections_parameterized(self, variant, kwargs):
        """All expected section markers present in blocks output for all variants."""
        args = _default_prompt_args(**kwargs)
        blocks = _build_system_prompt_blocks(**args)
        combined = blocks[0]["text"] + "\n\n" + blocks[1]["text"]

        # These markers must be present regardless of variant
        required_markers = [
            "You are a music remix planner",
            "CRITICAL MIXING RULES",
            "MIXING ADVISORY",
            "STEM LOUDNESS AWARENESS",
            "TRANSITIONS:",
            "SECTION RULES:",
            "GENRE GUIDANCE",
            "TEMPO MATCHING:",
            "HANDLING AMBIGUOUS PROMPTS",
            "STEM SEPARATION ARTIFACTS",
            "EXPLANATION: Write 2-3",
            "SONG DATA:",
            "1 bar = 4 beats",
            "STEM GAIN REFERENCE",
        ]
        for marker in required_markers:
            assert marker in combined, (
                f"Variant {variant}: marker missing from blocks: {marker}"
            )

    def test_stretch_advisory_in_dynamic_block(self):
        """When stretch_pct > 12, the stretch warning appears in dynamic block only."""
        args = _default_prompt_args(stretch_pct=18.5)
        blocks = _build_system_prompt_blocks(**args)
        assert "STRETCH WARNING (18.5%)" in blocks[1]["text"]
        assert "STRETCH WARNING" not in blocks[0]["text"]

    def test_lufs_guidance_in_static_block(self):
        """STEM LOUDNESS AWARENESS section appears in static (cached) block."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        assert "STEM LOUDNESS AWARENESS" in blocks[0]["text"]
        assert "Stems below -25 LUFS" in blocks[0]["text"]
        assert "Stems below -30 LUFS" in blocks[0]["text"]
        assert "Stems above -18 LUFS" in blocks[0]["text"]

    def test_lufs_values_not_in_static_block(self):
        """Per-stem LUFS values should never appear in the static (cached) block."""
        args = _default_prompt_args(
            vocal_stem_lufs={"vocals": -16.4},
            inst_stem_lufs={"drums": -18.8, "bass": -20.2},
        )
        blocks = _build_system_prompt_blocks(**args)
        assert "-16.4 LUFS" not in blocks[0]["text"]
        assert "-18.8 LUFS" not in blocks[0]["text"]

    def test_works_without_lufs_params(self):
        """Backward compat: function works when LUFS params are None."""
        args = _default_prompt_args()
        # Explicitly ensure LUFS params are None (the default)
        assert args["vocal_stem_lufs"] is None
        assert args["inst_stem_lufs"] is None
        blocks = _build_system_prompt_blocks(**args)
        assert len(blocks) == 2
        # LUFS guidance section is still present in static block
        assert "STEM LOUDNESS AWARENESS" in blocks[0]["text"]

    def test_lufs_values_in_dynamic_block_with_stem_analysis(self):
        """When LUFS dicts and stem_analysis are provided, LUFS values appear in dynamic block."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # Add stem_analysis to both so _build_stem_character produces output
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.05, p50=0.10, p85=0.20)
        meta_a.stem_analysis = StemAnalysis(
            bar_rms={"vocals": np.full(60, 0.15)},
            combined_energy=np.full(60, 0.5),
            vocal_active=np.full(60, True),
            vocal_gaps=[],
            bucket_thresholds=buckets,
        )
        meta_b.stem_analysis = StemAnalysis(
            bar_rms={
                "drums": np.full(60, 0.18),
                "bass": np.full(60, 0.12),
            },
            combined_energy=np.full(60, 0.5),
            vocal_active=np.full(60, False),
            vocal_gaps=[],
            bucket_thresholds=buckets,
        )

        from musicmixer.services.interpreter import _compute_key_guidance, estimate_target_bpm, TARGET_REMIX_DURATION_SECONDS
        _key_available, key_detail = _compute_key_guidance(meta_a, meta_b)
        target_bpm = estimate_target_bpm(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)
        total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)

        blocks = _build_system_prompt_blocks(
            song_a_meta=meta_a,
            song_b_meta=meta_b,
            key_matching_available=_key_available,
            key_matching_detail=key_detail,
            total_available_beats=total_available_beats,
            vocal_stem_lufs={"vocals": -16.4},
            inst_stem_lufs={"drums": -18.8, "bass": -22.7},
        )

        dynamic_text = blocks[1]["text"]
        assert "-16.4 LUFS" in dynamic_text
        assert "-18.8 LUFS" in dynamic_text
        assert "-22.7 LUFS" in dynamic_text

    def test_lufs_values_absent_when_none(self):
        """When LUFS dicts are None but stem_analysis exists, no LUFS values in output."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        buckets = EnergyBuckets(noise_floor=0.02, p10=0.05, p50=0.10, p85=0.20)
        meta_a.stem_analysis = StemAnalysis(
            bar_rms={"vocals": np.full(60, 0.15)},
            combined_energy=np.full(60, 0.5),
            vocal_active=np.full(60, True),
            vocal_gaps=[],
            bucket_thresholds=buckets,
        )
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        from musicmixer.services.interpreter import _compute_key_guidance, estimate_target_bpm, TARGET_REMIX_DURATION_SECONDS
        _key_available, key_detail = _compute_key_guidance(meta_a, meta_b)
        target_bpm = estimate_target_bpm(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)
        total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)

        blocks = _build_system_prompt_blocks(
            song_a_meta=meta_a,
            song_b_meta=meta_b,
            key_matching_available=_key_available,
            key_matching_detail=key_detail,
            total_available_beats=total_available_beats,
            vocal_stem_lufs=None,
            inst_stem_lufs=None,
        )

        dynamic_text = blocks[1]["text"]
        assert "LUFS)" not in dynamic_text


# ---------------------------------------------------------------------------
# Phase 2 tests: _build_few_shot_messages cache_control
# ---------------------------------------------------------------------------


class TestFewShotMessagesCaching:
    """Tests for cache_control on few-shot messages."""

    def test_few_shot_last_message_has_cache_control(self):
        """Last few-shot message's content block has cache_control: {type: ephemeral}."""
        messages = _build_few_shot_messages()
        last_msg = messages[-1]
        assert last_msg["role"] == "user"
        # The last content block in the last message
        last_content_block = last_msg["content"][-1]
        assert last_content_block["cache_control"] == {"type": "ephemeral"}

    def test_few_shot_messages_always_6stem(self):
        """All 6 stems present in every section's stem_gains in the few-shot examples."""
        messages = _build_few_shot_messages()
        expected_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            assert set(section["stem_gains"].keys()) == expected_stems, (
                                f"Missing stems in example section {section['label']}: "
                                f"got {set(section['stem_gains'].keys())}"
                            )

    def test_few_shot_non_last_messages_no_cache_control(self):
        """Only the last message has cache_control; earlier tool_results do not."""
        messages = _build_few_shot_messages()
        # Check all but the last message
        for msg in messages[:-1]:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict):
                        assert "cache_control" not in block, (
                            f"Unexpected cache_control in non-last message: {block}"
                        )


# ---------------------------------------------------------------------------
# Phase 3 tests: API call integration + cache stats logging
# ---------------------------------------------------------------------------


class TestInterpretPromptCaching:
    """Tests for interpret_prompt() with prompt caching integration."""

    def test_interpret_prompt_passes_blocks_to_api(self):
        """interpret_prompt() passes list[dict] with 2 entries to system= kwarg."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # Build a mock response that mimics Anthropic's API
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "test_id"
        mock_tool_use.input = {
            "vocal_source": "song_a",
            "start_time_vocal": 0.0,
            "end_time_vocal": 200.0,
            "start_time_instrumental": 0.0,
            "end_time_instrumental": 200.0,
            "sections": [
                {"label": "intro", "start_beat": 0, "end_beat": 32,
                 "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5},
                 "transition_in": "fade", "transition_beats": 4},
                {"label": "verse", "start_beat": 32, "end_beat": 128,
                 "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 0.7, "guitar": 0.3, "piano": 0.2, "other": 0.3},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "breakdown", "start_beat": 128, "end_beat": 192,
                 "stem_gains": {"vocals": 0.3, "drums": 0.3, "bass": 0.5, "guitar": 0.6, "piano": 0.5, "other": 0.4},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "drop", "start_beat": 192, "end_beat": 352,
                 "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.3, "piano": 0.2, "other": 0.3},
                 "transition_in": "cut", "transition_beats": 0},
                {"label": "outro", "start_beat": 352, "end_beat": 416,
                 "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.4, "piano": 0.3, "other": 0.4},
                 "transition_in": "crossfade", "transition_beats": 8},
            ],
            "tempo_source": "song_b",
            "key_source": "none",
            "explanation": "Test explanation.",
            "warnings": [],
        }

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_use]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock()
        mock_response.usage.cache_read_input_tokens = 1000
        mock_response.usage.cache_creation_input_tokens = 5000

        with patch("musicmixer.services.interpreter.settings") as mock_settings, \
             patch("musicmixer.services.interpreter.anthropic") as mock_anthropic:
            mock_settings.stem_backend = "modal"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_timeout_seconds = 30
            mock_settings.llm_max_retries = 1

            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            plan = interpret_prompt("test prompt", meta_a, meta_b)

            # Verify the system= kwarg was a list of dicts
            call_kwargs = mock_client.messages.create.call_args
            system_arg = call_kwargs.kwargs["system"]

            assert isinstance(system_arg, list), f"system= should be list, got {type(system_arg)}"
            assert len(system_arg) == 2, f"system= should have 2 blocks, got {len(system_arg)}"
            assert system_arg[0]["type"] == "text"
            assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
            assert system_arg[1]["type"] == "text"
            assert "cache_control" not in system_arg[1]

    def test_cache_stats_logging_no_crash(self):
        """Cache stats logging works even when usage object lacks cache fields."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "test_id"
        mock_tool_use.input = {
            "vocal_source": "song_a",
            "start_time_vocal": 0.0,
            "end_time_vocal": 200.0,
            "start_time_instrumental": 0.0,
            "end_time_instrumental": 200.0,
            "sections": [
                {"label": "intro", "start_beat": 0, "end_beat": 32,
                 "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5},
                 "transition_in": "fade", "transition_beats": 4},
                {"label": "verse", "start_beat": 32, "end_beat": 128,
                 "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 0.7, "guitar": 0.3, "piano": 0.2, "other": 0.3},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "drop", "start_beat": 128, "end_beat": 352,
                 "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.3, "piano": 0.2, "other": 0.3},
                 "transition_in": "cut", "transition_beats": 0},
                {"label": "outro", "start_beat": 352, "end_beat": 416,
                 "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.4, "piano": 0.3, "other": 0.4},
                 "transition_in": "crossfade", "transition_beats": 8},
            ],
            "tempo_source": "song_b",
            "key_source": "none",
            "explanation": "Test.",
            "warnings": [],
        }

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_tool_use]
        mock_response.model = "claude-sonnet-4-20250514"
        # Simulate old SDK without cache fields -- use a plain object
        mock_usage = type("Usage", (), {"input_tokens": 100, "output_tokens": 50})()
        mock_response.usage = mock_usage

        with patch("musicmixer.services.interpreter.settings") as mock_settings, \
             patch("musicmixer.services.interpreter.anthropic") as mock_anthropic:
            mock_settings.stem_backend = "modal"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_timeout_seconds = 30
            mock_settings.llm_max_retries = 1

            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            # Should not raise even without cache_read_input_tokens / cache_creation_input_tokens
            plan = interpret_prompt("test prompt", meta_a, meta_b)
            assert isinstance(plan, RemixPlan)

    def test_stem_backend_local_raises(self):
        """interpret_prompt() raises ValueError when stem_backend is local."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        with patch("musicmixer.services.interpreter.settings") as mock_settings:
            mock_settings.stem_backend = "local"

            with pytest.raises(ValueError, match="stem_backend='local' is not supported"):
                interpret_prompt("test prompt", meta_a, meta_b)


# ---------------------------------------------------------------------------
# Prompt revision tests: verify new prompt text
# ---------------------------------------------------------------------------


class TestPromptRevisionText:
    """Tests verifying the revised system prompt text for fuller mixes."""

    def test_vocal_instrumental_balance_in_static_block(self):
        """Static block contains revised Rule 2 about vocal-instrumental balance."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        static_text = blocks[0]["text"]
        assert "VOCAL-INSTRUMENTAL BALANCE" in static_text
        assert "FULL MIX" in static_text
        assert "spectral ducking" in static_text
        assert "BAND PLAYING TOGETHER" in static_text

    def test_stem_muting_policy_in_static_block(self):
        """Static block contains revised stem muting policy."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        static_text = blocks[0]["text"]
        assert "Stem Muting Policy" in static_text
        assert "COMPLETE REMOVAL" in static_text
        assert "3+ muted stems = thin mix" in static_text
        assert "0.25-0.35" in static_text

    def test_stem_gain_reference_in_static_block(self):
        """Static block contains the new Stem Gain Reference section."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        static_text = blocks[0]["text"]
        assert "STEM GAIN REFERENCE" in static_text
        assert "GAIN SCALE (linear amplitude)" in static_text
        assert "ENERGY BUDGET BY SECTION TYPE" in static_text
        assert "Chorus/Drop: target sum 4.5-5.5" in static_text
        assert "drum-bass foundation" in static_text

    def test_rule1_updated_minimum_gain(self):
        """Rule 1 uses updated minimum gain of 0.30-0.35 for other stem."""
        args = _default_prompt_args()
        blocks = _build_system_prompt_blocks(**args)
        static_text = blocks[0]["text"]
        assert "gain 0.30-0.35" in static_text

    def test_few_shot_gains_are_fuller(self):
        """Few-shot examples use higher gains (no aggressive muting of instrumentals)."""
        messages = _build_few_shot_messages()
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            gains = section["stem_gains"]
                            # Count muted instrumental stems (gain == 0.0, excluding vocals)
                            muted_inst = sum(
                                1 for stem, g in gains.items()
                                if stem != "vocals" and g == 0.0
                            )
                            # No section should have more than 2 muted instrumental stems
                            assert muted_inst <= 2, (
                                f"Section '{section['label']}' has {muted_inst} muted "
                                f"instrumental stems: {gains}"
                            )

    def test_few_shot_verse_gains_follow_guidance(self):
        """Verse sections in few-shot examples have gains matching the new guidance."""
        messages = _build_few_shot_messages()
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            if section["label"] == "verse":
                                gains = section["stem_gains"]
                                if gains["vocals"] > 0.0:  # vocal verse
                                    # Drums should be 0.75-0.95
                                    assert 0.70 <= gains["drums"] <= 0.95, (
                                        f"Verse drums={gains['drums']}, expected 0.75-0.95"
                                    )
                                    # Bass should be 0.70-0.90
                                    assert 0.65 <= gains["bass"] <= 0.90, (
                                        f"Verse bass={gains['bass']}, expected 0.70-0.90"
                                    )


# ---------------------------------------------------------------------------
# Arrangement validation tests
# ---------------------------------------------------------------------------


def _make_plan_with_sections(sections: list[Section]) -> RemixPlan:
    """Create a RemixPlan with the given sections for validation testing."""
    return RemixPlan(
        vocal_source="song_a",
        start_time_vocal=0.0,
        end_time_vocal=200.0,
        start_time_instrumental=0.0,
        end_time_instrumental=200.0,
        sections=sections,
        tempo_source="weighted_midpoint",
        key_source="none",
        explanation="Test plan.",
        warnings=[],
        used_fallback=False,
    )


class TestArrangementValidation:
    """Tests for arrangement quality validation in _validate_remix_plan()."""

    def test_thin_section_warning(self):
        """Sections with < 2 active instrumental stems trigger a warning."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.4}, "fade", 4),
            # Only bass is active -- drums, guitar, piano, other all muted
            Section("verse", 32, 128, {"vocals": 0.9, "drums": 0.0, "bass": 0.5, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 4),
            Section("drop", 128, 352, {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.4}, "cut", 0),
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.4, "piano": 0.4, "other": 0.4}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        thin_warnings = [w for w in result.warnings if "active instrumental stems" in w]
        assert len(thin_warnings) >= 1
        assert "verse" in thin_warnings[0]

    def test_thin_section_allowed_for_intro_outro(self):
        """Intro and outro sections do NOT trigger thin section warnings."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            # Intro with only drums -- should NOT warn
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.8, "bass": 0.0, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "fade", 4),
            Section("verse", 32, 128, {"vocals": 0.9, "drums": 0.8, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.4}, "crossfade", 4),
            Section("drop", 128, 352, {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 0.4}, "cut", 0),
            # Outro with only drums -- should NOT warn
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.5, "bass": 0.0, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        thin_warnings = [w for w in result.warnings if "active instrumental stems" in w]
        assert len(thin_warnings) == 0

    def test_globally_muted_stem_warning(self):
        """A stem muted in ALL sections triggers a warning."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # Piano is 0.0 in every section
        sections = [
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.5, "piano": 0.0, "other": 0.4}, "fade", 4),
            Section("verse", 32, 128, {"vocals": 0.9, "drums": 0.8, "bass": 0.7, "guitar": 0.5, "piano": 0.0, "other": 0.4}, "crossfade", 4),
            Section("drop", 128, 352, {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.6, "piano": 0.0, "other": 0.5}, "cut", 0),
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.4, "piano": 0.0, "other": 0.4}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        muted_warnings = [w for w in result.warnings if "muted (0.0) in every section" in w]
        assert len(muted_warnings) >= 1
        assert "piano" in muted_warnings[0]

    def test_high_muting_rate_warning(self):
        """High overall muting percentage (>30%) triggers a warning."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # Every section has 3+ muted instrumental stems = high muting rate
        sections = [
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "fade", 4),
            Section("verse", 32, 128, {"vocals": 0.9, "drums": 0.7, "bass": 0.5, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 4),
            Section("drop", 128, 352, {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "cut", 0),
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        mute_rate_warnings = [w for w in result.warnings if "muting rate" in w]
        assert len(mute_rate_warnings) >= 1

    def test_good_plan_no_arrangement_warnings(self):
        """A well-balanced plan produces no arrangement quality warnings."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.80, "bass": 0.85, "guitar": 0.55, "piano": 0.45, "other": 0.40}, "fade", 4),
            Section("verse", 32, 128, {"vocals": 0.90, "drums": 0.80, "bass": 0.75, "guitar": 0.55, "piano": 0.45, "other": 0.40}, "crossfade", 4),
            Section("breakdown", 128, 192, {"vocals": 0.0, "drums": 0.50, "bass": 0.70, "guitar": 0.80, "piano": 0.60, "other": 0.45}, "crossfade", 4),
            Section("drop", 192, 352, {"vocals": 1.0, "drums": 0.90, "bass": 0.85, "guitar": 0.65, "piano": 0.55, "other": 0.50}, "cut", 0),
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.55, "bass": 0.65, "guitar": 0.50, "piano": 0.50, "other": 0.45}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        # No arrangement quality warnings
        arrangement_keywords = ["active instrumental stems", "muted (0.0) in every", "muting rate"]
        arrangement_warnings = [
            w for w in result.warnings
            if any(kw in w for kw in arrangement_keywords)
        ]
        assert len(arrangement_warnings) == 0, f"Unexpected warnings: {arrangement_warnings}"

    def test_validation_warnings_are_non_blocking(self):
        """Arrangement warnings do not prevent the plan from being returned."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # A plan with issues but still structurally valid
        sections = [
            Section("intro", 0, 32, {"vocals": 0.0, "drums": 0.8, "bass": 0.7, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "fade", 4),
            Section("verse", 32, 128, {"vocals": 0.9, "drums": 0.0, "bass": 0.5, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 4),
            Section("drop", 128, 352, {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "cut", 0),
            Section("outro", 352, 416, {"vocals": 0.0, "drums": 0.5, "bass": 0.5, "guitar": 0.0, "piano": 0.0, "other": 0.0}, "crossfade", 8),
        ]
        plan = _make_plan_with_sections(sections)
        result = _validate_remix_plan(plan, meta_a, meta_b)

        # Plan is returned (not blocked), has warnings, and has correct structure
        assert isinstance(result, RemixPlan)
        assert len(result.warnings) > 0
        assert len(result.sections) >= 2
