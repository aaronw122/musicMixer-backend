"""Tests for musicmixer.services.interpreter -- deterministic fallback plan + prompt caching."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicmixer.models import (
    AudioMetadata,
    ChordEvent,
    ChordProgression,
    DrumPattern,
    EnergyBuckets,
    IntentPlan,
    IntentSection,
    LyricLine,
    LyricsData,
    PolyphonyInfo,
    RemixPlan,
    Section,
    StemAnalysis,
    WordAlignment,
    WordEvent,
)
from musicmixer.services.interpreter import (
    LLM_MAX_TOKENS,
    TARGET_REMIX_DURATION_SECONDS,
    _build_cross_song_layer,
    _build_dynamic_context,
    _build_few_shot_messages,
    _build_lyrics_layer,
    _build_song_info,
    _build_stem_character,
    _build_system_prompt_block,
    _format_word_timing_sample,
    _validate_intent_duration,
    _validate_intent_plan,
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
        assert plan.tempo_source == "weighted_midpoint"

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


def _default_dynamic_args(
    bpm_a: float = 120.0,
    bpm_b: float = 118.0,
    duration_a: float = 240.0,
    duration_b: float = 210.0,
    stretch_pct: float | None = None,
    lyrics_a: LyricsData | None = None,
    lyrics_b: LyricsData | None = None,
) -> dict:
    """Build kwargs dict for _build_dynamic_context."""
    meta_a = _make_metadata(bpm=bpm_a, duration=duration_a)
    meta_b = _make_metadata(bpm=bpm_b, duration=duration_b)
    from musicmixer.services.interpreter import estimate_target_bpm, TARGET_REMIX_DURATION_SECONDS
    target_bpm = estimate_target_bpm(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)
    total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    return dict(
        song_a_meta=meta_a,
        song_b_meta=meta_b,
        total_available_beats=total_available_beats,
        stretch_pct=stretch_pct,
        lyrics_a=lyrics_a,
        lyrics_b=lyrics_b,
    )


# ---------------------------------------------------------------------------
# Phase 2 tests: _build_system_prompt_blocks
# ---------------------------------------------------------------------------


class TestBuildSystemPromptBlock:
    """Tests for _build_system_prompt_block() and _build_dynamic_context()."""

    def test_returns_single_block(self):
        """Returns a single content block dict."""
        block = _build_system_prompt_block()
        assert isinstance(block, dict)
        assert block["type"] == "text"

    def test_block_has_cache_control(self):
        """Block has cache_control: {type: ephemeral}."""
        block = _build_system_prompt_block()
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_static_block_is_constant(self):
        """Calling _build_system_prompt_block() always returns the same text."""
        block1 = _build_system_prompt_block()
        block2 = _build_system_prompt_block()
        assert block1["text"] == block2["text"]

    def test_dynamic_context_varies_with_song_data(self):
        """Dynamic context changes when song metadata changes."""
        args1 = _default_dynamic_args(bpm_a=90.0, bpm_b=140.0)
        args2 = _default_dynamic_args(bpm_a=120.0, bpm_b=100.0)
        ctx1 = _build_dynamic_context(**args1)
        ctx2 = _build_dynamic_context(**args2)
        assert ctx1 != ctx2

    def test_dynamic_context_contains_song_metadata(self):
        """Song BPMs appear in dynamic context, not in static block."""
        args = _default_dynamic_args(bpm_a=95.0, bpm_b=125.0)
        ctx = _build_dynamic_context(**args)
        block = _build_system_prompt_block()
        # BPM values appear in dynamic context (song data)
        assert "95 BPM" in ctx
        assert "125 BPM" in ctx
        # BPM values should NOT appear in static block
        assert "95 BPM" not in block["text"]
        assert "125 BPM" not in block["text"]

    def test_combined_contains_all_expected_sections(self):
        """Combined static + dynamic text contains all expected section markers."""
        block = _build_system_prompt_block()
        args = _default_dynamic_args()
        ctx = _build_dynamic_context(**args)
        combined = block["text"] + "\n\n" + ctx

        # Key strings that uniquely identify each section
        section_markers = [
            "You are an expert music mashup artist",   # Section 1
            "ARRANGEMENT RULES",                       # Section 2
            "TRANSITIONS:",                            # Section 3
            "ARRANGEMENT APPROACH:",                   # Section 4
            "Standard Mashup: intro ->",               # Section 4 (reference)
            "STEM ROLE GUIDELINES",                    # Section 5
            "ENERGY LEVELS AND ARC",                   # Section 5
            "MIXING ADVISORY",                         # Section 5
            "GENRE GUIDANCE",                          # Section 6
            "CRITICAL MIXING RULES",                   # Section 7
            "STEM SEPARATION ARTIFACTS",               # Section 8
            "TEMPO MATCHING:",                         # Section 9
            "EXPLANATION: Write 2-3",                  # Section 10
            "SONG DATA:",                              # Dynamic
            "1 bar = 4 beats",                         # Dynamic (duration)
        ]
        for marker in section_markers:
            assert marker in combined, f"Marker missing: {marker}"

    def test_removed_gain_sections_absent(self):
        """Gain-specific sections are no longer in the system prompt."""
        block = _build_system_prompt_block()
        args = _default_dynamic_args()
        ctx = _build_dynamic_context(**args)
        combined = block["text"] + "\n\n" + ctx

        absent_markers = [
            "STEM GAIN REFERENCE",
            "STEM LOUDNESS AWARENESS",
            "ENERGY BUDGET BY SECTION TYPE",
            "GAIN SCALE (linear amplitude)",
            "HANDLING AMBIGUOUS PROMPTS",
        ]
        for marker in absent_markers:
            assert marker not in combined, f"Removed marker still present: {marker}"

    @pytest.mark.parametrize("variant,kwargs", [
        ("no_lyrics_no_stretch", dict()),
        ("with_lyrics", dict(lyrics_a=_make_lyrics())),
        ("with_stretch_above_12", dict(stretch_pct=15.0)),
        ("with_key_detail", dict(bpm_a=100.0, bpm_b=105.0)),
    ])
    def test_combined_contains_all_sections_parameterized(self, variant, kwargs):
        """All expected section markers present for all variants."""
        block = _build_system_prompt_block()
        args = _default_dynamic_args(**kwargs)
        ctx = _build_dynamic_context(**args)
        combined = block["text"] + "\n\n" + ctx

        # These markers must be present regardless of variant
        required_markers = [
            "You are an expert music mashup artist",
            "CRITICAL MIXING RULES",
            "STEM ROLE GUIDELINES",
            "MIXING ADVISORY",
            "TRANSITIONS:",
            "ARRANGEMENT RULES",
            "GENRE GUIDANCE",
            "TEMPO MATCHING:",
            "STEM SEPARATION ARTIFACTS",
            "EXPLANATION: Write 2-3",
            "SONG DATA:",
            "1 bar = 4 beats",
        ]
        for marker in required_markers:
            assert marker in combined, (
                f"Variant {variant}: marker missing: {marker}"
            )

    def test_stretch_advisory_in_dynamic_context(self):
        """When stretch_pct > 12, the stretch warning appears in dynamic context only."""
        args = _default_dynamic_args(stretch_pct=18.5)
        ctx = _build_dynamic_context(**args)
        block = _build_system_prompt_block()
        assert "STRETCH ADVISORY (18.5% stretch on vocals)" in ctx
        assert "STRETCH ADVISORY" not in block["text"]

    def test_stem_roles_guidance_in_static_block(self):
        """STEM ROLE GUIDELINES section appears in static (cached) block."""
        block = _build_system_prompt_block()
        static_text = block["text"]
        assert "STEM ROLE GUIDELINES" in static_text
        assert '"lead"' in static_text
        assert '"support"' in static_text
        assert '"background"' in static_text
        assert '"texture"' in static_text
        assert '"silent"' in static_text


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
        """All 6 stems present in every section's stem_roles in the few-shot examples."""
        messages = _build_few_shot_messages()
        expected_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
        valid_roles = {"lead", "support", "background", "texture", "silent"}
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            assert set(section["stem_roles"].keys()) == expected_stems, (
                                f"Missing stems in example section {section['label']}: "
                                f"got {set(section['stem_roles'].keys())}"
                            )
                            for stem, role in section["stem_roles"].items():
                                assert role in valid_roles, (
                                    f"Invalid role '{role}' for stem '{stem}' in "
                                    f"section '{section['label']}'"
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

        # Build a mock response that mimics Anthropic's API with new intent format
        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "test_id"
        mock_tool_use.input = {
            "start_time_vocal": 0.0,
            "end_time_vocal": 200.0,
            "start_time_instrumental": 0.0,
            "end_time_instrumental": 200.0,
            "sections": [
                {"label": "intro", "start_beat": 0, "end_beat": 32,
                 "energy": "low",
                 "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"},
                 "transition_in": "fade", "transition_beats": 4},
                {"label": "verse", "start_beat": 32, "end_beat": 128,
                 "energy": "medium",
                 "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "breakdown", "start_beat": 128, "end_beat": 192,
                 "energy": "low",
                 "stem_roles": {"vocals": "background", "drums": "background", "bass": "support", "guitar": "lead", "piano": "background", "other": "texture"},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "drop", "start_beat": 192, "end_beat": 352,
                 "energy": "peak",
                 "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"},
                 "transition_in": "cut", "transition_beats": 0},
                {"label": "outro", "start_beat": 352, "end_beat": 416,
                 "energy": "low",
                 "stem_roles": {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"},
                 "transition_in": "crossfade", "transition_beats": 8},
            ],
            "vocal_type": "sung",
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
            assert len(system_arg) == 1, f"system= should have 1 block, got {len(system_arg)}"
            assert system_arg[0]["type"] == "text"
            assert system_arg[0]["cache_control"] == {"type": "ephemeral"}

            # Verify the plan is an IntentPlan
            assert isinstance(plan, IntentPlan)
            assert len(plan.sections) >= 2
            assert all(isinstance(s, IntentSection) for s in plan.sections)

    def test_cache_stats_logging_no_crash(self):
        """Cache stats logging works even when usage object lacks cache fields."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        mock_tool_use = MagicMock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "test_id"
        mock_tool_use.input = {
            "start_time_vocal": 0.0,
            "end_time_vocal": 200.0,
            "start_time_instrumental": 0.0,
            "end_time_instrumental": 200.0,
            "sections": [
                {"label": "intro", "start_beat": 0, "end_beat": 32,
                 "energy": "low",
                 "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"},
                 "transition_in": "fade", "transition_beats": 4},
                {"label": "verse", "start_beat": 32, "end_beat": 128,
                 "energy": "medium",
                 "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"},
                 "transition_in": "crossfade", "transition_beats": 4},
                {"label": "drop", "start_beat": 128, "end_beat": 352,
                 "energy": "peak",
                 "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"},
                 "transition_in": "cut", "transition_beats": 0},
                {"label": "outro", "start_beat": 352, "end_beat": 416,
                 "energy": "low",
                 "stem_roles": {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"},
                 "transition_in": "crossfade", "transition_beats": 8},
            ],
            "vocal_type": "sung",
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
            assert isinstance(plan, IntentPlan)

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


class TestPromptIntentText:
    """Tests verifying the system prompt uses intent-based role terminology."""

    def test_vocal_instrumental_balance_uses_roles(self):
        """Static block contains Rule 2 about vocal-instrumental balance using role language."""
        block = _build_system_prompt_block()
        static_text = block["text"]
        assert "VOCAL-INSTRUMENTAL BALANCE" in static_text
        assert "FULL BAND" in static_text
        assert '"lead"' in static_text

    def test_role_variation_rule_in_static_block(self):
        """Static block contains Rule 7 about role variation (not gain variation)."""
        block = _build_system_prompt_block()
        static_text = block["text"]
        assert "ROLE VARIATION" in static_text

    def test_few_shot_sections_have_energy_and_roles(self):
        """Few-shot examples use energy + stem_roles (not stem_gains)."""
        messages = _build_few_shot_messages()
        valid_energies = {"low", "medium", "high", "peak"}
        valid_roles = {"lead", "support", "background", "texture", "silent"}
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            assert "stem_roles" in section, (
                                f"Section '{section['label']}' missing stem_roles"
                            )
                            assert "energy" in section, (
                                f"Section '{section['label']}' missing energy"
                            )
                            assert "stem_gains" not in section, (
                                f"Section '{section['label']}' still has stem_gains"
                            )
                            assert section["energy"] in valid_energies, (
                                f"Invalid energy '{section['energy']}' in section '{section['label']}'"
                            )
                            for stem, role in section["stem_roles"].items():
                                assert role in valid_roles, (
                                    f"Invalid role '{role}' for stem '{stem}' "
                                    f"in section '{section['label']}'"
                                )

    def test_few_shot_verse_roles_follow_guidance(self):
        """Verse sections in few-shot examples have vocals as lead and drums/bass as support."""
        messages = _build_few_shot_messages()
        for msg in messages:
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_use":
                        for section in block["input"].get("sections", []):
                            if section["label"] == "verse":
                                roles = section["stem_roles"]
                                if roles["vocals"] != "silent":
                                    assert roles["vocals"] == "lead", (
                                        f"Verse vocals should be 'lead', got '{roles['vocals']}'"
                                    )
                                    assert roles["drums"] in ("lead", "support"), (
                                        f"Verse drums should be 'lead' or 'support', got '{roles['drums']}'"
                                    )


# ---------------------------------------------------------------------------
# Arrangement validation tests
# ---------------------------------------------------------------------------


def _make_intent_plan_with_sections(sections: list[IntentSection]) -> IntentPlan:
    """Create an IntentPlan with the given sections for validation testing."""
    return IntentPlan(
        start_time_vocal=0.0,
        end_time_vocal=200.0,
        start_time_instrumental=0.0,
        end_time_instrumental=200.0,
        sections=sections,
        vocal_type="sung",
        explanation="Test plan.",
        warnings=[],
    )


class TestIntentValidation:
    """Tests for structural validation in _validate_intent_plan()."""

    def test_contiguous_sections_pass(self):
        """Contiguous sections pass validation without warnings."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            IntentSection("verse", 32, 128, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 4),
            IntentSection("drop", 128, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        assert isinstance(result, IntentPlan)
        assert len(result.sections) == 4

    def test_gap_sections_fixed(self):
        """Gaps between sections are fixed by extending the previous section."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            # Gap: 32-40 missing
            IntentSection("verse", 40, 128, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 4),
            IntentSection("drop", 128, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # Intro should have been extended to cover the gap
        assert result.sections[0].end_beat == result.sections[1].start_beat

    def test_overlap_sections_fixed(self):
        """Overlapping sections are fixed by truncating the earlier section."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 40, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            # Overlap: verse starts at 32 but intro ends at 40
            IntentSection("verse", 32, 128, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 4),
            IntentSection("drop", 128, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # Sections should be fixed to be contiguous
        for i in range(1, len(result.sections)):
            assert result.sections[i].start_beat >= result.sections[i - 1].end_beat

    def test_missing_stem_roles_filled(self):
        """Missing stem_roles keys are filled with 'texture' default."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # Section missing 'piano' and 'other' keys
        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background"}, "fade", 4),
            IntentSection("drop", 32, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # All 6 stems should be present in every section
        expected_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
        for section in result.sections:
            assert set(section.stem_roles.keys()) == expected_stems
            # Missing ones should be filled with "texture"
            if section.label == "intro":
                assert section.stem_roles["piano"] == "texture"
                assert section.stem_roles["other"] == "texture"

    def test_invalid_role_corrected(self):
        """Invalid stem role values are corrected to 'texture'."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "INVALID_ROLE", "piano": "background", "other": "texture"}, "fade", 4),
            IntentSection("drop", 32, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # Invalid role should be corrected to "texture"
        assert result.sections[0].stem_roles["guitar"] == "texture"

    def test_single_section_gets_intro_prepended(self):
        """A plan with only 1 section gets an intro prepended to make at least 2."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("drop", 0, 416, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        assert len(result.sections) >= 2
        assert result.sections[0].label == "intro"
        assert result.sections[0].stem_roles["vocals"] == "silent"

    def test_time_range_clamped(self):
        """Time ranges beyond song duration are clamped."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            IntentSection("drop", 32, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)
        plan.end_time_vocal = 999.0  # Way beyond song duration
        plan.end_time_instrumental = 999.0

        result = _validate_intent_plan(plan, meta_a, meta_b)

        assert result.end_time_vocal <= meta_a.duration_seconds
        assert result.end_time_instrumental <= meta_b.duration_seconds

    def test_early_return_filters_sub4beat_sections(self):
        """Issue #12: Early return at min-length filter assigns filtered sections to plan."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        # All sections are < 4 beats — should trigger early return
        sections = [
            IntentSection("intro", 0, 2, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 0),
            IntentSection("verse", 2, 5, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 0),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # After filtering, plan.sections should be empty (all < 4 beats)
        assert result.sections == []
        assert any("no sections >= 4 beats" in w for w in result.warnings)

    def test_early_return_keeps_valid_sections_removes_short(self):
        """Issue #12: When some sections are < 4 beats, they're removed; valid ones remain."""
        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        sections = [
            IntentSection("intro", 0, 2, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 0),
            IntentSection("verse", 2, 128, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 0),
            IntentSection("drop", 128, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 4),
        ]
        plan = _make_intent_plan_with_sections(sections)
        result = _validate_intent_plan(plan, meta_a, meta_b)

        # The 2-beat intro should be removed; 3 valid sections remain
        labels = [s.label for s in result.sections]
        assert "intro" not in labels
        assert len(result.sections) >= 2  # at least verse + drop (outro may be kept or intro prepended)


class TestIntentDurationValidation:
    """Tests for _validate_intent_duration() — Issue #11 off-by-one fix."""

    def test_last_section_clamped_after_filter(self):
        """Issue #11: Last section's end_beat is clamped AFTER filtering, not before."""
        target_bpm = 120.0
        max_duration = TARGET_REMIX_DURATION_SECONDS * 1.5  # 315s
        max_beats = int(max_duration * target_bpm / 60)  # 630

        # Create sections where the last one spans the max_beats boundary
        # Section C starts before max_beats but ends after it
        sections = [
            IntentSection("intro", 0, 100, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            IntentSection("verse", 100, 400, "medium", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 4),
            IntentSection("drop", 400, 600, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 600, 800, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)

        result, ok = _validate_intent_duration(plan, target_bpm)

        assert ok is True
        # The filter should keep sections with start_beat < max_beats (630)
        # intro(0-100), verse(100-400), drop(400-600), outro(600-800) — outro starts at 600 < 630, kept
        # The post-filter last section's end_beat should be clamped to max_beats
        assert result.sections[-1].end_beat == max_beats
        assert all(s.start_beat < max_beats for s in result.sections)

    def test_all_sections_beyond_max_beats_returns_false(self):
        """When all sections start beyond max_beats, returns (plan, False)."""
        target_bpm = 120.0
        max_duration = TARGET_REMIX_DURATION_SECONDS * 1.5
        max_beats = int(max_duration * target_bpm / 60)

        # All sections start beyond max_beats — need total to exceed max_duration
        sections = [
            IntentSection("drop", max_beats + 10, max_beats + 100, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
        ]
        plan = _make_intent_plan_with_sections(sections)

        result, ok = _validate_intent_duration(plan, target_bpm)

        assert ok is False
        assert result.sections == []

    def test_within_duration_unchanged(self):
        """Sections within duration limits pass through unchanged."""
        target_bpm = 120.0
        sections = [
            IntentSection("intro", 0, 32, "low", {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "background", "other": "texture"}, "fade", 4),
            IntentSection("drop", 32, 352, "peak", {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "cut", 0),
            IntentSection("outro", 352, 416, "low", {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "crossfade", 8),
        ]
        plan = _make_intent_plan_with_sections(sections)

        result, ok = _validate_intent_duration(plan, target_bpm)

        assert ok is True
        assert len(result.sections) == 3
        assert result.sections[-1].end_beat == 416


# ---------------------------------------------------------------------------
# LLM max_tokens truncation and retry behaviour tests
# ---------------------------------------------------------------------------


def _valid_plan_input(total_end_beat: int = 416) -> dict:
    """Return a valid tool_use input dict for create_remix_plan."""
    return {
        "start_time_vocal": 0.0,
        "end_time_vocal": 200.0,
        "start_time_instrumental": 0.0,
        "end_time_instrumental": 200.0,
        "sections": [
            {"label": "intro", "start_beat": 0, "end_beat": 32,
             "energy": "low",
             "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support",
                            "guitar": "background", "piano": "background", "other": "texture"},
             "transition_in": "fade", "transition_beats": 4},
            {"label": "verse", "start_beat": 32, "end_beat": 128,
             "energy": "medium",
             "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support",
                            "guitar": "background", "piano": "texture", "other": "texture"},
             "transition_in": "crossfade", "transition_beats": 4},
            {"label": "breakdown", "start_beat": 128, "end_beat": 192,
             "energy": "low",
             "stem_roles": {"vocals": "background", "drums": "background", "bass": "support",
                            "guitar": "lead", "piano": "background", "other": "texture"},
             "transition_in": "crossfade", "transition_beats": 4},
            {"label": "drop", "start_beat": 192, "end_beat": total_end_beat - 64,
             "energy": "peak",
             "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support",
                            "guitar": "support", "piano": "background", "other": "background"},
             "transition_in": "cut", "transition_beats": 0},
            {"label": "outro", "start_beat": total_end_beat - 64, "end_beat": total_end_beat,
             "energy": "low",
             "stem_roles": {"vocals": "silent", "drums": "background", "bass": "background",
                            "guitar": "background", "piano": "background", "other": "texture"},
             "transition_in": "crossfade", "transition_beats": 8},
        ],
        "vocal_type": "sung",
        "explanation": "Test explanation.",
        "warnings": [],
    }


def _mock_response(stop_reason: str, plan_input: dict | None = None,
                   input_tokens: int = 100, output_tokens: int = 50):
    """Build a mock Anthropic response."""
    content = []
    if plan_input is not None:
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "test_id"
        tool_block.input = plan_input
        content.append(tool_block)

    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.usage.cache_read_input_tokens = 0
    resp.usage.cache_creation_input_tokens = 0
    return resp


def _interpret_with_mock(mock_responses):
    """Run interpret_prompt with mocked LLM returning given responses in sequence."""
    meta_a = _make_metadata(bpm=120.0, duration=240.0)
    meta_b = _make_metadata(bpm=118.0, duration=210.0)

    with patch("musicmixer.services.interpreter.settings") as mock_settings, \
         patch("musicmixer.services.interpreter.anthropic") as mock_anthropic:
        mock_settings.stem_backend = "modal"
        mock_settings.anthropic_api_key = "test-key"
        mock_settings.llm_model = "claude-sonnet-4-20250514"
        mock_settings.llm_timeout_seconds = 30
        mock_settings.llm_max_retries = 1

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = mock_responses
        mock_anthropic.Anthropic.return_value = mock_client

        plan = interpret_prompt("test prompt", meta_a, meta_b)
        return plan, mock_client


class TestMaxTokensPartialParse:
    """Tests for partial JSON parse recovery when LLM hits max_tokens."""

    def test_max_tokens_salvages_valid_partial_plan(self):
        """When stop_reason is max_tokens but tool_use has valid plan, salvage it."""
        valid_input = _valid_plan_input(total_end_beat=416)
        response = _mock_response("max_tokens", plan_input=valid_input, output_tokens=1536)

        plan, _ = _interpret_with_mock([response])

        assert isinstance(plan, IntentPlan)
        assert len(plan.sections) >= 2
        assert any("truncated" in w.lower() for w in plan.warnings)

    def test_max_tokens_unparseable_falls_back(self):
        """When stop_reason is max_tokens and tool_use input is garbage, fall back."""
        garbage_input = {"sections": "not a list", "explanation": "x"}
        response = _mock_response("max_tokens", plan_input=garbage_input, output_tokens=1536)

        plan, _ = _interpret_with_mock([response])

        # Should get a fallback RemixPlan, not an IntentPlan
        assert isinstance(plan, RemixPlan)
        assert plan.used_fallback is True

    def test_max_tokens_no_tool_block_falls_back(self):
        """When stop_reason is max_tokens with no tool_use block, fall back."""
        response = _mock_response("max_tokens", plan_input=None, output_tokens=1536)

        plan, _ = _interpret_with_mock([response])

        assert isinstance(plan, RemixPlan)
        assert plan.used_fallback is True


class TestRetryMessageReplacement:
    """Tests for retry message replacement (not accumulation) on _DurationTooShortError."""

    def test_retry_does_not_accumulate_messages(self):
        """After _DurationTooShortError, messages list length should be unchanged."""
        # First response: too-short plan (only 100 beats ~ 50s at 120 BPM, well under 147s min)
        short_input = _valid_plan_input(total_end_beat=100)
        short_response = _mock_response("tool_use", plan_input=short_input)

        # Second response: valid-length plan
        good_input = _valid_plan_input(total_end_beat=416)
        good_response = _mock_response("tool_use", plan_input=good_input)

        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        with patch("musicmixer.services.interpreter.settings") as mock_settings, \
             patch("musicmixer.services.interpreter.anthropic") as mock_anthropic:
            mock_settings.stem_backend = "modal"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_timeout_seconds = 30
            mock_settings.llm_max_retries = 1

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [short_response, good_response]
            mock_anthropic.Anthropic.return_value = mock_client

            plan = interpret_prompt("test prompt", meta_a, meta_b)

            # Both calls should have the same number of messages
            first_call_msgs = mock_client.messages.create.call_args_list[0].kwargs["messages"]
            second_call_msgs = mock_client.messages.create.call_args_list[1].kwargs["messages"]
            assert len(first_call_msgs) == len(second_call_msgs), (
                f"Message count changed: {len(first_call_msgs)} -> {len(second_call_msgs)}. "
                "Retry should replace, not accumulate messages."
            )

    def test_retry_prepends_guidance_to_user_message(self):
        """Retry should prepend rejection guidance to the user message content."""
        short_input = _valid_plan_input(total_end_beat=100)
        short_response = _mock_response("tool_use", plan_input=short_input)

        good_input = _valid_plan_input(total_end_beat=416)
        good_response = _mock_response("tool_use", plan_input=good_input)

        meta_a = _make_metadata(bpm=120.0, duration=240.0)
        meta_b = _make_metadata(bpm=118.0, duration=210.0)

        with patch("musicmixer.services.interpreter.settings") as mock_settings, \
             patch("musicmixer.services.interpreter.anthropic") as mock_anthropic:
            mock_settings.stem_backend = "modal"
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.llm_model = "claude-sonnet-4-20250514"
            mock_settings.llm_timeout_seconds = 30
            mock_settings.llm_max_retries = 1

            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [short_response, good_response]
            mock_anthropic.Anthropic.return_value = mock_client

            interpret_prompt("test prompt", meta_a, meta_b)

            # Second call's last user message should contain retry guidance
            second_call_msgs = mock_client.messages.create.call_args_list[1].kwargs["messages"]
            last_user_msg = second_call_msgs[-1]
            assert "IMPORTANT" in last_user_msg["content"]
            assert "REJECTED" in last_user_msg["content"]
            # Original prompt content should still be present
            assert "test prompt" in last_user_msg["content"]


class TestLLMMaxTokensConstant:
    """Tests for the LLM_MAX_TOKENS constant."""

    def test_llm_max_tokens_is_3072(self):
        """LLM_MAX_TOKENS should be 3072."""
        assert LLM_MAX_TOKENS == 3072

    def test_interpret_prompt_uses_llm_max_tokens(self):
        """interpret_prompt passes LLM_MAX_TOKENS to the API call."""
        valid_input = _valid_plan_input(total_end_beat=416)
        response = _mock_response("tool_use", plan_input=valid_input)

        _, mock_client = _interpret_with_mock([response])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == LLM_MAX_TOKENS


# ---------------------------------------------------------------------------
# PulseMap prompt integration tests
# ---------------------------------------------------------------------------


def _make_metadata_with_pulsemap(
    bpm: float = 120.0,
    duration: float = 240.0,
    chord_progression: ChordProgression | None = None,
    polyphony_info: PolyphonyInfo | None = None,
    drum_pattern: DrumPattern | None = None,
    word_alignment: WordAlignment | None = None,
    stem_analysis: StemAnalysis | None = None,
) -> AudioMetadata:
    """Create AudioMetadata with optional PulseMap fields for testing."""
    beat_interval_frames = int(60 / bpm * 44100 / 512)
    num_beats = int(bpm * duration / 60)
    beat_frames = np.arange(0, num_beats) * beat_interval_frames
    total_beats = round(bpm * duration / 60 / 4) * 4
    return AudioMetadata(
        bpm=bpm,
        bpm_confidence=0.9,
        beat_frames=beat_frames,
        duration_seconds=duration,
        total_beats=total_beats,
        chord_progression=chord_progression,
        polyphony_info=polyphony_info,
        drum_pattern=drum_pattern,
        word_alignment=word_alignment,
        stem_analysis=stem_analysis,
    )


def _sample_chord_progression() -> ChordProgression:
    return ChordProgression(
        chords=[
            ChordEvent(0, 2000, "Cmaj7"),
            ChordEvent(2000, 4000, "Am"),
            ChordEvent(4000, 6000, "F"),
            ChordEvent(6000, 8000, "G"),
        ],
        unique_chords=["Cmaj7", "Am", "F", "G"],
        most_common_chord="Cmaj7",
        progression_summary="I-vi-IV-V in C major",
    )


def _sample_polyphony_solo() -> PolyphonyInfo:
    return PolyphonyInfo(polyphonic=False, method="mid_side", gate1_ratio=0.02, gate2_ratio=None)


def _sample_polyphony_duet() -> PolyphonyInfo:
    return PolyphonyInfo(polyphonic=True, method="mid_side", gate1_ratio=0.25, gate2_ratio=45.0)


def _sample_drum_pattern() -> DrumPattern:
    return DrumPattern(
        kick_count=48, snare_count=24, hihat_count=96,
        total_hits=168, duration_ms=180000,
        style_hint="four_on_floor",
    )


def _sample_word_alignment() -> WordAlignment:
    return WordAlignment(
        words=[
            WordEvent(start_ms=12340, text="Never", end=12560),
            WordEvent(start_ms=12560, text="gonna", end=12780),
            WordEvent(start_ms=12780, text="give", end=12950),
            WordEvent(start_ms=12950, text="you", end=13100),
            WordEvent(start_ms=13100, text="up", end=13300),
        ],
        source="whisperx",
        lrclib_validated=True,
        lrclib_offset_ms=50,
    )


def _sample_stem_analysis() -> StemAnalysis:
    """Minimal stem analysis for testing Layer 3."""
    n_bars = 60
    return StemAnalysis(
        bar_rms={
            "vocals": np.full(n_bars, 0.1, dtype=np.float32),
            "drums": np.full(n_bars, 0.15, dtype=np.float32),
            "bass": np.full(n_bars, 0.12, dtype=np.float32),
            "guitar": np.full(n_bars, 0.08, dtype=np.float32),
            "piano": np.full(n_bars, 0.05, dtype=np.float32),
            "other": np.full(n_bars, 0.03, dtype=np.float32),
        },
        combined_energy=np.full(n_bars, 0.5, dtype=np.float32),
        vocal_active=np.ones(n_bars, dtype=bool),
        vocal_gaps=[],
        bucket_thresholds=EnergyBuckets(
            noise_floor=0.02, p10=0.04, p50=0.08, p85=0.14,
        ),
    )


class TestMixingRulesInPrompt:
    """Tests that mixing rules (polyphony, chords, drums, word timing) are in the static system prompt."""

    def test_no_pulsemap_section_header(self):
        """PulseMap analysis rules section has been dissolved into existing sections."""
        block = _build_system_prompt_block()
        text = block["text"]
        assert "PULSEMAP ANALYSIS RULES" not in text

    def test_polyphony_rules_present(self):
        """Polyphony rules appear in Stem Role Guidelines."""
        block = _build_system_prompt_block()
        text = block["text"]
        assert "polyphonic vocals" in text.lower()
        assert "Solo vocals" in text

    def test_chord_rules_present(self):
        """Chord progression rules appear in Arrangement/Transitions sections."""
        block = _build_system_prompt_block()
        text = block["text"]
        assert "chord changes" in text.lower()
        assert "Shared chords" in text

    def test_drum_rules_present(self):
        """Drum pattern rules appear in Genre Guidance."""
        block = _build_system_prompt_block()
        text = block["text"]
        assert "groove compatibility" in text.lower()

    def test_word_timing_rules_present(self):
        """Word-level timing rules appear in Transitions and Stem Role Guidelines."""
        block = _build_system_prompt_block()
        text = block["text"]
        assert "Vocal gaps >500ms" in text
        assert "breathe WITH the vocal" in text


class TestPulseMapLayer1:
    """Tests for PulseMap data in Layer 1 (Song Overview)."""

    def test_chord_progression_in_song_info(self):
        """Chord progression summary appears in Layer 1 when present."""
        meta = _make_metadata_with_pulsemap(chord_progression=_sample_chord_progression())
        info = _build_song_info("Song A", meta, meta.total_beats)
        assert "Chords: I-vi-IV-V in C major" in info

    def test_no_chords_no_chord_line(self):
        """No chord line when chord_progression is None."""
        meta = _make_metadata_with_pulsemap()
        info = _build_song_info("Song A", meta, meta.total_beats)
        assert "Chords:" not in info

    def test_polyphony_solo_in_song_info(self):
        """Solo voice label appears when polyphony is solo."""
        meta = _make_metadata_with_pulsemap(polyphony_info=_sample_polyphony_solo())
        info = _build_song_info("Song A", meta, meta.total_beats)
        assert "solo voice" in info

    def test_polyphony_duet_in_song_info(self):
        """Harmony/duet label appears when polyphony is polyphonic."""
        meta = _make_metadata_with_pulsemap(polyphony_info=_sample_polyphony_duet())
        info = _build_song_info("Song A", meta, meta.total_beats)
        assert "harmony/duet detected" in info

    def test_no_polyphony_no_line(self):
        """No polyphony line when polyphony_info is None."""
        meta = _make_metadata_with_pulsemap()
        info = _build_song_info("Song A", meta, meta.total_beats)
        assert "polyphony" not in info.lower()


class TestPulseMapLayer3:
    """Tests for PulseMap data in Layer 3 (Stem Character)."""

    def test_drum_pattern_in_stem_character(self):
        """Drum pattern info appears in Layer 3 when present."""
        meta = _make_metadata_with_pulsemap(
            drum_pattern=_sample_drum_pattern(),
            stem_analysis=_sample_stem_analysis(),
        )
        char = _build_stem_character("Song B", meta)
        assert "Drum pattern: four-on-floor" in char
        assert "kick=48" in char
        assert "snare=24" in char
        assert "hihat=96" in char

    def test_no_drum_pattern_no_line(self):
        """No drum pattern line when drum_pattern is None."""
        meta = _make_metadata_with_pulsemap(stem_analysis=_sample_stem_analysis())
        char = _build_stem_character("Song B", meta)
        assert "Drum pattern:" not in char

    def test_silent_drum_pattern_no_line(self):
        """No drum pattern line when total_hits is 0."""
        silent_drums = DrumPattern(
            kick_count=0, snare_count=0, hihat_count=0,
            total_hits=0, duration_ms=180000,
            style_hint="silent",
        )
        meta = _make_metadata_with_pulsemap(
            drum_pattern=silent_drums,
            stem_analysis=_sample_stem_analysis(),
        )
        char = _build_stem_character("Song B", meta)
        assert "Drum pattern:" not in char


class TestPulseMapLayer4:
    """Tests for PulseMap data in Layer 4 (Cross-Song)."""

    def test_shared_chords_compatibility(self):
        """Chord compatibility line shows shared chords when both songs have them."""
        chords_a = ChordProgression(
            chords=[], unique_chords=["Am", "F", "C", "G"],
            most_common_chord="Am", progression_summary="test",
        )
        chords_b = ChordProgression(
            chords=[], unique_chords=["Am", "F", "Dm", "E"],
            most_common_chord="Am", progression_summary="test",
        )
        meta_a = _make_metadata_with_pulsemap(chord_progression=chords_a)
        meta_b = _make_metadata_with_pulsemap(chord_progression=chords_b)
        cross = _build_cross_song_layer(meta_a, meta_b)
        assert "Chord compatibility:" in cross
        assert "Am" in cross
        assert "F" in cross
        assert "good harmonic overlap" in cross

    def test_no_shared_chords(self):
        """Warns about harmonic clashes when no chords are shared."""
        chords_a = ChordProgression(
            chords=[], unique_chords=["C", "G"],
            most_common_chord="C", progression_summary="test",
        )
        chords_b = ChordProgression(
            chords=[], unique_chords=["Dm", "E"],
            most_common_chord="Dm", progression_summary="test",
        )
        meta_a = _make_metadata_with_pulsemap(chord_progression=chords_a)
        meta_b = _make_metadata_with_pulsemap(chord_progression=chords_b)
        cross = _build_cross_song_layer(meta_a, meta_b)
        assert "no shared chords" in cross

    def test_no_chords_no_compatibility_line(self):
        """No chord compatibility line when chord data is absent."""
        meta_a = _make_metadata_with_pulsemap()
        meta_b = _make_metadata_with_pulsemap()
        cross = _build_cross_song_layer(meta_a, meta_b)
        assert "Chord compatibility:" not in cross


class TestPulseMapLayer5:
    """Tests for PulseMap word alignment in Layer 5 (Lyrics)."""

    def test_word_timing_appended_to_lyrics(self):
        """Word timing sample appears after bar-level lyrics when word_alignment is set."""
        lyrics = _make_lyrics()
        wa = _sample_word_alignment()
        result = _build_lyrics_layer(lyrics, None, word_alignment_a=wa)
        # Bar-level lyrics must still be present
        assert "bar" in result
        assert "Hello world" in result
        # Word timing should also appear
        assert "word timing (sample)" in result
        assert "[12340ms] Never" in result
        assert "[13100ms] up" in result

    def test_bar_level_preserved_without_word_alignment(self):
        """Bar-level lyrics are preserved when word_alignment is None."""
        lyrics = _make_lyrics()
        result = _build_lyrics_layer(lyrics, None)
        assert "bar" in result
        assert "Hello world" in result
        assert "word timing" not in result

    def test_no_lyrics_no_word_timing(self):
        """No word timing shown when there are no lyrics (even if alignment exists)."""
        wa = _sample_word_alignment()
        result = _build_lyrics_layer(None, None, word_alignment_a=wa)
        assert result == ""

    def test_format_word_timing_sample_basic(self):
        """_format_word_timing_sample produces correct compact format."""
        words = [
            WordEvent(start_ms=1000, text="hello", end=1200),
            WordEvent(start_ms=2000, text="world", end=2200),
        ]
        result = _format_word_timing_sample(words)
        assert "[1000ms] hello" in result
        assert "[2000ms] world" in result

    def test_format_word_timing_sample_truncation(self):
        """Long word lists are sampled down to max_words."""
        words = [WordEvent(start_ms=i * 100, text=f"w{i}", end=i * 100 + 50) for i in range(100)]
        result = _format_word_timing_sample(words, max_words=10)
        # Should contain at most 10 word entries
        assert result.count("ms]") == 10

    def test_format_word_timing_sample_empty(self):
        """Empty word list returns placeholder."""
        result = _format_word_timing_sample([])
        assert "no words" in result


class TestPulseMapDynamicContext:
    """Tests for PulseMap data flowing through _build_dynamic_context end-to-end."""

    def test_all_pulsemap_fields_in_dynamic_context(self):
        """When all PulseMap fields are populated, they appear in dynamic context."""
        meta_a = _make_metadata_with_pulsemap(
            chord_progression=_sample_chord_progression(),
            polyphony_info=_sample_polyphony_duet(),
            word_alignment=_sample_word_alignment(),
            stem_analysis=_sample_stem_analysis(),
        )
        meta_b = _make_metadata_with_pulsemap(
            chord_progression=ChordProgression(
                chords=[], unique_chords=["Am", "Dm"],
                most_common_chord="Am", progression_summary="test",
            ),
            drum_pattern=_sample_drum_pattern(),
            stem_analysis=_sample_stem_analysis(),
        )
        lyrics_a = _make_lyrics()
        args = dict(
            song_a_meta=meta_a,
            song_b_meta=meta_b,
            total_available_beats=400,
            lyrics_a=lyrics_a,
        )
        ctx = _build_dynamic_context(**args)

        # Layer 1: chords and polyphony
        assert "I-vi-IV-V in C major" in ctx
        assert "harmony/duet detected" in ctx
        # Layer 3: drum pattern
        assert "four-on-floor" in ctx
        # Layer 4: chord compatibility
        assert "Chord compatibility:" in ctx
        assert "Am" in ctx
        # Layer 5: word timing
        assert "word timing (sample)" in ctx
        assert "[12340ms] Never" in ctx

    def test_no_pulsemap_fields_graceful_degradation(self):
        """When all PulseMap fields are None, context is still valid without them."""
        meta_a = _make_metadata_with_pulsemap()
        meta_b = _make_metadata_with_pulsemap()
        args = dict(
            song_a_meta=meta_a,
            song_b_meta=meta_b,
            total_available_beats=400,
        )
        ctx = _build_dynamic_context(**args)
        assert "SONG DATA:" in ctx
        assert "LAYER 1: SONG OVERVIEW" in ctx
        # PulseMap-specific content should be absent
        assert "Chords:" not in ctx
        assert "polyphony" not in ctx.lower()
        assert "Drum pattern:" not in ctx
        assert "Chord compatibility:" not in ctx
        assert "word timing" not in ctx
