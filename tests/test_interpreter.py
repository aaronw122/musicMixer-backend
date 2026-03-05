"""Tests for musicmixer.services.interpreter -- deterministic fallback plan + prompt caching."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from musicmixer.models import AudioMetadata, EnergyBuckets, IntentPlan, IntentSection, LyricLine, LyricsData, RemixPlan, Section, StemAnalysis
from musicmixer.services.interpreter import (
    _build_dynamic_context,
    _build_few_shot_messages,
    _build_system_prompt_block,
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
    from musicmixer.services.interpreter import _compute_key_guidance, estimate_target_bpm, TARGET_REMIX_DURATION_SECONDS
    _key_available, key_detail = _compute_key_guidance(meta_a, meta_b)
    target_bpm = estimate_target_bpm(vocal_bpm=meta_a.bpm, instrumental_bpm=meta_b.bpm)
    total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    return dict(
        song_a_meta=meta_a,
        song_b_meta=meta_b,
        key_matching_detail=key_detail,
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
            "You are an expert music mashup artist",           # Section 1
            "CRITICAL MIXING RULES",                   # Section 2
            "STEM ROLE GUIDELINES",                    # Section 3
            "ENERGY LEVELS AND ARC",                   # Section 3
            "MIXING ADVISORY",                         # Section 3
            "TRANSITIONS:",                            # Section 4
            "Template A (Standard Mashup)",            # Section 5
            "ARRANGEMENT RULES",                       # Section 6
            "GENRE GUIDANCE",                          # Section 7
            "TEMPO MATCHING:",                         # Section 8
            "STEM SEPARATION ARTIFACTS",               # Section 10
            "EXPLANATION: Write 2-3",                  # Section 11
            "SONG DATA:",                              # Section 12
            "1 bar = 4 beats",                         # Duration in Section 9
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
        assert "STRETCH WARNING (18.5%)" in ctx
        assert "STRETCH WARNING" not in block["text"]

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
        key_source="none",
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
