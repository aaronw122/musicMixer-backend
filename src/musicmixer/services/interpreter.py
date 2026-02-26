"""Remix plan interpreter: LLM-powered prompt interpretation with deterministic fallback.

Day 3: Converts user prompts + song metadata into structured RemixPlan objects
using Anthropic's tool_use API. Falls back to deterministic arrangement on any
LLM failure.

Day 2 fallback functions (generate_fallback_plan, default_arrangement) are
preserved at the bottom of this file.
"""

from __future__ import annotations

import copy
import json
import logging
import time

import anthropic

from musicmixer.config import settings
from musicmixer.models import AudioMetadata, RemixPlan, Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]
FOUR_STEMS = ["vocals", "drums", "bass", "other"]

# Stems that only exist in 6-stem (Modal) mode
EXTRA_STEMS = {"guitar", "piano"}


# ---------------------------------------------------------------------------
# tool_use schema
# ---------------------------------------------------------------------------

REMIX_PLAN_TOOL: dict = {
    "name": "create_remix_plan",
    "description": "Create a structured remix plan based on the user's prompt and song analysis data.",
    "input_schema": {
        "type": "object",
        "required": [
            "vocal_source", "start_time_vocal", "end_time_vocal",
            "start_time_instrumental", "end_time_instrumental",
            "sections", "tempo_source", "key_source",
            "explanation", "warnings",
        ],
        "properties": {
            "vocal_source": {
                "type": "string",
                "enum": ["song_a", "song_b"],
                "description": "Which song provides the vocals. The other song provides ALL instrumentals.",
            },
            "start_time_vocal": {
                "type": "number",
                "minimum": 0,
                "description": "Start time (seconds) in the vocal source song. Choose the most interesting/energetic region.",
            },
            "end_time_vocal": {
                "type": "number",
                "minimum": 5,
                "description": "End time (seconds) in the vocal source song. Must be > start_time_vocal.",
            },
            "start_time_instrumental": {
                "type": "number",
                "minimum": 0,
                "description": "Start time (seconds) in the instrumental source song.",
            },
            "end_time_instrumental": {
                "type": "number",
                "minimum": 5,
                "description": "End time (seconds) in the instrumental source song. Must be > start_time_instrumental.",
            },
            "sections": {
                "type": "array",
                "minItems": 2,
                "maxItems": 12,
                "description": "Ordered list of remix sections. Must be contiguous (end_beat of one = start_beat of next). First section starts at beat 0.",
                "items": {
                    "type": "object",
                    "required": ["label", "start_beat", "end_beat", "stem_gains", "transition_in", "transition_beats"],
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["intro", "verse", "breakdown", "drop", "outro"],
                            "description": "Section type. Determines the energy character.",
                        },
                        "start_beat": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Starting beat (inclusive). Must be a multiple of 4 for bar alignment.",
                        },
                        "end_beat": {
                            "type": "integer",
                            "minimum": 4,
                            "description": "Ending beat (exclusive). Section length = end_beat - start_beat. Should be 4, 8, or 16.",
                        },
                        "stem_gains": {
                            "type": "object",
                            "required": ["vocals", "drums", "bass", "guitar", "piano", "other"],
                            "description": "Volume level for each stem in this section. 0.0 = silent, 1.0 = full volume.",
                            "properties": {
                                "vocals": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "drums": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "bass": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "guitar": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "piano": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "other": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "additionalProperties": False,
                        },
                        "transition_in": {
                            "type": "string",
                            "enum": ["fade", "crossfade", "cut"],
                            "description": "How this section transitions from the previous one.",
                        },
                        "transition_beats": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 8,
                            "description": "Length of transition in beats. Must be < half the section length. Use 0 for 'cut'.",
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "tempo_source": {
                "type": "string",
                "enum": ["song_a", "song_b", "average"],
                "description": "Which song's tempo to use as target. 'average' only when BPMs differ by <15%.",
            },
            "key_source": {
                "type": "string",
                "enum": ["song_a", "song_b", "none"],
                "description": "Which song's key to match to. 'none' to skip key matching.",
            },
            "explanation": {
                "type": "string",
                "maxLength": 500,
                "description": "2-3 non-technical sentences explaining what you did and why. Shown directly to the user.",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Caveats: what you couldn't fulfill, prompt ambiguities, quality concerns. Empty array if none.",
            },
        },
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# 4-stem adaptation
# ---------------------------------------------------------------------------

def _adapt_schema_for_4stem(tool_schema: dict) -> dict:
    """Remove guitar and piano from the tool schema for 4-stem (local) mode."""
    schema = copy.deepcopy(tool_schema)
    stem_gains = schema["input_schema"]["properties"]["sections"]["items"]["properties"]["stem_gains"]

    # Remove guitar and piano from required and properties
    stem_gains["required"] = [s for s in stem_gains["required"] if s not in EXTRA_STEMS]
    for stem in EXTRA_STEMS:
        stem_gains["properties"].pop(stem, None)

    return schema


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt(
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
    key_matching_available: bool,
    key_matching_detail: str,
    total_available_beats: int,
) -> str:
    """Construct the full system prompt with song metadata injected."""

    # Determine which stems are available
    is_4stem = settings.stem_backend == "local"
    stem_list = "vocals, drums, bass, other" if is_4stem else "vocals, drums, bass, guitar, piano, other"
    stem_count = 4 if is_4stem else 6

    # Compute per-song beat counts (approximate)
    total_beats_a = song_a_meta.total_beats
    total_beats_b = song_b_meta.total_beats

    # Approximate target BPM for beat-to-seconds conversion
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)

    # Build song metadata lines (degrade gracefully if fields don't exist)
    song_a_info = _build_song_info("Song A", song_a_meta, total_beats_a)
    song_b_info = _build_song_info("Song B", song_b_meta, total_beats_b)

    # Stem gains note for 4-stem mode
    stem_gains_note = ""
    if is_4stem:
        stem_gains_note = (
            "\n- Only 4 stems are available: vocals, drums, bass, other. "
            "Guitar and piano are not separated in this mode."
        )

    sections = []

    # Section 1: Role and MVP Constraints
    sections.append(f"""You are a music remix planner. You decide how to combine two songs into a mashup remix.

CONSTRAINTS:
- Vocals ALWAYS come from one song. All instrumentals ({stem_list} minus vocals) ALWAYS come from the other song.
- You CANNOT mix stems across songs (e.g., no "drums from Song A with bass from Song B").
- You CANNOT add effects, generate new sounds, or use vocals from both songs.
- "other" contains synths, strings, wind instruments, and anything not captured by the {stem_count - 1} named stems.
- If the user asks for something impossible, acknowledge it in the `warnings` field and produce the best plan within these limits.{stem_gains_note}

CAPABILITIES:
- Choose which song provides vocals (vocal_source)
- Select source regions from each song (start/end times in seconds)
- Design a section-based arrangement with per-stem volume control (0.0-1.0 for each of: {stem_list})
- Choose transitions between sections (fade, crossfade, cut)
- Set tempo and key matching strategy""")

    # Section 2: Mixing Philosophy
    sections.append("""MIXING PRINCIPLES:
- Contrast creates energy: if a section has drums at 0.0, the next section's drums at 1.0 will feel powerful
- When vocals are active, reduce competing stems (guitar, piano, other) to 0.3-0.5 unless the user asks for a "full" sound
- Every remix should have an energy arc: intro/low -> verse-mid -> breakdown dip -> drop/peak -> outro resolve
- If a prompt implies pumping/ducking, treat non-4/4 or syncopated grooves conservatively (sub-bass-only, gentle depth) and avoid obvious whole-mix pumping
- Muted stems (0.0) are a tool, not a failure -- silence in the right place is more powerful than sound
- Use the full 0.0-1.0 range. Avoid keeping all stems at 0.5-0.8 throughout -- that produces a flat, unengaging mix""")

    # Section 3: Transition Guidance
    sections.append("""TRANSITIONS:
- "cut": Use between sections at similar energy levels for a punchy feel. Good for drop-to-verse or chorus-to-chorus.
- "crossfade": Use when energy changes significantly between sections. Default choice for most transitions.
- "fade": Use for the first section (intro) and last section (outro). Also good for bringing vocals in from silence.""")

    # Section 4: Arrangement Templates
    sections.append(f"""Your sections must sum to approximately {total_available_beats} beats.

Template A (Standard Mashup): intro(~15%) -> verse(~30%) -> breakdown(~15%) -> drop(~30%) -> outro(~10%)
Template B (DJ Set): build(~25%) -> vocals in(~25%) -> peak(~25%) -> vocals out(~12%) -> outro(~13%)
Template C (Quick Hit): intro(~15%) -> vocal drop(~70%) -> outro(~15%)
Template D (Chill): intro(~25%) -> vocals(~50%) -> outro(~25%)

If total beats < 48, use Template C (Quick Hit). If 48-96, use Standard Mashup. If > 96, you may use DJ Set or add a second verse.""")

    # Section 5: Section Rules
    stem_gains_required = stem_list
    sections.append(f"""SECTION RULES:
- Sections should be 4, 8, or 16 beats long (max 16 beats per section)
- Default: start with instrumental only (establishes the beat before vocals enter), unless the prompt suggests otherwise
- Always end with instrumental only or a fade
- section labels: "intro", "verse", "breakdown", "drop", "outro"
- stem_gains values must be between 0.0 and 1.0 (never exceed 1.0 -- it causes distortion)
- stem_gains must include all stems: {stem_gains_required}
- transition_in: "fade", "crossfade", or "cut"
- transition_beats: how many beats the transition lasts (0-8, must be less than half the section length)""")

    # Section 6: Genre-Aware Arrangement
    sections.append("""GENRE GUIDANCE (infer from BPM + energy profile):
- Hip-hop/rap (80-100 BPM): Keep drums consistent throughout. Build energy through vocal intensity and layering, not drum drops.
- EDM/dance (120-130 BPM): Use breakdown -> build -> drop patterns.
- Pop/rock (100-130 BPM): Use verse-chorus dynamics -- stripped for verses, full for choruses.
- R&B/soul (60-90 BPM): Smooth transitions, no abrupt changes. Layer elements gradually.""")

    # Section 7: Tempo and Key Guidance
    sections.append(f"""TEMPO MATCHING:
- tempo_source "average" only when BPMs differ by <15%.
- 15-30% gap: prefer vocal source tempo (the song providing vocals gets stretched less).
- >30% gap: system will stretch vocals only. Note this in your explanation.

KEY MATCHING:
{key_matching_detail}

PITCH LIMIT:
- Do not plan shifts above +/-4 semitones. If compatibility would require more, keep original key and add a warning.""")

    # Section 8: Ambiguity Handling
    sections.append("""HANDLING AMBIGUOUS PROMPTS:
- Vague ("make it cool"): Use energy profiles. Pick vocals from the song with more prominent vocals. Use Standard Mashup template.
- Contradictory ("vocals from both"): Acknowledge in warnings. Pick the better vocal source and explain why.
- Genre jargon ("trap", "lo-fi"): Translate to volume/structure decisions. "Trap" = heavy bass, sparse hi-hats. "Lo-fi" = reduce other, gentle, Template D.
- Time references ("guitar solo at 2:30"): Use the time range in source region selection. Add a warning that you can't verify what's there.

DURATION: Target remix duration 60-120 seconds. Minimum 30s, maximum 180s.""")

    # Section 9: Stem Artifact Awareness
    sections.append("""STEM SEPARATION ARTIFACTS:
Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals.
Bleed is less noticeable during high-energy sections. When the instrumental source song has prominent vocals, avoid purely-instrumental sections longer than 8 beats -- ghost vocals bleed through.""")

    # Section 10: Explanation and Warnings
    sections.append("""EXPLANATION: Write 2-3 non-technical sentences explaining what you did and why. No internal jargon. This is shown directly to the user.

WARNINGS: Populate this array when:
- The prompt is vague and you had to make assumptions
- The prompt asks for something impossible (cross-song stem mixing, effects)
- You're uncertain about a time reference or genre interpretation
- Tempo/key gap is large and the remix may sound noticeably different from the originals""")

    # Section 11: Song Metadata
    sections.append(f"""SONG DATA:

{song_a_info}

{song_b_info}

1 beat = {60 / target_bpm:.2f}s at {target_bpm:.0f} BPM.""")

    return "\n\n".join(sections)


def _build_song_info(label: str, meta: AudioMetadata, total_beats: int) -> str:
    """Build a metadata summary line for a single song, degrading gracefully."""
    parts = [f"{meta.bpm:.1f} BPM"]

    # Key/scale (Day 3+ fields -- may not exist)
    key = getattr(meta, "key", None)
    scale = getattr(meta, "scale", None)
    if key and scale:
        parts.append(f"{key} {scale}")

    parts.append(f"{meta.duration_seconds:.0f}s")
    parts.append(f"{total_beats} beats")

    # Vocal prominence (Day 3+ field)
    vocal_prom = getattr(meta, "vocal_prominence_db", None)
    if vocal_prom is not None:
        parts.append(f"vocal_prominence: {vocal_prom:.0f} dB")

    header = f"{label} ({', '.join(parts)}):"

    # Energy profile (Day 3+ field)
    energy_text = _condense_energy_profile(meta)
    if energy_text:
        return f"{header}\n{energy_text}"

    return header


def _condense_energy_profile(meta: AudioMetadata) -> str:
    """Convert energy_regions into compact text for LLM context.

    Degrades gracefully if energy_regions is not available (returns empty string).
    """
    energy_regions = getattr(meta, "energy_regions", None)
    if not energy_regions:
        return ""

    # Group regions by character label
    groups: dict[str, list] = {}
    for r in energy_regions:
        groups.setdefault(r.character, []).append(r)

    label_map = {
        "rhythmic": "Rhythmic (chorus/drop)",
        "sustained": "Sustained (breakdown)",
        "sparse": "Sparse (verse)",
        "moderate": "Moderate (intro/outro)",
    }

    lines = []
    for character in ["rhythmic", "sustained", "sparse", "moderate"]:
        regions = groups.get(character, [])
        if not regions:
            continue
        entries = ", ".join(
            f"{r.start_sec:.0f}s-{r.end_sec:.0f}s [{r.relative_energy:.2f}]"
            for r in regions
        )
        lines.append(f"  {label_map[character]}: {entries}")

    # Add temporal structure summary
    structure = " -> ".join(
        f"{r.character}({r.start_sec:.0f}-{r.end_sec:.0f}s)"
        for r in sorted(energy_regions, key=lambda r: r.start_sec)
    )
    lines.append(f"  Structure: {structure}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key matching guidance
# ---------------------------------------------------------------------------

def _compute_key_guidance(
    meta_a: AudioMetadata, meta_b: AudioMetadata,
) -> tuple[bool, str]:
    """Pre-compute key matching decision for the LLM context.

    Returns (available, detail_string). Degrades gracefully if key detection
    fields are not present on AudioMetadata.
    """
    key_confidence_a = getattr(meta_a, "key_confidence", None)
    key_confidence_b = getattr(meta_b, "key_confidence", None)

    # If key detection is not implemented yet, bail out
    if key_confidence_a is None or key_confidence_b is None:
        return False, "Key matching: unavailable (key detection not yet implemented)"

    has_mod_a = getattr(meta_a, "has_modulation", False)
    has_mod_b = getattr(meta_b, "has_modulation", False)

    if has_mod_a or has_mod_b:
        return False, "Key matching: unavailable (one or both songs modulate key mid-song)"

    # key_confidence here is from BPM confidence; for key we use a separate
    # attribute if/when it exists. For now degrade if not available.
    key_conf_a = getattr(meta_a, "key_confidence", 0.0)
    key_conf_b = getattr(meta_b, "key_confidence", 0.0)
    min_confidence = min(key_conf_a, key_conf_b)

    if min_confidence < 0.40:
        return False, f"Key matching: unavailable (low confidence: {min_confidence:.2f})"
    elif min_confidence < 0.55:
        return True, f"Key matching: available with half shift (moderate confidence: {min_confidence:.2f})"
    else:
        return True, f"Key matching: available (confidence: {min_confidence:.2f})"


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

def _build_few_shot_messages() -> list[dict]:
    """Build 3 few-shot examples as user/assistant message pairs."""
    is_4stem = settings.stem_backend == "local"

    if is_4stem:
        # Adapt examples for 4-stem mode
        return _build_few_shot_messages_4stem()

    return [
        # Example 1: Clear directive, matched tempos
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "Put Song A\'s vocals over Song B\'s beat, boost the bass"\n\n'
                "Song A: 120 BPM, C major, 240s, 480 beats\n"
                "Song B: 118 BPM, C major, 210s, 413 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_1",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_a",
                        "start_time_vocal": 16.0,
                        "end_time_vocal": 136.0,
                        "start_time_instrumental": 8.0,
                        "end_time_instrumental": 128.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.8, "bass": 1.0, "guitar": 0.6, "piano": 0.5, "other": 0.8}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 1.0, "guitar": 0.4, "piano": 0.3, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.8, "drums": 0.0, "bass": 0.6, "guitar": 0.7, "piano": 0.8, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 1.0, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.6, "bass": 0.8, "guitar": 0.5, "piano": 0.5, "other": 0.7}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_b",
                        "key_source": "none",
                        "explanation": "I put Song A's vocals over Song B's instrumental with the bass boosted to full. The remix starts with an instrumental intro, builds into the vocal verse, has a drum-free breakdown for contrast, then hits a full drop before fading out.",
                        "warnings": [],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_1", "content": "Plan accepted."}]},

        # Example 2: Vague prompt, tempo mismatch
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "mix them together, make it sound good"\n\n'
                "Song A: 88 BPM, G minor, 195s, 286 beats\n"
                "Song B: 125 BPM, A minor, 210s, 437 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_2",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_b",
                        "start_time_vocal": 30.0,
                        "end_time_vocal": 120.0,
                        "start_time_instrumental": 10.0,
                        "end_time_instrumental": 100.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.8, "guitar": 0.6, "piano": 0.5, "other": 1.0}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 0.6, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 48, "end_beat": 80, "stem_gains": {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "guitar": 0.5, "piano": 0.3, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 2},
                            {"label": "outro", "start_beat": 80, "end_beat": 96, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.6, "guitar": 0.6, "piano": 0.7, "other": 0.8}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I chose Song B's vocals since they're more prominent. The vocals are stretched to match Song A's slower tempo, which gives the remix a laid-back feel. The arrangement builds from an instrumental intro into the vocal sections.",
                        "warnings": ["These songs have very different tempos, so the vocals have been slowed down to fit the beat. They may sound a bit different from the original."],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_2", "content": "Plan accepted."}]},

        # Example 3: Contradictory prompt, genre clash
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "I want the drums from Song A with the vocals from both songs"\n\n'
                "Song A: 140 BPM, F# minor, 180s, 420 beats\n"
                "Song B: 72 BPM, Bb major, 240s, 288 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_3",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_b",
                        "start_time_vocal": 20.0,
                        "end_time_vocal": 100.0,
                        "start_time_instrumental": 0.0,
                        "end_time_instrumental": 60.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 8, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.7, "guitar": 0.0, "piano": 0.0, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 8, "end_beat": 24, "stem_gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.6, "guitar": 0.3, "piano": 0.5, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 24, "end_beat": 40, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.4, "piano": 0.3, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 40, "end_beat": 48, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.4, "guitar": 0.3, "piano": 0.4, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I used Song B's vocals since they're more prominent. The vocals are stretched significantly to match Song A's fast tempo, giving a unique energy. I used a Quick Hit template since the tempo difference is extreme.",
                        "warnings": [
                            "I can only use vocals from one song at a time -- I chose Song B since its vocals are clearer.",
                            "These songs have extremely different tempos. The vocals have been sped up significantly and may sound different from the original.",
                        ],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_3", "content": "Plan accepted."}]},
    ]


def _build_few_shot_messages_4stem() -> list[dict]:
    """Build few-shot examples adapted for 4-stem (local) mode."""
    # Same structure but without guitar/piano in stem_gains
    return [
        # Example 1
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "Put Song A\'s vocals over Song B\'s beat, boost the bass"\n\n'
                "Song A: 120 BPM, C major, 240s, 480 beats\n"
                "Song B: 118 BPM, C major, 210s, 413 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_1",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_a",
                        "start_time_vocal": 16.0,
                        "end_time_vocal": 136.0,
                        "start_time_instrumental": 8.0,
                        "end_time_instrumental": 128.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.8, "bass": 1.0, "other": 0.8}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 1.0, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.8, "drums": 0.0, "bass": 0.6, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 1.0, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.6, "bass": 0.8, "other": 0.7}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_b",
                        "key_source": "none",
                        "explanation": "I put Song A's vocals over Song B's instrumental with the bass boosted to full. The remix starts with an instrumental intro, builds into the vocal verse, has a drum-free breakdown for contrast, then hits a full drop before fading out.",
                        "warnings": [],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_1", "content": "Plan accepted."}]},

        # Example 2
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "mix them together, make it sound good"\n\n'
                "Song A: 88 BPM, G minor, 195s, 286 beats\n"
                "Song B: 125 BPM, A minor, 210s, 437 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_2",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_b",
                        "start_time_vocal": 30.0,
                        "end_time_vocal": 120.0,
                        "start_time_instrumental": 10.0,
                        "end_time_instrumental": 100.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.8, "other": 1.0}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 0.6, "drums": 0.7, "bass": 0.8, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 48, "end_beat": 80, "stem_gains": {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 2},
                            {"label": "outro", "start_beat": 80, "end_beat": 96, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.6, "other": 0.8}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I chose Song B's vocals since they're more prominent. The vocals are stretched to match Song A's slower tempo, which gives the remix a laid-back feel. The arrangement builds from an instrumental intro into the vocal sections.",
                        "warnings": ["These songs have very different tempos, so the vocals have been slowed down to fit the beat. They may sound a bit different from the original."],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_2", "content": "Plan accepted."}]},

        # Example 3
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "I want the drums from Song A with the vocals from both songs"\n\n'
                "Song A: 140 BPM, F# minor, 180s, 420 beats\n"
                "Song B: 72 BPM, Bb major, 240s, 288 beats"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "example_3",
                    "name": "create_remix_plan",
                    "input": {
                        "vocal_source": "song_b",
                        "start_time_vocal": 20.0,
                        "end_time_vocal": 100.0,
                        "start_time_instrumental": 0.0,
                        "end_time_instrumental": 60.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 8, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.7, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 8, "end_beat": 24, "stem_gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.6, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 24, "end_beat": 40, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "other": 0.5}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 40, "end_beat": 48, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.4, "other": 0.6}, "transition_in": "crossfade", "transition_beats": 4},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I used Song B's vocals since they're more prominent. The vocals are stretched significantly to match Song A's fast tempo, giving a unique energy. I used a Quick Hit template since the tempo difference is extreme.",
                        "warnings": [
                            "I can only use vocals from one song at a time -- I chose Song B since its vocals are clearer.",
                            "These songs have extremely different tempos. The vocals have been sped up significantly and may sound different from the original.",
                        ],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_3", "content": "Plan accepted."}]},
    ]


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_remix_plan(
    raw: dict,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
) -> RemixPlan:
    """Parse the raw tool_use output dict into a RemixPlan model."""
    sections = []
    for s in raw["sections"]:
        sections.append(Section(
            label=s["label"],
            start_beat=int(s["start_beat"]),
            end_beat=int(s["end_beat"]),
            stem_gains={k: float(v) for k, v in s["stem_gains"].items()},
            transition_in=s["transition_in"],
            transition_beats=int(s["transition_beats"]),
        ))

    return RemixPlan(
        vocal_source=raw["vocal_source"],
        start_time_vocal=float(raw["start_time_vocal"]),
        end_time_vocal=float(raw["end_time_vocal"]),
        start_time_instrumental=float(raw["start_time_instrumental"]),
        end_time_instrumental=float(raw["end_time_instrumental"]),
        sections=sections,
        tempo_source=raw["tempo_source"],
        key_source=raw["key_source"],
        explanation=raw["explanation"],
        warnings=list(raw.get("warnings", [])),
        used_fallback=False,
    )


# ---------------------------------------------------------------------------
# Post-LLM validation
# ---------------------------------------------------------------------------

def _validate_remix_plan(
    plan: RemixPlan,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
) -> RemixPlan:
    """Validate and fix the LLM's remix plan. Fixes issues in-place where possible."""
    clamped_fields: list[str] = []

    # Time range validation
    vocal_meta = song_a_meta if plan.vocal_source == "song_a" else song_b_meta
    inst_meta = song_b_meta if plan.vocal_source == "song_a" else song_a_meta

    plan.start_time_vocal = max(0, plan.start_time_vocal)
    plan.end_time_vocal = min(vocal_meta.duration_seconds, plan.end_time_vocal)
    if plan.end_time_vocal - plan.start_time_vocal < 5.0:
        plan.end_time_vocal = min(plan.start_time_vocal + 30.0, vocal_meta.duration_seconds)
        clamped_fields.append("vocal_time_range")

    plan.start_time_instrumental = max(0, plan.start_time_instrumental)
    plan.end_time_instrumental = min(inst_meta.duration_seconds, plan.end_time_instrumental)
    if plan.end_time_instrumental - plan.start_time_instrumental < 5.0:
        plan.end_time_instrumental = min(plan.start_time_instrumental + 30.0, inst_meta.duration_seconds)
        clamped_fields.append("instrumental_time_range")

    # Section validation (10-point checklist)
    sections = plan.sections

    # 1. Sort by start_beat
    sections.sort(key=lambda s: s.start_beat)

    # 2. No overlaps
    for i in range(len(sections) - 1):
        if sections[i].end_beat > sections[i + 1].start_beat:
            sections[i].end_beat = sections[i + 1].start_beat
            clamped_fields.append(f"overlap_section_{i}")

    # 3. No gaps > 1 beat
    for i in range(len(sections) - 1):
        gap = sections[i + 1].start_beat - sections[i].end_beat
        if gap > 1:
            sections[i].end_beat = sections[i + 1].start_beat
            clamped_fields.append(f"gap_section_{i}")

    # 4. Minimum section length (4 beats)
    sections = [s for s in sections if s.end_beat - s.start_beat >= 4]
    if not sections:
        # Completely unrecoverable -- use default arrangement
        total_beats = int(inst_meta.bpm * 90 / 60)
        plan.sections = default_arrangement(total_beats)
        plan.used_fallback = True
        plan.warnings.append("Section arrangement was regenerated automatically.")
        return plan

    # 5. transition_beats <= (end_beat - start_beat) / 2
    for s in sections:
        max_transition = (s.end_beat - s.start_beat) // 2
        if s.transition_beats > max_transition:
            s.transition_beats = max_transition
            clamped_fields.append(f"transition_beats_{s.label}")

    # 6. stem_gains keys (add missing, remove unknown)
    valid_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    for s in sections:
        for stem in valid_stems:
            if stem not in s.stem_gains:
                s.stem_gains[stem] = 0.0
        s.stem_gains = {k: v for k, v in s.stem_gains.items() if k in valid_stems}

    # 7. stem_gains values in [0.0, 1.0]
    for s in sections:
        for stem, gain in s.stem_gains.items():
            if gain < 0.0 or gain > 1.0:
                s.stem_gains[stem] = max(0.0, min(1.0, gain))
                clamped_fields.append(f"gain_{s.label}_{stem}")

    # 8. Total beat range within available audio (deferred to pipeline -- needs beat grid)

    # 9. At least 2 sections
    if len(sections) < 2:
        total_beats = sections[0].end_beat
        half = total_beats // 2
        # Snap to bar boundary
        half = (half // 4) * 4
        if half < 4:
            half = 4
        intro = Section("intro", 0, half, {**sections[0].stem_gains, "vocals": 0.0}, "fade", 4)
        main = sections[0]
        main.start_beat = half
        sections = [intro, main]

    # 10. Last section end_beat on bar boundary (multiple of 4)
    last = sections[-1]
    remainder = last.end_beat % 4
    if remainder != 0:
        last.end_beat += (4 - remainder)

    plan.sections = sections

    if clamped_fields:
        logger.info("Section validation clamped fields: %s", clamped_fields)

    # Duration validation
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)
    plan = _validate_duration(plan, target_bpm)

    return plan


def _validate_duration(plan: RemixPlan, target_bpm: float) -> RemixPlan:
    """Clamp total duration to 30-180 seconds."""
    if not plan.sections:
        return plan

    total_beats = plan.sections[-1].end_beat
    total_seconds = total_beats * 60 / target_bpm

    if total_seconds < 30:
        # Extend last section
        needed_beats = int(30 * target_bpm / 60) - total_beats
        plan.sections[-1].end_beat += max(needed_beats, 8)
        plan.warnings.append("Remix was extended to meet minimum duration.")
    elif total_seconds > 180:
        # Truncate
        max_beats = int(180 * target_bpm / 60)
        plan.sections[-1].end_beat = min(plan.sections[-1].end_beat, max_beats)
        plan.sections = [s for s in plan.sections if s.start_beat < max_beats]
        plan.warnings.append("Remix was shortened to fit maximum duration.")

    return plan


# ---------------------------------------------------------------------------
# Main LLM entry point
# ---------------------------------------------------------------------------

def interpret_prompt(
    prompt: str,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
) -> RemixPlan:
    """Convert user prompt + song metadata into a structured remix plan.

    Synchronous -- runs in the pipeline thread, NOT the async event loop.
    Falls back to generate_fallback_plan() on any LLM failure.
    """
    # Guard: if no API key configured, skip LLM entirely
    if not settings.anthropic_api_key:
        logger.warning("No ANTHROPIC_API_KEY configured, using fallback plan")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Pre-compute key matching decision
    _key_available, key_matching_detail = _compute_key_guidance(
        song_a_meta, song_b_meta,
    )

    # Compute total available beats (from instrumental source, approximated)
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)
    total_available_beats = int(target_bpm * 90 / 60)  # ~90 seconds worth

    system_prompt = _build_system_prompt(
        song_a_meta, song_b_meta,
        _key_available, key_matching_detail,
        total_available_beats,
    )

    # Build messages: few-shot examples + user prompt
    messages = _build_few_shot_messages() + [
        {"role": "user", "content": f'Create a remix plan for this prompt: "{prompt}"'},
    ]

    # Select and possibly adapt tool schema for 4-stem mode
    tool_schema = REMIX_PLAN_TOOL
    if settings.stem_backend == "local":
        tool_schema = _adapt_schema_for_4stem(tool_schema)

    # Log request
    logger.info(
        "LLM request: prompt=%r, song_a_bpm=%.1f, song_b_bpm=%.1f, model=%s",
        prompt, song_a_meta.bpm, song_b_meta.bpm, settings.llm_model,
    )

    start = time.monotonic()

    try:
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=2048,
            system=system_prompt,
            messages=messages,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": "create_remix_plan"},
            timeout=settings.llm_timeout_seconds,
        )
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 500, 529) and settings.llm_max_retries > 0:
            # Retry once with longer timeout
            logger.warning("LLM transient error %d, retrying", e.status_code)
            time.sleep(2)
            try:
                response = client.messages.create(
                    model=settings.llm_model,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=messages,
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": "create_remix_plan"},
                    timeout=30,
                )
            except Exception:
                logger.exception("LLM retry failed, using fallback")
                return generate_fallback_plan(song_a_meta, song_b_meta)
        else:
            logger.exception("LLM error (status=%d), using fallback", e.status_code)
            return generate_fallback_plan(song_a_meta, song_b_meta)
    except Exception:
        logger.exception("LLM error, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    latency_ms = (time.monotonic() - start) * 1000

    # Check stop reason
    if response.stop_reason == "max_tokens":
        logger.warning("LLM hit max_tokens, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    # Extract tool_use result
    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_use_block is None:
        logger.warning("LLM returned no tool_use block, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    raw_plan = tool_use_block.input

    # Log response
    logger.info(
        "LLM response: latency_ms=%.0f, model=%s, raw_plan=%s",
        latency_ms, response.model, json.dumps(raw_plan, indent=None),
    )

    # Parse and validate
    try:
        plan = _parse_remix_plan(raw_plan, song_a_meta, song_b_meta)
        plan = _validate_remix_plan(plan, song_a_meta, song_b_meta)
        return plan
    except Exception:
        logger.exception("LLM plan validation failed, using fallback")
        return generate_fallback_plan(song_a_meta, song_b_meta)


# ===========================================================================
# Day 2 deterministic fallback (preserved as-is)
# ===========================================================================

def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
    """Generate a deterministic remix plan from audio analysis metadata.

    Defaults: vocals from song_a, instrumentals from song_b.
    Uses the region starting at 25% into each song, capped at 90 seconds.
    Tempo target is the instrumental song's BPM.
    """
    # Vocal source defaults to song_a (Day 2 -- no vocal_prominence_db yet)
    vocal_src = "song_a"
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 25% into each song, up to 90 seconds
    v_start = vocal_meta.duration_seconds * 0.25
    v_end = min(v_start + 90.0, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.25
    i_end = min(i_start + 90.0, inst_meta.duration_seconds)

    tempo_src = "weighted_midpoint"  # Split stretch burden between both songs
    total_beats = int(inst_meta.bpm * 90 / 60)  # Beats in 90 seconds

    logger.info(
        "Generated fallback plan: vocals=%s, tempo=%s, total_beats=%d",
        vocal_src,
        tempo_src,
        total_beats,
    )

    return RemixPlan(
        vocal_source=vocal_src,
        start_time_vocal=v_start,
        end_time_vocal=v_end,
        start_time_instrumental=i_start,
        end_time_instrumental=i_end,
        sections=default_arrangement(total_beats),
        tempo_source=tempo_src,
        key_source="none",
        explanation=(
            "We created a remix using the strongest sections of each song. "
            "Vocals from Song A layered over Song B's instrumentals."
        ),
        warnings=["Using automatic remix layout (no prompt interpretation yet)."],
        used_fallback=True,
    )


def default_arrangement(total_beats: int) -> list[Section]:
    """Build a 5- or 6-section fallback arrangement.

    6-section (>= 96 beats): intro -> build -> main -> breakdown -> drop -> outro
    5-section (< 96 beats):  intro -> build -> main -> breakdown -> outro

    Beat boundaries are snapped to 4-bar (16-beat) phrase boundaries for
    musically coherent transitions.
    """
    MIN_SECTION_BEATS = 8  # Absolute minimum -- shorter sections are musically meaningless

    def snap_to_phrase(beat: int, phrase_beats: int = 16) -> int:
        """Snap a beat index DOWN to the nearest phrase boundary (default 4 bars).
        Uses floor (not round) for predictable section proportions."""
        return max(0, (beat // phrase_beats) * phrase_beats)

    # For very short arrangements, skip phrase snapping
    if total_beats < 48:
        eighth = total_beats // 8
        quarter = total_beats // 4
        three_quarter = total_beats * 3 // 4
        seven_eighth = total_beats * 7 // 8
    else:
        eighth = snap_to_phrase(total_beats // 8)
        quarter = snap_to_phrase(total_beats // 4)
        three_quarter = snap_to_phrase(total_beats * 3 // 4)
        seven_eighth = snap_to_phrase(total_beats * 7 // 8)

        # Guard: ensure monotonically increasing boundaries with minimum section size.
        if eighth < MIN_SECTION_BEATS:
            eighth = min(16, total_beats // 6)
        if quarter <= eighth:
            quarter = min(eighth + 16, total_beats - 3 * MIN_SECTION_BEATS)
        if three_quarter <= quarter:
            three_quarter = min(quarter + 16, total_beats - 2 * MIN_SECTION_BEATS)
        if seven_eighth <= three_quarter:
            seven_eighth = min(three_quarter + 16, total_beats - MIN_SECTION_BEATS)

        # Final validation: all boundaries within [0, total_beats]
        eighth = max(MIN_SECTION_BEATS, min(eighth, total_beats - 4 * MIN_SECTION_BEATS))
        quarter = max(eighth + MIN_SECTION_BEATS, min(quarter, total_beats - 3 * MIN_SECTION_BEATS))
        three_quarter = max(quarter + MIN_SECTION_BEATS, min(three_quarter, total_beats - 2 * MIN_SECTION_BEATS))
        seven_eighth = max(three_quarter + MIN_SECTION_BEATS, min(seven_eighth, total_beats - MIN_SECTION_BEATS))

    # Build and main MUST share identical instrumental gains to prevent volume dips
    # at the build->main transition. The auto-leveler's detector_audio uses the
    # instrumental bus -- different gains would change detected energy at boundaries,
    # re-triggering the volume dip bug fixed in the 2026-02-25 investigation.
    inst_body =      {"drums": 0.7, "bass": 0.7, "guitar": 0.5, "piano": 0.4, "other": 0.5}
    inst_intro =     {"drums": 0.6, "bass": 0.5, "guitar": 0.3, "piano": 0.2, "other": 0.3}
    inst_breakdown = {"drums": 0.1, "bass": 0.4, "guitar": 0.6, "piano": 0.7, "other": 0.5}
    inst_outro =     {"drums": 0.5, "bass": 0.5, "guitar": 0.3, "piano": 0.3, "other": 0.4}

    # 6-section: if enough beats, add a "drop" section between breakdown and outro
    if total_beats >= 96:
        # Recalculate boundaries for 6 sections
        # main ends earlier to make room; breakdown and drop each get 1/8
        five_eighth = snap_to_phrase(total_beats * 5 // 8)
        six_eighth = snap_to_phrase(total_beats * 6 // 8)
        drop_start = six_eighth
        drop_end = snap_to_phrase(total_beats * 7 // 8)
        outro_start = drop_end

        # Validate drop section has minimum beats
        if drop_end - drop_start < MIN_SECTION_BEATS:
            drop_end = min(drop_start + 16, total_beats - MIN_SECTION_BEATS)
            outro_start = drop_end

        return [
            Section(label="intro", start_beat=0, end_beat=eighth,
                    stem_gains={"vocals": 0.0, **inst_intro},
                    transition_in="fade", transition_beats=4),
            Section(label="build", start_beat=eighth, end_beat=quarter,
                    stem_gains={"vocals": 0.5, **inst_body},
                    transition_in="crossfade", transition_beats=min(8, (quarter - eighth) // 3)),
            Section(label="main", start_beat=quarter, end_beat=five_eighth,
                    stem_gains={"vocals": 1.0, **inst_body},
                    transition_in="crossfade", transition_beats=4),
            Section(label="breakdown", start_beat=five_eighth, end_beat=drop_start,
                    stem_gains={"vocals": 0.4, **inst_breakdown},
                    transition_in="crossfade", transition_beats=min(8, (drop_start - five_eighth) // 3)),
            Section(label="drop", start_beat=drop_start, end_beat=drop_end,
                    stem_gains={"vocals": 0.8, **inst_body},
                    transition_in="crossfade", transition_beats=2),
            Section(label="outro", start_beat=outro_start, end_beat=total_beats,
                    stem_gains={"vocals": 0.0, **inst_outro},
                    transition_in="crossfade", transition_beats=min(8, (total_beats - outro_start) // 2)),
        ]

    # 5-section fallback (not enough beats for drop section)
    return [
        Section(label="intro", start_beat=0, end_beat=eighth,
                stem_gains={"vocals": 0.0, **inst_intro},
                transition_in="fade", transition_beats=4),
        Section(label="build", start_beat=eighth, end_beat=quarter,
                stem_gains={"vocals": 0.5, **inst_body},
                transition_in="crossfade", transition_beats=4),
        Section(label="main", start_beat=quarter, end_beat=three_quarter,
                stem_gains={"vocals": 1.0, **inst_body},
                transition_in="crossfade", transition_beats=4),
        Section(label="breakdown", start_beat=three_quarter, end_beat=seven_eighth,
                stem_gains={"vocals": 0.4, **inst_breakdown},
                transition_in="crossfade", transition_beats=4),
        Section(label="outro", start_beat=seven_eighth, end_beat=total_beats,
                stem_gains={"vocals": 0.0, **inst_outro},
                transition_in="crossfade", transition_beats=4),
    ]
