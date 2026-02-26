"""Remix plan interpreter: LLM-powered prompt interpretation with deterministic fallback.

Day 3: Converts user prompts + song metadata into structured RemixPlan objects
using Anthropic's tool_use API. Falls back to deterministic arrangement on any
LLM failure.

Day 2 fallback functions (generate_fallback_plan, default_arrangement) are
preserved at the bottom of this file.

Day 3 song structure integration: 4-layer system prompt with section maps,
stem character, cross-song relationships, and 7 failure mode guards.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path

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

# Target remix duration in seconds. Controls beat budget, LLM guidance, and fallback plans.
TARGET_REMIX_DURATION_SECONDS = 210  # 3.5 minutes


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
                "maxItems": 20,
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
                            "description": "Ending beat (exclusive). Section length = end_beat - start_beat. Should be 4, 8, 16, or 32.",
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
    stretch_pct: float | None = None,
) -> str:
    """Construct the full system prompt with 4-layer song data and failure mode guards."""

    # Determine which stems are available
    is_4stem = settings.stem_backend == "local"
    stem_list = "vocals, drums, bass, other" if is_4stem else "vocals, drums, bass, guitar, piano, other"
    stem_count = 4 if is_4stem else 6

    # Compute per-song beat counts (approximate)
    total_beats_a = song_a_meta.total_beats
    total_beats_b = song_b_meta.total_beats

    # Approximate target BPM for beat-to-seconds conversion
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)

    # Stem gains note for 4-stem mode
    stem_gains_note = ""
    if is_4stem:
        stem_gains_note = (
            "\n- Only 4 stems are available: vocals, drums, bass, other. "
            "Guitar and piano are not separated in this mode."
        )

    sections: list[str] = []

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

    # Section 2: Failure Mode Guards (7 rules from spec section 7)
    sections.append("""CRITICAL MIXING RULES (violations produce bad audio):
1. INSTRUMENTAL SECTIONS: Prefer sections with no vocals (vox:no, labeled GOOD INSTRUMENTAL SOURCE). For the "other" stem: low-energy sections -> gain 0.2; medium+ energy -> gain 0.4-0.6 (preserves genre identity).
2. VOCAL CLARITY: When vocals are active, reduce mid-frequency stems (guitar, piano, other) by 30% or more. Only drums + bass should be at full volume alongside vocals.
3. NO RHYTHMIC COLLISION: Never use drums from both songs simultaneously. Never overlap bass lines from both songs.
4. ENERGY MATCHING: Match vocal energy to instrumental energy level. Exception: quiet vocal over minimal beat is acceptable as an intentional artistic choice.
5. DYNAMIC RANGE: The remix MUST have at least 1 contrast moment (e.g., breakdown -> drop) and use a minimum of 3 different energy levels across sections.
6. ENDING: End with 4-8 bars of reduced energy or a natural outro. NEVER cut the remix at full energy -- it sounds broken.
7. GAIN VARIATION: Vary gain profiles across sections. Strip down to drums+bass+vocals for contrast, then use full arrangement for impact. Flat gain across all sections produces a lifeless mix.""")

    # Section 3: Mixing Advisory Notes
    sections.append("""MIXING ADVISORY:
- Stagger stem entries over 2-4 bars for natural-sounding builds (don't bring everything in at once).
- Begin vocal sections 1-2 beats early for pickup notes (vocals often start before the downbeat).
- Two peak-level stems at full volume (1.0) will clip. Reduce one by 3-6 dB (gain 0.5-0.7).
- Section labels in the song data are approximate guidance, not rigid constraints. Use them to understand song structure, but your arrangement should serve the user's prompt.
- Contrast creates energy: if a section has drums at 0.0, the next section's drums at 1.0 will feel powerful.
- Muted stems (0.0) are a tool, not a failure -- silence in the right place is more powerful than sound.
- Use the full 0.0-1.0 range. Avoid keeping all stems at 0.5-0.8 throughout -- that produces a flat, unengaging mix.""")

    # Section 4: Stretch Advisory
    stretch_advisory = ""
    if stretch_pct is not None and stretch_pct > 12:
        stretch_advisory = f"""
STRETCH WARNING ({stretch_pct:.1f}%):
- At >12% stretch, limit stretched sections: max 8 bars at up to 15%, max 4 bars above 15%.
- Prefer stretching instruments over vocals (vocals degrade faster under stretch).
- This is advisory -- use musical judgment."""

    # Section 5: Transition Guidance
    sections.append(f"""TRANSITIONS:
- "cut": Use between sections at similar energy levels for a punchy feel. Good for drop-to-verse or chorus-to-chorus.
- "crossfade": Use when energy changes significantly between sections. Default choice for most transitions.
- "fade": Use for the first section (intro) and last section (outro). Also good for bringing vocals in from silence.{stretch_advisory}""")

    # Section 6: Arrangement Templates
    sections.append(f"""Your sections must sum to approximately {total_available_beats} beats.

Template A (Standard Mashup): intro(~15%) -> verse(~30%) -> breakdown(~15%) -> drop(~30%) -> outro(~10%)
Template B (DJ Set): build(~25%) -> vocals in(~25%) -> peak(~25%) -> vocals out(~12%) -> outro(~13%)
Template C (Quick Hit): intro(~15%) -> vocal drop(~70%) -> outro(~15%)
Template D (Chill): intro(~25%) -> vocals(~50%) -> outro(~25%)
Template E (Extended Mix): intro(~8%) -> verse(~15%) -> chorus(~12%) -> breakdown(~8%) -> verse(~15%) -> chorus(~12%) -> bridge(~10%) -> drop(~12%) -> outro(~8%)

If total beats < 48, use Template C. If 48-96, use Standard Mashup. If 96-192, use DJ Set or add a second verse. If > 192, use Extended Mix.""")

    # Section 7: Section Rules
    stem_gains_required = stem_list
    sections.append(f"""SECTION RULES:
- Sections should be 4, 8, 16, or 32 beats long (max 32 beats per section)
- Default: start with instrumental only (establishes the beat before vocals enter), unless the prompt suggests otherwise
- Always end with instrumental only or a fade
- section labels: "intro", "verse", "breakdown", "drop", "outro"
- stem_gains values must be between 0.0 and 1.0 (never exceed 1.0 -- it causes distortion)
- stem_gains must include all stems: {stem_gains_required}
- transition_in: "fade", "crossfade", or "cut"
- transition_beats: how many beats the transition lasts (0-8, must be less than half the section length)""")

    # Section 8: Genre-Aware Arrangement
    sections.append("""GENRE GUIDANCE (infer from BPM + energy profile + section map):
- Hip-hop/rap (80-100 BPM): Keep drums consistent throughout. Build energy through vocal intensity and layering, not drum drops.
- EDM/dance (120-130 BPM): Use breakdown -> build -> drop patterns. Align drops with sections annotated DROP.
- Pop/rock (100-130 BPM): Use verse-chorus dynamics -- stripped for verses, full for choruses.
- R&B/soul (60-90 BPM): Smooth transitions, no abrupt changes. Layer elements gradually.
- Jam/rock (variable BPM): Use instrumental sections for extended jams. Vocal gaps are natural entry points.""")

    # Section 9: Tempo and Key Guidance
    sections.append(f"""TEMPO MATCHING:
- tempo_source "average" only when BPMs differ by <15%.
- 15-30% gap: prefer vocal source tempo (the song providing vocals gets stretched less).
- >30% gap: system will stretch vocals only. Note this in your explanation.

KEY MATCHING:
{key_matching_detail}

PITCH LIMIT:
- Do not plan shifts above +/-4 semitones. If compatibility would require more, keep original key and add a warning.""")

    # Section 10: Ambiguity Handling
    sections.append("""HANDLING AMBIGUOUS PROMPTS:
- Vague ("make it cool"): Use energy profiles and section maps. Pick vocals from the song with higher vocal prominence. Use Standard Mashup template. Align sections to GOOD INSTRUMENTAL SOURCE annotations.
- Contradictory ("vocals from both"): Acknowledge in warnings. Pick the better vocal source and explain why.
- Genre jargon ("trap", "lo-fi"): Translate to volume/structure decisions. "Trap" = heavy bass, sparse hi-hats. "Lo-fi" = reduce other, gentle, Template D.
- Time references ("guitar solo at 2:30"): Cross-reference with the section map to find the right region. Add a warning if unsure.

DURATION: Target remix duration 180-240 seconds. Minimum 60s, maximum 300s.""")

    # Section 11: Stem Artifact Awareness
    sections.append("""STEM SEPARATION ARTIFACTS:
Stem separation is imperfect. Vocal stem may contain instrument traces. Instrumental stems may contain ghost vocals.
Bleed is less noticeable during high-energy sections. Prefer sections annotated GOOD INSTRUMENTAL SOURCE for clean instrumental passages. When the instrumental source song has prominent vocals, avoid purely-instrumental sections longer than 8 beats -- ghost vocals bleed through.""")

    # Section 12: Explanation and Warnings
    sections.append("""EXPLANATION: Write 2-3 non-technical sentences explaining what you did and why. No internal jargon. This is shown directly to the user.

WARNINGS: Populate this array when:
- The prompt is vague and you had to make assumptions
- The prompt asks for something impossible (cross-song stem mixing, effects)
- You're uncertain about a time reference or genre interpretation
- Tempo/key gap is large and the remix may sound noticeably different from the originals""")

    # Section 13: 4-Layer Song Data
    song_a_info = _build_song_info("Song A", song_a_meta, total_beats_a)
    song_b_info = _build_song_info("Song B", song_b_meta, total_beats_b)
    cross_song = _build_cross_song_layer(song_a_meta, song_b_meta, stretch_pct)

    song_data_parts = [
        "=== LAYER 1: SONG OVERVIEW ===",
        song_a_info,
        song_b_info,
    ]

    # Layer 2: Section Maps (already embedded in song_info)
    has_section_data = (
        getattr(song_a_meta, "song_structure", None) is not None
        or getattr(song_b_meta, "song_structure", None) is not None
    )
    if has_section_data:
        song_data_parts.append("\n=== LAYER 2: SECTION MAP ===")
        # Re-build just the section map portions
        struct_a = getattr(song_a_meta, "song_structure", None)
        if struct_a and struct_a.sections:
            section_map_a = _build_section_map("Song A", struct_a, total_beats_a // 4)
            song_data_parts.append(section_map_a)
        struct_b = getattr(song_b_meta, "song_structure", None)
        if struct_b and struct_b.sections:
            section_map_b = _build_section_map("Song B", struct_b, total_beats_b // 4)
            song_data_parts.append(section_map_b)

    # Layer 3: Stem Character
    has_stem_data = (
        getattr(song_a_meta, "stem_analysis", None) is not None
        or getattr(song_b_meta, "stem_analysis", None) is not None
    )
    if has_stem_data:
        song_data_parts.append("\n=== LAYER 3: STEM CHARACTER ===")
        stem_a = _build_stem_character("Song A", song_a_meta)
        stem_b = _build_stem_character("Song B", song_b_meta)
        if stem_a:
            song_data_parts.append(stem_a)
        if stem_b:
            song_data_parts.append(stem_b)

    # Layer 4: Cross-Song Relationships
    if cross_song:
        song_data_parts.append("\n=== LAYER 4: CROSS-SONG ===")
        song_data_parts.append(cross_song)

    sections.append("SONG DATA:\n\n" + "\n".join(song_data_parts))

    # Beat reference
    sections.append(f"1 beat = {60 / target_bpm:.2f}s at {target_bpm:.0f} BPM. 1 bar = 4 beats.")

    return "\n\n".join(sections)


def _build_section_map(label: str, structure: object, total_bars: int) -> str:
    """Build aligned section map table for Layer 2."""
    lines = [f"{label} ({total_bars} bars):"]
    for sec in structure.sections:
        t_start = f"{int(sec.start_time) // 60}:{int(sec.start_time) % 60:02d}"
        t_end = f"{int(sec.end_time) // 60}:{int(sec.end_time) % 60:02d}"

        annotations_str = ""
        if sec.annotations:
            annotations_str = " | " + ", ".join(sec.annotations)

        energy_display = sec.energy_trajectory if sec.energy_trajectory else sec.energy_level

        lines.append(
            f"  {sec.start_bar:>3}-{sec.end_bar:<3}  "
            f"{sec.bar_count:>2}b  "
            f"{t_start}-{t_end} | "
            f"{sec.label:<14} | "
            f"{energy_display:<16} | "
            f"{sec.density:<10} | "
            f"{sec.vocal_status:<10}"
            f"{annotations_str}"
        )

    if structure.vocal_gaps:
        gap_strs = [f"{g.start_bar}-{g.end_bar}" for g in structure.vocal_gaps]
        lines.append(f"Vocal gaps: {', '.join(gap_strs)}")

    return "\n".join(lines)


def _build_stem_character(label: str, meta: AudioMetadata) -> str:
    """Build Layer 3 stem character line for a single song (MVP format)."""
    import numpy as np

    stem_analysis = getattr(meta, "stem_analysis", None)
    if not stem_analysis or not stem_analysis.bar_rms:
        return ""

    stem_descs: list[str] = []
    suppressed: list[str] = []
    buckets = stem_analysis.bucket_thresholds

    for stem_name in ALL_STEMS:
        if stem_name not in stem_analysis.bar_rms:
            continue
        rms_array = stem_analysis.bar_rms[stem_name]
        if rms_array is None or len(rms_array) == 0:
            suppressed.append(stem_name)
            continue

        mean_rms = float(np.mean(rms_array))

        if mean_rms < buckets.noise_floor:
            suppressed.append(f"{stem_name}: negligible")
            continue

        # Classify energy bucket
        if mean_rms < buckets.p10:
            energy_bucket = "low"
        elif mean_rms < buckets.p50:
            energy_bucket = "medium"
        elif mean_rms < buckets.p85:
            energy_bucket = "high"
        else:
            energy_bucket = "peak"

        # Classify density
        active_bars = int((rms_array > buckets.noise_floor).sum())
        active_frac = active_bars / max(len(rms_array), 1)
        if active_frac < 0.3:
            density = "sparse"
        elif active_frac < 0.6:
            density = "mid"
        else:
            density = "full"

        stem_descs.append(f"{stem_name}: {energy_bucket}-energy, {density}")

    result = f"{label}: " + ". ".join(stem_descs) + "."
    if suppressed:
        result += f" ({' | '.join(suppressed)})"
    return result


def _build_song_info(
    label: str,
    meta: AudioMetadata,
    total_beats: int,
) -> str:
    """Build Layer 1 (Song Overview) for a single song.

    Returns a multi-line string with overview, vocal prominence, and energy profile.
    Layers 2-3 are built separately by _build_section_map() and _build_stem_character().
    Degrades gracefully when new fields are absent.
    """
    key = getattr(meta, "key", None)
    scale = getattr(meta, "scale", None)
    key_str = f"{key}{scale}" if key and scale else "unknown key"

    # Estimate total bars from total_beats (4 beats per bar)
    total_bars = total_beats // 4

    # Duration formatted as m:ss
    dur_min = int(meta.duration_seconds) // 60
    dur_sec = int(meta.duration_seconds) % 60

    overview = (
        f'{label}: "{_song_filename(meta)}" -- '
        f"{meta.bpm:.0f} BPM, {key_str}, {dur_min}:{dur_sec:02d}, {total_bars} bars."
    )

    # Vocal prominence
    vocal_prom = getattr(meta, "vocal_prominence_db", None)
    if vocal_prom is not None:
        if vocal_prom > 6:
            vox_desc = f"dominant, +{vocal_prom:.0f} dB above instrumental, clean separation"
        elif vocal_prom > 3:
            vox_desc = f"moderate, +{vocal_prom:.0f} dB above instrumental"
        else:
            vox_desc = f"buried, +{vocal_prom:.0f} dB above instrumental, bleed expected"
        overview += f"\nVocals: {vox_desc}."

    # Energy profile from song_structure
    song_structure = getattr(meta, "song_structure", None)
    if song_structure and song_structure.sections:
        energy_levels = [s.energy_level for s in song_structure.sections]
        unique_levels = set(energy_levels)
        if len(unique_levels) <= 2 and "high" in unique_levels:
            overview += " Energy: compressed."
        elif len(unique_levels) >= 4:
            overview += " Energy: wide dynamic range."
        else:
            overview += " Energy: moderate dynamics."

    return overview


def _song_filename(meta: AudioMetadata) -> str:
    """Extract a display-friendly filename from AudioMetadata, if available."""
    # Try to get source filename from metadata attributes added during upload
    source_path = getattr(meta, "source_path", None)
    if source_path:
        return Path(source_path).stem
    return "uploaded song"


def _build_cross_song_layer(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    stretch_pct: float | None = None,
) -> str:
    """Build Layer 4: Cross-Song Relationships.

    Degrades gracefully when new fields are absent.
    """
    lines: list[str] = []

    # Loudness difference
    mean_rms_a = getattr(meta_a, "mean_rms", None)
    mean_rms_b = getattr(meta_b, "mean_rms", None)
    if mean_rms_a and mean_rms_b and mean_rms_a > 0.001 and mean_rms_b > 0.001:
        loudness_diff = 20 * math.log10(max(mean_rms_a, 1e-10) / max(mean_rms_b, 1e-10))
        if abs(loudness_diff) < 2:
            lines.append("Loudness: similar levels.")
        elif loudness_diff > 0:
            lines.append(
                f"Loudness: Song A ~{abs(loudness_diff):.0f} dB louder. "
                f"Reduce Song A stems ~{abs(loudness_diff):.0f} dB."
            )
        else:
            lines.append(
                f"Loudness: Song B ~{abs(loudness_diff):.0f} dB louder. "
                f"Reduce Song B stems ~{abs(loudness_diff):.0f} dB."
            )

    # Vocal source recommendation
    vp_a = getattr(meta_a, "vocal_prominence_db", None)
    vp_b = getattr(meta_b, "vocal_prominence_db", None)
    if vp_a is not None and vp_b is not None:
        if vp_a > vp_b:
            lines.append(
                f"Vocal source: Song A (+{vp_a:.0f} dB, cleaner). "
                f"Song B vocals have more bleed (+{vp_b:.0f} dB)."
            )
        else:
            lines.append(
                f"Vocal source: Song B (+{vp_b:.0f} dB, cleaner). "
                f"Song A vocals have more bleed (+{vp_a:.0f} dB)."
            )

    # Instrumental sections from the recommended instrumental source
    # (default: Song B provides instrumentals)
    struct_b = getattr(meta_b, "song_structure", None)
    if struct_b and struct_b.sections:
        inst_sections = [
            f"bars {s.start_bar}-{s.end_bar}"
            for s in struct_b.sections
            if "GOOD INSTRUMENTAL SOURCE" in (s.annotations or [])
        ]
        if inst_sections:
            lines.append(f"Instrumental source: Song B clean sections at {', '.join(inst_sections)}.")

    # Stretch info
    if stretch_pct is not None and stretch_pct > 0:
        lines.append(f"Tempo stretch: {stretch_pct:.1f}%.")

    return "\n".join(lines) if lines else ""


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
    """Build 3 few-shot examples showing how to interpret the 4-layer song data.

    Examples demonstrate:
    1. Clear directive with section map alignment
    2. Vague prompt using vocal gaps and energy profiles
    3. Contradictory prompt with extreme tempo mismatch
    """
    is_4stem = settings.stem_backend == "local"

    if is_4stem:
        return _build_few_shot_messages_4stem()

    return [
        # Example 1: Clear directive, uses section map + GOOD INSTRUMENTAL SOURCE
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "Put Song A\'s vocals over Song B\'s beat, boost the bass"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Track One" -- 120 BPM, Cmin, 4:00, 120 bars.\n'
                "Vocals: dominant, +8 dB above instrumental, clean separation. Energy: compressed.\n"
                'Song B: "Track Two" -- 118 BPM, Cmaj, 3:30, 103 bars.\n'
                "Vocals: moderate, +3 dB above instrumental. Energy: wide dynamic range.\n\n"
                "=== LAYER 2: SECTION MAP ===\n"
                "Song A (120 bars):\n"
                "    1-8    8b  0:00-0:16 | intro          | medium           | mid        | vox:no    | GOOD INSTRUMENTAL SOURCE\n"
                "    9-40  32b  0:16-1:20 | verse          | high             | full       | vox:yes\n"
                "   41-56  16b  1:20-1:52 | chorus         | high             | full+extra | vox:yes   | DROP\n"
                "   57-72  16b  1:52-2:24 | verse          | high             | full       | vox:yes\n"
                "   73-88  16b  2:24-2:56 | chorus         | high             | full+extra | vox:yes\n"
                "   89-104 16b  2:56-3:28 | breakdown      | high->medium     | mid        | vox:yes\n"
                "  105-120 16b  3:28-4:00 | outro          | medium->low      | sparse     | vox:fading\n"
                "Vocal gaps: 1-8\n"
                "Song B (103 bars):\n"
                "    1-8    8b  0:00-0:16 | intro          | low              | sparse     | vox:no    | GOOD INSTRUMENTAL SOURCE\n"
                "    9-40  32b  0:16-1:21 | verse          | medium           | mid        | vox:yes\n"
                "   41-72  32b  1:21-2:26 | instrumental   | high->peak       | full+extra | vox:no    | GOOD INSTRUMENTAL SOURCE\n"
                "   73-88  16b  2:26-2:58 | verse          | medium           | mid        | vox:yes\n"
                "   89-103 15b  2:58-3:30 | outro          | medium->low      | sparse     | vox:no\n"
                "Vocal gaps: 1-8, 41-72, 89-103\n\n"
                "=== LAYER 3: STEM CHARACTER ===\n"
                "Song A: vocals: high-energy, full. drums: high-energy, full. bass: high-energy, full. other: high-energy, full. (guitar: negligible | piano: minor)\n"
                "Song B: guitar: high-energy, full. drums: high-energy, full. bass: medium-energy, mid. vocals: medium-energy, mid. piano: low-energy, sparse. (other: minimal)\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: Song A ~5 dB louder. Reduce Song A stems ~5 dB.\n"
                "Vocal source: Song A (+8 dB, clean). Song B has more bleed (+3 dB).\n"
                "Instrumental source: Song B clean sections at bars 1-8, 41-72, 89-103."
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
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.8, "bass": 1.0, "guitar": 0.5, "piano": 0.4, "other": 0.3}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 1.0, "guitar": 0.3, "piano": 0.2, "other": 0.3}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.0, "drums": 0.0, "bass": 0.5, "guitar": 0.6, "piano": 0.7, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 1.0, "guitar": 0.4, "piano": 0.3, "other": 0.3}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.7, "guitar": 0.5, "piano": 0.5, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_b",
                        "key_source": "none",
                        "explanation": "I put Song A's vocals over Song B's instrumental with the bass boosted to full. I used Song B's clean instrumental sections (bars 41-72) for the main groove, starting with their sparse intro. Mid-frequency stems are reduced when vocals are active for clarity.",
                        "warnings": [],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_1", "content": "Plan accepted."}]},

        # Example 2: Vague prompt -- uses vocal gaps and energy profiles
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "mix them together, make it sound good"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Slow Jam" -- 88 BPM, Gmin, 3:15, 71 bars.\n'
                "Vocals: moderate, +4 dB above instrumental. Energy: moderate dynamics.\n"
                'Song B: "Upbeat Track" -- 125 BPM, Amin, 3:30, 109 bars.\n'
                "Vocals: dominant, +7 dB above instrumental, clean separation. Energy: wide dynamic range.\n\n"
                "=== LAYER 2: SECTION MAP ===\n"
                "Song B (109 bars):\n"
                "    1-8    8b  0:00-0:15 | intro          | low              | sparse     | vox:no    | GOOD INSTRUMENTAL SOURCE\n"
                "    9-32  24b  0:15-0:57 | verse          | medium           | mid        | vox:yes\n"
                "   33-48  16b  0:57-1:21 | chorus         | high             | full       | vox:yes   | DROP\n"
                "   49-64  16b  1:21-1:45 | instrumental   | high             | full       | vox:no    | GOOD INSTRUMENTAL SOURCE\n"
                "   65-80  16b  1:45-2:09 | verse          | medium           | mid        | vox:yes\n"
                "   81-96  16b  2:09-2:33 | chorus         | high             | full+extra | vox:yes\n"
                "   97-109 13b  2:33-3:30 | outro          | medium->low      | sparse     | vox:no\n"
                "Vocal gaps: 1-8, 49-64, 97-109\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: similar levels.\n"
                "Vocal source: Song B (+7 dB, cleaner)."
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
                        "start_time_vocal": 15.0,
                        "end_time_vocal": 133.0,
                        "start_time_instrumental": 10.0,
                        "end_time_instrumental": 100.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.8, "guitar": 0.5, "piano": 0.4, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 0.8, "drums": 0.7, "bass": 0.8, "guitar": 0.3, "piano": 0.3, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.0, "drums": 0.3, "bass": 0.5, "guitar": 0.6, "piano": 0.5, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 64, "end_beat": 88, "stem_gains": {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "guitar": 0.4, "piano": 0.2, "other": 0.3}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 88, "end_beat": 104, "stem_gains": {"vocals": 0.0, "drums": 0.4, "bass": 0.5, "guitar": 0.5, "piano": 0.6, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I chose Song B's vocals since they have better separation (+7 dB). The instrumental comes from Song A's laid-back 88 BPM groove. I used Song B's vocal gaps for the instrumental intro and breakdown, giving contrast before the vocal drop.",
                        "warnings": ["These songs have very different tempos, so the vocals have been slowed down to fit the beat. They may sound a bit different from the original."],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_2", "content": "Plan accepted."}]},

        # Example 3: Contradictory prompt, extreme tempo mismatch
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "I want the drums from Song A with the vocals from both songs"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Fast EDM" -- 140 BPM, F#min, 3:00, 105 bars.\n'
                "Vocals: buried, +1 dB above instrumental, bleed expected. Energy: compressed.\n"
                'Song B: "Slow RnB" -- 72 BPM, Bbmaj, 4:00, 72 bars.\n'
                "Vocals: dominant, +9 dB above instrumental, clean separation. Energy: moderate dynamics.\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: Song A ~4 dB louder.\n"
                "Vocal source: Song B (+9 dB, much cleaner).\n"
                "Tempo stretch: 48.6%."
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
                            {"label": "intro", "start_beat": 0, "end_beat": 8, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.7, "guitar": 0.0, "piano": 0.0, "other": 0.4}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 8, "end_beat": 24, "stem_gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.5, "guitar": 0.2, "piano": 0.3, "other": 0.3}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 24, "end_beat": 40, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "guitar": 0.3, "piano": 0.2, "other": 0.3}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 40, "end_beat": 48, "stem_gains": {"vocals": 0.0, "drums": 0.4, "bass": 0.4, "guitar": 0.3, "piano": 0.4, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I used Song B's vocals since they have much better separation (+9 dB vs +1 dB). The drums-only verse creates contrast before the full drop. I kept it short due to the extreme tempo difference.",
                        "warnings": [
                            "I can only use vocals from one song at a time -- I chose Song B since its vocals are much cleaner (+9 dB separation vs Song A's +1 dB).",
                            "These songs have extremely different tempos (48.6% stretch). The vocals have been sped up significantly and may sound different from the original.",
                        ],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_3", "content": "Plan accepted."}]},
    ]


def _build_few_shot_messages_4stem() -> list[dict]:
    """Build few-shot examples adapted for 4-stem (local) mode.

    Same structure as 6-stem but without guitar/piano in stem_gains.
    Uses the new 4-layer song data format.
    """
    return [
        # Example 1: Clear directive with section map
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "Put Song A\'s vocals over Song B\'s beat, boost the bass"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Track One" -- 120 BPM, Cmin, 4:00, 120 bars.\n'
                "Vocals: dominant, +8 dB. Energy: compressed.\n"
                'Song B: "Track Two" -- 118 BPM, Cmaj, 3:30, 103 bars.\n'
                "Vocals: moderate, +3 dB. Energy: wide dynamic range.\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: Song A ~5 dB louder.\n"
                "Vocal source: Song A (+8 dB, clean)."
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
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.8, "bass": 1.0, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 1.0, "drums": 0.7, "bass": 1.0, "other": 0.3}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "breakdown", "start_beat": 48, "end_beat": 64, "stem_gains": {"vocals": 0.0, "drums": 0.0, "bass": 0.5, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 64, "end_beat": 96, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 1.0, "other": 0.3}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 96, "end_beat": 112, "stem_gains": {"vocals": 0.0, "drums": 0.5, "bass": 0.7, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_b",
                        "key_source": "none",
                        "explanation": "I put Song A's vocals over Song B's instrumental with the bass boosted to full. Mid-frequency stems are reduced when vocals are active for clarity. The breakdown strips to bass and other for contrast before the full drop.",
                        "warnings": [],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_1", "content": "Plan accepted."}]},

        # Example 2: Vague prompt
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "mix them together, make it sound good"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Slow Jam" -- 88 BPM, Gmin, 3:15, 71 bars.\n'
                'Song B: "Upbeat" -- 125 BPM, Amin, 3:30, 109 bars.\n'
                "Vocals: dominant, +7 dB.\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Vocal source: Song B (+7 dB, cleaner)."
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
                        "start_time_vocal": 15.0,
                        "end_time_vocal": 133.0,
                        "start_time_instrumental": 10.0,
                        "end_time_instrumental": 100.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 16, "stem_gains": {"vocals": 0.0, "drums": 0.7, "bass": 0.8, "other": 0.5}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 16, "end_beat": 48, "stem_gains": {"vocals": 0.8, "drums": 0.7, "bass": 0.8, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 48, "end_beat": 80, "stem_gains": {"vocals": 1.0, "drums": 0.8, "bass": 0.9, "other": 0.3}, "transition_in": "crossfade", "transition_beats": 2},
                            {"label": "outro", "start_beat": 80, "end_beat": 96, "stem_gains": {"vocals": 0.0, "drums": 0.4, "bass": 0.5, "other": 0.5}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I chose Song B's vocals since they have better separation. The laid-back 88 BPM instrumental groove from Song A gives a chill vibe. The 'other' stem is kept low when vocals are active for clarity.",
                        "warnings": ["These songs have very different tempos, so the vocals have been slowed down to fit the beat."],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_2", "content": "Plan accepted."}]},

        # Example 3: Contradictory prompt
        {
            "role": "user",
            "content": (
                'Create a remix plan for this prompt: "I want the drums from Song A with the vocals from both songs"\n\n'
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Fast EDM" -- 140 BPM, F#min, 3:00, 105 bars.\n'
                'Song B: "Slow RnB" -- 72 BPM, Bbmaj, 4:00, 72 bars.\n'
                "Vocals: dominant, +9 dB. Tempo stretch: 48.6%."
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
                            {"label": "intro", "start_beat": 0, "end_beat": 8, "stem_gains": {"vocals": 0.0, "drums": 0.9, "bass": 0.7, "other": 0.4}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 8, "end_beat": 24, "stem_gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.5, "other": 0.3}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "drop", "start_beat": 24, "end_beat": 40, "stem_gains": {"vocals": 1.0, "drums": 0.9, "bass": 0.8, "other": 0.3}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 40, "end_beat": 48, "stem_gains": {"vocals": 0.0, "drums": 0.4, "bass": 0.4, "other": 0.4}, "transition_in": "crossfade", "transition_beats": 4},
                        ],
                        "tempo_source": "song_a",
                        "key_source": "none",
                        "explanation": "I used Song B's vocals since they have much better separation (+9 dB). The drums-only verse creates contrast before the full drop. Kept it short due to the extreme tempo difference.",
                        "warnings": [
                            "I can only use vocals from one song at a time -- I chose Song B since its vocals are much cleaner.",
                            "These songs have extremely different tempos (48.6% stretch). The vocals have been sped up significantly and may sound different.",
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
        total_beats = int(inst_meta.bpm * TARGET_REMIX_DURATION_SECONDS / 60)
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
    """Clamp total duration to 60-300 seconds."""
    if not plan.sections:
        return plan

    total_beats = plan.sections[-1].end_beat
    total_seconds = total_beats * 60 / target_bpm

    min_duration = 60
    max_duration = 300

    if total_seconds < min_duration:
        # Extend last section
        needed_beats = int(min_duration * target_bpm / 60) - total_beats
        plan.sections[-1].end_beat += max(needed_beats, 8)
        plan.warnings.append("Remix was extended to meet minimum duration.")
    elif total_seconds > max_duration:
        # Truncate
        max_beats = int(max_duration * target_bpm / 60)
        plan.sections[-1].end_beat = min(plan.sections[-1].end_beat, max_beats)
        plan.sections = [s for s in plan.sections if s.start_beat < max_beats]
        plan.warnings.append("Remix was shortened to fit maximum duration.")

    return plan


# ---------------------------------------------------------------------------
# Stretch calculation + post-plan validation
# ---------------------------------------------------------------------------

def _compute_stretch_pct(bpm_a: float, bpm_b: float) -> float:
    """Compute the approximate stretch percentage between two songs.

    Uses a simple average target BPM estimate and returns the max stretch
    either song would undergo. This matches the logic in compute_tempo_plan().
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0
    # Approximate: use the weighted midpoint approach from compute_tempo_plan
    target_bpm = bpm_b * 0.65 + bpm_a * 0.35
    stretch_a = abs(target_bpm - bpm_a) / bpm_a * 100
    stretch_b = abs(target_bpm - bpm_b) / bpm_b * 100
    return max(stretch_a, stretch_b)


def _warn_vocal_stretch_limits(plan: RemixPlan, stretch_pct: float) -> None:
    """Advisory warning if vocal sections exceed recommended bar limits at stretch ratio.

    Per spec: >12% stretch -> max 8 bars at up to 15%, max 4 bars above 15%.
    This is advisory logging only -- does NOT truncate or modify the plan.
    """
    if stretch_pct <= 12:
        return

    # Determine bar limit based on stretch percentage
    if stretch_pct <= 15:
        max_bars = 8
    else:
        max_bars = 4

    for section in plan.sections:
        # Check sections with active vocals
        vocal_gain = section.stem_gains.get("vocals", 0.0)
        if vocal_gain < 0.1:
            continue

        section_beats = section.end_beat - section.start_beat
        section_bars = section_beats // 4

        if section_bars > max_bars:
            logger.warning(
                "Vocal stretch advisory: section '%s' (bars %d-%d, %d bars) exceeds "
                "recommended %d-bar limit at %.1f%% stretch. Audio quality may degrade.",
                section.label,
                section.start_beat // 4,
                section.end_beat // 4,
                section_bars,
                max_bars,
                stretch_pct,
            )


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

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    # Pre-compute key matching decision
    _key_available, key_matching_detail = _compute_key_guidance(
        song_a_meta, song_b_meta,
    )

    # Compute total available beats (from instrumental source, approximated)
    target_bpm = max(song_a_meta.bpm, song_b_meta.bpm)
    total_available_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)

    # Compute stretch percentage for advisory context
    stretch_pct = _compute_stretch_pct(song_a_meta.bpm, song_b_meta.bpm)

    system_prompt = _build_system_prompt(
        song_a_meta, song_b_meta,
        _key_available, key_matching_detail,
        total_available_beats,
        stretch_pct=stretch_pct,
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

        # Post-plan validation: warn if vocal sections exceed stretch bar limits
        if stretch_pct is not None and stretch_pct > 12:
            _warn_vocal_stretch_limits(plan, stretch_pct)

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
    Uses the region starting at 25% into each song, capped at TARGET_REMIX_DURATION_SECONDS.
    Tempo target is the instrumental song's BPM.
    """
    # Vocal source defaults to song_a (Day 2 -- no vocal_prominence_db yet)
    vocal_src = "song_a"
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 25% into each song, up to TARGET_REMIX_DURATION_SECONDS
    v_start = vocal_meta.duration_seconds * 0.25
    v_end = min(v_start + TARGET_REMIX_DURATION_SECONDS, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.25
    i_end = min(i_start + TARGET_REMIX_DURATION_SECONDS, inst_meta.duration_seconds)

    tempo_src = "average"  # Split stretch burden between both songs
    total_beats = int(inst_meta.bpm * TARGET_REMIX_DURATION_SECONDS / 60)

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
    """Build a 5-, 6-, or 8-section fallback arrangement.

    8-section (>= 192 beats): intro -> verse -> chorus -> breakdown -> verse -> chorus -> drop -> outro
    6-section (96-191 beats): intro -> build -> main -> breakdown -> drop -> outro
    5-section (< 96 beats):   intro -> build -> main -> breakdown -> outro

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

    # 8-section extended: intro -> verse -> chorus -> breakdown -> verse -> chorus -> drop -> outro
    if total_beats >= 192:
        # Proportions: intro(~8%) verse(~15%) chorus(~12%) breakdown(~8%) verse(~15%) chorus(~12%) drop(~12%) outro(~8%)
        # ~90% allocated, rounding handled by snap
        b1 = snap_to_phrase(int(total_beats * 0.08))                    # end intro
        b2 = snap_to_phrase(int(total_beats * 0.23))                    # end verse 1
        b3 = snap_to_phrase(int(total_beats * 0.35))                    # end chorus 1
        b4 = snap_to_phrase(int(total_beats * 0.43))                    # end breakdown
        b5 = snap_to_phrase(int(total_beats * 0.58))                    # end verse 2
        b6 = snap_to_phrase(int(total_beats * 0.70))                    # end chorus 2
        b7 = snap_to_phrase(int(total_beats * 0.82))                    # end drop (outro starts)

        # Guard: ensure monotonically increasing boundaries with minimum section size
        boundaries = [b1, b2, b3, b4, b5, b6, b7]
        for idx in range(len(boundaries)):
            prev = boundaries[idx - 1] if idx > 0 else 0
            if boundaries[idx] <= prev:
                boundaries[idx] = prev + MIN_SECTION_BEATS
        # Ensure last boundary leaves room for outro
        if boundaries[-1] >= total_beats - MIN_SECTION_BEATS:
            boundaries[-1] = total_beats - MIN_SECTION_BEATS
        b1, b2, b3, b4, b5, b6, b7 = boundaries

        inst_bridge = {"drums": 0.3, "bass": 0.5, "guitar": 0.7, "piano": 0.6, "other": 0.5}

        return [
            Section(label="intro", start_beat=0, end_beat=b1,
                    stem_gains={"vocals": 0.0, **inst_intro},
                    transition_in="fade", transition_beats=4),
            Section(label="verse", start_beat=b1, end_beat=b2,
                    stem_gains={"vocals": 0.9, **inst_body},
                    transition_in="crossfade", transition_beats=min(8, (b2 - b1) // 3)),
            Section(label="drop", start_beat=b2, end_beat=b3,
                    stem_gains={"vocals": 1.0, **inst_body},
                    transition_in="crossfade", transition_beats=4),
            Section(label="breakdown", start_beat=b3, end_beat=b4,
                    stem_gains={"vocals": 0.3, **inst_breakdown},
                    transition_in="crossfade", transition_beats=min(8, (b4 - b3) // 3)),
            Section(label="verse", start_beat=b4, end_beat=b5,
                    stem_gains={"vocals": 0.9, **inst_body},
                    transition_in="crossfade", transition_beats=4),
            Section(label="drop", start_beat=b5, end_beat=b6,
                    stem_gains={"vocals": 1.0, **inst_body},
                    transition_in="crossfade", transition_beats=4),
            Section(label="breakdown", start_beat=b6, end_beat=b7,
                    stem_gains={"vocals": 0.5, **inst_bridge},
                    transition_in="crossfade", transition_beats=min(8, (b7 - b6) // 3)),
            Section(label="outro", start_beat=b7, end_beat=total_beats,
                    stem_gains={"vocals": 0.0, **inst_outro},
                    transition_in="crossfade", transition_beats=min(8, (total_beats - b7) // 2)),
        ]

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
