"""Remix plan interpreter: LLM-powered prompt interpretation with deterministic fallback.

Day 3: Converts user prompts + song metadata into structured RemixPlan objects
using Anthropic's tool_use API. Falls back to deterministic arrangement on any
LLM failure.

Day 2 fallback functions (generate_fallback_plan, default_arrangement) are
preserved at the bottom of this file.

Day 3 song structure integration: 5-layer system prompt with section maps,
stem character, cross-song relationships, lyrics, and 8 failure mode guards.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import anthropic

from musicmixer.config import settings
from musicmixer.models import VOCAL_SOURCE, AudioMetadata, IntentPlan, IntentSection, LyricsData, RemixPlan, Section
from musicmixer.services.tempo import compute_stretch_pct, estimate_material_budget, estimate_target_bpm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_STEMS = ["vocals", "drums", "bass", "guitar", "piano", "other"]

# Target remix duration in seconds. Controls beat budget, LLM guidance, and fallback plans.
TARGET_REMIX_DURATION_SECONDS = 210  # 3.5 minutes

# Max output tokens for LLM remix plan generation.
# 3072 gives comfortable headroom for 9-12 section plans (~1450 tokens typical).
LLM_MAX_TOKENS = 3072


class _DurationTooShortError(Exception):
    """Raised when the LLM's arrangement is too short for the target duration."""

    def __init__(self, plan: IntentPlan, target_bpm: float):
        self.plan = plan
        self.target_bpm = target_bpm
        total_beats = plan.sections[-1].end_beat if plan.sections else 0
        total_seconds = total_beats * 60 / target_bpm if target_bpm > 0 else 0
        super().__init__(
            f"Arrangement too short: {total_seconds:.0f}s "
            f"({total_beats} beats at {target_bpm:.0f} BPM), "
            f"target={TARGET_REMIX_DURATION_SECONDS}s"
        )


# ---------------------------------------------------------------------------
# tool_use schema
# ---------------------------------------------------------------------------

REMIX_PLAN_TOOL: dict = {
    "name": "create_remix_plan",
    "description": "Create a structured remix plan based on the user's prompt and song analysis data.",
    "input_schema": {
        "type": "object",
        "required": [
            "start_time_vocal", "end_time_vocal",
            "start_time_instrumental", "end_time_instrumental",
            "sections", "vocal_type",
            "explanation", "warnings",
        ],
        "properties": {
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
                    "required": ["label", "start_beat", "end_beat", "energy", "stem_roles", "transition_in", "transition_beats"],
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["intro", "verse", "chorus", "bridge", "breakdown", "drop", "outro"],
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
                            "description": "Ending beat (exclusive). Section length = end_beat - start_beat. Should be 4, 8, 16, 32, or 64.",
                        },
                        "energy": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "peak"],
                            "description": "Overall energy level of this section.",
                        },
                        "stem_roles": {
                            "type": "object",
                            "required": ["vocals", "drums", "bass", "guitar", "piano", "other"],
                            "description": "Role for each stem: lead, support, background, texture, or silent.",
                            "properties": {
                                "vocals": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
                                "drums": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
                                "bass": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
                                "guitar": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
                                "piano": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
                                "other": {"type": "string", "enum": ["lead", "support", "background", "texture", "silent"], "description": "lead=primary, support=audible contributor, background=fullness, texture=atmospheric, silent=absent"},
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
            # tempo_source removed — always computed algorithmically via weighted_midpoint
            "vocal_type": {
                "type": "string",
                "enum": ["sung", "rap"],
                "description": "Whether Song A's vocals are melodic/sung or rap/spoken word. Flag 'rap' if the vocals are predominantly rapped or spoken word, even if there are brief melodic hooks or ad-libs. Only use 'sung' if the vocals are primarily melodic singing throughout. When in doubt, lean toward 'rap' — pitch-shifting rap vocals sounds worse than skipping key matching on a slightly melodic track.",
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
# System prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt_block() -> dict:
    """Construct the static system prompt as a single cached content block.

    All static instruction sections are combined into one block with
    cache_control for Anthropic prompt caching. Dynamic content (song
    metadata, tempo, templates) is built separately by _build_dynamic_context()
    and injected into the user message.
    """
    # 6-stem separation (Modal / BS-RoFormer) -- hardcoded post Phase 1
    stem_list = "vocals, drums, bass, guitar, piano, other"
    stem_count = 6

    # Section 1: Role and MVP Constraints
    section_1 = f"""You are an expert music mashup artist with impeccable taste. You think like a jazz musician — every choice is intentional, every silence is earned, every transition serves the groove. You plan how to combine two songs into a mashup remix that sounds like it was always meant to exist.

You will receive two songs with detailed metadata — BPM, key, section maps, energy profiles, stem analysis, and (when available) synced lyrics. Song A is your vocal source. Song B is your instrumental source. Your goal is to create a cohesive arrangement that layers Song A's vocals over Song B's instrumentals — choosing the best regions, structuring sections with purpose, and assigning stem roles that serve the music. Use the metadata to make informed decisions, not guesses.

CONSTRAINTS:
- Vocals ALWAYS come from Song A. Instrumentals ALWAYS come from Song B. This is fixed — do not reference vocal_source in your output.
- You CANNOT mix stems across songs (e.g., no "drums from Song A with bass from Song B").
- You CANNOT add effects, generate new sounds, or use vocals from both songs.
- "other" contains synths, strings, wind instruments, and anything not captured by the {stem_count - 1} named stems.
- All end times must be greater than their corresponding start times.
- If the songs are a poor match (extreme tempo/key gap, incompatible energy profiles), acknowledge it in the `warnings` field and produce the best plan within these limits.

CAPABILITIES:
- Choose which portion of each song to use (start/end times in seconds — you don't have to use the entire song)
- Design a section-based arrangement with per-stem role assignment (lead, support, background, texture, silent for each of: {stem_list})
- Choose transitions between sections (fade, crossfade, cut)"""

    # Section 7: Critical Mixing Rules (failure mode guards)
    section_7 = """CRITICAL MIXING RULES (violations produce bad audio):
1. INSTRUMENTAL SECTIONS: Prefer sections with no vocals (vox:--, labeled GOOD INSTRUMENTAL SOURCE). For instrumental breakdowns, assign at least one stem as "lead".
2. VOCAL-INSTRUMENTAL BALANCE: When vocals are active, assign them "lead" and ensure at least 2-3 instrumental stems are "support" or "background". A mashup should sound like a FULL BAND, not a vocal solo.
3. VOCAL BLEED AWARENESS: Song A's vocal stem may contain faint drums/bass from the original mix. In low-energy sections where Song B's drums are quiet or silent, this ghost rhythm can become audible and clash. Keep Song B drums, bass, or other active stems at "support" or higher when vocals are active — they mask the bleed. The concern is sections where ALL Song B stems are quiet or "texture" while vocals play — that's where ghost rhythm becomes audible.
4. ENERGY MATCHING: Match vocal energy to instrumental energy level. Exception: quiet vocal over minimal beat is acceptable as an intentional artistic choice.
5. DYNAMIC RANGE: The remix MUST have at least 1 contrast moment (e.g., breakdown -> drop) and use a minimum of 3 different energy levels across sections.
6. ENDING: End with 4-8 bars of reduced energy or a natural outro. NEVER cut the remix at full energy -- it sounds broken.
7. ROLE VARIATION: Vary stem roles across sections. Strip down to drums+bass+vocals for contrast, then promote more stems to "support" for impact. Flat roles across all sections produces a lifeless mix.
8. LYRIC-AWARE CUTS: When lyrics are available, prefer placing section boundaries at natural lyric breaks (end of line/verse). Cross-reference Layer 5 bar numbers with Layer 2 section boundaries. If lyrics show a hook or repeated phrase, that's a prime candidate for the "drop" section.
9. VOCAL PRESENCE: Both songs carry musical identity. If Song B's instrumentals are only audible during intros and outros, the mashup is karaoke — Song B becomes wallpaper. Give Song B at least one section of 8-16 bars where it stands on its own: a breakdown, a drop, or an instrumental bridge in the middle of the arrangement. This lets the listener hear both songs as participants. If a stretch advisory is present in the dynamic context, defer to its vocal budget. Override freely when the source material or user prompt calls for vocal-forward treatment."""

    # Section 5: Stem Role Guidelines (roles, frequency awareness, energy arc, mixing advisory, phrase alignment)
    section_5 = """STEM ROLE GUIDELINES:
- Vocal sections: vocals as "lead", at least 2-3 instrumental stems as "support" or "background"
- Instrumental sections (breakdowns, intros): at least one "lead" instrumental
- Drum-bass pair: typically "support" or higher in any rhythmic section
- Vary roles across sections for dynamics: verse guitar as "background", chorus promotes it to "support"
- A mashup should sound like a full band. Err toward "background" over "silent" — use "silent" SPARINGLY

FREQUENCY AWARENESS (role assignment guide):
- Vocals occupy the mid-range (300 Hz - 5 kHz). Guitar, piano, and "other" (synths) overlap this range and WILL mask vocals if too loud. When vocals are "lead", prefer guitar/piano/other at "background" or "texture" unless the stem is rhythmic (choppy guitar) rather than sustained (pad/lead synth).
- Drums and bass rarely conflict with vocals — they are safe at "support" alongside vocal "lead". Exception: heavy sub-bass (808s, deep bass synths) can mud low male vocals or baritone singers. In bass-heavy genres (trap, hip-hop), demote bass to "background" during vocal leads if the vocal energy sits low.
- The "other" stem is the most dangerous vocal mask when it contains sustained mid-range content (synth leads, pads, sustained strings). Default it to "texture" in vocal sections unless Layer 3 shows it is low-energy or sparse. In hip-hop/R&B, "other" is often horn stabs or samples that sit fine at "background".
- In medium-energy sections, keep max 3 stems at "support" or above — push the rest to "background". Peak sections are exempt: a full-band climax with 4-5 stems at "support" is what makes it peak.

ENERGY LEVELS AND ARC:
- "low": Sparse, minimal. If leading INTO a higher section, keep 1-2 stems at "support" to maintain momentum. If fading out, everything "background" or lower.
- "medium": Working level — verses, bridges. Where most of the remix lives.
- "high": Driving, elevated — pre-chorus, energetic verses, builds before drops.
- "peak": Maximum intensity. The payoff moment. Reserve for 1-2 sections MAX — more than 2 causes fatigue and nothing feels special.

ARC PRINCIPLES:
- CONTRAST creates impact. A "peak" after "low" hits harder than "peak" after "high". Always include at least one low→peak jump.
- Match arc to template: Standard/DJ = single peak. Extended = double peak with a valley between. Chill = gentle wave, rarely exceeds "high".
- The LAST section must always be "low" or "medium". Never end at peak — it sounds broken.
- Strip stems to CREATE contrast, not just to be quiet. A breakdown with only drums+bass makes the following chorus feel massive.

MIXING ADVISORY:
- Stagger stem entries over 2-4 bars for natural-sounding builds (don't bring everything in at once). Exception: after a breakdown or silence, slamming all stems in on beat one IS the point of a drop — use a "cut" transition with full stem activation.
- Begin vocal sections 1-2 beats early for pickup notes (vocals often start before the downbeat).
- PHRASE ALIGNMENT: When choosing start_time_vocal, prefer a point where vocals begin at a phrase boundary (bar 1 of a verse or chorus in Layer 2). Cross-reference Song A's section map to ensure the vocal entry lands on a strong downbeat in your arrangement. Vocal phrases typically run in 4-bar or 8-bar groups.
- Section labels in the song data are approximate guidance, not rigid constraints. Use them to understand song structure, but your arrangement should serve the remix.
- Contrast creates energy: if a section has drums as "silent", the next section's drums at "support" will feel powerful."""

    # Section 2: Arrangement Rules
    section_2 = """ARRANGEMENT RULES (for the `sections` array in your remix plan):
- Each entry should be 4, 8, 16, 32, or 64 beats long (max 64 beats per entry)
- Default: start with instrumental only (establishes the beat before vocals enter)
- Always end with instrumental only or a fade
- transition_beats must be less than half the entry length, and never more than 8 beats. Long crossfades destroy punch.
- Label meanings: "chorus" = vocal-led high energy. "drop" = instrumental-led high energy. "bridge" = transitional. "breakdown" = energy decreasing."""

    # Section 6: Genre Guidance
    section_6 = """GENRE GUIDANCE (infer from BPM + energy profile + section map):
- Hip-hop/boom-bap (80-100 BPM, straight): Consistent drums throughout. Build energy through vocal intensity and layering, not drum drops.
- Trap/modern hip-hop (130-160 BPM, reported as full-time): Sparse hi-hats, heavy 808 bass. Half-time feel — the kick hits every other beat, so it SOUNDS like 65-80 BPM despite the metadata showing double that. Use breakdown->drop sparingly; energy comes from bass and vocal flow.
- R&B/soul (70-110 BPM): Smooth transitions, no abrupt changes. Layer elements gradually. Vocals are always the star.
- Pop/rock (100-140 BPM): Verse-chorus dynamics — stripped for verses, full for choruses. Guitar often drives energy shifts.
- EDM/dance (120-160 BPM): Breakdown -> build -> drop. Align drops with sections annotated DROP. The "other" stem often carries the main synth hook.
- Jam/rock (variable BPM): Extended instrumental sections. Vocal gaps are natural entry points.
- If BPM alone is ambiguous (e.g., 130 BPM could be pop, EDM, or trap), use the section map and energy profile to disambiguate. Trap has sparse density; EDM has full+extra density at drops; pop has verse-chorus alternation."""

    # Section 8: Stem Separation Artifacts
    section_8 = """STEM SEPARATION ARTIFACTS:
Stem separation leaves residual bleed — ghost vocals in instrumental stems, instrument traces in vocal stems. Song B's per-section vox: values show how prominent vocals were in the original mix; higher values mean more bleed in that section's instrumental stems. Sections marked vox:-- or GOOD INSTRUMENTAL SOURCE have the least bleed. Active stems — especially drums and bass — naturally mask bleed, so it matters most in sparse, exposed passages."""

    # Section 3: Transitions (static definitions; stretch advisory is dynamic)
    section_3 = """TRANSITIONS:
- "cut": Hard switch with no overlap. Best for maximum impact when moving UP in energy (breakdown-to-drop, build-to-chorus) or for same-energy lateral transitions (verse-to-verse). Avoid for large energy drops — sounds broken.
- "crossfade": Gradual blend over transition_beats. Default choice — works for ascending, descending, and same-level transitions. Prefer over cut when energy change is gradual.
- "fade": Volume ramp from/to silence. Use for the first section (fade in) and last section (fade out). Also works for bringing vocals in from nothing.
- Transitions should land on bar boundaries (multiples of 4 beats). A crossfade starting mid-bar sounds sloppy."""

    # Section 4: Arrangement Approach (reference patterns, not rigid templates)
    section_4 = """ARRANGEMENT APPROACH:
Build the arrangement from the source material outward. Examine both songs' section maps, phrase lengths, and energy profiles FIRST, then design sections that align with natural phrase boundaries (4, 8, 16, or 32 bars).

Reference arrangements (use as inspiration, not rigid structures):
- Standard Mashup: intro -> verse -> chorus -> breakdown -> chorus -> outro
- DJ Set: intro -> verse -> breakdown -> drop -> outro
- Quick Hit (short remixes): intro -> chorus -> verse -> chorus -> outro
- Chill: intro -> verse -> bridge -> verse -> outro
- Extended Mix: intro -> verse -> chorus -> breakdown -> verse -> chorus -> bridge -> drop -> outro

Every section boundary must land on a phrase boundary from at least one source song. Section durations come from the source material's natural phrase lengths, not fixed percentages. Prefer fewer, longer sections over many short ones — minimum 8 bars per section. Short remixes (under 48 total bars) should have 3-4 sections; longer remixes can have up to 8. You may combine, reorder, or omit sections from the references above."""

    # Section 9: Tempo Rules and Vocal Type
    section_9 = """TEMPO MATCHING:
Tempo is handled automatically by the system using an algorithm that balances vocal and instrumental stretch.
You do NOT choose tempo_source — it is not in your tool schema. Focus on arrangement and stem roles.
If the BPM gap is large (>20%), mention in your explanation that some tempo stretching was applied.

VOCAL TYPE:
Classify Song A's vocal_type as 'rap' if the vocals are predominantly rapped or spoken word, even with brief melodic hooks or ad-libs. Only use 'sung' if the vocals are primarily melodic singing throughout. When in doubt, lean toward 'rap' — a 7-semitone pitch shift on rap vocals sounds far worse than skipping key matching on a slightly melodic track."""

    # Section 10: Explanation and Warnings
    section_10 = """EXPLANATION: Write 2-3 non-technical sentences explaining what you did and why. No internal jargon. This is shown directly to the user.

WARNINGS: Populate this array ONLY for issues that actually degrade audio quality:
- Tempo/key gap is large and the remix may sound noticeably different from the originals
- Key shift exceeds 3 semitones
Do NOT warn about normal characteristics like one song having more vocals than the other — that is expected (Song A provides vocals, Song B provides instrumentals). Empty array is fine."""

    # Ordering principle: definitions first, rules after.
    # Block 1 (ontology): role → arrangement rules → transitions → arrangement approach → stem roles → genre
    # Block 2 (constraints): guards → artifact awareness → tempo/key rules → explanation/warnings
    static_sections = [
        section_1, section_2, section_3, section_4, section_5, section_6,
        section_7, section_8, section_9, section_10,
    ]
    return {
        "type": "text",
        "text": "\n\n".join(static_sections),
        "cache_control": {"type": "ephemeral"},
    }


def _build_dynamic_context(
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
    total_available_beats: int,
    stretch_pct: float | None = None,
    lyrics_a: LyricsData | None = None,
    lyrics_b: LyricsData | None = None,
    material_duration: float | None = None,
    ideal_beats: int | None = None,
) -> str:
    """Build dynamic context string for injection into the user message.

    Contains per-request content: song data layers, stretch advisory,
    and duration target. This content varies between requests and is NOT
    cached in the system prompt.
    """
    # Compute per-song beat counts (approximate)
    total_beats_a = song_a_meta.total_beats
    total_beats_b = song_b_meta.total_beats

    # Approximate target BPM for beat-to-seconds conversion.
    target_bpm = estimate_target_bpm(
        vocal_bpm=song_a_meta.bpm,
        instrumental_bpm=song_b_meta.bpm,
    )

    # Dynamic: Stretch advisory (conditional, appended only when stretch > 12%)
    stretch_section = ""
    if stretch_pct is not None and stretch_pct > 12:
        if stretch_pct <= 15:
            impact = "moderate — artifacts noticeable on sustained vocal notes"
            vocal_budget = "40%"
        else:
            impact = "significant — artifacts clearly audible throughout vocals"
            vocal_budget = "30%"
        stretch_section = f"""STRETCH ADVISORY ({stretch_pct:.1f}% stretch on vocals):
Vocal time-stretching above 12% introduces audible artifacts (phasiness, vowel warble). At {stretch_pct:.1f}%, quality impact is {impact}.

Arrangement guidance to minimize degradation:
1. VOCAL BUDGET: Keep vocal sections to ≤{vocal_budget} of total beats. Fill remaining time with instrumental sections (intro, breakdown, outro).
2. FRAMING: Open and close with instrumental sections — first and last impressions must be clean.
3. VOCAL SECTION LENGTH: Prefer shorter vocal sections (4-8 bars) over long ones. The ear locks onto stretch artifacts over time; breaks reset perception.
4. MASKING: Back vocal sections with full, busy instrumental beds (drums + bass + harmony). Sparse accompaniment exposes artifacts. Avoid vocals over minimal or solo-instrument backing.
5. ENERGY: Place vocals in higher-energy sections where the dense mix provides natural masking.

These are quality-driven defaults. Override with musical judgment if the prompt demands a vocal-focused arrangement."""

    # Dynamic: Duration target (varies per song pair)
    min_ref = min(TARGET_REMIX_DURATION_SECONDS, material_duration) if material_duration is not None else TARGET_REMIX_DURATION_SECONDS
    duration_section = f"""DURATION: Target = {TARGET_REMIX_DURATION_SECONDS}s = ~{total_available_beats} beats at {target_bpm:.0f} BPM (1 beat = {60 / target_bpm:.2f}s, 1 bar = 4 beats).
Arrangements shorter than {int(min_ref * 0.7)}s will be REJECTED."""

    if ideal_beats is not None and total_available_beats < ideal_beats and material_duration is not None:
        duration_section += (
            f"\nNOTE: Source material limits this remix to ~{total_available_beats} beats "
            f"(~{material_duration:.0f}s) after tempo matching. "
            f"Do NOT plan beyond {total_available_beats} beats."
        )

    # Dynamic: Song Data (5 layers)
    song_a_info = _build_song_info("Song A", song_a_meta, total_beats_a)
    song_b_info = _build_song_info("Song B", song_b_meta, total_beats_b)
    cross_song = _build_cross_song_layer(song_a_meta, song_b_meta, stretch_pct)

    song_data_parts = [
        "=== LAYER 1: SONG OVERVIEW ===",
        song_a_info,
        song_b_info,
    ]

    has_section_data = (
        getattr(song_a_meta, "song_structure", None) is not None
        or getattr(song_b_meta, "song_structure", None) is not None
    )
    if has_section_data:
        song_data_parts.append(
            "\n=== LAYER 2: SECTION MAP ===\n"
            "Columns: bars, duration, time, label, energy, density, vocal prominence.\n"
            "vox:+XdB = vocals X dB above accompaniment (higher = more dominant). "
            "vox:-- = no vocals detected. (fading) = vocal exit transition."
        )
        struct_a = getattr(song_a_meta, "song_structure", None)
        if struct_a and struct_a.sections:
            section_map_a = _build_section_map("Song A", struct_a, total_beats_a // 4)
            song_data_parts.append(section_map_a)
        struct_b = getattr(song_b_meta, "song_structure", None)
        if struct_b and struct_b.sections:
            section_map_b = _build_section_map("Song B", struct_b, total_beats_b // 4)
            song_data_parts.append(section_map_b)

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

    if cross_song:
        song_data_parts.append("\n=== LAYER 4: CROSS-SONG ===")
        song_data_parts.append(cross_song)

    lyrics_layer = _build_lyrics_layer(lyrics_a, lyrics_b)
    if lyrics_layer:
        song_data_parts.append(lyrics_layer)

    song_data = "SONG DATA:\n\n" + "\n".join(song_data_parts)

    # Song data first so the LLM grounds itself in the material before reading rules.
    # Only truly per-request content remains dynamic.
    dynamic_sections = [song_data]
    if stretch_section:
        dynamic_sections.append(stretch_section)
    dynamic_sections.append(duration_section)
    return "\n\n".join(dynamic_sections)


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

        # Render vocal column with dB values when available
        if sec.vocal_prominence_db is not None and sec.vocal_status == "vox:fading":
            vox_display = f"vox:{sec.vocal_prominence_db:+.0f}dB(fading)"
        elif sec.vocal_prominence_db is not None:
            vox_display = f"vox:{sec.vocal_prominence_db:+.0f}dB"
        elif sec.vocal_prominence_db is None and sec.vocal_status == "vox:no":
            vox_display = "vox:--"
        elif sec.vocal_prominence_db is None and sec.vocal_status == "vox:fading":
            vox_display = "vox:fading"
        else:
            vox_display = sec.vocal_status

        lines.append(
            f"  {sec.start_bar:>3}-{sec.end_bar:<3}  "
            f"{sec.bar_count:>2}b  "
            f"{t_start}-{t_end} | "
            f"{sec.label:<14} | "
            f"{energy_display:<16} | "
            f"{sec.density:<10} | "
            f"{vox_display:<18}"
            f"{annotations_str}"
        )

    if structure.vocal_gaps:
        gap_strs = [f"{g.start_bar}-{g.end_bar}" for g in structure.vocal_gaps]
        lines.append(f"Vocal gaps: {', '.join(gap_strs)}")

    return "\n".join(lines)


def _build_stem_character(
    label: str,
    meta: AudioMetadata,
) -> str:
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

    result = f"{label} stems: " + ". ".join(stem_descs) + "."
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


def _build_lyrics_layer(
    lyrics_a: LyricsData | None,
    lyrics_b: LyricsData | None,
    max_lines_per_song: int = 60,
) -> str:
    """Build Layer 5: Lyrics text for the system prompt.

    Formats synced lyrics with bar numbers, plain lyrics with just text.
    Caps at max_lines_per_song per song; samples evenly if longer.
    Returns empty string if no lyrics exist for either song.
    """
    if not lyrics_a and not lyrics_b:
        return ""

    parts: list[str] = [
        "\n=== LAYER 5: LYRICS ===",
        "Use these lyrics to avoid cutting mid-phrase, identify hooks, and match themes to the prompt.",
    ]

    for label, lyrics in [("Song A", lyrics_a), ("Song B", lyrics_b)]:
        if not lyrics or not lyrics.lines:
            parts.append(f"\n{label} lyrics: no lyrics found.")
            continue

        sync_label = "synced" if lyrics.is_synced else "plain"
        parts.append(f"\n{label} lyrics ({sync_label}, {len(lyrics.lines)} lines):")

        lines = lyrics.lines
        # Sample evenly if too many lines
        if len(lines) > max_lines_per_song:
            step = len(lines) / max_lines_per_song
            lines = [lines[int(i * step)] for i in range(max_lines_per_song)]

        for line in lines:
            if line.bar_number is not None:
                parts.append(f"  bar {line.bar_number:>3}: {line.text}")
            else:
                parts.append(f"  {line.text}")

    return "\n".join(parts)


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

    # Instrumental sections from Song B (fixed: Song B provides instrumentals)
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
# Few-shot examples
# ---------------------------------------------------------------------------

def _build_few_shot_messages() -> list[dict]:
    """Build 2 few-shot examples showing how to interpret the 5-layer song data.

    Examples demonstrate:
    A. "Bread and Butter": Well-matched songs, full metadata + lyrics, 8 sections,
       progressive stem variation, chorus labels, lyric-aware placement.
    B. "Edge Case": Moderate tempo gap, sparse metadata (no lyrics),
       6 sections, sparser arrangements, bridge label.
    """
    # The actual default prompt used when user provides nothing
    default_prompt = (
        "Create a mashup using vocals from Song A over the instrumentals "
        "from Song B. Analyze the song structures and make smart arrangement decisions."
    )

    return [
        # Example A: "Bread and Butter" — well-matched songs, full data, lyrics
        {
            "role": "user",
            "content": (
                f'Create a remix plan for this prompt: "{default_prompt}"\n\n'
                "DURATION: Target = 210s = ~413 beats at 118 BPM (1 beat = 0.51s, 1 bar = 4 beats).\n"
                "Arrangements shorter than 147s will be REJECTED.\n\n"
                "SONG DATA:\n\n"
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Night Ride" -- 120 BPM, Cmin, 4:00, 120 bars.\n'
                "Energy: compressed.\n"
                'Song B: "City Groove" -- 118 BPM, Cmaj, 3:30, 103 bars.\n'
                "Energy: wide dynamic range.\n\n"
                "=== LAYER 2: SECTION MAP ===\n"
                "Columns: bars, duration, time, label, energy, density, vocal prominence.\n"
                "vox:+XdB = vocals X dB above accompaniment (higher = more dominant). "
                "vox:-- = no vocals detected. (fading) = vocal exit transition.\n"
                "Song A (120 bars):\n"
                "    1-8    8b  0:00-0:16 | intro          | medium           | mid        | vox:--              | GOOD INSTRUMENTAL SOURCE\n"
                "    9-40  32b  0:16-1:20 | verse          | high             | full       | vox:+8dB\n"
                "   41-56  16b  1:20-1:52 | chorus         | high             | full+extra | vox:+8dB            | DROP\n"
                "   57-72  16b  1:52-2:24 | verse          | high             | full       | vox:+8dB\n"
                "   73-88  16b  2:24-2:56 | chorus         | high             | full+extra | vox:+8dB\n"
                "   89-104 16b  2:56-3:28 | breakdown      | high->medium     | mid        | vox:+6dB\n"
                "  105-120 16b  3:28-4:00 | outro          | medium->low      | sparse     | vox:+4dB(fading)\n"
                "Vocal gaps: 1-8\n"
                "Song B (103 bars):\n"
                "    1-8    8b  0:00-0:16 | intro          | low              | sparse     | vox:--              | GOOD INSTRUMENTAL SOURCE\n"
                "    9-40  32b  0:16-1:21 | verse          | medium           | mid        | vox:+3dB\n"
                "   41-72  32b  1:21-2:26 | instrumental   | high->peak       | full+extra | vox:--              | GOOD INSTRUMENTAL SOURCE\n"
                "   73-88  16b  2:26-2:58 | verse          | medium           | mid        | vox:+3dB\n"
                "   89-103 15b  2:58-3:30 | outro          | medium->low      | sparse     | vox:--\n"
                "Vocal gaps: 1-8, 41-72, 89-103\n\n"
                "=== LAYER 3: STEM CHARACTER ===\n"
                "Song A stems: vocals: high-energy, full. drums: high-energy, full. bass: high-energy, full. other: high-energy, full. (guitar: negligible | piano: minor)\n"
                "Song B stems: guitar: high-energy, full. drums: high-energy, full. bass: medium-energy, mid. vocals: medium-energy, mid. piano: low-energy, sparse. (other: minimal)\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: Song A ~5 dB louder. Reduce Song A stems ~5 dB.\n"
                "Instrumental source: Song B clean sections at bars 1-8, 41-72, 89-103.\n\n"
                "=== LAYER 5: LYRICS ===\n"
                "Use these lyrics to avoid cutting mid-phrase, identify hooks, and match themes to the prompt.\n\n"
                "Song A lyrics (synced, 12 lines):\n"
                "  bar   9: Rolling through the city late at night\n"
                "  bar  11: Every streetlight burning way too bright\n"
                "  bar  14: Said she wanna ride so we ride all night\n"
                "  bar  17: Windows down and feeling so alive\n"
                "  bar  41: Roll with me roll with me now\n"
                "  bar  43: Roll with me roll with me now\n"
                "  bar  45: Everybody knows that we run this town\n"
                "  bar  47: Roll with me roll with me now\n"
                "  bar  57: Back on the block where it all began\n"
                "  bar  59: Same old crew with the same old plan\n"
                "  bar  73: Roll with me roll with me now\n"
                "  bar  75: Everybody knows that we run this town\n"
                "Song B lyrics: no lyrics found."
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
                        "start_time_vocal": 0.0,
                        "end_time_vocal": 220.0,
                        "start_time_instrumental": 0.0,
                        "end_time_instrumental": 210.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 32, "energy": "low", "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "intro", "start_beat": 32, "end_beat": 64, "energy": "medium", "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 64, "end_beat": 128, "energy": "medium", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "chorus", "start_beat": 128, "end_beat": 192, "energy": "high", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "breakdown", "start_beat": 192, "end_beat": 224, "energy": "low", "stem_roles": {"vocals": "silent", "drums": "background", "bass": "support", "guitar": "lead", "piano": "background", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 224, "end_beat": 288, "energy": "medium", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "chorus", "start_beat": 288, "end_beat": 368, "energy": "peak", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 368, "end_beat": 416, "energy": "low", "stem_roles": {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "background", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "vocal_type": "sung",
                        "explanation": "Night Ride's vocals over City Groove's beat with a warm, building arrangement. The hook ('Roll with me') lands at the first chorus for impact, and the second verse promotes guitar from background to support for variety.",
                        "warnings": [],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_1", "content": "Plan accepted."}]},

        # Example B: "Edge Case" — moderate tempo gap, sparse metadata, no lyrics
        {
            "role": "user",
            "content": (
                f'Create a remix plan for this prompt: "{default_prompt}"\n\n'
                "DURATION: Target = 210s = ~308 beats at 88 BPM (1 beat = 0.68s, 1 bar = 4 beats).\n"
                "Arrangements shorter than 147s will be REJECTED.\n\n"
                "SONG DATA:\n\n"
                "=== LAYER 1: SONG OVERVIEW ===\n"
                'Song A: "Slow Jam" -- 88 BPM, Gmin, 3:45, 82 bars.\n'
                "Energy: moderate dynamics.\n"
                'Song B: "Upbeat Track" -- 98 BPM, Amin, 3:30, 86 bars.\n'
                "Energy: wide dynamic range.\n\n"
                "=== LAYER 2: SECTION MAP ===\n"
                "Columns: bars, duration, time, label, energy, density, vocal prominence.\n"
                "vox:+XdB = vocals X dB above accompaniment (higher = more dominant). "
                "vox:-- = no vocals detected. (fading) = vocal exit transition.\n"
                "Song B (86 bars):\n"
                "    1-8    8b  0:00-0:15 | intro          | low              | sparse     | vox:--              | GOOD INSTRUMENTAL SOURCE\n"
                "    9-32  24b  0:15-0:57 | verse          | medium           | mid        | vox:+4dB\n"
                "   33-48  16b  0:57-1:21 | chorus         | high             | full       | vox:+4dB            | DROP\n"
                "   49-64  16b  1:21-1:45 | instrumental   | high             | full       | vox:--              | GOOD INSTRUMENTAL SOURCE\n"
                "   65-80  16b  1:45-2:09 | verse          | medium           | mid        | vox:+4dB\n"
                "   81-86   6b  2:09-2:30 | outro          | medium->low      | sparse     | vox:--\n"
                "Vocal gaps: 1-8, 49-64, 81-86\n\n"
                "=== LAYER 4: CROSS-SONG ===\n"
                "Loudness: similar levels.\n"
                "Instrumental source: Song B clean sections at bars 1-8, 49-64, 81-86.\n"
                "Tempo stretch: 10.2%."
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
                        "start_time_vocal": 0.0,
                        "end_time_vocal": 225.0,
                        "start_time_instrumental": 0.0,
                        "end_time_instrumental": 210.0,
                        "sections": [
                            {"label": "intro", "start_beat": 0, "end_beat": 32, "energy": "low", "stem_roles": {"vocals": "silent", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "fade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 32, "end_beat": 96, "energy": "medium", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "bridge", "start_beat": 96, "end_beat": 128, "energy": "medium", "stem_roles": {"vocals": "background", "drums": "support", "bass": "support", "guitar": "lead", "piano": "background", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "verse", "start_beat": 128, "end_beat": 192, "energy": "medium", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 4},
                            {"label": "chorus", "start_beat": 192, "end_beat": 264, "energy": "high", "stem_roles": {"vocals": "lead", "drums": "support", "bass": "support", "guitar": "support", "piano": "background", "other": "background"}, "transition_in": "cut", "transition_beats": 0},
                            {"label": "outro", "start_beat": 264, "end_beat": 312, "energy": "low", "stem_roles": {"vocals": "silent", "drums": "background", "bass": "background", "guitar": "background", "piano": "texture", "other": "texture"}, "transition_in": "crossfade", "transition_beats": 8},
                        ],
                        "vocal_type": "sung",
                        "explanation": "Slow Jam's vocals sit over Upbeat Track's instrumental, keeping a relaxed feel. The bridge section uses guitar as lead for contrast before the final vocal chorus.",
                        "warnings": ["Song B's tempo was adjusted ~10% to match Song A. Minor artifacts may be audible in sustained instruments."],
                    },
                }
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "example_2", "content": "Plan accepted.", "cache_control": {"type": "ephemeral"}}]},
    ]


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_intent_plan(
    raw: dict,
) -> IntentPlan:
    """Parse the raw tool_use output dict into an IntentPlan model."""
    sections = []
    for s in raw["sections"]:
        sections.append(IntentSection(
            label=s["label"],
            start_beat=int(s["start_beat"]),
            end_beat=int(s["end_beat"]),
            energy=s["energy"],
            stem_roles={k: str(v) for k, v in s["stem_roles"].items()},
            transition_in=s["transition_in"],
            transition_beats=int(s["transition_beats"]),
        ))

    # Log what LLM would have chosen for tempo (data collection), but always use algorithmic
    llm_tempo = raw.get("tempo_source", "not_provided")
    if llm_tempo != "not_provided":
        logger.info("LLM suggested tempo_source=%r (ignored, using weighted_midpoint)", llm_tempo)

    return IntentPlan(
        start_time_vocal=float(raw["start_time_vocal"]),
        end_time_vocal=float(raw["end_time_vocal"]),
        start_time_instrumental=float(raw["start_time_instrumental"]),
        end_time_instrumental=float(raw["end_time_instrumental"]),
        sections=sections,
        vocal_type=raw.get("vocal_type", "sung"),
        explanation=raw["explanation"],
        warnings=list(raw.get("warnings", [])),
    )


# ---------------------------------------------------------------------------
# Post-LLM validation
# ---------------------------------------------------------------------------

def _validate_intent_plan(
    plan: IntentPlan,
    song_a_meta: AudioMetadata,
    song_b_meta: AudioMetadata,
    material_duration: float | None = None,
) -> IntentPlan:
    """Validate and fix the LLM's intent plan. Fixes structural issues in-place where possible.

    Gain-specific validation (minimum active stems, muting rate, etc.) is now
    the gain mapper's responsibility. This function only validates structural
    correctness: time ranges, beat contiguity, section lengths, and duration.
    """
    clamped_fields: list[str] = []

    # Time range validation — Song A always provides vocals, Song B always provides instrumentals
    vocal_meta = song_a_meta
    inst_meta = song_b_meta

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

    # Section validation
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
        # Completely unrecoverable -- caller should use fallback
        plan.sections = sections
        plan.warnings.append("Section arrangement was invalid (no sections >= 4 beats).")
        return plan

    # 5. transition_beats <= (end_beat - start_beat) / 2
    for s in sections:
        max_transition = (s.end_beat - s.start_beat) // 2
        if s.transition_beats > max_transition:
            s.transition_beats = max_transition
            clamped_fields.append(f"transition_beats_{s.label}")

    # 6. stem_roles keys (add missing with default "texture", remove unknown)
    valid_stems = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    valid_roles = {"lead", "support", "background", "texture", "silent"}
    for s in sections:
        for stem in valid_stems:
            if stem not in s.stem_roles:
                s.stem_roles[stem] = "texture"
        s.stem_roles = {k: v for k, v in s.stem_roles.items() if k in valid_stems}
        # Validate role values
        for stem, role in s.stem_roles.items():
            if role not in valid_roles:
                s.stem_roles[stem] = "texture"
                clamped_fields.append(f"role_{s.label}_{stem}")

    # 7. At least 2 sections
    if len(sections) < 2:
        total_beats = sections[0].end_beat
        half = total_beats // 2
        # Snap to bar boundary
        half = (half // 4) * 4
        if half < 4:
            half = 4
        intro_roles = {**sections[0].stem_roles, "vocals": "silent"}
        intro = IntentSection("intro", 0, half, "low", intro_roles, "fade", 4)
        main = sections[0]
        main.start_beat = half
        sections = [intro, main]

    # 8. Last section end_beat on bar boundary (multiple of 4)
    last = sections[-1]
    remainder = last.end_beat % 4
    if remainder != 0:
        last.end_beat += (4 - remainder)

    plan.sections = sections

    if clamped_fields:
        logger.info("Section validation clamped fields: %s", clamped_fields)

    # Duration validation — Song A is always vocal, Song B is always instrumental
    target_bpm = estimate_target_bpm(song_a_meta.bpm, song_b_meta.bpm)

    # Pre-render logging: arrangement stats before duration validation
    total_beats = sections[-1].end_beat if sections else 0
    estimated_seconds = total_beats * 60 / target_bpm if target_bpm > 0 else 0
    logger.info(
        "Arrangement validated: %d sections, %d total beats, "
        "estimated %.0fs at %.0f BPM (target: %ds)",
        len(sections), total_beats, estimated_seconds,
        target_bpm, TARGET_REMIX_DURATION_SECONDS,
    )

    plan, duration_ok = _validate_intent_duration(plan, target_bpm, material_duration=material_duration)
    if not duration_ok:
        raise _DurationTooShortError(plan, target_bpm)

    return plan


def _validate_intent_duration(
    plan: IntentPlan, target_bpm: float,
    material_duration: float | None = None,
) -> tuple[IntentPlan, bool]:
    """Validate arrangement duration against target range.

    Returns (plan, is_acceptable). If is_acceptable is False, the caller
    should retry the LLM with duration feedback rather than blindly extending.

    When material_duration is provided and is shorter than the default target,
    the minimum duration threshold is based on the material budget instead.
    """
    if not plan.sections:
        return plan, False

    total_beats = plan.sections[-1].end_beat
    total_seconds = total_beats * 60 / target_bpm

    # When material budget is much shorter than target, adjust reference
    reference_duration = TARGET_REMIX_DURATION_SECONDS
    if material_duration is not None:
        reference_duration = min(TARGET_REMIX_DURATION_SECONDS, material_duration)
    min_duration = reference_duration * 0.7
    max_duration = TARGET_REMIX_DURATION_SECONDS * 1.5   # 315s

    if total_seconds < min_duration:
        logger.warning(
            "Arrangement too short: %.0fs (%.0f beats at %.0f BPM), "
            "min=%.0fs, target=%ds",
            total_seconds, total_beats, target_bpm,
            min_duration, TARGET_REMIX_DURATION_SECONDS,
        )
        return plan, False
    elif total_seconds > max_duration:
        max_beats = int(max_duration * target_bpm / 60)
        plan.sections = [s for s in plan.sections if s.start_beat < max_beats]
        if not plan.sections:
            return plan, False
        plan.sections[-1].end_beat = min(plan.sections[-1].end_beat, max_beats)
        plan.warnings.append("Remix was shortened to fit maximum duration.")
        return plan, True

    return plan, True


# ---------------------------------------------------------------------------
# Stretch calculation + post-plan validation
# ---------------------------------------------------------------------------

def _compute_stretch_pct(bpm_a: float, bpm_b: float) -> float:
    """Compute the approximate stretch percentage between two songs.

    Delegates to the shared tempo module (single source of truth).
    Assumes song_a=vocal, song_b=instrumental as default convention.
    """
    return compute_stretch_pct(vocal_bpm=bpm_a, instrumental_bpm=bpm_b)


def _warn_vocal_stretch_limits(plan: IntentPlan, stretch_pct: float) -> None:
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
        # Check sections with active vocals (any role other than "silent")
        vocal_role = section.stem_roles.get("vocals", "silent")
        if vocal_role == "silent":
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
    prompt: str = "",
    song_a_meta: AudioMetadata = None,
    song_b_meta: AudioMetadata = None,
    lyrics_a: LyricsData | None = None,
    lyrics_b: LyricsData | None = None,
) -> IntentPlan | RemixPlan:
    """Convert user prompt + song metadata into a structured IntentPlan.

    Returns IntentPlan (musical intent with stem roles + energy levels) on
    LLM success. The gain mapper module converts IntentPlan -> RemixPlan.

    Falls back to generate_fallback_plan() on any LLM failure, which returns
    a RemixPlan directly (bypasses gain mapping). Callers should check
    isinstance() or used_fallback to determine which path was taken.

    Synchronous -- runs in the pipeline thread, NOT the async event loop.

    When no prompt is provided, uses a default prompt that lets the LLM
    analyze song structure and make intelligent mixing decisions.
    """
    # Default prompt when user doesn't provide one
    if not prompt or not prompt.strip():
        prompt = "Create a mashup using vocals from Song A over the instrumentals from Song B. Analyze the song structures and make smart arrangement decisions."
        logger.info("No user prompt provided, using default prompt for LLM interpretation")

    # Guard: interpreter requires 6-stem separation (Modal)
    if settings.stem_backend != "modal":
        raise ValueError(
            f"stem_backend={settings.stem_backend!r} is not supported. "
            "The interpreter requires 6-stem separation (stem_backend='modal')."
        )

    # Guard: if no API key configured, skip LLM entirely
    if not settings.anthropic_api_key:
        logger.warning("No ANTHROPIC_API_KEY configured, using fallback plan")
        return generate_fallback_plan(song_a_meta, song_b_meta)

    client = anthropic.Anthropic(
        api_key=settings.anthropic_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    # Compute total available beats (Song A=vocal, Song B=instrumental).
    target_bpm = estimate_target_bpm(
        vocal_bpm=song_a_meta.bpm,
        instrumental_bpm=song_b_meta.bpm,
    )
    ideal_beats = int(target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    material_duration = estimate_material_budget(
        vocal_bpm=song_a_meta.bpm,
        vocal_duration=song_a_meta.duration_seconds,
        instrumental_bpm=song_b_meta.bpm,
        instrumental_duration=song_b_meta.duration_seconds,
        target_bpm=target_bpm,
    )
    total_available_beats = min(ideal_beats, int(target_bpm * material_duration / 60))

    # Compute stretch percentage for advisory context
    stretch_pct = compute_stretch_pct(song_a_meta.bpm, song_b_meta.bpm)

    system_blocks = [_build_system_prompt_block()]

    # Build dynamic context for user message
    dynamic_context = _build_dynamic_context(
        song_a_meta, song_b_meta,
        total_available_beats,
        stretch_pct=stretch_pct,
        lyrics_a=lyrics_a,
        lyrics_b=lyrics_b,
        material_duration=material_duration,
        ideal_beats=ideal_beats,
    )

    # Build messages: few-shot examples + user prompt with dynamic context
    user_content = f'{dynamic_context}\n\nCreate a remix plan for this prompt: "{prompt}"'
    messages = _build_few_shot_messages() + [
        {"role": "user", "content": user_content},
    ]

    tool_schema = REMIX_PLAN_TOOL

    # Log request
    logger.info(
        "LLM request: prompt=%r, song_a_bpm=%.1f, song_b_bpm=%.1f, model=%s",
        prompt, song_a_meta.bpm, song_b_meta.bpm, settings.llm_model,
    )

    start = time.monotonic()
    max_duration_attempts = 2  # 1 original + 1 retry for short duration

    for attempt in range(max_duration_attempts):
        # --- LLM call (with existing transient error retry) ---
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=LLM_MAX_TOKENS,
                system=system_blocks,
                messages=messages,
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": "create_remix_plan"},
            )
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 500, 529) and settings.llm_max_retries > 0:
                logger.warning("LLM transient error %d, retrying", e.status_code)
                time.sleep(2)
                try:
                    response = client.messages.create(
                        model=settings.llm_model,
                        max_tokens=LLM_MAX_TOKENS,
                        system=system_blocks,
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

        # Log cache stats and cost
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cache_read = getattr(usage, "cache_read_input_tokens", 0)
        cache_created = getattr(usage, "cache_creation_input_tokens", 0)

        # Sonnet 4 pricing: $3/MTok input, $15/MTok output,
        # $0.30/MTok cache read, $3.75/MTok cache write
        uncached_input = input_tokens - cache_read - cache_created
        cost_input = uncached_input * 3.0 / 1_000_000
        cost_cache_read = cache_read * 0.30 / 1_000_000
        cost_cache_write = cache_created * 3.75 / 1_000_000
        cost_output = output_tokens * 15.0 / 1_000_000
        cost_total = cost_input + cost_cache_read + cost_cache_write + cost_output

        logger.info(
            "LLM cost: $%.4f (in=%d/$%.4f, cache_read=%d/$%.4f, "
            "cache_write=%d/$%.4f, out=%d/$%.4f)",
            cost_total,
            uncached_input, cost_input,
            cache_read, cost_cache_read,
            cache_created, cost_cache_write,
            output_tokens, cost_output,
        )

        # Check stop reason
        if response.stop_reason == "max_tokens":
            logger.warning(
                "LLM hit max_tokens (attempt %d/%d, output=%d tokens)",
                attempt + 1, max_duration_attempts, output_tokens,
            )
            partial_block = next(
                (b for b in response.content if b.type == "tool_use"), None
            )
            if partial_block is not None:
                try:
                    partial_plan = _parse_intent_plan(partial_block.input)
                    partial_plan = _validate_intent_plan(partial_plan, song_a_meta, song_b_meta)
                    logger.info(
                        "Salvaged partial plan from truncated response: %d sections",
                        len(partial_plan.sections),
                    )
                    partial_plan.warnings.append("Plan recovered from truncated LLM response.")
                    return partial_plan
                except _DurationTooShortError:
                    if attempt < max_duration_attempts - 1:
                        continue  # retry with duration feedback
                    logger.warning("Partial plan also too short, using fallback")
                except Exception:
                    logger.warning("Could not parse partial plan", exc_info=True)
            return generate_fallback_plan(song_a_meta, song_b_meta)

        # Extract tool_use result
        tool_use_block = next(
            (b for b in response.content if b.type == "tool_use"), None
        )
        if tool_use_block is None:
            logger.warning("LLM returned no tool_use block, using fallback")
            return generate_fallback_plan(song_a_meta, song_b_meta)

        raw_plan = tool_use_block.input

        logger.info(
            "LLM response (attempt %d/%d): latency_ms=%.0f, model=%s, raw_plan=%s",
            attempt + 1, max_duration_attempts,
            latency_ms, response.model, json.dumps(raw_plan, indent=None),
        )

        # Parse and validate
        try:
            plan = _parse_intent_plan(raw_plan)
        except Exception:
            # Parse error -- on retry this can happen if LLM returns garbage
            logger.exception(
                "LLM plan parse failed (attempt %d/%d), using fallback",
                attempt + 1, max_duration_attempts,
            )
            return generate_fallback_plan(song_a_meta, song_b_meta)

        try:
            plan = _validate_intent_plan(plan, song_a_meta, song_b_meta, material_duration=material_duration)
        except _DurationTooShortError as e:
            if attempt < max_duration_attempts - 1:
                needed_beats = int(
                    TARGET_REMIX_DURATION_SECONDS * e.target_bpm / 60
                )
                actual_beats = e.plan.sections[-1].end_beat if e.plan.sections else 0
                beat_delta = needed_beats - actual_beats
                suggested_sections = max(6, needed_beats // 32)
                logger.warning(
                    "LLM plan too short (attempt %d/%d): %d beats, need ~%d (+%d). Retrying.",
                    attempt + 1, max_duration_attempts,
                    actual_beats, needed_beats, beat_delta,
                )
                # Replace original user message with condensed retry guidance
                # (avoids duplicating the full plan JSON in input tokens and
                # prevents the LLM from anchoring on its failed output)
                retry_guidance = (
                    f"IMPORTANT: Your previous arrangement was REJECTED — "
                    f"only {actual_beats} beats ({actual_beats * 60 / e.target_bpm:.0f}s), "
                    f"need ~{needed_beats} beats ({TARGET_REMIX_DURATION_SECONDS}s). "
                    f"Create at least {suggested_sections} sections."
                )
                original_content = messages[-1]["content"]
                messages[-1] = {
                    "role": "user",
                    "content": f"{retry_guidance}\n\n{original_content}",
                }
                continue  # retry
            else:
                logger.warning(
                    "LLM plan too short after %d attempts, using fallback",
                    max_duration_attempts,
                )
                return generate_fallback_plan(song_a_meta, song_b_meta)
        except Exception:
            logger.exception("LLM plan validation failed, using fallback")
            return generate_fallback_plan(song_a_meta, song_b_meta)

        # Validation passed
        if stretch_pct is not None and stretch_pct > 12:
            _warn_vocal_stretch_limits(plan, stretch_pct)
        return plan

    # Safety fallback (should not reach here)
    return generate_fallback_plan(song_a_meta, song_b_meta)


# ===========================================================================
# Day 2 deterministic fallback (preserved as-is)
# ===========================================================================

def generate_fallback_plan(meta_a: AudioMetadata, meta_b: AudioMetadata) -> RemixPlan:
    """Generate a deterministic remix plan from audio analysis metadata.

    Fixed convention: Song A provides vocals, Song B provides instrumentals.
    Uses the region starting at 25% into each song, capped at TARGET_REMIX_DURATION_SECONDS.
    Tempo target is the instrumental song's BPM.
    """
    # Fixed convention: Song A provides vocals
    vocal_src = VOCAL_SOURCE
    vocal_meta = meta_a
    inst_meta = meta_b

    # Use region starting at 25% into each song, up to TARGET_REMIX_DURATION_SECONDS
    v_start = vocal_meta.duration_seconds * 0.25
    v_end = min(v_start + TARGET_REMIX_DURATION_SECONDS, vocal_meta.duration_seconds)
    i_start = inst_meta.duration_seconds * 0.25
    i_end = min(i_start + TARGET_REMIX_DURATION_SECONDS, inst_meta.duration_seconds)

    tempo_src = "weighted_midpoint"  # Bias toward instrumental BPM to minimize vocal stretch
    fallback_target_bpm = estimate_target_bpm(
        vocal_bpm=vocal_meta.bpm,
        instrumental_bpm=inst_meta.bpm,
        tempo_source=tempo_src,
    )
    ideal_beats = int(fallback_target_bpm * TARGET_REMIX_DURATION_SECONDS / 60)
    material_duration = estimate_material_budget(
        vocal_bpm=vocal_meta.bpm,
        vocal_duration=vocal_meta.duration_seconds,
        instrumental_bpm=inst_meta.bpm,
        instrumental_duration=inst_meta.duration_seconds,
        target_bpm=fallback_target_bpm,
    )
    total_beats = min(ideal_beats, int(fallback_target_bpm * material_duration / 60))

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
        explanation=(
            "We created a remix using the strongest sections of each song. "
            "Vocals from Song A layered over Song B's instrumentals."
        ),
        warnings=["Using automatic remix layout (LLM was unavailable)."],
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
