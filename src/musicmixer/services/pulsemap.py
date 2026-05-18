"""PulseMap audio analysis: chords, polyphony, drums, word alignment.

Adapted from PulseMap's standalone analysis scripts into musicMixer's
service layer.  Each function takes a file path and returns a typed
dataclass.  Optional heavy dependencies are guarded with feature flags
so the module imports cleanly even when libraries are missing.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from musicmixer.models import (
    ChordEvent,
    ChordProgression,
    DrumPattern,
    LyricsData,
    PolyphonyInfo,
    WordAlignment,
    WordEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards (same pattern as analysis.py)
# ---------------------------------------------------------------------------

try:
    import essentia.standard as es
    _HAS_ESSENTIA = True
except ImportError:
    _HAS_ESSENTIA = False

try:
    import lv_chordia
    _HAS_LV_CHORDIA = True
except ImportError:
    _HAS_LV_CHORDIA = False

try:
    import whisperx as _whisperx
    _HAS_WHISPERX = True
except ImportError:
    _HAS_WHISPERX = False


# ---------------------------------------------------------------------------
# Roman numeral mapping for chord summary
# ---------------------------------------------------------------------------

_MAJOR_SCALE_DEGREES = {
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "F": ["F", "G", "A", "Bb", "B", "C", "D"],
    "G": ["G", "A", "B", "C", "D", "E", "F#"],
    "A": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "B": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "Db": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"],
    "Eb": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"],
    "Ab": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "Bb": ["Bb", "C", "D", "Eb", "F", "G", "A"],
}

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


def _chord_root(chord: str) -> str:
    """Extract root note from a chord symbol like 'Cmaj7' or 'F#m'."""
    if len(chord) >= 2 and chord[1] in ("#", "b"):
        return chord[:2]
    return chord[:1]


def _to_roman(chord: str, key_root: str) -> str:
    """Convert a chord to Roman numeral relative to key_root, or return as-is."""
    root = _chord_root(chord)
    degrees = _MAJOR_SCALE_DEGREES.get(key_root)
    if degrees is None:
        return chord
    try:
        idx = degrees.index(root)
    except ValueError:
        return chord
    numeral = _ROMAN[idx]
    # Lowercase for minor chords
    is_minor = "m" in chord[len(root):] and "maj" not in chord[len(root):]
    return numeral.lower() if is_minor else numeral


def _build_progression_summary(chords: list[ChordEvent]) -> str:
    """Build a human-readable progression summary from chord events."""
    if not chords:
        return "no chords detected"

    # Deduplicate consecutive chords to get the progression
    progression: list[str] = []
    for ce in chords:
        if not progression or progression[-1] != ce.chord:
            progression.append(ce.chord)

    # Find the most common chord root to guess the key
    roots = [_chord_root(c) for c in progression]
    key_root = Counter(roots).most_common(1)[0][0]

    # Convert first 8 unique chords to Roman numerals
    roman_parts = [_to_roman(c, key_root) for c in progression[:8]]
    roman_str = "-".join(roman_parts)
    if len(progression) > 8:
        roman_str += "-..."

    return f"{roman_str} in {key_root}"


# ---------------------------------------------------------------------------
# 1. Polyphony detection
# ---------------------------------------------------------------------------

def detect_polyphony(vocal_stem_path: Path) -> PolyphonyInfo:
    """Detect whether a vocal stem contains polyphonic (multi-voice) content.

    Gate 1: Mid/side RMS ratio on stereo stems.
      - <0.05 side/mid ratio = solo voice
      - >0.15 = polyphonic (harmony, duet, choir)

    Gate 2: Essentia MultiPitchKlapuri on loudest 15s window.
      - >30% of frames with 2+ simultaneous pitches = polyphonic

    Mono files skip Gate 1 and go directly to Gate 2.
    If essentia is unavailable, only Gate 1 runs (mono files default to solo).
    """
    audio, sr = sf.read(str(vocal_stem_path), dtype="float32")

    # Stereo gate 1: mid/side ratio
    if audio.ndim == 2 and audio.shape[1] == 2:
        mid = (audio[:, 0] + audio[:, 1]) / 2.0
        side = (audio[:, 0] - audio[:, 1]) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        ratio = side_rms / mid_rms if mid_rms > 1e-10 else 0.0

        if ratio < 0.05:
            logger.info("Polyphony gate 1: solo (side/mid ratio=%.4f)", ratio)
            return PolyphonyInfo(
                polyphonic=False, method="mid_side",
                gate1_ratio=ratio, gate2_percent=None,
            )
        if ratio > 0.15:
            logger.info("Polyphony gate 1: polyphonic (side/mid ratio=%.4f)", ratio)
            return PolyphonyInfo(
                polyphonic=True, method="mid_side",
                gate1_ratio=ratio, gate2_percent=None,
            )
        # Ambiguous range (0.05 - 0.15): fall through to Gate 2
        gate1_ratio: float | None = ratio
    else:
        # Mono: skip Gate 1
        gate1_ratio = None
        if audio.ndim == 2:
            audio = audio[:, 0]

    # Gate 2: MultiPitchKlapuri (essentia)
    if not _HAS_ESSENTIA:
        logger.info("Polyphony: essentia unavailable, defaulting to solo")
        return PolyphonyInfo(
            polyphonic=False, method="mid_side",
            gate1_ratio=gate1_ratio, gate2_percent=None,
        )

    # Convert to mono float32 for essentia
    if audio.ndim == 2:
        mono = ((audio[:, 0] + audio[:, 1]) / 2.0).astype(np.float32)
    else:
        mono = audio.astype(np.float32)

    # Find loudest 15s window
    window_samples = min(int(15.0 * sr), len(mono))
    if len(mono) <= window_samples:
        segment = mono
    else:
        # Compute rolling RMS to find loudest window
        hop = sr  # 1-second hops
        best_start = 0
        best_energy = 0.0
        for start in range(0, len(mono) - window_samples, hop):
            chunk = mono[start:start + window_samples]
            energy = float(np.mean(chunk ** 2))
            if energy > best_energy:
                best_energy = energy
                best_start = start
        segment = mono[best_start:best_start + window_samples]

    # Run MultiPitchKlapuri
    multipitch = es.MultiPitchKlapuri(sampleRate=float(sr))
    frames_pitches = multipitch(segment)

    # Count frames with 2+ pitches
    total_frames = len(frames_pitches)
    if total_frames == 0:
        logger.info("Polyphony gate 2: no frames from Klapuri")
        return PolyphonyInfo(
            polyphonic=False, method="klapuri",
            gate1_ratio=gate1_ratio, gate2_percent=0.0,
        )

    multi_count = sum(1 for pitches in frames_pitches if len(pitches) >= 2)
    pct = multi_count / total_frames
    polyphonic = pct > 0.30

    logger.info(
        "Polyphony gate 2: %s (%.1f%% frames with 2+ pitches)",
        "polyphonic" if polyphonic else "solo", pct * 100,
    )
    return PolyphonyInfo(
        polyphonic=polyphonic, method="klapuri",
        gate1_ratio=gate1_ratio, gate2_percent=pct,
    )


# ---------------------------------------------------------------------------
# 2. Chord detection
# ---------------------------------------------------------------------------

def detect_chords(audio_path: Path) -> ChordProgression:
    """Detect chord progression using lv-chordia.

    Wraps lv-chordia's chord_recognition(), converts JAMS output to
    ChordEvent list, and derives a human-readable summary.

    Raises RuntimeError if lv-chordia is not installed.
    """
    if not _HAS_LV_CHORDIA:
        raise RuntimeError(
            "lv-chordia is not installed. Install with: uv add lv-chordia"
        )

    jams_result = lv_chordia.chord_recognition(str(audio_path))

    # Parse JAMS annotations
    chord_events: list[ChordEvent] = []
    annotations = jams_result.annotations
    if annotations:
        chord_annot = annotations[0]
        for obs in chord_annot.data:
            chord_label = obs.value
            if chord_label == "N" or not chord_label:
                continue
            start_ms = int(obs.time * 1000)
            end_ms = int((obs.time + obs.duration) * 1000)
            chord_events.append(ChordEvent(
                start_ms=start_ms,
                end_ms=end_ms,
                chord=chord_label,
            ))

    unique = list(dict.fromkeys(ce.chord for ce in chord_events))
    most_common = ""
    if chord_events:
        counter = Counter(ce.chord for ce in chord_events)
        most_common = counter.most_common(1)[0][0]

    summary = _build_progression_summary(chord_events)

    return ChordProgression(
        chords=chord_events,
        unique_chords=unique,
        most_common_chord=most_common,
        progression_summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Drum pattern transcription
# ---------------------------------------------------------------------------

def transcribe_drum_pattern(drum_stem_path: Path) -> DrumPattern:
    """Classify drum hits in a drum stem into kick, snare, and hi-hat.

    Uses librosa onset detection + spectral band classification:
    - Kick: dominant energy <200 Hz
    - Snare: dominant energy 200 Hz - 2 kHz
    - Hi-hat: dominant energy >2 kHz

    Derives a style hint from the ratios of hit types.
    """
    y, sr = librosa.load(str(drum_stem_path), sr=22050, mono=True)
    duration_ms = int(len(y) / sr * 1000)

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")

    if len(onset_frames) == 0:
        logger.info("Drums: no onsets detected")
        return DrumPattern(
            kick_count=0, snare_count=0, hihat_count=0,
            total_hits=0, duration_ms=duration_ms, style_hint="silent",
        )

    # Classify each onset by spectral centroid of a short window around it
    kick_count = 0
    snare_count = 0
    hihat_count = 0

    hop_length = 512  # librosa default
    window_frames = 4  # ~93ms at 22050/512

    for frame in onset_frames:
        start_sample = frame * hop_length
        end_sample = min(start_sample + window_frames * hop_length, len(y))
        segment = y[start_sample:end_sample]

        if len(segment) < hop_length:
            continue

        # Compute spectral centroid for this segment
        centroid = librosa.feature.spectral_centroid(
            y=segment, sr=sr, hop_length=hop_length,
        )
        mean_centroid = float(np.mean(centroid))

        if mean_centroid < 200:
            kick_count += 1
        elif mean_centroid < 2000:
            snare_count += 1
        else:
            hihat_count += 1

    total = kick_count + snare_count + hihat_count
    style_hint = _derive_drum_style(kick_count, snare_count, hihat_count, total)

    logger.info(
        "Drums: kick=%d snare=%d hihat=%d total=%d style=%s",
        kick_count, snare_count, hihat_count, total, style_hint,
    )
    return DrumPattern(
        kick_count=kick_count,
        snare_count=snare_count,
        hihat_count=hihat_count,
        total_hits=total,
        duration_ms=duration_ms,
        style_hint=style_hint,
    )


def _derive_drum_style(kick: int, snare: int, hihat: int, total: int) -> str:
    """Derive a style hint from drum hit type ratios."""
    if total == 0:
        return "silent"

    kick_ratio = kick / total
    hihat_ratio = hihat / total

    # Very few hits overall — sparse pattern
    if total < 10:
        return "sparse"

    # High kick ratio with even spacing suggests four-on-the-floor
    if kick_ratio > 0.35 and hihat_ratio < 0.4:
        return "four_on_floor"

    # Kick-heavy with dominant hi-hats suggests trap (check before breakbeat)
    if kick_ratio > 0.3 and hihat_ratio > 0.4:
        return "trap"

    # High hi-hat ratio with moderate kick/snare suggests breakbeat or busy pattern
    if hihat_ratio > 0.5:
        return "breakbeat"

    return "standard"


# ---------------------------------------------------------------------------
# 4. Word-level alignment
# ---------------------------------------------------------------------------

def align_words(
    vocal_stem_path: Path,
    lyrics_data: LyricsData | None = None,
) -> WordAlignment:
    """Align words in a vocal stem using WhisperX.

    Loads WhisperX base model, transcribes the vocal stem, and aligns
    with wav2vec2 phoneme model for word-level timing.

    Optionally validates against existing LRCLIB lyrics timestamps from
    the in-memory LyricsData to detect offset/mismatch.

    Raises RuntimeError if whisperx is not installed.
    """
    if not _HAS_WHISPERX:
        raise RuntimeError(
            "whisperx is not installed. Install with: uv add whisperx"
        )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Load and transcribe
    model = _whisperx.load_model("base", device, compute_type=compute_type)
    audio = _whisperx.load_audio(str(vocal_stem_path))
    result = model.transcribe(audio, batch_size=16)

    # Align with wav2vec2
    align_model, metadata = _whisperx.load_align_model(
        language_code=result.get("language", "en"), device=device,
    )
    aligned = _whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    # Extract word events
    word_events: list[WordEvent] = []
    for segment in aligned.get("segments", []):
        for word_info in segment.get("words", []):
            start = word_info.get("start")
            end = word_info.get("end")
            text = word_info.get("word", "")
            if start is not None and end is not None and text:
                word_events.append(WordEvent(
                    t=int(start * 1000),
                    text=text.strip(),
                    end=int(end * 1000),
                ))

    # Validate against LRCLIB timestamps if available
    lrclib_validated = False
    lrclib_offset_ms: int | None = None

    if lyrics_data is not None and lyrics_data.is_synced and word_events:
        lrclib_validated, lrclib_offset_ms = _validate_against_lrclib(
            word_events, lyrics_data,
        )

    logger.info(
        "Word alignment: %d words, lrclib_validated=%s, offset=%s",
        len(word_events), lrclib_validated, lrclib_offset_ms,
    )
    return WordAlignment(
        words=word_events,
        source="whisperx",
        lrclib_validated=lrclib_validated,
        lrclib_offset_ms=lrclib_offset_ms,
    )


def _validate_against_lrclib(
    words: list[WordEvent],
    lyrics_data: LyricsData,
) -> tuple[bool, int | None]:
    """Compare WhisperX word timing against LRCLIB synced line timestamps.

    For each LRCLIB synced line, find the closest WhisperX word that
    starts a matching text fragment.  Compute the median offset.
    If the median offset is <2000ms, consider it validated.

    Returns (validated: bool, offset_ms: int | None).
    """
    offsets: list[int] = []

    for line in lyrics_data.lines:
        if line.timestamp_seconds is None:
            continue
        line_ms = int(line.timestamp_seconds * 1000)
        line_words = line.text.strip().lower().split()
        if not line_words:
            continue
        first_word = line_words[0]

        # Find closest matching word in WhisperX output
        best_offset: int | None = None
        best_dist = float("inf")
        for we in words:
            if we.text.strip().lower() == first_word:
                dist = abs(we.t - line_ms)
                if dist < best_dist:
                    best_dist = dist
                    best_offset = we.t - line_ms
        if best_offset is not None and best_dist < 5000:
            offsets.append(best_offset)

    if not offsets:
        return False, None

    median_offset = int(np.median(offsets))
    validated = abs(median_offset) < 2000
    return validated, median_offset
