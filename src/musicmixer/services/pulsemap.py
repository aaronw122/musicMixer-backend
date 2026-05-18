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

_HAS_WHISPERX = bool(__import__("importlib").util.find_spec("whisperx"))


# ---------------------------------------------------------------------------
# Roman numeral mapping for chord summary
# ---------------------------------------------------------------------------

_MAJOR_SCALE_DEGREES = {
    "C": ["C", "D", "E", "F", "G", "A", "B"],
    "D": ["D", "E", "F#", "G", "A", "B", "C#"],
    "E": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "F": ["F", "G", "A", "Bb", "C", "D", "E"],
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

    from lv_chordia.chord_recognition import chord_recognition

    results = chord_recognition(str(audio_path), chord_dict_name="submission")

    # Parse chord annotations (list of dicts with start_time, end_time, chord)
    chord_events: list[ChordEvent] = []
    for entry in results:
        chord_label = entry.get("chord", "")
        if chord_label == "N" or chord_label == "X" or not chord_label:
            continue
        # Convert JAMS notation (C:maj7) to standard notation (Cmaj7)
        chord_name = _jams_to_standard(chord_label)
        start_ms = round(entry["start_time"] * 1000)
        end_ms = round(entry["end_time"] * 1000)
        chord_events.append(ChordEvent(
            start_ms=start_ms,
            end_ms=end_ms,
            chord=chord_name,
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
# JAMS chord notation conversion (from PulseMap detect_chords.py)
# ---------------------------------------------------------------------------

_JAMS_QUALITY_MAP = {
    "maj": "", "min": "m", "aug": "aug", "dim": "dim",
    "maj7": "maj7", "min7": "m7", "7": "7", "dim7": "dim7",
    "hdim7": "m7b5", "minmaj7": "mMaj7", "maj6": "6", "min6": "m6",
    "9": "9", "maj9": "maj9", "min9": "m9", "11": "11", "13": "13",
    "sus2": "sus2", "sus4": "sus4", "sus4(b7)": "7sus4",
    "sus4(b7,9)": "9sus4", "1": "5", "(1,5)": "5",
}

_NOTES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_INTERVAL_SEMITONES = {
    "1": 0, "b2": 1, "2": 2, "b3": 3, "3": 4, "4": 5,
    "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "b7": 10, "7": 11,
}


def _note_to_index(note: str) -> int | None:
    for i, n in enumerate(_NOTES_SHARP):
        if n == note:
            return i
    for i, n in enumerate(_NOTES_FLAT):
        if n == note:
            return i
    return None


def _interval_to_note(root: str, interval: str) -> str | None:
    root_idx = _note_to_index(root)
    semitones = _INTERVAL_SEMITONES.get(interval)
    if root_idx is None or semitones is None:
        return None
    target_idx = (root_idx + semitones) % 12
    return _NOTES_FLAT[target_idx] if "b" in root else _NOTES_SHARP[target_idx]


def _jams_to_standard(jams_chord: str) -> str:
    """Convert JAMS chord notation to standard lead sheet notation.

    JAMS: 'C:maj7', 'F#:min', 'Bb:7', 'G:sus4', 'D:min7/b3'
    Standard: 'Cmaj7', 'F#m', 'Bb7', 'Gsus4', 'Dm7/Bb'
    """
    if ":" not in jams_chord:
        return jams_chord

    parts = jams_chord.split(":")
    root = parts[0]
    quality_and_bass = parts[1] if len(parts) > 1 else ""

    bass_part = ""
    if "/" in quality_and_bass:
        quality, bass_interval = quality_and_bass.rsplit("/", 1)
        bass_note = _interval_to_note(root, bass_interval)
        if bass_note:
            bass_part = f"/{bass_note}"
    else:
        quality = quality_and_bass

    std_quality = _JAMS_QUALITY_MAP.get(quality, quality)
    return f"{root}{std_quality}{bass_part}"


# ---------------------------------------------------------------------------
# 3. Drum pattern transcription
# ---------------------------------------------------------------------------

_KICK_FREQ_MAX = 200
_SNARE_FREQ_MAX = 2000


def transcribe_drum_pattern(drum_stem_path: Path) -> DrumPattern:
    """Classify drum hits in a drum stem into kick, snare, and hi-hat.

    Uses librosa onset detection + spectral band energy ratios
    (matching PulseMap's approach):
    - Kick: >40% energy below 200 Hz
    - Snare: >30% energy between 200 Hz and 2 kHz
    - Hi-hat: >30% energy above 2 kHz
    A single onset can trigger multiple categories.

    Derives a style hint from the ratios of hit types.
    """
    y, sr = librosa.load(str(drum_stem_path), sr=44100, mono=True)
    duration_ms = int(len(y) / sr * 1000)

    hop_length = 512

    # Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length, backtrack=True, units="frames",
    )

    if len(onset_frames) == 0:
        logger.info("Drums: no onsets detected")
        return DrumPattern(
            kick_count=0, snare_count=0, hihat_count=0,
            total_hits=0, duration_ms=duration_ms, style_hint="silent",
        )

    # Compute STFT for spectral band energy classification
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    kick_bins = freqs < _KICK_FREQ_MAX
    snare_bins = (freqs >= _KICK_FREQ_MAX) & (freqs < _SNARE_FREQ_MAX)
    hihat_bins = freqs >= _SNARE_FREQ_MAX

    kick_count = 0
    snare_count = 0
    hihat_count = 0

    for frame_idx in onset_frames:
        if frame_idx >= S.shape[1]:
            continue

        spectrum = S[:, frame_idx]
        kick_energy = float(np.sum(spectrum[kick_bins]))
        snare_energy = float(np.sum(spectrum[snare_bins]))
        hihat_energy = float(np.sum(spectrum[hihat_bins]))

        total_energy = kick_energy + snare_energy + hihat_energy
        if total_energy < 1e-6:
            continue

        kick_ratio = kick_energy / total_energy
        snare_ratio = snare_energy / total_energy
        hihat_ratio = hihat_energy / total_energy

        if kick_ratio > 0.4:
            kick_count += 1
        if snare_ratio > 0.3:
            snare_count += 1
        if hihat_ratio > 0.3:
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

    import logging as _logging
    import sys
    import torch

    # Patch whisperx logging to use stderr (it defaults to stdout, polluting output)
    import whisperx as _whisperx  # noqa: N813 — lazy import to avoid loading torch at module level
    try:
        import whisperx.log_utils as _log_utils
        def _setup_stderr(level="warning", log_file=None):
            wx_logger = _logging.getLogger("whisperx")
            wx_logger.handlers.clear()
            handler = _logging.StreamHandler(sys.stderr)
            handler.setLevel(_logging.WARNING)
            wx_logger.addHandler(handler)
            wx_logger.setLevel(_logging.WARNING)
            wx_logger.propagate = False
        _log_utils.setup_logging = _setup_stderr
        _setup_stderr()
    except (ImportError, AttributeError):
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Load and transcribe
    model = _whisperx.load_model("base", device, compute_type=compute_type, language="en")
    audio = _whisperx.load_audio(str(vocal_stem_path))
    result = model.transcribe(audio, batch_size=8, language="en")

    # Align with wav2vec2
    align_model, metadata = _whisperx.load_align_model(
        language_code="en", device=device,
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
            text = word_info.get("word", "")
            if start is not None and text:
                text = text.strip()
                if text:
                    word_events.append(WordEvent(
                        t=round(start * 1000),
                        text=text,
                        end=round(word_info.get("end", start + 0.3) * 1000),
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
    """Validate LRCLIB line timestamps against WhisperX word clusters.

    Uses PulseMap's cluster-based approach: group WhisperX words into
    clusters (>1s gap = new cluster), then match LRCLIB line starts to
    cluster starts. Validated if median offset <5s and >60% consistency.

    Returns (validated: bool, offset_ms: int | None).
    """
    # Build sorted list of word start times
    stt_words = sorted(w.t for w in words)
    if len(stt_words) < 3:
        return False, None

    # Group into clusters (>1000ms gap = new cluster)
    cluster_starts = [stt_words[0]]
    for i in range(1, len(stt_words)):
        if stt_words[i] - stt_words[i - 1] > 1000:
            cluster_starts.append(stt_words[i])

    if not cluster_starts:
        return False, None

    # Filter LRCLIB lines: skip empty and parenthetical (backing vocals, etc.)
    lyric_lines = []
    for line in lyrics_data.lines:
        if line.timestamp_seconds is None:
            continue
        text = line.text.strip()
        if not text or (text.startswith("(") and text.endswith(")")):
            continue
        lyric_lines.append(line)

    if not lyric_lines:
        return False, None

    # Match each LRCLIB line to nearest cluster start
    offsets: list[int] = []
    for ll in lyric_lines:
        ll_t = int(ll.timestamp_seconds * 1000)
        best_dist = float("inf")
        best_offset: int | None = None
        for cs in cluster_starts:
            dist = abs(cs - ll_t)
            if dist < best_dist:
                best_dist = dist
                best_offset = ll_t - cs
        if best_offset is not None and best_dist < 60000:
            offsets.append(best_offset)

    if len(offsets) < 3:
        return False, None

    offsets.sort()
    median = offsets[len(offsets) // 2]
    within_threshold = sum(1 for o in offsets if abs(o - median) < 3000)
    consistency = within_threshold / len(offsets)

    validated = abs(median) < 5000 and consistency >= 0.6
    return validated, round(median)
