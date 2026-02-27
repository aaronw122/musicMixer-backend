"""Lyrics lookup, parsing, and bar mapping for LLM arrangement intelligence.

Fetches known-correct lyrics from free online databases (LRCLIB, Musixmatch)
via the syncedlyrics library, parses LRC format into timestamped lines, and
maps those lines to bar numbers derived from beat detection data. The result
is injected as Layer 5 in the LLM system prompt so it can make smarter
arrangement decisions (avoid cutting mid-phrase, identify hooks, match themes).

If no lyrics are found, the pipeline continues exactly as before — this is
purely additive.
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path

import librosa
import numpy as np
import syncedlyrics
from mutagen.easyid3 import EasyID3
from mutagen import MutagenError

from musicmixer.models import LyricLine, LyricsData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of lyric lines to include (cap token budget)
MAX_LYRIC_LINES = 60

# Timeout for syncedlyrics network lookup (seconds)
LYRICS_TIMEOUT_SECONDS = 15

# Default sample rate matching analysis.py
ANALYSIS_SR = 22050

# ---------------------------------------------------------------------------
# 3a. Filename parsing
# ---------------------------------------------------------------------------

# Common suffixes to strip from filenames before parsing
_SUFFIX_PATTERN = re.compile(
    r"\s*"
    r"[\(\[\{]"
    r"(?:Official\s*(?:Audio|Video|Music\s*Video|Lyric\s*Video|HD)|"
    r"Remaster(?:ed)?|"
    r"HD|HQ|4K|"
    r"Lyric(?:s)?\s*Video|"
    r"Audio|"
    r"ft\.?\s*.+|"
    r"feat\.?\s*.+|"
    r"\d{4}\s*Remaster(?:ed)?|"
    r"Live(?:\s+.+)?)"
    r"[\)\]\}]",
    re.IGNORECASE,
)

# YouTube video ID pattern (11 chars at end of filename)
_YOUTUBE_ID_PATTERN = re.compile(r"\s*[-_]\s*[A-Za-z0-9_-]{11}$")

# Track number prefix pattern (e.g., "01 - ", "01. ", "1 ")
_TRACK_NUMBER_PATTERN = re.compile(r"^\d{1,3}[\.\s\-]+\s*")


def parse_filename(filename: str) -> tuple[str, str] | None:
    """Parse artist and title from a filename like 'Artist - Title (Official Audio).mp3'.

    Handles:
    - Standard "Artist - Title" format
    - Common suffixes: (Official Audio), [Remaster], [HD], etc.
    - Track number prefixes: "01 - ", "01. "
    - YouTube video IDs at end of filename
    - Underscore separators (treated as spaces)

    Returns (artist, title) or None if no valid split found.
    """
    if not filename:
        return None

    # Remove file extension
    stem = Path(filename).stem

    # Replace underscores with spaces
    stem = stem.replace("_", " ")

    # Strip YouTube video ID
    stem = _YOUTUBE_ID_PATTERN.sub("", stem)

    # Strip common suffixes (may appear multiple times)
    prev = ""
    while prev != stem:
        prev = stem
        stem = _SUFFIX_PATTERN.sub("", stem)

    # Strip track number prefix
    stem = _TRACK_NUMBER_PATTERN.sub("", stem)

    # Clean up whitespace
    stem = stem.strip()

    if not stem:
        return None

    # Split on " - " (the standard separator)
    # Use the first occurrence to handle "Artist - Title - Subtitle" → ("Artist", "Title - Subtitle")
    parts = stem.split(" - ", maxsplit=1)
    if len(parts) == 2:
        artist = parts[0].strip()
        title = parts[1].strip()
        if artist and title:
            return (artist, title)

    # No valid split found
    return None


# ---------------------------------------------------------------------------
# 3b. ID3 tag reading
# ---------------------------------------------------------------------------

def read_id3_tags(audio_path: Path) -> tuple[str, str] | None:
    """Extract artist and title from ID3 tags via mutagen.

    Returns (artist, title) or None if tags are missing/unreadable.
    """
    try:
        tags = EasyID3(str(audio_path))
        artist = tags.get("artist", [None])[0]
        title = tags.get("title", [None])[0]
        if artist and title:
            return (artist.strip(), title.strip())
    except MutagenError:
        logger.debug("Could not read ID3 tags from %s", audio_path)
    except Exception:
        logger.debug("Unexpected error reading ID3 tags from %s", audio_path, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# 3c. Identity resolver
# ---------------------------------------------------------------------------

def resolve_song_identity(
    audio_path: Path,
    original_filename: str,
) -> tuple[str, str, str] | None:
    """Determine artist, title, and source for a song.

    Priority: filename parsing > ID3 tags > cleaned filename stem as search query.
    Returns (artist, title, source) or None if nothing useful can be extracted.
    """
    # Try filename first (most reliable for YouTube downloads)
    parsed = parse_filename(original_filename)
    if parsed:
        return (parsed[0], parsed[1], "filename")

    # Try ID3 tags from the actual file on disk
    id3 = read_id3_tags(audio_path)
    if id3:
        return (id3[0], id3[1], "id3")

    # Fallback: use the cleaned original filename stem as a search query
    if original_filename:
        stem = Path(original_filename).stem
        stem = stem.replace("_", " ")
        stem = _YOUTUBE_ID_PATTERN.sub("", stem)
        stem = _TRACK_NUMBER_PATTERN.sub("", stem)
        stem = stem.strip()
        if stem:
            return ("", stem, "filename")

    return None


# ---------------------------------------------------------------------------
# 3d. Lyrics fetch
# ---------------------------------------------------------------------------

# Regex to detect synced (LRC timestamped) lyrics
_SYNC_DETECT_PATTERN = re.compile(r"\[\d{1,2}:\d{2}")


def fetch_lyrics(artist: str, title: str) -> tuple[str, bool] | None:
    """Fetch lyrics via syncedlyrics library.

    Returns (raw_text, is_synced) or None if no lyrics found.
    """
    query = f"{artist} {title}".strip() if artist else title
    if not query:
        return None

    try:
        result = syncedlyrics.search(query)
        if not result:
            logger.info("No lyrics found for query: %s", query)
            return None

        is_synced = bool(_SYNC_DETECT_PATTERN.search(result))
        return (result, is_synced)

    except Exception:
        logger.warning("Lyrics fetch failed for query: %s", query, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 3e. LRC parser
# ---------------------------------------------------------------------------

# Permissive timestamp regex: [mm:ss], [mm:ss.cc], [mm:ss.ccc], [m:ss.cc]
_LRC_TIMESTAMP_PATTERN = re.compile(
    r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]"
)

# Metadata line tags to filter out
_LRC_METADATA_TAGS = {"ar", "ti", "al", "by", "offset", "re", "ve", "length", "au"}

# Offset tag pattern
_LRC_OFFSET_PATTERN = re.compile(r"\[offset:\s*([+-]?\d+)\s*\]", re.IGNORECASE)


def parse_lrc(lrc_text: str) -> list[LyricLine]:
    """Parse LRC-formatted lyrics into LyricLine objects.

    Handles:
    - Standard [mm:ss.cc] timestamps
    - Millisecond variant [mm:ss.ccc]
    - No fractional seconds [mm:ss]
    - Single-digit minutes
    - Multiple timestamps per line
    - Metadata line filtering ([ar:], [ti:], etc.)
    - [offset:] value applied to all timestamps
    """
    if not lrc_text:
        return []

    # Extract offset (milliseconds)
    offset_ms = 0.0
    offset_match = _LRC_OFFSET_PATTERN.search(lrc_text)
    if offset_match:
        offset_ms = float(offset_match.group(1))
    offset_seconds = offset_ms / 1000.0

    lines: list[LyricLine] = []

    for raw_line in lrc_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Check for metadata lines like [ar:Artist]
        meta_match = re.match(r"\[([a-zA-Z]+):", raw_line)
        if meta_match and meta_match.group(1).lower() in _LRC_METADATA_TAGS:
            continue

        # Find all timestamps in this line
        timestamps = _LRC_TIMESTAMP_PATTERN.findall(raw_line)
        if not timestamps:
            continue

        # Extract the text after all timestamps
        text = _LRC_TIMESTAMP_PATTERN.sub("", raw_line).strip()
        if not text:
            continue

        # Create a LyricLine for each timestamp (multiple timestamps = same text at different times)
        for minutes_str, seconds_str, frac_str in timestamps:
            minutes = int(minutes_str)
            seconds = int(seconds_str)
            frac_seconds = 0.0
            if frac_str:
                # Normalize to fractional seconds: "12" → 0.12, "123" → 0.123, "1" → 0.1
                frac_seconds = int(frac_str) / (10 ** len(frac_str))
            timestamp = minutes * 60 + seconds + frac_seconds + offset_seconds
            timestamp = max(0.0, timestamp)  # Clamp to non-negative
            lines.append(LyricLine(text=text, timestamp_seconds=timestamp))

    # Sort by timestamp
    lines.sort(key=lambda l: l.timestamp_seconds if l.timestamp_seconds is not None else 0.0)

    return lines


def parse_plain_lyrics(text: str) -> list[LyricLine]:
    """Parse plain (unsynced) lyrics into LyricLine objects without timestamps."""
    if not text:
        return []

    lines: list[LyricLine] = []
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if raw_line:
            lines.append(LyricLine(text=raw_line))
    return lines


# ---------------------------------------------------------------------------
# 3f. Bar mapping
# ---------------------------------------------------------------------------

def _compute_bar_times_from_beats(
    beat_frames: np.ndarray,
    sr: int = ANALYSIS_SR,
) -> np.ndarray:
    """Derive bar boundary times from beat frames.

    Takes every 4th beat frame (bar start) and converts to seconds.
    Returns array of bar start times in seconds.
    """
    if beat_frames is None or len(beat_frames) < 4:
        return np.array([])

    bar_frames = beat_frames[::4]
    bar_times = librosa.frames_to_time(bar_frames, sr=sr)
    return bar_times


def _compute_bar_times_from_bpm(
    bpm: float,
    duration_seconds: float = 300.0,
) -> np.ndarray:
    """Compute bar boundary times from BPM (fallback when beat_frames unavailable).

    Assumes 4/4 time: bar_duration = (60/bpm) * 4.
    """
    if bpm <= 0:
        return np.array([])

    bar_duration = (60.0 / bpm) * 4
    num_bars = int(math.ceil(duration_seconds / bar_duration))
    return np.arange(num_bars) * bar_duration


def map_lyrics_to_bars(
    lines: list[LyricLine],
    beat_frames: np.ndarray | None,
    bpm: float,
    sr: int = ANALYSIS_SR,
    duration_seconds: float = 300.0,
) -> list[LyricLine]:
    """Map synced lyric lines to bar numbers using beat detection data.

    Uses beat_frames[::4] via librosa.frames_to_time() for accurate mapping.
    Falls back to BPM-based calculation if beat_frames unavailable.
    """
    if not lines:
        return lines

    # Compute bar boundary times
    bar_times = _compute_bar_times_from_beats(beat_frames, sr=sr)
    if len(bar_times) == 0:
        bar_times = _compute_bar_times_from_bpm(bpm, duration_seconds)
    if len(bar_times) == 0:
        return lines

    # Map each line to a bar using searchsorted
    for line in lines:
        if line.timestamp_seconds is not None:
            # searchsorted returns the index where the timestamp would be inserted
            # to keep the array sorted — this is the bar number (0-indexed).
            # We use side='right' and subtract 1 to get the bar that contains the timestamp.
            bar_idx = int(np.searchsorted(bar_times, line.timestamp_seconds, side="right")) - 1
            line.bar_number = max(0, bar_idx)

    return lines


def map_plain_lyrics_to_bars(
    lines: list[LyricLine],
    vocal_active: np.ndarray | None,
    total_bars: int,
) -> list[LyricLine]:
    """Distribute plain (unsynced) lyrics across vocal-active bars proportionally.

    For plain lyrics without timestamps, we spread them evenly across bars
    where vocals are detected as active.
    """
    if not lines or total_bars <= 0:
        return lines

    # Find vocal-active bar indices
    if vocal_active is not None and len(vocal_active) > 0:
        active_bars = np.where(vocal_active)[0].tolist()
    else:
        # Fallback: assume all bars are vocal-active
        active_bars = list(range(total_bars))

    if not active_bars:
        active_bars = list(range(total_bars))

    # Distribute lines evenly across active bars
    num_lines = len(lines)
    num_active = len(active_bars)

    for i, line in enumerate(lines):
        # Map line index to active bar index proportionally
        bar_list_idx = int(i * num_active / num_lines)
        bar_list_idx = min(bar_list_idx, num_active - 1)
        line.bar_number = active_bars[bar_list_idx]

    return lines


# ---------------------------------------------------------------------------
# 3g. Top-level function
# ---------------------------------------------------------------------------

def lookup_lyrics_for_song(
    audio_path: Path,
    original_filename: str,
) -> LyricsData | None:
    """Look up lyrics for a song, parse them, and return structured data.

    Parameters:
        audio_path: Path to the actual audio file on disk (for ID3 tag extraction).
        original_filename: The original upload filename (for artist/title parsing).

    Returns:
        LyricsData with parsed lyrics, or None if no lyrics found.
    """
    start_time = time.monotonic()

    try:
        # Step 1: Resolve song identity
        identity = resolve_song_identity(audio_path, original_filename)
        if identity is None:
            logger.info("Could not identify song from filename or tags: %s", original_filename)
            return None

        artist, title, source = identity
        logger.info("Song identified as: %s - %s (source: %s)", artist, title, source)

        # Step 2: Fetch lyrics
        fetch_result = fetch_lyrics(artist, title)
        if fetch_result is None:
            return None

        raw_text, is_synced = fetch_result

        # Step 3: Parse lyrics
        if is_synced:
            lines = parse_lrc(raw_text)
        else:
            lines = parse_plain_lyrics(raw_text)

        if not lines:
            logger.info("Parsed lyrics yielded no lines for: %s - %s", artist, title)
            return None

        # Cap lines to avoid token budget blow-up
        if len(lines) > MAX_LYRIC_LINES:
            # Sample evenly to preserve song structure
            indices = np.linspace(0, len(lines) - 1, MAX_LYRIC_LINES, dtype=int)
            lines = [lines[i] for i in indices]

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Lyrics found: %d lines, synced=%s, elapsed=%.0fms for: %s - %s",
            len(lines), is_synced, elapsed_ms, artist, title,
        )

        return LyricsData(
            artist=artist,
            title=title,
            source=source,
            is_synced=is_synced,
            lines=lines,
            raw_text=raw_text,
            lookup_duration_ms=elapsed_ms,
        )

    except Exception:
        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.warning(
            "Lyrics lookup failed after %.0fms for: %s",
            elapsed_ms, original_filename, exc_info=True,
        )
        return None
