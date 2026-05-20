"""Redis-backed per-song cache keyed by YouTube video ID.

Caches AudioMetadata (including stem_analysis, song_structure, PulseMap),
LyricsData, and stem file paths. On cache hit, the pipeline skips download,
separation, and analysis — jumping straight to LLM + DSP.

Redis stores small data (~40-80KB per song). Stems (~480MB per song) live
on the filesystem at data/song_cache/{video_id}/.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import redis

from musicmixer.config import settings
from musicmixer.models import (
    AudioMetadata,
    CachedSong,
    ChordEvent,
    ChordProgression,
    DrumPattern,
    EnergyBuckets,
    LyricLine,
    LyricsData,
    PolyphonyInfo,
    SectionInfo,
    SongStructure,
    StemAnalysis,
    VocalGap,
    WordAlignment,
    WordEvent,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection
# ---------------------------------------------------------------------------

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Return a shared Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=1,
        )
    return _redis_client


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _song_key(video_id: str) -> str:
    return f"song:{video_id}"


# ---------------------------------------------------------------------------
# Serialization: Python objects → JSON-safe dicts
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy arrays and types to plain Python."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def _serialize_audio_metadata(meta: AudioMetadata) -> str:
    """Serialize AudioMetadata (with all nested types) to a JSON string.

    asdict() recursively converts all nested dataclasses to plain dicts.
    _NumpyEncoder catches numpy arrays/types and converts them to plain Python.
    """
    return json.dumps(asdict(meta), cls=_NumpyEncoder)


def _serialize_lyrics(lyrics: LyricsData) -> str:
    """Serialize LyricsData to a JSON string."""
    return json.dumps(asdict(lyrics))


# ---------------------------------------------------------------------------
# Deserialization: JSON-safe dicts → Python objects
# ---------------------------------------------------------------------------

def _deserialize_audio_metadata(json_str: str) -> AudioMetadata:
    """Reconstruct AudioMetadata from a JSON string."""
    d = json.loads(json_str)
    return _dict_to_meta(d)


def _dict_to_meta(d: dict) -> AudioMetadata:
    """Reconstruct AudioMetadata from a plain dict."""
    return AudioMetadata(
        bpm=d["bpm"],
        bpm_confidence=d["bpm_confidence"],
        beat_frames=np.array(d["beat_frames"]) if d.get("beat_frames") is not None else np.array([]),
        duration_seconds=d["duration_seconds"],
        total_beats=d["total_beats"],
        beat_times=np.array(d["beat_times"]) if d.get("beat_times") is not None else None,
        downbeat_times=np.array(d["downbeat_times"]) if d.get("downbeat_times") is not None else None,
        key=d.get("key"),
        scale=d.get("scale"),
        key_confidence=d.get("key_confidence"),
        has_modulation=d.get("has_modulation", False),
        source_quality=d.get("source_quality"),
        mean_rms=d.get("mean_rms"),
        stem_analysis=_dict_to_stem_analysis(d["stem_analysis"]) if d.get("stem_analysis") else None,
        song_structure=_dict_to_song_structure(d["song_structure"]) if d.get("song_structure") else None,
        chord_progression=_dict_to_chord_progression(d["chord_progression"]) if d.get("chord_progression") else None,
        polyphony_info=PolyphonyInfo(**d["polyphony_info"]) if d.get("polyphony_info") else None,
        drum_pattern=DrumPattern(**d["drum_pattern"]) if d.get("drum_pattern") else None,
        word_alignment=_dict_to_word_alignment(d["word_alignment"]) if d.get("word_alignment") else None,
    )


def _dict_to_stem_analysis(d: dict) -> StemAnalysis:
    return StemAnalysis(
        bar_rms={name: np.array(arr) for name, arr in d["bar_rms"].items()},
        combined_energy=np.array(d["combined_energy"]),
        vocal_active=np.array(d["vocal_active"], dtype=bool),
        vocal_gaps=[VocalGap(**vg) for vg in d["vocal_gaps"]],
        bucket_thresholds=EnergyBuckets(**d["bucket_thresholds"]),
    )


def _dict_to_song_structure(d: dict) -> SongStructure:
    return SongStructure(
        sections=[SectionInfo(**sec) for sec in d["sections"]],
        vocal_gaps=[VocalGap(**vg) for vg in d["vocal_gaps"]],
        total_bars=d["total_bars"],
    )


def _dict_to_chord_progression(d: dict) -> ChordProgression:
    return ChordProgression(
        chords=[ChordEvent(**ce) for ce in d["chords"]],
        unique_chords=d["unique_chords"],
        most_common_chord=d["most_common_chord"],
        progression_summary=d["progression_summary"],
    )


def _dict_to_word_alignment(d: dict) -> WordAlignment:
    return WordAlignment(
        words=[WordEvent(**w) for w in d["words"]],
        source=d["source"],
        lrclib_validated=d["lrclib_validated"],
        lrclib_offset_ms=d.get("lrclib_offset_ms"),
    )


def _deserialize_lyrics(json_str: str) -> LyricsData:
    """Reconstruct LyricsData from a JSON string."""
    d = json.loads(json_str)
    return LyricsData(
        artist=d["artist"],
        title=d["title"],
        source=d["source"],
        is_synced=d["is_synced"],
        lines=[LyricLine(**ll) for ll in d["lines"]],
        raw_text=d["raw_text"],
        lookup_duration_ms=d.get("lookup_duration_ms", 0.0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cached_song(video_id: str) -> CachedSong | None:
    """Look up a song in Redis by video ID. Returns None on miss or error."""
    try:
        r = _get_redis()
        data: dict[str, str] = r.hgetall(_song_key(video_id))  # type: ignore[assignment]
    except redis.RedisError:
        logger.warning("Redis unavailable, skipping cache lookup", exc_info=True)
        return None

    if not data or "meta" not in data:
        return None

    try:
        meta = _deserialize_audio_metadata(data["meta"])
        lyrics = _deserialize_lyrics(data["lyrics"]) if data.get("lyrics") else None
        stems_path = data.get("stems_path")

        # Validate stems still exist on disk
        has_stems = False
        if stems_path and Path(stems_path).is_dir():
            wav_files = list(Path(stems_path).glob("*.wav"))
            has_stems = len(wav_files) >= 4  # at least 4 stems present

        return CachedSong(
            video_id=video_id,
            title=data.get("title", ""),
            artist=data.get("artist", ""),
            meta=meta,
            lyrics=lyrics,
            stems_path=stems_path,
            has_stems=has_stems,
        )
    except Exception:
        logger.warning("Failed to deserialize cached song %s", video_id, exc_info=True)
        return None


def cache_song_metadata(
    video_id: str,
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
) -> None:
    """Write song analysis results to Redis."""
    try:
        r = _get_redis()
        mapping: dict[str, str] = {
            "title": title,
            "artist": artist,
            "meta": _serialize_audio_metadata(meta),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        if lyrics is not None:
            mapping["lyrics"] = _serialize_lyrics(lyrics)

        r.hset(_song_key(video_id), mapping=mapping)
        logger.info("Cached metadata for video %s", video_id)
    except redis.RedisError:
        logger.warning("Redis unavailable, skipping cache write", exc_info=True)


def cache_song_stems(video_id: str, stems_dir: Path) -> None:
    """Copy stems to the song cache directory and update Redis."""
    cache_dir = settings.song_cache_dir / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Copy each WAV file
    for wav_file in stems_dir.glob("*.wav"):
        shutil.copy2(wav_file, cache_dir / wav_file.name)

    # Update Redis with the path
    try:
        r = _get_redis()
        r.hset(_song_key(video_id), "stems_path", str(cache_dir))
        logger.info("Cached stems for video %s at %s", video_id, cache_dir)
    except redis.RedisError:
        logger.warning("Redis unavailable, stems cached to disk only", exc_info=True)


def get_cached_stems(video_id: str, output_dir: Path) -> bool:
    """Copy cached stems to output_dir. Returns True if stems were found."""
    cache_dir = settings.song_cache_dir / video_id
    if not cache_dir.is_dir():
        return False

    wav_files = list(cache_dir.glob("*.wav"))
    if len(wav_files) < 4:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for wav_file in wav_files:
        shutil.copy2(wav_file, output_dir / wav_file.name)

    logger.info("Restored cached stems for video %s (%d files)", video_id, len(wav_files))
    return True
