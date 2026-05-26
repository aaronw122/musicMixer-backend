"""Redis-backed per-song cache keyed by YouTube video ID.

Caches AudioMetadata (including stem_analysis, song_structure, PulseMap),
LyricsData, and stem file paths. On cache hit, the pipeline skips download,
separation, and analysis — jumping straight to LLM + DSP.

Redis stores small data (~40-80KB per song). Stems (~480MB per song) live
on the filesystem at data/song_cache/{video_id}/{role}/.

Two Redis key patterns:
- `song:{video_id}:meta` — metadata hash (shared across roles)
- `song:{video_id}:{role}:stems` — stems_path as a plain string (role-specific)

Metadata is role-independent (BPM, key, energy describe the original song).
Only stem separation is role-dependent, so stems get their own role-qualified key.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
# Role-aware stem count thresholds
# ---------------------------------------------------------------------------

SongRole = Literal["vocal", "instrumental"]
ROLE_VOCAL: SongRole = "vocal"
ROLE_INSTRUMENTAL: SongRole = "instrumental"

_VALID_ROLES: tuple[SongRole, ...] = (ROLE_VOCAL, ROLE_INSTRUMENTAL)
_MIN_VOCAL_STEMS = 3        # lead_vocals, backing_vocals, instrumental
_MIN_INSTRUMENTAL_STEMS = 4  # vocals, drums, bass, guitar, piano, other (at least 4)

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

def _meta_key(video_id: str) -> str:
    """Redis key for shared metadata hash: song:{video_id}:meta"""
    return f"song:{video_id}:meta"


def _stems_key(video_id: str, role: SongRole) -> str:
    """Redis key for role-specific stems path: song:{video_id}:{role}:stems"""
    if role not in _VALID_ROLES:
        raise ValueError(f"Invalid role {role!r}, must be one of {_VALID_ROLES}")
    return f"song:{video_id}:{role}:stems"


def _min_stems_for_role(role: SongRole) -> int:
    """Return the minimum stem file count for a given role."""
    if role == ROLE_VOCAL:
        return _MIN_VOCAL_STEMS
    if role == ROLE_INSTRUMENTAL:
        return _MIN_INSTRUMENTAL_STEMS
    raise ValueError(f"Invalid role {role!r}, must be one of {_VALID_ROLES}")


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

def get_cached_song(video_id: str, role: SongRole) -> CachedSong | None:
    """Look up a song in Redis by video ID and role. Returns None on miss or error.

    Two-step lookup:
    1. Metadata from song:{video_id}:meta (shared across roles)
    2. Stems path from song:{video_id}:{role}:stems (role-specific)

    Returns None if metadata is missing. Returns CachedSong with has_stems=False
    if metadata exists but stems are missing or invalid for the requested role.
    """
    try:
        r = _get_redis()
        data: dict[str, str] = r.hgetall(_meta_key(video_id))  # type: ignore[assignment]
    except redis.RedisError:
        logger.warning("Redis unavailable, skipping cache lookup", exc_info=True)
        return None

    if not data or "meta" not in data:
        return None

    try:
        meta = _deserialize_audio_metadata(data["meta"])
        lyrics = _deserialize_lyrics(data["lyrics"]) if data.get("lyrics") else None

        # Step 2: look up role-specific stems path
        stems_path: str | None = None
        try:
            stems_path = r.get(_stems_key(video_id, role))  # type: ignore[assignment]
        except redis.RedisError:
            logger.warning("Redis unavailable for stems lookup", exc_info=True)

        # Validate stems still exist on disk (role-aware threshold)
        has_stems = False
        if stems_path and Path(stems_path).is_dir():
            wav_files = list(Path(stems_path).glob("*.wav"))
            has_stems = len(wav_files) >= _min_stems_for_role(role)

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
        logger.warning("Failed to deserialize cached song %s (role=%s)", video_id, role, exc_info=True)
        return None


def cache_song_metadata(
    video_id: str,
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
) -> None:
    """Write song analysis results to the shared metadata key in Redis.

    Writes to song:{video_id}:meta (no role — metadata is role-independent).
    Always does a full overwrite of the hash to ensure consistency.
    """
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

        pipe = r.pipeline()
        pipe.delete(_meta_key(video_id))
        pipe.hset(_meta_key(video_id), mapping=mapping)
        pipe.execute()
        logger.info("Cached metadata for video %s", video_id)
    except redis.RedisError:
        logger.warning("Redis unavailable, skipping cache write", exc_info=True)


def cache_song_stems(video_id: str, role: SongRole, stems_dir: Path) -> None:
    """Copy stems to a role-qualified cache directory and update Redis.

    Writes stems_path to song:{video_id}:{role}:stems as a plain Redis string.
    """
    cache_dir = settings.song_cache_dir / video_id / role
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Copy each WAV file
    for wav_file in stems_dir.glob("*.wav"):
        shutil.copy2(wav_file, cache_dir / wav_file.name)

    # Update Redis with the path (plain string, not a hash field)
    try:
        r = _get_redis()
        r.set(_stems_key(video_id, role), str(cache_dir))
        logger.info("Cached stems for video %s (role=%s) at %s", video_id, role, cache_dir)
    except redis.RedisError:
        logger.warning("Redis unavailable, stems cached to disk only", exc_info=True)


def get_cached_stems(video_id: str, role: SongRole, output_dir: Path) -> bool:
    """Copy cached stems to output_dir. Returns True if stems were found.

    Uses role-aware stem count validation: vocal needs >= 3, instrumental >= 4.
    """
    cache_dir = settings.song_cache_dir / video_id / role
    if not cache_dir.is_dir():
        return False

    wav_files = list(cache_dir.glob("*.wav"))
    if len(wav_files) < _min_stems_for_role(role):
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for wav_file in wav_files:
        shutil.copy2(wav_file, output_dir / wav_file.name)

    logger.info("Restored cached stems for video %s (role=%s, %d files)", video_id, role, len(wav_files))
    return True
