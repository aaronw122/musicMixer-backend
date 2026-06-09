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
from musicmixer.services.separation import VOCAL_SONG_STEMS
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
# Song roles and stem identity validation
# ---------------------------------------------------------------------------

SongRole = Literal["vocal", "instrumental"]
ROLE_VOCAL: SongRole = "vocal"
ROLE_INSTRUMENTAL: SongRole = "instrumental"

# A cached stem directory is only valid for a role if its WAV file names (the
# stem identities) exactly match one of the intended separation shapes for that
# role. This rejects mismatched blobs (e.g. a 6-stem instrumental separation
# cached under the vocal role) and partial caches that pass a count-only check.
#
# vocal:
#   - modal/MelBand:   {lead_vocals, backing_vocals, instrumental}
#   - local fallback:  {lead_vocals, instrumental}  (intentionally accepted)
# instrumental:
#   - modal/BS-RoFormer: {vocals, drums, bass, guitar, piano, other}
#   - local htdemucs_ft: {vocals, drums, bass, other}
_VALID_STEM_SETS_BY_ROLE: dict[SongRole, tuple[frozenset[str], ...]] = {
    ROLE_VOCAL: (
        frozenset(VOCAL_SONG_STEMS),
        frozenset({"lead_vocals", "instrumental"}),
    ),
    ROLE_INSTRUMENTAL: (
        frozenset({"vocals", "drums", "bass", "guitar", "piano", "other"}),
        frozenset({"vocals", "drums", "bass", "other"}),
    ),
}

# Secondary floor for the instrumental full-cache contract: even a known
# identity set must contain at least this many stems. Guards against a future
# accepted shape with fewer than 4 stems satisfying the full-cache fast path.
_MIN_INSTRUMENTAL_STEMS = 4


def _stems_valid_for_role(role: SongRole, wav_files: list[Path]) -> bool:
    """Return True iff the WAV file names exactly match a known shape for role.

    Identities are derived from each path's ``.stem`` (filename without
    extension). Replaces the old count-only validation so a role's cache can
    only be considered usable when it actually holds that role's stems.
    """
    if role not in _VALID_STEM_SETS_BY_ROLE:
        raise ValueError(
            f"Invalid role {role!r}, must be one of {tuple(_VALID_STEM_SETS_BY_ROLE)}"
        )
    names = {f.stem for f in wav_files}
    if role == ROLE_INSTRUMENTAL and len(names) < _MIN_INSTRUMENTAL_STEMS:
        return False
    return frozenset(names) in _VALID_STEM_SETS_BY_ROLE[role]


def _is_degenerate_energy(stem_analysis: StemAnalysis | None) -> bool:
    """Return True iff a cached ``StemAnalysis`` carries zeroed-out energy.

    A vocal-source analysis written before the Phase 1 fix (and any legacy entry
    where the instrumental stem was dropped) stores an empty/all-zero
    ``combined_energy`` with ``vocal_active`` all-false. Such metadata is shared
    role-independently via ``song:{video_id}:meta``, so reusing it for an
    instrumental source feeds the arrangement flat energy.

    Degenerate iff ``combined_energy`` is empty OR all-zero, AND ``vocal_active``
    is all-false. The conjunction avoids flagging a genuinely quiet-but-valid
    analysis: real energy is never both flat *and* devoid of any vocal activity.
    A ``None`` analysis is not "degenerate energy" — it simply has no analysis
    yet, which the medium-cache path handles separately.
    """
    if stem_analysis is None:
        return False

    combined = np.asarray(stem_analysis.combined_energy)
    energy_dead = combined.size == 0 or not np.any(combined)

    vocal = np.asarray(stem_analysis.vocal_active)
    vocal_silent = vocal.size == 0 or not np.any(vocal)

    return bool(energy_dead and vocal_silent)

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
    if role not in _VALID_STEM_SETS_BY_ROLE:
        raise ValueError(f"Invalid role {role!r}, must be one of {tuple(_VALID_STEM_SETS_BY_ROLE)}")
    return f"song:{video_id}:{role}:stems"


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


def _resolve_stems_path(r: redis.Redis, video_id: str, role: SongRole) -> tuple[str | None, bool]:
    """Look up stems path from Redis and validate on disk.

    Returns (stems_path, has_stems).
    """
    try:
        stems_path = r.get(_stems_key(video_id, role))
    except redis.RedisError:
        logger.warning("Redis unavailable for stems lookup", exc_info=True)
        return None, False

    if isinstance(stems_path, bytes):
        stems_path = stems_path.decode()
    if not stems_path:
        return None, False

    cached_stems_dir = Path(stems_path)
    if not cached_stems_dir.is_dir():
        return stems_path, False

    wav_files = list(cached_stems_dir.glob("*.wav"))
    return stems_path, _stems_valid_for_role(role, wav_files)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _self_heal_degenerate_energy(
    video_id: str,
    role: SongRole,
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
    stems_path: str,
) -> bool:
    """Recompute zeroed energy metadata in place from the role's cached stems.

    Mirrors the #232 stem-cache self-heal, but for the shared meta key: when a
    medium-cache hit carries a degenerate ``stem_analysis`` (energy written by a
    pre-Phase-1 vocal-source analysis) and the requested role's stems are cached
    on disk, recompute ``analyze_stems`` from those on-disk stems — no
    re-download, no re-separation — and overwrite ``song:{video_id}:meta`` so the
    cache self-corrects. ``meta.stem_analysis`` / ``meta.song_structure`` are
    updated in place; returns True iff the recompute produced non-degenerate
    energy and the meta was rewritten.

    Self-limiting: only fires on a degenerate hit with valid on-disk stems.
    """
    # Imported lazily: analysis pulls in librosa/soundfile, which are heavy and
    # only needed on the rare degenerate path (keeps cache lookups cheap).
    from musicmixer.services.analysis import analyze_stems

    cache_dir = Path(stems_path)
    wav_files = list(cache_dir.glob("*.wav"))
    if not _stems_valid_for_role(role, wav_files):
        return False

    stem_paths = {f.stem: f for f in wav_files}
    try:
        # No audio_path / ml_segments: the original mix is not cached on disk, so
        # energy falls back to the per-stem sum. The role's cached stems
        # reconstruct the full mix, so the sum yields real (non-degenerate)
        # energy — sufficient to heal. song_structure from this recompute is
        # heuristic (no ml_segments), so we do NOT use it (see below).
        recomputed_stem_analysis, recomputed_structure = analyze_stems(
            stem_paths=stem_paths,
            beat_frames=meta.beat_frames,
            bpm=meta.bpm,
        )
    except Exception:
        logger.warning(
            "Self-heal recompute failed for video %s (role=%s)", video_id, role, exc_info=True
        )
        return False

    if _is_degenerate_energy(recomputed_stem_analysis):
        logger.warning(
            "Self-heal recompute for video %s (role=%s) still degenerate; leaving meta as-is",
            video_id, role,
        )
        return False

    # Heal ONLY stem_analysis. Section labels (song_structure) were never part of
    # the energy defect — SongFormer runs on raw audio and is role-independent —
    # so preserve any existing (likely ML-derived) song_structure rather than
    # downgrade it to this recompute's heuristic version. Fall back to the
    # recomputed structure only if the cached one is absent/empty.
    # NOTE: a healed entry carries per-stem-sum (not raw-mix-anchored) energy, so
    # role-identity isn't guaranteed for healed entries (still non-degenerate and
    # strictly better than zeroed).
    meta.stem_analysis = recomputed_stem_analysis
    if meta.song_structure is None or not meta.song_structure.sections:
        meta.song_structure = recomputed_structure
    cache_song_metadata(video_id, title, artist, meta, lyrics)
    logger.info(
        "Self-healed degenerate energy metadata for video %s (role=%s) from cached stems",
        video_id, role,
    )
    return True


def get_cached_song(video_id: str, role: SongRole) -> CachedSong | None:
    """Look up a song in Redis by video ID and role. Returns None on miss or error.

    Two-step lookup:
    1. Metadata from song:{video_id}:meta (shared across roles)
    2. Stems path from song:{video_id}:{role}:stems (role-specific)

    Returns None if metadata is missing. Returns CachedSong with has_stems=False
    if metadata exists but stems are missing or invalid for the requested role.

    Self-heal: if the cached meta is otherwise valid but its ``stem_analysis``
    carries degenerate (zeroed) energy AND the role's stems are cached on disk,
    recompute the analysis from those stems and rewrite ``:meta`` before
    returning (see ``_self_heal_degenerate_energy``).
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
        stems_path, has_stems = _resolve_stems_path(r, video_id, role)

        # Self-heal a degenerate (zeroed) energy meta when the role's stems are
        # cached on disk. Cheap recompute, no re-download/re-separation.
        if has_stems and stems_path and _is_degenerate_energy(meta.stem_analysis):
            _self_heal_degenerate_energy(
                video_id=video_id,
                role=role,
                title=data.get("title", ""),
                artist=data.get("artist", ""),
                meta=meta,
                lyrics=lyrics,
                stems_path=stems_path,
            )

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
    """Replace the role-qualified cache directory with fresh stems and update Redis.

    The role directory is pruned (rmtree + recreate) before copying so a recache
    cannot leave stale WAVs (e.g. legacy 6-stem blobs) beside the new stems —
    making the cache self-healing. Writes stems_path to
    song:{video_id}:{role}:stems as a plain Redis string only after the copied
    stem names validate for the role.
    """
    cache_dir = settings.song_cache_dir / video_id / role

    # Prune any existing (possibly stale/corrupt) role directory, then recreate.
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for wav_file in stems_dir.glob("*.wav"):
        shutil.copy2(wav_file, cache_dir / wav_file.name)

    copied = list(cache_dir.glob("*.wav"))
    if not _stems_valid_for_role(role, copied):
        logger.warning(
            "Refusing to cache stems for video %s (role=%s): copied stems %s do not "
            "match a valid shape for this role",
            video_id,
            role,
            sorted(f.stem for f in copied),
        )
        return

    try:
        r = _get_redis()
        r.set(_stems_key(video_id, role), str(cache_dir))
        logger.info("Cached stems for video %s (role=%s) at %s", video_id, role, cache_dir)
    except redis.RedisError:
        logger.warning("Redis unavailable, stems cached to disk only", exc_info=True)


def get_cached_stems(video_id: str, role: SongRole, output_dir: Path) -> bool:
    """Copy cached stems to output_dir. Returns True if stems were found.

    Uses role-aware stem identity validation: the cached WAV names must exactly
    match a known separation shape for the role (see ``_stems_valid_for_role``).
    """
    cache_dir = settings.song_cache_dir / video_id / role
    if not cache_dir.is_dir():
        return False

    wav_files = list(cache_dir.glob("*.wav"))
    if not _stems_valid_for_role(role, wav_files):
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    for wav_file in wav_files:
        shutil.copy2(wav_file, output_dir / wav_file.name)

    logger.info("Restored cached stems for video %s (role=%s, %d files)", video_id, role, len(wav_files))
    return True
