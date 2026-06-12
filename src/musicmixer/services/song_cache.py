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
import os
import shutil
import tempfile
import threading
import time
import uuid
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
# On-disk metadata path (source of truth) + atomic write helpers
# ---------------------------------------------------------------------------

def _meta_path(video_id: str) -> Path:
    """On-disk metadata file: data/song_cache/{video_id}/meta.json.

    meta.json is the source of truth for song metadata (BPM, key, stem_analysis,
    lyrics). Redis is a pure accelerator — on a Redis miss we fall back to this
    file and re-warm Redis from it. See the dual-writer invariant in
    ``_write_song_metadata``.
    """
    return settings.song_cache_dir / video_id / "meta.json"


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically (temp file in the same dir + os.replace).

    A reader either sees the previous file or the fully-written new one — never a
    truncated mid-write file. ``os.replace`` is atomic on POSIX when src and dest
    are on the same filesystem, so the temp file is created beside ``dest``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=f".{dest.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_text(dest: Path, text: str) -> None:
    """Atomic text write (see ``_atomic_write_bytes``)."""
    _atomic_write_bytes(dest, text.encode("utf-8"))


def _atomic_replace_dir(src_dir: Path, dest_dir: Path) -> None:
    """Atomically replace ``dest_dir`` with ``src_dir`` via rename-into-place.

    Both directories must be on the same filesystem (they are — both under
    ``song_cache_dir``). The swap is: rename any existing dest aside, rename src
    into dest, then remove the old one. A concurrent reader sees either the old
    complete dir or the new complete dir — never a half-filled directory (which
    a ``rmtree``-then-copy approach would expose as a poison cache hit).
    """
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    old_aside = dest_dir.with_name(f".{dest_dir.name}.old.{uuid.uuid4().hex}")
    had_existing = dest_dir.exists()
    if had_existing:
        os.replace(dest_dir, old_aside)
    try:
        os.replace(src_dir, dest_dir)
    except BaseException:
        # Restore the previous dir if the swap-in failed.
        if had_existing and old_aside.exists() and not dest_dir.exists():
            os.replace(old_aside, dest_dir)
        raise
    finally:
        if old_aside.exists():
            shutil.rmtree(old_aside, ignore_errors=True)


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


def _stems_dir_for(video_id: str, role: SongRole) -> Path:
    """Deterministic on-disk stems directory for a (video_id, role).

    Stems live at data/song_cache/{video_id}/{role}/. The Redis stems key only
    ever points here, so we can resolve stems from disk alone when Redis is gone.
    """
    return settings.song_cache_dir / video_id / role


def _resolve_stems_path(r: redis.Redis, video_id: str, role: SongRole) -> tuple[str | None, bool]:
    """Resolve the role's stems path and validate on disk.

    Disk is the source of truth for cached *artifacts*: we resolve against the
    deterministic on-disk directory (``_stems_dir_for``) rather than gating on a
    Redis key, so stems remain reachable after a Redis flush. Redis is consulted
    only opportunistically to re-warm the stems key on a disk hit.

    Returns (stems_path, has_stems).
    """
    cached_stems_dir = _stems_dir_for(video_id, role)
    if not cached_stems_dir.is_dir():
        return None, False

    wav_files = list(cached_stems_dir.glob("*.wav"))
    valid = _stems_valid_for_role(role, wav_files)
    stems_path = str(cached_stems_dir)

    if valid:
        # Re-warm the Redis stems key if it's missing (best-effort accelerator).
        try:
            if not r.exists(_stems_key(video_id, role)):
                r.set(_stems_key(video_id, role), stems_path)
        except redis.RedisError:
            logger.debug("Could not re-warm stems key for %s (%s)", video_id, role, exc_info=True)

    return stems_path, valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _recompute_analysis_from_cached_stems(
    video_id: str,
    role: SongRole,
    meta: AudioMetadata,
    stems_path: str,
) -> tuple[StemAnalysis, SongStructure] | None:
    """Recompute ``analyze_stems`` from the role's on-disk stems, or None if unusable.

    Returns None when the cached stems are invalid for the role, the recompute
    raises, or the recomputed energy is still degenerate. No audio_path/ml_segments
    are passed: the original mix is not cached on disk, so energy falls back to the
    per-stem sum. The role's cached stems reconstruct the full mix, so that sum
    yields real (non-degenerate) energy — sufficient to heal.
    """
    # Imported lazily: analysis pulls in librosa/soundfile, which are heavy and
    # only needed on the rare degenerate path (keeps cache lookups cheap).
    from musicmixer.services.analysis import analyze_stems

    wav_files = list(Path(stems_path).glob("*.wav"))
    if not _stems_valid_for_role(role, wav_files):
        return None

    stem_paths = {f.stem: f for f in wav_files}
    try:
        recomputed_stem_analysis, recomputed_structure = analyze_stems(
            stem_paths=stem_paths,
            beat_frames=meta.beat_frames,
            bpm=meta.bpm,
            downbeat_times=meta.downbeat_times,
        )
    except Exception:
        logger.warning(
            "Self-heal recompute failed for video %s (role=%s)", video_id, role, exc_info=True
        )
        return None

    if _is_degenerate_energy(recomputed_stem_analysis):
        logger.warning(
            "Self-heal recompute for video %s (role=%s) still degenerate; leaving meta as-is",
            video_id, role,
        )
        return None

    return recomputed_stem_analysis, recomputed_structure


def _apply_healed_stem_analysis(
    meta: AudioMetadata,
    recomputed_stem_analysis: StemAnalysis,
    recomputed_structure: SongStructure,
) -> None:
    """Swap fresh ``stem_analysis`` into ``meta`` while preserving its song structure.

    Heal ONLY stem_analysis. Section labels (song_structure) were never part of the
    energy defect — SongFormer runs on raw audio and is role-independent — so
    preserve any existing (likely ML-derived) song_structure rather than downgrade
    it to this recompute's heuristic version. Fall back to the recomputed structure
    only if the cached one is absent/empty.

    NOTE: a healed entry carries per-stem-sum (not raw-mix-anchored) energy, so
    role-identity isn't guaranteed for healed entries (still non-degenerate and
    strictly better than zeroed).
    """
    meta.stem_analysis = recomputed_stem_analysis
    if meta.song_structure is None or not meta.song_structure.sections:
        meta.song_structure = recomputed_structure


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

    Background: a vocal-source analysis written before the Phase 1 fix (and any
    legacy entry where the instrumental stem was dropped) stores zeroed energy.
    Because metadata is shared role-independently via ``song:{video_id}:meta``,
    reusing it for an instrumental source feeds the arrangement flat energy. This
    mirrors the #232 stem-cache self-heal but targets the shared meta key.

    Policy: recompute from the role's on-disk stems (no re-download/re-separation),
    apply the fresh stem_analysis while preserving existing song_structure, then
    overwrite ``song:{video_id}:meta``. ``meta`` is updated in place; returns True
    iff the recompute produced non-degenerate energy and the meta was rewritten.

    Self-limiting: only fires on a degenerate hit with valid on-disk stems.
    """
    recomputed = _recompute_analysis_from_cached_stems(video_id, role, meta, stems_path)
    if recomputed is None:
        return False

    recomputed_stem_analysis, recomputed_structure = recomputed
    _apply_healed_stem_analysis(meta, recomputed_stem_analysis, recomputed_structure)
    cache_song_metadata(video_id, title, artist, meta, lyrics)
    logger.info(
        "Self-healed degenerate energy metadata for video %s (role=%s) from cached stems",
        video_id, role,
    )
    return True


def _load_song_meta_fields(video_id: str) -> dict[str, str] | None:
    """Load the :meta field mapping, preferring Redis, falling back to disk.

    Source-of-truth model (Part 0): meta.json on disk is authoritative; Redis is
    an accelerator. So:
    1. Try Redis. On a hit, use it (fast path).
    2. On a Redis MISS or Redis error, fall back to meta.json on disk. If found,
       re-warm Redis from disk so the next lookup is fast again.

    Returns None only when BOTH Redis and disk lack metadata.

    NOTE: this does NOT short-circuit to None on a Redis miss — that gate-on-:meta
    behavior is exactly the bug that made stems-on-disk unreachable after a flush.
    """
    redis_ok = True
    try:
        r = _get_redis()
        data: dict[str, str] = r.hgetall(_meta_key(video_id))  # type: ignore[assignment]
    except redis.RedisError:
        logger.warning("Redis unavailable, falling back to disk meta", exc_info=True)
        redis_ok = False
        data = {}

    if data and "meta" in data:
        return data

    # Redis miss (or error) — fall back to the on-disk source of truth.
    disk = _load_meta_from_disk(video_id)
    if disk is None:
        return None

    logger.info("Redis :meta miss for video %s; loaded meta.json from disk", video_id)
    if redis_ok:
        # Re-warm Redis from the authoritative disk copy.
        try:
            _write_redis_meta(video_id, disk)
        except redis.RedisError:
            logger.warning("Failed to re-warm Redis meta for video %s", video_id, exc_info=True)
    return disk


def get_cached_song(video_id: str, role: SongRole) -> CachedSong | None:
    """Look up a song by video ID and role. Returns None on miss or error.

    Two-step lookup:
    1. Metadata — Redis :meta first, falling back to on-disk meta.json (source of
       truth) so a Redis flush does not make cached artifacts unreachable.
    2. Stems path from song:{video_id}:{role}:stems (role-specific), validated on
       disk (the stems dir's WAV names must match the role shape).

    Returns None if metadata is missing from BOTH Redis and disk. Returns
    CachedSong with has_stems=False if metadata exists but stems are missing or
    invalid for the requested role.

    Self-heal: if the cached meta is otherwise valid but its ``stem_analysis``
    carries degenerate (zeroed) energy AND the role's stems are cached on disk,
    recompute the analysis from those stems and rewrite ``:meta`` before
    returning (see ``_self_heal_degenerate_energy``).
    """
    data = _load_song_meta_fields(video_id)
    if data is None:
        return None

    try:
        r = _get_redis()
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


def _build_meta_mapping(
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
    cached_at: str,
) -> dict[str, str]:
    """Build the field mapping shared by meta.json and the Redis :meta hash."""
    mapping: dict[str, str] = {
        "title": title,
        "artist": artist,
        "meta": _serialize_audio_metadata(meta),
        "cached_at": cached_at,
    }
    if lyrics is not None:
        mapping["lyrics"] = _serialize_lyrics(lyrics)
    return mapping


def _write_redis_meta(video_id: str, mapping: dict[str, str]) -> None:
    """Overwrite the Redis :meta hash from a field mapping (full replace)."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.delete(_meta_key(video_id))
    pipe.hset(_meta_key(video_id), mapping=mapping)
    pipe.execute()


def _write_song_metadata(
    video_id: str,
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
) -> None:
    """The ONE dual-writer for song metadata. Routes ALL metadata mutations.

    Consistency invariant (Part 0): meta.json on disk is the source of truth and
    Redis is an accelerator. Every metadata mutation — ``cache_song_metadata``
    and the self-heal path — MUST go through here so disk and Redis never
    diverge into "Redis fresh, disk stale" (which would serve WRONG data after a
    flush, since disk wins on a fallback read).

    Order: write meta.json atomically FIRST (source of truth), then update Redis.
    If the Redis update fails, disk is still correct and a later read re-warms
    Redis from disk. A meta.json write failure aborts before Redis is touched.
    """
    cached_at = datetime.now(timezone.utc).isoformat()
    mapping = _build_meta_mapping(title, artist, meta, lyrics, cached_at)

    # 1. Disk first — the source of truth. A failure here must abort (don't leave
    #    Redis fresh while disk is stale/absent).
    _atomic_write_text(_meta_path(video_id), json.dumps(mapping))

    # 2. Redis accelerator. Best-effort: disk already holds the truth.
    try:
        _write_redis_meta(video_id, mapping)
    except redis.RedisError:
        logger.warning(
            "Redis unavailable writing meta for video %s; disk meta.json is authoritative",
            video_id, exc_info=True,
        )


def _load_meta_from_disk(video_id: str) -> dict[str, str] | None:
    """Load the meta.json field mapping from disk, or None if absent/corrupt."""
    path = _meta_path(video_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Corrupt/unreadable meta.json for video %s", video_id, exc_info=True)
        return None
    if not isinstance(data, dict) or "meta" not in data:
        return None
    return data


def cache_song_metadata(
    video_id: str,
    title: str,
    artist: str,
    meta: AudioMetadata,
    lyrics: LyricsData | None,
) -> None:
    """Write song analysis results to the metadata store (meta.json + Redis).

    Routes through the single dual-writer (``_write_song_metadata``) so meta.json
    on disk (source of truth) and the Redis :meta accelerator stay consistent.
    Always does a full overwrite to ensure consistency.
    """
    _write_song_metadata(video_id, title, artist, meta, lyrics)
    logger.info("Cached metadata for video %s", video_id)


def cache_song_stems(video_id: str, role: SongRole, stems_dir: Path) -> None:
    """Atomically install fresh stems into the role-qualified cache + update Redis.

    Atomicity (Part 0 contract): stems are copied into a sibling temp directory
    first; only after the copied stem names validate for the role is the temp dir
    renamed into place (``_atomic_replace_dir``). A reader therefore sees either
    the previous complete dir or the new complete dir — never a half-filled one
    (which a ``rmtree``-then-copy approach would expose as a poison cache hit on a
    crash mid-write). The rename also drops any stale WAVs (e.g. legacy 6-stem
    blobs), keeping the cache self-healing.

    Writes stems_path to song:{video_id}:{role}:stems only after the rename, so a
    Redis stems key never points at an incomplete directory.
    """
    cache_dir = _stems_dir_for(video_id, role)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)

    # Stage into a sibling temp dir on the same filesystem (so the rename is atomic).
    staging = cache_dir.with_name(f".{role}.staging.{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for wav_file in stems_dir.glob("*.wav"):
            shutil.copy2(wav_file, staging / wav_file.name)

        copied = list(staging.glob("*.wav"))
        if not _stems_valid_for_role(role, copied):
            logger.warning(
                "Refusing to cache stems for video %s (role=%s): copied stems %s do not "
                "match a valid shape for this role",
                video_id,
                role,
                sorted(f.stem for f in copied),
            )
            return

        _atomic_replace_dir(staging, cache_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    try:
        r = _get_redis()
        r.set(_stems_key(video_id, role), str(cache_dir))
        logger.info("Cached stems for video %s (role=%s) at %s", video_id, role, cache_dir)
    except redis.RedisError:
        logger.warning("Redis unavailable, stems cached to disk only", exc_info=True)


# ---------------------------------------------------------------------------
# Audio cache (role-agnostic raw download, by video_id)
# ---------------------------------------------------------------------------
#
# The audio cache stores the raw downloaded audio ONCE per video_id, independent
# of role: data/song_cache/{video_id}/audio.mp3. This is the role-agnostic
# invariant — one download per video, re-separated per required role. It is the
# Tier-3→Tier-2 bridge: a Tier-2 hit (audio cached, stems missing for the role)
# avoids YouTube entirely and only re-runs separation.

_AUDIO_FILENAME = "audio.mp3"
_AUDIO_META_FILENAME = "audio_meta.json"


def _audio_path(video_id: str) -> Path:
    """On-disk audio cache path: data/song_cache/{video_id}/audio.mp3."""
    return settings.song_cache_dir / video_id / _AUDIO_FILENAME


def _audio_meta_path(video_id: str) -> Path:
    """Sidecar holding the audio's title/duration/codec for cache-hit reuse."""
    return settings.song_cache_dir / video_id / _AUDIO_META_FILENAME


def get_cached_audio(video_id: str) -> tuple[Path, dict[str, Any]] | None:
    """Return ``(audio_path, meta_dict)`` for ``video_id`` if cached, else None.

    ``meta_dict`` carries title/duration/codec/bitrate from the original download
    (so a Tier-2 hit reconstructs the download result without touching YouTube).
    A non-empty file is required so a zero-byte artifact from a crashed download
    never counts as a hit. TTL is enforced lazily: an expired file is a miss and
    is removed.
    """
    path = _audio_path(video_id)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
    except OSError:
        return None

    ttl_hours = settings.audio_cache_ttl_hours
    if ttl_hours > 0:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > ttl_hours * 3600:
            logger.info("Cached audio for video %s expired (age %.1fh)", video_id, age_seconds / 3600)
            path.unlink(missing_ok=True)
            _audio_meta_path(video_id).unlink(missing_ok=True)
            return None

    meta: dict[str, Any] = {}
    meta_path = _audio_meta_path(video_id)
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Unreadable audio_meta.json for %s", video_id, exc_info=True)

    return path, meta


def cache_audio(video_id: str, audio_bytes: bytes, meta: dict[str, Any] | None = None) -> Path:
    """Atomically persist raw downloaded audio bytes (+ sidecar meta) for ``video_id``.

    Writes via temp + os.replace so a crash mid-write never leaves a truncated
    file that would pass an existence-only check (Part 0 atomic-write contract).
    The audio file is written LAST so the sidecar is never newer than the audio
    it describes. Returns the cache path.
    """
    if meta is not None:
        _atomic_write_text(_audio_meta_path(video_id), json.dumps(meta))
    dest = _audio_path(video_id)
    _atomic_write_bytes(dest, audio_bytes)
    logger.info("Cached audio for video %s (%.1fMB) at %s", video_id, len(audio_bytes) / 1e6, dest)
    _enforce_audio_cache_size_cap(protected_video_id=video_id)
    return dest


def _enforce_audio_cache_size_cap(protected_video_id: str | None = None) -> None:
    """Evict oldest cached audio files when total size exceeds the configured cap.

    Conservative policy (see plan Open decisions):
    - Only audio.mp3 files are eligible for eviction — meta.json and stem dirs are
      NEVER touched, so a Tier-1 stems hit is never dropped while leaving a
      dangling audio reference, and meta.json (source of truth) always survives.
    - ``protected_video_id`` (the just-written or in-flight video) is never
      evicted, so a concurrent separation that is about to read it is safe.
    - Eviction is least-recently-modified first.
    """
    cap_gb = settings.audio_cache_max_gb
    if cap_gb <= 0:
        return

    cap_bytes = int(cap_gb * 1024**3)
    cache_root = settings.song_cache_dir
    if not cache_root.is_dir():
        return

    files: list[tuple[float, int, Path, str]] = []
    total = 0
    for vid_dir in cache_root.iterdir():
        if not vid_dir.is_dir():
            continue
        audio = vid_dir / _AUDIO_FILENAME
        try:
            st = audio.stat()
        except OSError:
            continue
        files.append((st.st_mtime, st.st_size, audio, vid_dir.name))
        total += st.st_size

    if total <= cap_bytes:
        return

    in_flight = _in_flight_video_ids()
    files.sort(key=lambda t: t[0])  # oldest mtime first
    for _mtime, size, audio, vid in files:
        if total <= cap_bytes:
            break
        if vid == protected_video_id or vid in in_flight:
            continue
        try:
            audio.unlink(missing_ok=True)
            total -= size
            logger.info("Evicted cached audio for video %s (size-cap, freed %.1fMB)", vid, size / 1e6)
        except OSError:
            logger.debug("Failed to evict cached audio for %s", vid, exc_info=True)


# ---------------------------------------------------------------------------
# Single-flight dedup (per video_id)
# ---------------------------------------------------------------------------
#
# Two concurrent requests for the same video_id (cross-session, or the intra-
# request future_a/future_b pair in inverse-songs mixes) must not both
# download/separate. The second waits on a per-video_id lock and reuses the
# first's cached result. We use an in-process lock registry (the backend runs a
# single process); a multi-process deployment would swap this for a Redis lock.

_single_flight_lock = threading.Lock()
_video_locks: dict[str, threading.Lock] = {}
_in_flight: set[str] = set()


def _video_lock(video_id: str) -> threading.Lock:
    """Return the process-wide lock for ``video_id``, creating it on first use."""
    with _single_flight_lock:
        lock = _video_locks.get(video_id)
        if lock is None:
            lock = threading.Lock()
            _video_locks[video_id] = lock
        return lock


def _in_flight_video_ids() -> frozenset[str]:
    """Snapshot of video_ids currently being downloaded/separated."""
    with _single_flight_lock:
        return frozenset(_in_flight)


class single_flight:
    """Context manager that serializes work for a given ``video_id``.

    The first caller acquires the lock and runs; concurrent callers for the same
    video_id block until it releases, then proceed (and should re-check the cache,
    finding the first caller's result). Marks the video as in-flight so the
    size-cap sweeper won't evict an artifact mid-use.
    """

    def __init__(self, video_id: str):
        self._video_id = video_id
        self._lock = _video_lock(video_id)

    def __enter__(self) -> "single_flight":
        self._lock.acquire()
        with _single_flight_lock:
            _in_flight.add(self._video_id)
        return self

    def __exit__(self, *exc: object) -> None:
        with _single_flight_lock:
            _in_flight.discard(self._video_id)
        self._lock.release()


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
