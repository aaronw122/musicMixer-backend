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

import errno
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

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


# A valid WAV begins with a 12-byte "RIFF....WAVE" header. A truncated WAV from a
# disk-full write keeps a nonzero size, so a bare size check is insufficient; we
# verify the RIFF/WAVE magic and a sane floor instead of a full (costly) decode.
_WAV_MIN_BYTES = 1024


def _is_intact_wav(path: Path) -> bool:
    """Lightweight WAV integrity check: readable RIFF/WAVE header + sane size.

    Confirms the file is at least ``_WAV_MIN_BYTES`` large and starts with a
    ``RIFF....WAVE`` header. Deliberately NOT a full decode — too costly per file
    at publication time. Catches zero-byte, truncated, and non-WAV files that a
    bare existence/size check would wrongly accept (see plan §8 disk-full).
    """
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < _WAV_MIN_BYTES:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"


def _build_manifest(wav_files: list[Path]) -> tuple[str, ...]:
    """Sorted tuple of WAV filenames for a validated role directory.

    The manifest records exactly which stem files a ``ready`` record references,
    so a later read can confirm every expected file still exists and is intact.
    """
    return tuple(sorted(f.name for f in wav_files))


def _validate_role_dir(role: SongRole, stems_dir: Path) -> tuple[str, ...] | None:
    """Validate a role's stem directory and return its manifest, or None.

    A directory is publishable iff its WAV identities exactly match a known shape
    for the role (``_stems_valid_for_role``) AND every WAV passes the lightweight
    integrity check (``_is_intact_wav``). Returns the manifest on success.
    """
    if not stems_dir.is_dir():
        return None
    wav_files = list(stems_dir.glob("*.wav"))
    if not _stems_valid_for_role(role, wav_files):
        return None
    if not all(_is_intact_wav(f) for f in wav_files):
        return None
    return _build_manifest(wav_files)


def _manifest_files_intact(stems_dir: Path, manifest: tuple[str, ...] | None) -> bool:
    """Confirm every file named in ``manifest`` exists and is an intact WAV."""
    if not manifest:
        return False
    for name in manifest:
        if not _is_intact_wav(stems_dir / name):
            return False
    return True


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

    _atomic_write_text(_meta_path(video_id), json.dumps(mapping))

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
    file that would pass an existence-only check. The audio file is written LAST
    so the sidecar is never newer than the audio it describes. Returns the path.
    """
    if meta is not None:
        _atomic_write_text(_audio_meta_path(video_id), json.dumps(meta))
    dest = _audio_path(video_id)
    _atomic_write_bytes(dest, audio_bytes)
    logger.info("Cached audio for video %s (%.1fMB) at %s", video_id, len(audio_bytes) / 1e6, dest)
    _enforce_audio_cache_size_cap(protected_video_id=video_id)
    return dest


def _enforce_audio_cache_size_cap(protected_video_id: str | None = None) -> None:
    """Evict oldest cached audio files (least-recently-modified first) when total
    size exceeds the configured cap.

    Only audio.mp3 files are eligible — meta.json and stem dirs are never touched,
    and ``protected_video_id`` (the just-written or in-flight video) is never
    evicted, so a concurrent separation about to read it is safe.
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
# In-process lock registry; a multi-process deployment would swap this for Redis.

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


def cached_stems_exist(video_id: str, role: SongRole) -> bool:
    """Return True if valid cached stems exist on disk for (video_id, role).

    Non-mutating existence/validity check using the SAME role-aware identity
    logic ``get_cached_stems`` gates on (``_stems_valid_for_role``), so a
    "validate" and a subsequent "copy" can never disagree. Copies nothing —
    callers can confirm presence before mutating any session dir.
    """
    cache_dir = settings.song_cache_dir / video_id / role
    if not cache_dir.is_dir():
        return False
    return _stems_valid_for_role(role, list(cache_dir.glob("*.wav")))


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


# ---------------------------------------------------------------------------
# Distributed stem-cache coordination (Redis lease + state machine)
# ---------------------------------------------------------------------------
# Two Redis keys per (video_id, role) coordinate stem separation across workers:
#   song:{video_id}:{role}:stem_lock   — string holding the owner_token (ownership)
#   song:{video_id}:{role}:stem_state  — hash describing the lifecycle (observability)
# The lock is the ownership authority; the state hash is descriptive. Renew and
# release are token-checked via Lua so a worker that lost its lease can never act
# on another owner's lock.

StemCacheStatus = Literal["processing", "ready", "failed"]
StemErrorCode = Literal["transient", "invalid_input"]

# Bounded failure categories that map to a retry policy. A raw traceback or
# arbitrary exception text is never stored in the state record.
STEM_ERROR_TRANSIENT: StemErrorCode = "transient"
STEM_ERROR_INVALID_INPUT: StemErrorCode = "invalid_input"


def _stem_state_key(video_id: str, role: SongRole) -> str:
    """Redis key for the lifecycle hash: song:{video_id}:{role}:stem_state"""
    if role not in _VALID_STEM_SETS_BY_ROLE:
        raise ValueError(f"Invalid role {role!r}, must be one of {tuple(_VALID_STEM_SETS_BY_ROLE)}")
    return f"song:{video_id}:{role}:stem_state"


def _stem_lock_key(video_id: str, role: SongRole) -> str:
    """Redis key for the ownership token: song:{video_id}:{role}:stem_lock"""
    if role not in _VALID_STEM_SETS_BY_ROLE:
        raise ValueError(f"Invalid role {role!r}, must be one of {tuple(_VALID_STEM_SETS_BY_ROLE)}")
    return f"song:{video_id}:{role}:stem_lock"


@dataclass(frozen=True)
class StemCacheState:
    """A snapshot of the stem-coordination state hash for a (video_id, role).

    Mirrors the Redis hash fields. Optional fields are absent (None) when not yet
    set; ``manifest`` is the list of expected WAV filenames.
    """

    status: StemCacheStatus
    owner_token: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    lease_expires_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    retry_after: str | None = None
    attempt: int = 0
    path: str | None = None
    manifest: tuple[str, ...] | None = None
    separator_version: str | None = None
    error_code: StemErrorCode | None = None

    def to_hash(self) -> dict[str, str]:
        """Serialize to the Redis hash field mapping (string values only)."""
        mapping: dict[str, str] = {"status": self.status, "attempt": str(self.attempt)}
        for field_name in (
            "owner_token",
            "started_at",
            "updated_at",
            "lease_expires_at",
            "completed_at",
            "failed_at",
            "retry_after",
            "path",
            "separator_version",
            "error_code",
        ):
            value = getattr(self, field_name)
            if value is not None:
                mapping[field_name] = value
        if self.manifest is not None:
            mapping["manifest"] = json.dumps(list(self.manifest))
        return mapping

    @classmethod
    def from_hash(cls, data: dict[str, str]) -> "StemCacheState":
        """Reconstruct from a Redis hash field mapping."""
        manifest_raw = data.get("manifest")
        manifest = tuple(json.loads(manifest_raw)) if manifest_raw else None
        error_code = data.get("error_code") or None
        return cls(
            status=data["status"],  # type: ignore[arg-type]
            owner_token=data.get("owner_token") or None,
            started_at=data.get("started_at") or None,
            updated_at=data.get("updated_at") or None,
            lease_expires_at=data.get("lease_expires_at") or None,
            completed_at=data.get("completed_at") or None,
            failed_at=data.get("failed_at") or None,
            retry_after=data.get("retry_after") or None,
            attempt=int(data.get("attempt", "0") or "0"),
            path=data.get("path") or None,
            manifest=manifest,
            separator_version=data.get("separator_version") or None,
            error_code=error_code,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class StemLease:
    """An owned lease over a (video_id, role) separation."""

    video_id: str
    role: SongRole
    owner_token: str


# Token-checked Lua. Renew/release act only if the lock value still equals the
# caller's owner_token, so a worker whose lease expired (and was re-acquired by
# another) cannot extend or delete the new owner's lock. Run via EVAL each call —
# the scripts are tiny and per-(video,role) coordination is not hot.
_RENEW_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
else
    return 0
end
"""

_RELEASE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class StemCacheCoordinator:
    """Distributed coordination primitives for per-(video_id, role) stem separation.

    Owns token-checked lease acquire/renew/release, state reads/writes, and the
    ready/failed transitions with attempt-driven backoff. Reuses the module-level
    ``_get_redis`` client and key-construction helpers — no second Redis client.
    """

    def __init__(self, redis_client: redis.Redis | None = None) -> None:
        self._redis_client = redis_client

    def _redis(self) -> redis.Redis:
        return self._redis_client if self._redis_client is not None else _get_redis()

    def get_state(self, video_id: str, role: SongRole) -> StemCacheState | None:
        """Return the current state, or None when the state key is absent (missing)."""
        data = self._redis().hgetall(_stem_state_key(video_id, role))
        if not data:
            return None
        return StemCacheState.from_hash(data)  # type: ignore[arg-type]

    def acquire(self, video_id: str, role: SongRole) -> StemLease | None:
        """Try to become the separation owner for (video_id, role).

        Atomically sets the lock with ``SET NX PX <lease>``. Returns a StemLease on
        success (this caller owns it) or None on contention (another live owner holds
        the lock). A stale/expired lock has been auto-removed by Redis, so a fresh
        acquire after expiry succeeds (takeover).
        """
        owner_token = uuid.uuid4().hex
        lease_ms = settings.stem_lock_lease_seconds * 1000
        acquired = self._redis().set(
            _stem_lock_key(video_id, role), owner_token, nx=True, px=lease_ms
        )
        if not acquired:
            return None

        now = _utcnow()
        expires = now + timedelta(seconds=settings.stem_lock_lease_seconds)
        existing = self.get_state(video_id, role)
        state = StemCacheState(
            status="processing",
            owner_token=owner_token,
            started_at=_iso(now),
            updated_at=_iso(now),
            lease_expires_at=_iso(expires),
            attempt=existing.attempt if existing is not None else 0,
            path=str(_stems_dir_for(video_id, role)),
            separator_version=settings.stem_separator_version,
        )
        self._write_state(video_id, role, state)
        return StemLease(video_id=video_id, role=role, owner_token=owner_token)

    def renew(self, lease: StemLease) -> bool:
        """Extend the lease iff the caller still owns the lock. Returns success."""
        lease_ms = settings.stem_lock_lease_seconds * 1000
        result = self._redis().eval(
            _RENEW_LUA, 1, _stem_lock_key(lease.video_id, lease.role),
            lease.owner_token, str(lease_ms),
        )
        renewed = bool(result)
        if renewed:
            now = _utcnow()
            expires = now + timedelta(seconds=settings.stem_lock_lease_seconds)
            self._redis().hset(
                _stem_state_key(lease.video_id, lease.role),
                mapping={"updated_at": _iso(now), "lease_expires_at": _iso(expires)},
            )
        return renewed

    def release(self, lease: StemLease) -> bool:
        """Delete the lock iff the caller still owns it. Returns whether it deleted."""
        result = self._redis().eval(
            _RELEASE_LUA, 1, _stem_lock_key(lease.video_id, lease.role),
            lease.owner_token,
        )
        return bool(result)

    def mark_failed(self, lease: StemLease, error_code: StemErrorCode) -> StemCacheState:
        """Record a categorized failure, schedule retry per backoff, release the lock.

        Only the lock owner may fail: the lock is released token-checked as part of
        failing. ``attempt`` is incremented; ``retry_after`` follows the retry schedule:
        transient failures back off exponentially from ``stem_retry_transient_base_seconds``
        capped at ``stem_retry_backoff_cap_seconds``; invalid-input uses a fixed window.
        Once ``attempt >= stem_retry_max_attempts`` the state holds ``failed`` with no
        ``retry_after`` until the 24h failed-state TTL expires. Returns the written state.
        """
        if error_code not in (STEM_ERROR_TRANSIENT, STEM_ERROR_INVALID_INPUT):
            raise ValueError(f"Invalid error_code {error_code!r}")

        existing = self.get_state(lease.video_id, lease.role)
        prior_attempt = existing.attempt if existing is not None else 0
        attempt = prior_attempt + 1

        now = _utcnow()
        retry_after: str | None = None
        if attempt < settings.stem_retry_max_attempts:
            retry_after = _iso(now + timedelta(seconds=_retry_delay_seconds(error_code, attempt)))

        state = StemCacheState(
            status="failed",
            owner_token=None,
            started_at=existing.started_at if existing is not None else None,
            updated_at=_iso(now),
            failed_at=_iso(now),
            retry_after=retry_after,
            attempt=attempt,
            path=str(_stems_dir_for(lease.video_id, lease.role)),
            separator_version=settings.stem_separator_version,
            error_code=error_code,
        )
        self._write_state(
            lease.video_id, lease.role, state, ttl_seconds=settings.stem_failed_ttl_seconds
        )
        self.release(lease)
        return state

    def mark_ready(self, lease: StemLease, staging_dir: Path) -> StemCacheState:
        """Fenced publication of a validated staging dir as the role's ready stems.

        Publish order: validate staging (exact role stem names + WAV
        integrity) and build the manifest → confirm the caller STILL owns the lease
        (token-checked) before any destructive replace → atomically swap the
        published ``{role}/`` dir into place (reuses ``_atomic_replace_dir``) →
        write ``status=ready`` with path, manifest, ``separator_version``, and
        timestamps → update the legacy ``:stems`` pointer. The lock is NOT released
        here; the caller releases it in its ``finally``.

        Raises ``PermissionError`` if the caller no longer owns the lease (a worker
        that lost its lease must never publish). Raises ``ValueError`` if the staging
        directory does not validate. ``ENOSPC`` during the replace propagates as
        ``OSError`` so the caller can categorize it transient.
        """
        manifest = _validate_role_dir(lease.role, staging_dir)
        if manifest is None:
            raise ValueError(
                f"Staging dir {staging_dir} is not a valid {lease.role} stem set"
            )

        if not self._owns_lock(lease):
            raise PermissionError(
                f"Lease for {lease.video_id}/{lease.role} no longer owned; refusing to publish"
            )

        cache_dir = _stems_dir_for(lease.video_id, lease.role)
        _atomic_replace_dir(staging_dir, cache_dir)

        now = _utcnow()
        existing = self.get_state(lease.video_id, lease.role)
        state = StemCacheState(
            status="ready",
            started_at=existing.started_at if existing is not None else None,
            updated_at=_iso(now),
            completed_at=_iso(now),
            attempt=existing.attempt if existing is not None else 0,
            path=str(cache_dir),
            manifest=manifest,
            separator_version=settings.stem_separator_version,
        )
        self._write_state(lease.video_id, lease.role, state)
        self._write_legacy_pointer(lease.video_id, lease.role, str(cache_dir))
        logger.info(
            "Published ready stems for video %s (role=%s, %d files)",
            lease.video_id, lease.role, len(manifest),
        )
        return state

    def invalidate_ready(self, video_id: str, role: SongRole) -> None:
        """Clear a ready record so a new lease can be competed for.

        Used when a ready record's ``separator_version`` is incompatible or its
        manifest/files fail validation. Deletes the state key (returning to
        ``missing``) only when the current record is ``ready`` — never disturbs an
        active ``processing`` owner. The legacy ``:stems`` pointer is also cleared.
        """
        state = self.get_state(video_id, role)
        if state is None or state.status != "ready":
            return
        r = self._redis()
        pipe = r.pipeline()
        pipe.delete(_stem_state_key(video_id, role))
        pipe.delete(_stems_key(video_id, role))
        pipe.execute()
        _log_stem_cache_outcome(
            "invalidated", video_id, role, separator_version=state.separator_version,
        )

    def reconcile_disk(self, video_id: str, role: SongRole) -> StemCacheState | None:
        """Lazily adopt a valid on-disk role dir as ready WITHOUT separation.

        Only acts when the state key is absent (missing). Inspects the deterministic
        ``{role}/`` directory; if it validates (exact role stems + intact WAVs),
        builds a manifest and publishes ``ready`` stamped with the CURRENT
        ``separator_version`` (legacy on-disk dirs carry no version stamp, so they
        are treated as current). If the directory is invalid, leaves
        state missing and does NOT promote a stale legacy ``:stems`` pointer.

        Returns the published ready state, or None if nothing was adopted.
        """
        if self.get_state(video_id, role) is not None:
            return None

        cache_dir = _stems_dir_for(video_id, role)
        manifest = _validate_role_dir(role, cache_dir)
        if manifest is None:
            return None

        now = _utcnow()
        state = StemCacheState(
            status="ready",
            updated_at=_iso(now),
            completed_at=_iso(now),
            path=str(cache_dir),
            manifest=manifest,
            separator_version=settings.stem_separator_version,
        )
        self._write_state(video_id, role, state)
        self._write_legacy_pointer(video_id, role, str(cache_dir))
        logger.info(
            "Adopted on-disk stems as ready for video %s (role=%s, %d files)",
            video_id, role, len(manifest),
        )
        return state

    def _owns_lock(self, lease: StemLease) -> bool:
        """True iff the lock value still equals the lease's owner_token (live lock)."""
        held = self._redis().get(_stem_lock_key(lease.video_id, lease.role))
        return held == lease.owner_token

    def _write_legacy_pointer(self, video_id: str, role: SongRole, path: str) -> None:
        """Best-effort write of the legacy ``:stems`` pointer (backwards compatibility)."""
        try:
            self._redis().set(_stems_key(video_id, role), path)
        except redis.RedisError:
            logger.debug(
                "Could not write legacy :stems pointer for %s (%s)",
                video_id, role, exc_info=True,
            )

    def _write_state(
        self,
        video_id: str,
        role: SongRole,
        state: StemCacheState,
        ttl_seconds: int | None = None,
    ) -> None:
        """Replace the state hash with ``state`` (full overwrite), optional TTL."""
        key = _stem_state_key(video_id, role)
        r = self._redis()
        pipe = r.pipeline()
        pipe.delete(key)
        pipe.hset(key, mapping=state.to_hash())
        if ttl_seconds is not None and ttl_seconds > 0:
            pipe.expire(key, ttl_seconds)
        pipe.execute()


def _retry_delay_seconds(error_code: StemErrorCode, attempt: int) -> int:
    """Seconds until retry for a failed attempt.

    Invalid input is a fixed window. Transient failures back off exponentially
    from the configured base, capped at the configured ceiling.
    """
    if error_code == STEM_ERROR_INVALID_INPUT:
        return settings.stem_retry_invalid_input_seconds
    delay = settings.stem_retry_transient_base_seconds * (2 ** (attempt - 1))
    return min(delay, settings.stem_retry_backoff_cap_seconds)


# Sibling dirs left beside a published {role}/ dir: owner-scoped staging
# (".{role}.staging.{token}") and the transient swap-aside (".{role}.old.{uuid}").
# A hard crash or OOM-kill can orphan either.
_STAGING_INFIX = ".staging."
_OLD_INFIX = ".old."


def sweep_orphaned_staging_dirs(coordinator: StemCacheCoordinator | None = None) -> int:
    """Remove orphaned ``.staging.*`` / ``.old.*`` siblings under the song cache.

    A ``.staging.{token}`` dir is orphaned unless its trailing ``token`` still holds
    a live lock for its ``(video_id, role)`` — a worker that died never released its
    lock, which Redis expires, so a staging dir whose token no longer matches the
    live lock is safe to drop. ``.old.*`` swap-aside dirs are always transient (the
    atomic replace removes them on success), so any left on disk are crash debris.

    Returns the number of directories removed. Best-effort; logs and continues on
    per-dir errors so one bad entry never aborts the sweep.
    """
    coordinator = coordinator if coordinator is not None else StemCacheCoordinator()
    root = settings.song_cache_dir
    if not root.is_dir():
        return 0

    removed = 0
    for video_dir in root.iterdir():
        try:
            if not video_dir.is_dir():
                continue
        except OSError:
            continue
        video_id = video_dir.name
        for sibling in video_dir.iterdir():
            name = sibling.name
            try:
                if not sibling.is_dir() or not name.startswith("."):
                    continue
            except OSError:
                continue

            if _OLD_INFIX in name:
                if _remove_dir(sibling):
                    removed += 1
                continue

            if _STAGING_INFIX not in name:
                continue

            role, token = _parse_staging_name(name)
            if role is not None and token is not None and _staging_token_is_live(
                coordinator, video_id, role, token
            ):
                continue
            if _remove_dir(sibling):
                removed += 1

    if removed:
        logger.info("Swept %d orphaned staging/old stem dirs", removed)
    return removed


def _parse_staging_name(name: str) -> tuple[SongRole | None, str | None]:
    """Parse ``.{role}.staging.{token}`` into (role, token); (None, None) if unmatched."""
    body = name[1:] if name.startswith(".") else name
    prefix, sep, token = body.partition(_STAGING_INFIX[1:])  # "staging."
    if not sep or not token:
        return None, None
    role = prefix.rstrip(".")
    if role not in _VALID_STEM_SETS_BY_ROLE:
        return None, None
    return role, token  # type: ignore[return-value]


def _staging_token_is_live(
    coordinator: StemCacheCoordinator, video_id: str, role: SongRole, token: str
) -> bool:
    """True iff ``token`` currently holds the live lock for (video_id, role)."""
    try:
        held = coordinator._redis().get(_stem_lock_key(video_id, role))
    except redis.RedisError:
        # Cannot confirm liveness; be conservative and keep the dir.
        return True
    return held == token


def _remove_dir(path: Path) -> bool:
    """Remove a directory tree; return whether it was removed. Best-effort."""
    try:
        shutil.rmtree(path)
        return True
    except OSError:
        logger.debug("Failed to sweep orphaned dir %s", path, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Cache-aware separation wrapper (renewable lease + wait / takeover)
# ---------------------------------------------------------------------------
# get_or_create_cached_stems wraps a role-specific separator behind the state
# machine: it elects one owner per (video_id, role) to run the paid separation,
# while concurrent callers wait and then reuse the owner's published stems. A
# Redis outage degrades to local separation rather than failing the request.

StemPaths = dict[str, Path | None]
SeparateFn = Callable[..., StemPaths]


class StemSeparationError(Exception):
    """A categorized stem-separation failure surfaced through the pipeline.

    ``error_code`` is one of the bounded categories. ``transient`` failures are
    retryable (infra/separator/timeouts, ENOSPC); ``invalid_input`` indicates the
    source audio is unusable and will not be retried for the invalid-input window.
    """

    def __init__(self, message: str, error_code: StemErrorCode):
        super().__init__(message)
        self.error_code = error_code


class StemWaitTimeout(StemSeparationError):
    """A waiter exceeded ``stem_wait_timeout_seconds`` without the owner publishing."""

    def __init__(self, video_id: str, role: SongRole):
        super().__init__(
            f"Timed out waiting for stems for {video_id}/{role}", STEM_ERROR_TRANSIENT
        )


def _categorize_separation_exception(exc: BaseException) -> StemErrorCode:
    """Map a separation exception to a bounded error code.

    A disk-full (``ENOSPC``) mid-separation/publish is transient. Plain
    ``ValueError`` from staging validation means the produced stems do not match the
    role shape — treat as invalid input. Everything else (Modal RPC errors, timeouts,
    network) is transient and retryable.
    """
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return STEM_ERROR_TRANSIENT
    if isinstance(exc, ValueError):
        return STEM_ERROR_INVALID_INPUT
    return STEM_ERROR_TRANSIENT


def _is_redis_outage(exc: BaseException) -> bool:
    """True iff ``exc`` indicates Redis is unreachable (connection-level failure)."""
    return isinstance(exc, (redis.ConnectionError, redis.TimeoutError))


# Bounded set of per-(video_id, role) stem-cache outcomes. Each is
# emitted exactly once on the path that already produces it; the set is closed so
# downstream queries can group on a known vocabulary.
StemCacheOutcome = Literal[
    "ready_hit",
    "owner_created",
    "waited_then_hit",
    "stale_takeover",
    "legacy_adopted",
    "failed_backoff",
    "invalidated",
]


def _log_stem_cache_outcome(
    outcome: StemCacheOutcome,
    video_id: str,
    role: SongRole,
    **fields: object,
) -> None:
    """Emit one structured stem-cache outcome record.

    Uses the project's ``extra=`` structured-logging convention (see
    ``pipeline_metrics``) so the outcome and its timing/category fields land as
    queryable JSON keys. Only bounded categories are recorded here — never raw
    Redis payloads, full URLs, or exception tracebacks.
    """
    logger.info(
        "stem_cache outcome=%s",
        outcome,
        extra={"stem_cache_outcome": outcome, "video_id": video_id, "role": role, **fields},
    )


class _LeaseRenewer:
    """Background thread that renews a lease until told to stop.

    Renews every ``stem_lock_renew_interval_seconds`` so the lease survives a long
    blocking separation (e.g. a remote Modal RPC with a cold start). If a renew
    fails (the lease was lost / Redis hiccup), ``lease_lost`` is set so the owner can
    refuse to publish over a new owner. Always stop in a ``finally``.
    """

    def __init__(self, coordinator: "StemCacheCoordinator", lease: StemLease):
        self._coordinator = coordinator
        self._lease = lease
        self._stop = threading.Event()
        self.lease_lost = threading.Event()
        self.renew_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def __enter__(self) -> "_LeaseRenewer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        interval = max(settings.stem_lock_renew_interval_seconds, 1)
        while not self._stop.wait(interval):
            try:
                renewed = self._coordinator.renew(self._lease)
            except redis.RedisError:
                logger.warning(
                    "Lease renew errored for %s/%s; will retry next interval",
                    self._lease.video_id, self._lease.role, exc_info=True,
                )
                continue
            if renewed:
                self.renew_count += 1
            else:
                logger.warning(
                    "Lost lease for %s/%s during separation; owner will not publish",
                    self._lease.video_id, self._lease.role,
                )
                self.lease_lost.set()
                return


def _copy_dir_wavs(src_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Copy every WAV in ``src_dir`` into ``output_dir``. Raises on a missing file.

    Used for the ready-copy path. ``FileNotFoundError`` propagates so the caller can
    treat a mid-copy ``_atomic_replace_dir`` (which deletes files out from under the
    glob) as a cache miss rather than a partial copy.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for wav_file in sorted(src_dir.glob("*.wav")):
        dest = output_dir / wav_file.name
        shutil.copy2(wav_file, dest)
        copied[wav_file.stem] = dest
    return copied


def _copy_ready_stems(
    coordinator: "StemCacheCoordinator",
    video_id: str,
    role: SongRole,
    output_dir: Path,
) -> dict[str, Path] | None:
    """Copy a ready record's validated stems into ``output_dir`` with one retry.

    Resolves the directory from the state record's ``path``, re-validates it, then
    copies. If the source dir/files vanish mid-copy (a concurrent invalidate +
    ``_atomic_replace_dir``), one bounded retry re-resolves and re-validates against
    the current record; a second disappearance is treated as a cache miss (returns
    None) so the caller falls through to compete for a lease.
    """
    for attempt in (1, 2):
        state = coordinator.get_state(video_id, role)
        if state is None or state.status != "ready" or not state.path:
            return None
        src_dir = Path(state.path)
        manifest = _validate_role_dir(role, src_dir)
        if manifest is None:
            return None
        try:
            stems = _copy_dir_wavs(src_dir, output_dir)
        except FileNotFoundError:
            logger.info(
                "Ready stems for %s/%s disappeared mid-copy (attempt %d); re-resolving",
                video_id, role, attempt,
            )
            shutil.rmtree(output_dir, ignore_errors=True)
            continue
        return stems
    logger.info(
        "Ready stems for %s/%s vanished after swap; treating as cache miss",
        video_id, role,
    )
    return None


def _run_uncached_separation(
    separate_fn: SeparateFn,
    audio_path: Path,
    session_output_dir: Path,
) -> StemPaths:
    """Run the separator straight into the session dir (no coordination)."""
    return separate_fn(audio_path, session_output_dir)


def _run_as_owner(
    coordinator: "StemCacheCoordinator",
    lease: StemLease,
    video_id: str,
    role: SongRole,
    audio_path: Path,
    session_output_dir: Path,
    separate_fn: SeparateFn,
    check_cancelled: Callable[[], None],
    took_over: bool = False,
) -> StemPaths:
    """Separate under a renewable lease, publish, then copy the result out.

    Separation runs into an owner-token-scoped staging dir. A background renewer
    keeps the lease alive across the blocking separation. On success the staging dir
    is published fenced (``mark_ready`` re-checks ownership before the replace); if
    the lease was lost, we discard staging and do NOT publish, letting a future caller
    finish. On failure we categorize, ``mark_failed`` (which releases the lock),
    and re-raise.
    """
    cache_dir = _stems_dir_for(video_id, role)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = cache_dir.with_name(f".{role}.staging.{lease.owner_token}")

    published = False
    sep_start = time.monotonic()
    try:
        with _LeaseRenewer(coordinator, lease) as renewer:
            check_cancelled()
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True, exist_ok=True)

            separate_fn(audio_path, staging)

            if renewer.lease_lost.is_set():
                logger.warning(
                    "stem_cache outcome=lease_lost video=%s role=%s; discarding staging",
                    video_id, role,
                )
                raise StemSeparationError(
                    f"Lost lease for {video_id}/{role} before publish", STEM_ERROR_TRANSIENT
                )

            coordinator.mark_ready(lease, staging)
            published = True
            _log_stem_cache_outcome(
                "stale_takeover" if took_over else "owner_created",
                video_id, role,
                separation_duration_s=round(time.monotonic() - sep_start, 3),
                lease_renew_count=renewer.renew_count,
                separator_version=settings.stem_separator_version,
            )
    except BaseException as exc:
        error_code = _categorize_separation_exception(exc)
        # Only fail the shared state if we still own the lock; mark_failed releases it.
        recorded_failed = False
        if coordinator._owns_lock(lease):
            try:
                coordinator.mark_failed(lease, error_code)
                recorded_failed = True
            except redis.RedisError:
                logger.warning("Could not record failed state for %s/%s", video_id, role, exc_info=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if recorded_failed:
            _log_stem_cache_outcome(
                "failed_backoff", video_id, role,
                error_code=error_code,
                separator_version=settings.stem_separator_version,
            )
        if isinstance(exc, StemSeparationError):
            raise
        if isinstance(exc, Exception):
            raise StemSeparationError(str(exc) or exc.__class__.__name__, error_code) from exc
        raise
    finally:
        try:
            coordinator.release(lease)
        except redis.RedisError:
            logger.debug("Lock release errored for %s/%s", video_id, role, exc_info=True)

    if not published:
        # Lease lost without raising shouldn't happen, but never copy in that case.
        raise StemSeparationError(
            f"Did not publish stems for {video_id}/{role}", STEM_ERROR_TRANSIENT
        )

    stems = _copy_ready_stems(coordinator, video_id, role, session_output_dir)
    if stems is None:
        raise StemSeparationError(
            f"Published stems for {video_id}/{role} vanished before copy",
            STEM_ERROR_TRANSIENT,
        )
    return stems


def _wait_for_owner(
    coordinator: "StemCacheCoordinator",
    video_id: str,
    role: SongRole,
    session_output_dir: Path,
    check_cancelled: Callable[[], None],
    on_wait: Callable[[], None] | None = None,
) -> StemPaths | None:
    """Poll until the owner publishes (copy), fails (raise), or the lease goes stale.

    Returns the copied stems on ``ready``; raises ``StemSeparationError`` on ``failed``
    or timeout; raises the cancellation error if ``check_cancelled`` fires. Returns
    None when the lease is stale (lock gone, state still processing) so the caller can
    compete for takeover.
    """
    deadline = time.monotonic() + settings.stem_wait_timeout_seconds
    wait_start = time.monotonic()
    notified = False
    while True:
        check_cancelled()
        state = coordinator.get_state(video_id, role)
        if not notified and state is not None and state.status == "processing":
            if on_wait is not None:
                on_wait()
            notified = True

        if state is None:
            return None  # state vanished -> re-enter the machine (compete)

        if state.status == "ready":
            stems = _copy_ready_stems(coordinator, video_id, role, session_output_dir)
            if stems is not None:
                _log_stem_cache_outcome(
                    "waited_then_hit", video_id, role,
                    wait_duration_s=round(time.monotonic() - wait_start, 3),
                    file_count=len(stems),
                )
                return stems
            return None  # ready vanished mid-copy -> compete

        if state.status == "failed":
            code: StemErrorCode = state.error_code or STEM_ERROR_TRANSIENT
            raise StemSeparationError(
                f"Separation failed for {video_id}/{role} ({code})", code
            )

        # processing: still owned?
        lock_held = coordinator._redis().get(_stem_lock_key(video_id, role))
        if not lock_held:
            return None  # stale lease -> caller competes for takeover

        if time.monotonic() >= deadline:
            if lock_held:
                raise StemWaitTimeout(video_id, role)
            return None

        time.sleep(max(settings.stem_wait_poll_seconds, 0))


def get_or_create_cached_stems(
    *,
    video_id: str | None,
    role: SongRole,
    audio_path: Path,
    session_output_dir: Path,
    separate_fn: SeparateFn,
    check_cancelled: Callable[[], None],
    on_wait: Callable[[], None] | None = None,
) -> StemPaths:
    """Cache-aware separation for one (video_id, role) with single-flight + reuse.

    - ``video_id is None`` (file upload): run the uncached path, no coordination.
    - ``ready``: copy validated cached stems into the session dir (resilient to a
      concurrent invalidate/replace).
    - ``owner``: acquire a lease, separate under renewal, publish fenced, copy out.
    - ``wait``: poll until ready/failed/timeout/cancel; on a stale lease, take over.

    On a Redis outage at any coordination point, degrade to local separation into the
    session dir — availability over dedup. Cancellation of a waiter stops polling
    and does not disturb the owner's lock/state.
    """
    if video_id is None:
        return _run_uncached_separation(separate_fn, audio_path, session_output_dir)

    coordinator = StemCacheCoordinator()

    try:
        return _coordinated_get_or_create(
            coordinator=coordinator,
            video_id=video_id,
            role=role,
            audio_path=audio_path,
            session_output_dir=session_output_dir,
            separate_fn=separate_fn,
            check_cancelled=check_cancelled,
            on_wait=on_wait,
        )
    except (redis.ConnectionError, redis.TimeoutError):
        logger.warning(
            "stem_cache outcome=redis_outage_fallback video=%s role=%s; running local separation",
            video_id, role, exc_info=True,
        )
        shutil.rmtree(session_output_dir, ignore_errors=True)
        return _run_uncached_separation(separate_fn, audio_path, session_output_dir)


def _coordinated_get_or_create(
    *,
    coordinator: "StemCacheCoordinator",
    video_id: str,
    role: SongRole,
    audio_path: Path,
    session_output_dir: Path,
    separate_fn: SeparateFn,
    check_cancelled: Callable[[], None],
    on_wait: Callable[[], None] | None = None,
) -> StemPaths:
    """Drive the state machine for a cached (video_id, role). Redis errors propagate.

    ``redis.ConnectionError``/``redis.TimeoutError`` bubble up to the caller's outage
    fallback. Loops a bounded number of times so a stale-lease takeover race or a
    mid-copy swap re-enters the machine rather than spinning forever.
    """
    max_rounds = settings.stem_retry_max_attempts + 2
    for _round in range(max_rounds):
        check_cancelled()

        # Lazily adopt a valid on-disk role dir as ready (legacy / crash recovery).
        adopted = coordinator.reconcile_disk(video_id, role)
        if adopted is not None:
            _log_stem_cache_outcome(
                "legacy_adopted", video_id, role,
                file_count=len(adopted.manifest or ()),
                separator_version=adopted.separator_version,
            )

        state = coordinator.get_state(video_id, role)

        if state is not None and state.status == "ready":
            stems = _copy_ready_stems(coordinator, video_id, role, session_output_dir)
            if stems is not None:
                # A just-adopted dir already emitted ``legacy_adopted``; don't also
                # count it as a plain ``ready_hit`` for the same request.
                if adopted is None:
                    _log_stem_cache_outcome(
                        "ready_hit", video_id, role, file_count=len(stems),
                    )
                return stems
            continue  # ready vanished mid-copy -> re-enter

        if state is not None and state.status == "failed":
            if state.retry_after is not None and _utcnow() < _parse_iso(state.retry_after):
                _log_stem_cache_outcome(
                    "failed_backoff", video_id, role,
                    error_code=state.error_code,
                    separator_version=state.separator_version,
                )
                raise StemSeparationError(
                    f"Separation for {video_id}/{role} is in backoff",
                    state.error_code or STEM_ERROR_TRANSIENT,
                )
            # retry_after elapsed (or held with no retry): compete for a fresh lease.

        # A pre-existing ``processing`` record whose lock we can now acquire means
        # the prior owner's lease expired: this acquire is a takeover.
        took_over = state is not None and state.status == "processing"
        lease = coordinator.acquire(video_id, role)
        if lease is not None:
            return _run_as_owner(
                coordinator=coordinator,
                lease=lease,
                video_id=video_id,
                role=role,
                audio_path=audio_path,
                session_output_dir=session_output_dir,
                separate_fn=separate_fn,
                check_cancelled=check_cancelled,
                took_over=took_over,
            )

        # Contention: someone else owns it. Wait, then copy / takeover.
        waited = _wait_for_owner(
            coordinator, video_id, role, session_output_dir, check_cancelled, on_wait
        )
        if waited is not None:
            return waited
        # None -> ready vanished mid-copy or stale lease: re-enter and compete.

    raise StemSeparationError(
        f"Exhausted coordination rounds for {video_id}/{role}", STEM_ERROR_TRANSIENT
    )


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp written by the coordinator."""
    return datetime.fromisoformat(ts)
