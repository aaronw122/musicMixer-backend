"""Pure, API-agnostic helpers for the remix endpoints.

These functions perform stage-adjacent input work (thumbnail derivation, upload
validation/write, duration probing) with no FastAPI, SSE, or ``SessionState``
coupling. They raise plain exceptions or return values; ``api/remix.py``
translates results into HTTP responses, preserving the route's exact status
codes and detail strings.
"""

import logging
import shutil
import subprocess
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Callable, Iterable

if TYPE_CHECKING:
    from musicmixer.config import Settings
    from musicmixer.models import AnalyzedSongs, CachedSong
    from musicmixer.services.pipeline_metrics import PipelineMetrics

logger = logging.getLogger("musicmixer.api.remix")


class UploadTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""


@dataclass(frozen=True)
class PreQueueCacheHit:
    """A served-from-cache remix resolved before the queue/processing lock.

    Carries the copied remix path and the metadata-derived session fields the
    route applies to its ``SessionState``. The route owns session lifecycle and
    the response; this is just the resolved data.
    """

    remix_path: str
    explanation: str
    used_fallback: bool
    key_warning: str | None
    warnings: list[str] = field(default_factory=list)


def restore_prequeue_cached_remix(
    url_a: str,
    url_b: str,
    prompt: str,
    *,
    session_id: str,
    cache_dir: Path,
    data_dir: Path,
) -> PreQueueCacheHit | None:
    """Resolve a URL-cache hit before queueing, or return ``None`` to fall through.

    Same URLs + prompt always produce the same remix, so a hit can be served
    instantly without consuming a processing slot. On a hit, the cached remix is
    copied into the session's output dir and the metadata-derived fields are
    returned. Any failure returns ``None`` so the route enqueues normally.
    """
    from musicmixer.services.remix_cache import (
        compute_url_cache_key,
        get_cached_remix,
        get_cached_metadata,
    )

    try:
        url_key = compute_url_cache_key(url_a, url_b, prompt)
        cached_path = get_cached_remix(url_key, cache_dir)
        if cached_path is None:
            return None

        logger.info(
            "Pre-queue URL cache hit (key=%s), serving instantly", url_key[:12]
        )
        meta = get_cached_metadata(url_key, cache_dir) or {}

        remix_dir = data_dir / "remixes" / session_id
        remix_dir.mkdir(parents=True, exist_ok=True)
        output_path = remix_dir / "remix.mp3"
        shutil.copy2(cached_path, output_path)

        return PreQueueCacheHit(
            remix_path=str(output_path),
            explanation=meta.get("explanation", ""),
            used_fallback=meta.get("used_fallback", False),
            key_warning=meta.get("key_warning"),
            warnings=meta.get("warnings", []),
        )
    except Exception:
        logger.debug(
            "Pre-queue URL cache check failed, proceeding normally", exc_info=True
        )
        return None


@dataclass(frozen=True)
class FullyCachedInputs:
    """Inputs for the fully-cached YouTube restore stage."""

    session_id: str
    cached_song_a: "CachedSong"
    cached_song_b: "CachedSong"


@dataclass
class FullyCachedCallbacks:
    """API-owned callbacks for the fully-cached restore stage.

    ``check_cancelled`` lets ``api/remix.py`` keep cancellation ownership; the
    stage only calls it. ``on_cache_skip`` lets the API emit its existing
    "skipping download" progress event through API-owned event construction —
    the stage never touches SSE payloads.
    """

    check_cancelled: Callable[[], None]
    on_cache_skip: Callable[[], None]


@dataclass(frozen=True)
class FullyCachedRestore:
    """Result of restoring a fully-cached YouTube remix.

    A cache hit (``used_cache=True``) carries everything the visible
    ``run_remix`` call in ``api/remix.py`` needs; a miss (``used_cache=False``)
    carries no payload and signals the wrapper to fall back to the normal
    download/separate pipeline. The stage never calls ``run_remix`` itself.
    """

    used_cache: bool
    analysis: "AnalyzedSongs | None" = None
    metrics: "PipelineMetrics | None" = None
    source_quality_a: str | None = None
    source_quality_b: str | None = None
    restored_stems_dir: Path | None = None


def restore_fully_cached_youtube_remix(
    inputs: FullyCachedInputs,
    *,
    callbacks: FullyCachedCallbacks,
    settings: "Settings",
) -> FullyCachedRestore:
    """Restore a fully-cached YouTube remix, or signal a cache miss.

    Both songs' stems + metadata are already on disk, so this skips downloads
    and analysis: it restores cached stems, measures per-stem LUFS, builds the
    ``AnalyzedSongs`` and ``PipelineMetrics`` the remix needs, and returns them.

    On a partial cache miss (one stem set missing) it returns a miss result
    BEFORE copying either stem set, so a degraded cache cannot orphan a
    half-restored stem dir. The wrapper owns the ``run_remix`` call and the
    fallback decision.

    Cancellation stays with ``api/remix.py`` via ``callbacks.check_cancelled``;
    this stage owns no session lifecycle.
    """
    from musicmixer.models import AnalyzedSongs
    from musicmixer.services.pipeline import _step_measure_stem_lufs
    from musicmixer.services.pipeline_metrics import PipelineMetrics
    from musicmixer.services.song_cache import (
        ROLE_INSTRUMENTAL,
        ROLE_VOCAL,
        cached_stems_exist,
        get_cached_stems,
    )

    session_id = inputs.session_id
    cached_song_a = inputs.cached_song_a
    cached_song_b = inputs.cached_song_b

    logger.info(
        "Session %s: Both songs fully cached, skipping downloads + analysis",
        session_id,
    )
    callbacks.on_cache_skip()

    callbacks.check_cancelled()

    stems_dir = settings.data_dir / "stems" / session_id
    song_a_stems_dir = stems_dir / "song_a"
    song_b_stems_dir = stems_dir / "song_b"

    # get_cached_stems copies as a side effect, so confirm BOTH are present before
    # copying either — otherwise "A ok, B missing" orphans A's stems on fallback.
    if not (
        cached_stems_exist(cached_song_a.video_id, ROLE_VOCAL)
        and cached_stems_exist(cached_song_b.video_id, ROLE_INSTRUMENTAL)
    ):
        logger.warning(
            "Session %s: Cached stems missing, falling back to full pipeline",
            session_id,
        )
        return FullyCachedRestore(used_cache=False)

    get_cached_stems(cached_song_a.video_id, ROLE_VOCAL, song_a_stems_dir)
    get_cached_stems(cached_song_b.video_id, ROLE_INSTRUMENTAL, song_b_stems_dir)

    song_a_stems = {f.stem: f for f in song_a_stems_dir.glob("*.wav")}
    song_b_stems = {f.stem: f for f in song_b_stems_dir.glob("*.wav")}

    vocal_stem_lufs, inst_stem_lufs = _step_measure_stem_lufs(
        session_id, song_a_stems_dir, song_b_stems_dir,
    )

    analysis = AnalyzedSongs(
        meta_a=cached_song_a.meta,
        meta_b=cached_song_b.meta,
        song_a_stems=song_a_stems,
        song_b_stems=song_b_stems,
        song_a_stems_dir=song_a_stems_dir,
        song_b_stems_dir=song_b_stems_dir,
        lyrics_a=cached_song_a.lyrics,
        lyrics_b=cached_song_b.lyrics,
        vocal_stem_lufs=vocal_stem_lufs,
        inst_stem_lufs=inst_stem_lufs,
    )

    metrics = PipelineMetrics(session_id=session_id)
    metrics.song_a_title = cached_song_a.title
    metrics.song_b_title = cached_song_b.title
    metrics.song_a_video_id = cached_song_a.video_id
    metrics.song_b_video_id = cached_song_b.video_id
    metrics.song_a_cache_hit = True
    metrics.song_b_cache_hit = True
    metrics.bpm_a = cached_song_a.meta.bpm
    metrics.bpm_b = cached_song_b.meta.bpm
    metrics.key_a = cached_song_a.meta.key or ""
    metrics.scale_a = cached_song_a.meta.scale or ""
    metrics.key_b = cached_song_b.meta.key or ""
    metrics.scale_b = cached_song_b.meta.scale or ""
    metrics.key_confidence_a = cached_song_a.meta.key_confidence or 0.0
    metrics.key_confidence_b = cached_song_b.meta.key_confidence or 0.0
    metrics.duration_a_s = cached_song_a.meta.duration_seconds
    metrics.duration_b_s = cached_song_b.meta.duration_seconds
    metrics.log_input()
    metrics.log_analysis()

    return FullyCachedRestore(
        used_cache=True,
        analysis=analysis,
        metrics=metrics,
        source_quality_a=cached_song_a.meta.source_quality,
        source_quality_b=cached_song_b.meta.source_quality,
        restored_stems_dir=stems_dir,
    )


def thumbnail_from_youtube_url(url: str) -> str | None:
    """Derive a YouTube thumbnail URL from a video URL. Returns None on failure."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ""
    path_parts = [p for p in parsed.path.split("/") if p]

    video_id = None
    if hostname == "youtu.be" and path_parts:
        video_id = path_parts[0]
    else:
        qs = urllib.parse.parse_qs(parsed.query)
        if qs.get("v"):
            video_id = qs["v"][0]
        elif path_parts and path_parts[0] in {"shorts", "embed"} and len(path_parts) > 1:
            video_id = path_parts[1]

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg" if video_id else None


def probe_duration(file_path: Path) -> float | None:
    """Return audio duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return None


def upload_extension(filename: str | None) -> str:
    """Return the lowercased file suffix for an uploaded file name."""
    return Path(filename or "").suffix.lower()


def extension_allowed(filename: str | None, allowed_extensions: Iterable[str]) -> bool:
    """Return whether the upload's extension is in the allowed set."""
    return upload_extension(filename) in allowed_extensions


def write_upload_file(file: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Stream an uploaded file to ``dest`` in 1 MB chunks.

    Raises ``UploadTooLargeError`` if the cumulative size exceeds ``max_bytes``.
    The size check happens mid-stream so an oversized upload is rejected without
    buffering the whole payload.
    """
    file.seek(0)
    chunks = []
    total = 0
    while True:
        chunk = file.read(1024 * 1024)  # 1 MB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError
        chunks.append(chunk)
    data = b"".join(chunks)
    dest.write_bytes(data)
