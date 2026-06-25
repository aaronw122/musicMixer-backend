"""Route-independent stage helpers for the remix endpoints.

Most stages perform input/stage work (thumbnail derivation, upload
validation/write, duration probing, downloads, cache restore) and return data;
they construct no SSE payloads and own no session lifecycle. ``api/remix.py``
translates their results into HTTP responses, preserving the route's exact
status codes and detail strings.

The one exception is ``analyze_and_checkpoint_youtube_pair``: it passes the
existing ``event_queue``/``session`` handles through to ``analyze_songs``, which
still owns progress emission and cancellation. The bridge constructs no SSE
payloads itself — it just threads those handles through.
"""

import logging
import shutil
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Callable, Iterable, Literal

if TYPE_CHECKING:
    import queue

    from musicmixer.config import Settings
    from musicmixer.models import AnalyzedSongs, CachedSong, SessionState
    from musicmixer.services.pipeline_metrics import PipelineMetrics
    from musicmixer.services.youtube import YouTubeAudioResult

logger = logging.getLogger("musicmixer.api.remix")

UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


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
        StemCacheCoordinator,
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

    # Require a ``ready`` state with a compatible validated manifest for BOTH
    # roles before skipping work. reconcile_disk lazily adopts a valid on-disk
    # role dir (legacy cache / Redis flush) into ``ready`` so existing caches
    # still short-circuit. A Redis outage leaves get_state None -> miss -> full
    # pipeline, which is the safe degraded behavior.
    coordinator = StemCacheCoordinator()

    def _role_ready(video_id: str, role) -> bool:
        try:
            coordinator.reconcile_disk(video_id, role)
            state = coordinator.get_state(video_id, role)
        except Exception:
            logger.warning(
                "Session %s: stem-cache state check failed for %s/%s; falling back",
                session_id, video_id, role, exc_info=True,
            )
            return False
        return state is not None and state.status == "ready"

    if not (
        _role_ready(cached_song_a.video_id, ROLE_VOCAL)
        and _role_ready(cached_song_b.video_id, ROLE_INSTRUMENTAL)
    ):
        logger.warning(
            "Session %s: Cached stems not ready, falling back to full pipeline",
            session_id,
        )
        return FullyCachedRestore(used_cache=False)

    # Copy only after BOTH roles are confirmed ready — otherwise "A ok, B missing"
    # would orphan A's stems on fallback.
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


@dataclass
class DownloadPairCallbacks:
    """API-owned callbacks for the YouTube download-pair stage.

    The stage never touches SSE payloads, the monotonic high-water mark, or the
    structured-error contract directly. It reaches all of those through these
    callbacks, whose implementations live in ``api/remix.py``:

    - ``check_cancelled`` keeps cancellation ownership in the API.
    - ``tag_failed_song`` wraps ``_tag_failed_song`` so a download failure carries
      its "A"/"B" slot for ``_build_error_event``. The stage re-raises the
      ORIGINAL exception object after tagging so ``error_class``/``failed_song``
      survive byte-identically.
    - ``on_download_pair_started`` / ``on_song_a_download_progress`` /
      ``on_song_b_download_progress`` / ``on_download_pair_finished`` drive the
      existing mid-download progress events through API-owned event construction
      (monotonic mapping included).
    """

    check_cancelled: Callable[[], None]
    tag_failed_song: Callable[[BaseException, Literal["A", "B"]], None]
    on_download_pair_started: Callable[[], None]
    on_song_a_download_progress: Callable[[float, str], None]
    on_song_b_download_progress: Callable[[float, str], None]
    on_download_pair_finished: Callable[[], None]


@dataclass(frozen=True)
class DownloadedPair:
    """Result of the YouTube download-pair stage.

    Carries the two download results plus the source-quality facts the
    analyze/checkpoint stage consumes (``source_quality_a``/``_b``). The wrapper
    keeps ownership of analysis, checkpointing, and the final ``run_remix`` call.
    """

    result_a: "YouTubeAudioResult"
    result_b: "YouTubeAudioResult"
    source_quality_a: str
    source_quality_b: str


def _run_sync(coro):
    """Run an async coroutine synchronously from a thread (no running event loop)."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _pre_trim_youtube_download(
    download_result: "YouTubeAudioResult | None",
    max_duration_seconds: int,
) -> None:
    """Trim long YouTube downloads before GPU-heavy processing."""
    if download_result is None:
        return

    if download_result.duration_seconds <= max_duration_seconds:
        return

    from musicmixer.services.processor import pre_trim_for_processing

    pre_trim_for_processing(
        download_result.wav_path,
        max_duration_seconds=max_duration_seconds,
    )
    download_result.duration_seconds = min(
        download_result.duration_seconds,
        max_duration_seconds,
    )


def download_youtube_pair(
    url_a: str,
    url_b: str,
    *,
    session_id: str,
    callbacks: DownloadPairCallbacks,
    cached_song_a: "CachedSong | None",
    cached_song_b: "CachedSong | None",
    settings: "Settings",
) -> DownloadedPair:
    """Download both YouTube songs concurrently, pre-trim, and tag failures.

    Both downloads are independent so a 2-thread pool overlaps them. On failure
    the offending exception is tagged with its "A"/"B" slot via
    ``callbacks.tag_failed_song`` and re-raised unchanged so the wrapper's
    ``_build_error_event`` still sees ``error_class``/``failed_song``.

    Mid-download progress events flow through API-owned callbacks; this stage
    never constructs SSE payloads. Cancellation stays with the API via
    ``callbacks.check_cancelled``.

    ``cached_song_a``/``_b`` are accepted for signature symmetry with the
    wrapper's cache-aware path; the fully-cached short-circuit is handled before
    this stage runs, so here both songs are always downloaded.
    """
    from musicmixer.services.youtube import (
        download_youtube_audio,
        extract_video_id,
    )

    upload_dir = settings.data_dir / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    callbacks.on_download_pair_started()

    # Resolve video IDs up-front so the audio cache + single-flight can key on them.
    video_id_a = extract_video_id(url_a)
    video_id_b = extract_video_id(url_b)

    def _download_a():
        return _run_sync(download_youtube_audio(
            url=url_a,
            output_dir=upload_dir,
            progress_callback=callbacks.on_song_a_download_progress,
            video_id=video_id_a,
        ))

    def _download_b():
        return _run_sync(download_youtube_audio(
            url=url_b,
            output_dir=upload_dir,
            progress_callback=callbacks.on_song_b_download_progress,
            video_id=video_id_b,
        ))

    # song_a_future → url_a → Song "A"; song_b_future → url_b → Song "B".
    # On failure we tag the exception with its slot so the SSE error handler
    # can emit `failed_song` to the frontend.
    with ThreadPoolExecutor(max_workers=2) as dl_executor:
        song_a_future = dl_executor.submit(_download_a)
        song_b_future = dl_executor.submit(_download_b)

        try:
            result_a = song_a_future.result(timeout=300)
        except Exception as exc:
            # If Song A fails, cancel Song B and re-raise
            callbacks.tag_failed_song(exc, "A")
            song_b_future.cancel()
            dl_executor.shutdown(wait=False, cancel_futures=True)
            raise

        try:
            result_b = song_b_future.result(timeout=300)
        except Exception as exc:
            # Song B failed after Song A succeeded
            callbacks.tag_failed_song(exc, "B")
            dl_executor.shutdown(wait=False, cancel_futures=True)
            raise

    callbacks.on_download_pair_finished()

    max_processing_duration = settings.processing_max_duration_seconds
    _pre_trim_youtube_download(result_a, max_processing_duration)
    _pre_trim_youtube_download(result_b, max_processing_duration)

    # Check cancellation before starting heavy pipeline work
    callbacks.check_cancelled()

    logger.info(
        "Session %s: YouTube downloads complete. A=%r (%ds, %s %dkbps), B=%r (%ds, %s %dkbps)",
        session_id,
        result_a.title, int(result_a.duration_seconds),
        result_a.source_codec, result_a.source_bitrate,
        result_b.title, int(result_b.duration_seconds),
        result_b.source_codec, result_b.source_bitrate,
    )

    source_quality_a = f"youtube-{result_a.source_codec}-{result_a.source_bitrate}kbps"
    source_quality_b = f"youtube-{result_b.source_codec}-{result_b.source_bitrate}kbps"

    logger.info(
        "Session %s: Source quality: A=%s, B=%s",
        session_id, source_quality_a, source_quality_b,
    )

    return DownloadedPair(
        result_a=result_a,
        result_b=result_b,
        source_quality_a=source_quality_a,
        source_quality_b=source_quality_b,
    )


def _checkpoint_song(
    *,
    url: str,
    role,
    title: str,
    meta,
    lyrics,
    already_cached: bool,
    session_id: str,
) -> None:
    """Persist one song's metadata to the cache, failure-isolated.

    Called per-song right after analysis. Any exception is swallowed and logged:
    a cache write must never crash the pipeline the user is waiting on, and one
    song's failure must not abort the other song's checkpoint. Metadata is skipped
    when the song came from the medium-cache path (it's already persisted).

    Stems are NOT published here: the lease-owning separation wrapper
    (``get_or_create_cached_stems``) is the sole stem publisher, fencing its
    atomic replace behind the Redis lease. Publishing here too would double-write
    and race the wrapper for the same (video_id, role).
    """
    from musicmixer.services.song_cache import cache_song_metadata
    from musicmixer.services.youtube import extract_video_id

    video_id = extract_video_id(url)
    if video_id is None:
        return
    if already_cached:
        return
    try:
        cache_song_metadata(
            video_id=video_id, title=title, artist="", meta=meta, lyrics=lyrics,
        )
        logger.info(
            "Session %s: checkpointed metadata for song %s (role=%s)", session_id, video_id, role,
        )
    except Exception:
        logger.warning(
            "Session %s: failed to checkpoint song %s (role=%s); continuing",
            session_id, video_id, role, exc_info=True,
        )


@dataclass(frozen=True)
class AnalyzedRemix:
    """Result of the YouTube analyze/checkpoint stage.

    Carries the ``AnalyzedSongs`` and ``PipelineMetrics`` the visible
    ``run_remix`` call in ``api/remix.py`` consumes, plus the source-quality
    strings threaded back through from the download stage. The wrapper keeps the
    final ``run_remix`` call and the average-duration update.
    """

    analysis: "AnalyzedSongs"
    metrics: "PipelineMetrics"
    source_quality_a: str
    source_quality_b: str


def analyze_and_checkpoint_youtube_pair(
    downloaded: DownloadedPair,
    *,
    url_a: str,
    url_b: str,
    session_id: str,
    event_queue: "queue.Queue",
    session: "SessionState",
    cached_song_a: "CachedSong | None",
    cached_song_b: "CachedSong | None",
) -> AnalyzedRemix:
    """Analyze the downloaded pair, then checkpoint each song to the cache.

    Builds ``PipelineMetrics`` (with video IDs populated), runs ``analyze_songs``
    over the two downloads, and persists each song's stems + metadata before the
    expensive remix/render via ``_checkpoint_song`` (failure-isolated, A then B).

    ``analyze_songs`` owns its own progress emission and cancellation checks
    through ``event_queue``/``session``; this stage builds no SSE payloads and
    owns no session lifecycle. ``session`` is a pass-through to that existing
    call, not an ownership handoff.
    """
    from musicmixer.services.pipeline import analyze_songs
    from musicmixer.services.pipeline_metrics import PipelineMetrics
    from musicmixer.services.song_cache import ROLE_INSTRUMENTAL, ROLE_VOCAL
    from musicmixer.services.youtube import extract_video_id

    result_a = downloaded.result_a
    result_b = downloaded.result_b
    source_quality_a = downloaded.source_quality_a
    source_quality_b = downloaded.source_quality_b

    video_id_a = extract_video_id(url_a)
    video_id_b = extract_video_id(url_b)

    metrics = PipelineMetrics(session_id=session_id)
    metrics.song_a_video_id = video_id_a or ""
    metrics.song_b_video_id = video_id_b or ""

    analysis = analyze_songs(
        session_id=session_id,
        song_a_path=str(result_a.wav_path),
        song_b_path=str(result_b.wav_path),
        event_queue=event_queue,
        session=session,
        song_a_original_filename=result_a.title,
        song_b_original_filename=result_b.title,
        source_quality_a=source_quality_a,
        source_quality_b=source_quality_b,
        metrics=metrics,
        cached_meta_a=cached_song_a.meta if cached_song_a else None,
        cached_meta_b=cached_song_b.meta if cached_song_b else None,
        cached_lyrics_a=cached_song_a.lyrics if cached_song_a else None,
        cached_lyrics_b=cached_song_b.lyrics if cached_song_b else None,
        video_id_a=video_id_a,
        video_id_b=video_id_b,
    )

    _checkpoint_song(
        url=url_a, role=ROLE_VOCAL, title=result_a.title,
        meta=analysis.meta_a, lyrics=analysis.lyrics_a,
        already_cached=cached_song_a is not None, session_id=session_id,
    )
    _checkpoint_song(
        url=url_b, role=ROLE_INSTRUMENTAL, title=result_b.title,
        meta=analysis.meta_b, lyrics=analysis.lyrics_b,
        already_cached=cached_song_b is not None, session_id=session_id,
    )

    return AnalyzedRemix(
        analysis=analysis,
        metrics=metrics,
        source_quality_a=source_quality_a,
        source_quality_b=source_quality_b,
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
    """Stream an uploaded file to ``dest`` in fixed-size chunks.

    Raises ``UploadTooLargeError`` if the cumulative size exceeds ``max_bytes``.
    The size check happens mid-stream so an oversized upload is rejected without
    buffering the whole payload.
    """
    file.seek(0)
    chunks = []
    total = 0
    while True:
        chunk = file.read(UPLOAD_CHUNK_SIZE_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError
        chunks.append(chunk)
    data = b"".join(chunks)
    dest.write_bytes(data)
