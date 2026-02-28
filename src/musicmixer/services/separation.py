from pathlib import Path
from typing import Callable
import logging
import io
import soundfile as sf

from musicmixer.config import settings

logger = logging.getLogger(__name__)


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate audio into stems. Returns mapping of stem name to WAV file path.

    Checks the stem cache first (keyed by SHA-256 of input file). On cache
    hit, copies cached stems to *output_dir* and returns immediately. On
    cache miss, dispatches to Modal or local backend, then caches the result.
    """
    from musicmixer.services.stem_cache import (
        cache_stems,
        get_cache_key,
        get_cached_stems,
    )

    # Check stem cache
    if settings.stem_cache_enabled:
        cache_key = get_cache_key(audio_path)
        if get_cached_stems(cache_key, output_dir):
            if progress_callback:
                progress_callback("Stems loaded from cache")
            # Build stem_paths from the copied files
            stem_paths = {}
            for wav in output_dir.glob("*.wav"):
                stem_paths[wav.stem] = wav
            logger.info("Using cached stems for %s (%s)", audio_path.name, cache_key[:12])
            return stem_paths
    else:
        cache_key = None

    # Cache miss -- run separation
    if settings.stem_backend == "modal":
        result = _separate_modal(audio_path, output_dir, progress_callback)
    else:
        result = _separate_local(audio_path, output_dir, progress_callback)

    # Cache the result for future use
    if cache_key is not None:
        try:
            cache_stems(cache_key, output_dir)
        except Exception:
            logger.warning("Failed to cache stems for %s", audio_path.name, exc_info=True)

    return result


def _separate_modal(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems via Modal cloud GPU (BS-RoFormer SW, 6-stem)."""
    import modal

    if progress_callback:
        progress_callback("Uploading to cloud GPU...")

    # Read audio file as bytes
    audio_bytes = audio_path.read_bytes()

    # Look up the deployed function by name
    separate_fn = modal.Function.from_name(
        "musicmixer-separation", "separate_stems_remote"
    )
    stem_bytes_map = separate_fn.remote(
        audio_bytes=audio_bytes,
        filename=audio_path.name,
    )

    if progress_callback:
        progress_callback("Stems received, saving...")

    # Validate stem count
    expected = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    received = set(stem_bytes_map.keys())
    if received != expected:
        missing = expected - received
        extra = received - expected
        raise RuntimeError(
            f"Expected 6 stems ({expected}), got {len(received)} ({received}). "
            f"Missing: {missing}. Extra: {extra}. Check BS-RoFormer checkpoint."
        )

    # Save stems to disk as WAV, validate float32
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_paths = {}
    for stem_name, wav_bytes in stem_bytes_map.items():
        # Validate float32
        info = sf.info(io.BytesIO(wav_bytes))
        if info.subtype != "FLOAT":
            logger.warning(
                f"Stem {stem_name} is {info.subtype}, expected FLOAT. "
                "Precision may be lost."
            )

        out_path = output_dir / f"{stem_name}.wav"
        out_path.write_bytes(wav_bytes)
        stem_paths[stem_name] = out_path

    return stem_paths


def _separate_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems locally using htdemucs_ft (4-stem fallback)."""
    from musicmixer.services.separation_local import separate_stems_local
    return separate_stems_local(audio_path, output_dir, progress_callback)
