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

    Dispatches to Modal (cloud GPU) or local backend based on settings.stem_backend.
    """
    if settings.stem_backend == "modal":
        return _separate_modal(audio_path, output_dir, progress_callback)
    else:
        return _separate_local(audio_path, output_dir, progress_callback)


def _separate_modal(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems via Modal cloud GPU (BS-RoFormer SW, 6-stem)."""
    from musicmixer.services.separation_modal import separate_stems_remote

    if progress_callback:
        progress_callback("Uploading to cloud GPU...")

    # Read audio file as bytes
    audio_bytes = audio_path.read_bytes()

    # Call Modal function
    stem_bytes_map = separate_stems_remote.remote(
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
