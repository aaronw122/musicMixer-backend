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
    """Separate audio into 6 instrumental stems via Modal or local backend (BS-RoFormer).

    Returns mapping of stem name to WAV file path.
    Used for Song B (instrumental source).
    """
    if settings.stem_backend == "modal":
        return _separate_modal(audio_path, output_dir, progress_callback)
    else:
        return _separate_local(audio_path, output_dir, progress_callback)


def separate_vocal_song(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate audio into lead/backing vocals via MelBand Roformer Karaoke.

    Returns mapping: {lead_vocals, backing_vocals, instrumental} -> WAV path.
    Used for Song A (vocal source).

    On Modal backend, calls the separate_vocal_song_remote Modal function.
    On local backend, falls back to BS-RoFormer separation and maps
    the "vocals" stem to "lead_vocals" (no backing vocal split available locally).
    """
    if settings.stem_backend == "modal":
        return _separate_vocal_modal(audio_path, output_dir, progress_callback)
    else:
        # Local fallback: use htdemucs_ft and remap "vocals" -> "lead_vocals"
        stems = _separate_local(audio_path, output_dir, progress_callback)
        return _remap_local_vocal_stems(stems, output_dir)


def _separate_vocal_modal(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate vocals via Modal cloud GPU (MelBand Roformer Karaoke, 3-stem)."""
    import modal

    if progress_callback:
        progress_callback("Uploading to cloud GPU (vocal separation)...")

    audio_bytes = audio_path.read_bytes()

    # Call the MelBand Roformer separation function (built by Agent A)
    separate_fn = modal.Function.from_name(
        "musicmixer-separation", "separate_vocal_song_remote"
    )
    stem_bytes_map = separate_fn.remote(
        audio_bytes=audio_bytes,
        filename=audio_path.name,
    )

    if progress_callback:
        progress_callback("Vocal stems received, saving...")

    # Validate expected stems
    expected = {"lead_vocals", "backing_vocals", "instrumental"}
    received = set(stem_bytes_map.keys())
    if received != expected:
        missing = expected - received
        extra = received - expected
        raise RuntimeError(
            f"Expected 3 stems ({expected}), got {len(received)} ({received}). "
            f"Missing: {missing}. Extra: {extra}. Check MelBand Roformer setup."
        )

    # Save stems to disk as WAV, validate float32
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_paths = {}
    for stem_name, wav_bytes in stem_bytes_map.items():
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


def _remap_local_vocal_stems(
    stems: dict[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Remap local fallback stems to the vocal-song vocabulary.

    Local separation (htdemucs_ft) produces "vocals" but not lead/backing split.
    Rename "vocals" -> "lead_vocals" and discard instrumental stems (Song A
    only needs vocal stems; Song B handles instrumentals).
    """
    import shutil

    remapped: dict[str, Path] = {}

    if "vocals" in stems:
        src = stems["vocals"]
        dst = output_dir / "lead_vocals.wav"
        if src != dst:
            shutil.copy2(src, dst)
        remapped["lead_vocals"] = dst

    # Keep "other" as potential backing vocal proxy if present
    # (htdemucs_ft doesn't separate backing vocals, so we skip it)

    # Keep instrumental stem if present (for structure analysis)
    for name in ("drums", "bass", "other"):
        if name in stems and stems[name] is not None:
            remapped[name] = stems[name]

    return remapped


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
