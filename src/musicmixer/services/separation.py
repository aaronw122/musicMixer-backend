from pathlib import Path
from typing import Callable
import logging
import io
import soundfile as sf

from musicmixer.config import settings

logger = logging.getLogger(__name__)

# Stem names for vocal-song separation (Song A via MelBand Roformer Karaoke)
VOCAL_SONG_STEMS = {"lead_vocals", "backing_vocals", "instrumental"}


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate audio into stems via Modal or local backend (BS-RoFormer, 6-stem).

    Used for Song B (instrumental source). Returns mapping of stem name to WAV file path.
    Stems: vocals, drums, bass, guitar, piano, other.
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
    """Separate a vocal-source song into lead vocals, backing vocals, and instrumental.

    Used for Song A (vocal source). Uses MelBand Roformer Karaoke on Modal,
    falls back to htdemucs_ft locally (which only produces vocals + instrumental;
    backing_vocals will be None in the local fallback).

    Returns mapping of stem name to WAV file path (or None for unavailable stems).
    Stems: lead_vocals, backing_vocals, instrumental.
    """
    if settings.stem_backend == "modal":
        return _separate_vocal_song_modal(audio_path, output_dir, progress_callback)
    else:
        return _separate_vocal_song_local(audio_path, output_dir, progress_callback)


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
    return _save_stem_bytes(stem_bytes_map, output_dir)


def _separate_vocal_song_modal(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate vocal song via Modal cloud GPU (MelBand Roformer Karaoke, 3-stem).

    Two-pass approach:
      Pass 1 (karaoke model): mix -> karaoke_track, lead_vocals = mix - karaoke
      Pass 2 (vocals model on karaoke_track): backing_vocals, instrumental
    """
    import modal

    if progress_callback:
        progress_callback("Uploading vocal song to cloud GPU...")

    audio_bytes = audio_path.read_bytes()

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
    expected = VOCAL_SONG_STEMS
    received = set(stem_bytes_map.keys())
    if received != expected:
        missing = expected - received
        extra = received - expected
        raise RuntimeError(
            f"Expected vocal-song stems ({expected}), got ({received}). "
            f"Missing: {missing}. Extra: {extra}."
        )

    return _save_stem_bytes(stem_bytes_map, output_dir)


def _save_stem_bytes(stem_bytes_map: dict[str, bytes], output_dir: Path) -> dict[str, Path]:
    """Save stem WAV bytes to disk, validating float32 format.

    Returns mapping of stem name to file path.
    """
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


def _separate_vocal_song_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate vocal song locally using htdemucs_ft (graceful fallback).

    htdemucs_ft only produces vocals + instrumental (no lead/backing split).
    Maps vocals -> lead_vocals, sets backing_vocals = None.
    """
    from musicmixer.services.separation_local import separate_vocal_song_local
    return separate_vocal_song_local(audio_path, output_dir, progress_callback)


def _separate_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems locally using htdemucs_ft (4-stem fallback)."""
    from musicmixer.services.separation_local import separate_stems_local
    return separate_stems_local(audio_path, output_dir, progress_callback)
