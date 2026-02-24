import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def overlay_and_export(
    vocal_stems: dict[str, Path],
    instrumental_stems: dict[str, Path],
    output_path: Path,
    target_sr: int = 44100,
) -> Path:
    """Overlay vocal and instrumental stems, export as MP3.

    Day 1: No tempo matching, no key matching, no LUFS normalization.
    Just load, standardize to same length, sum, and export.
    """
    # Load all stems as float32
    all_audio = []
    max_length = 0

    for stem_name, stem_path in {**vocal_stems, **instrumental_stems}.items():
        if stem_path is None:
            continue  # Skip missing stems (4-stem fallback)

        audio, sr = sf.read(str(stem_path), dtype="float32")

        # Resample if needed (should be 44.1kHz from BS-RoFormer, but verify)
        if sr != target_sr:
            import librosa
            logger.warning(f"Stem {stem_name} at {sr}Hz, resampling to {target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        all_audio.append((stem_name, audio))
        max_length = max(max_length, len(audio))

    # Pad shorter stems to match longest
    padded = []
    for stem_name, audio in all_audio:
        if len(audio) < max_length:
            pad_length = max_length - len(audio)
            audio = np.pad(audio, ((0, pad_length), (0, 0)), mode="constant")
        padded.append(audio)

    # Sum all stems (float32 -- no clipping during summation)
    mixed = np.sum(padded, axis=0)

    # Basic peak normalization to prevent clipping (no LUFS -- Day 1)
    peak = np.max(np.abs(mixed))
    if peak > 0.95:
        mixed = mixed * (0.95 / peak)

    # Export via ffmpeg (float32 WAV -> MP3 320kbps)
    # DO NOT use pydub for export -- it quantizes to 16-bit internally
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    sf.write(str(tmp_wav), mixed, target_sr, subtype="FLOAT")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "320k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr.decode()}")
        raise RuntimeError(f"ffmpeg export failed: {result.stderr.decode()[:500]}")

    # Clean up temp WAV
    tmp_wav.unlink(missing_ok=True)

    logger.info(f"Exported remix: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path
