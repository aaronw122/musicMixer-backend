import logging
import re
from pathlib import Path
from typing import Callable

import soundfile as sf

logger = logging.getLogger(__name__)

_STEM_TOKEN_RE = re.compile(r"[_\-.\s]+")


def _tokenize_stem_filename(filename_stem: str) -> list[str]:
    """Split a filename stem into lowercase tokens on common delimiters."""
    return [t for t in _STEM_TOKEN_RE.split(filename_stem.lower()) if t]


def separate_stems_local(
    audio_path: Path,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> dict[str, Path]:
    """Separate stems locally using htdemucs_ft (4-stem).

    Returns dict with keys: vocals, drums, bass, other.
    guitar and piano are None (htdemucs_ft only produces 4 stems).
    """
    from audio_separator.separator import Separator

    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Running local stem separation (htdemucs_ft)...")

    separator = Separator(output_dir=str(output_dir))
    separator.load_model("htdemucs_ft.yaml")
    separator.separate(str(audio_path))

    # Map output files to stem names using whole-token matching
    stems = {}
    expected_4stem = ["vocals", "drums", "bass", "other"]
    for stem_file in output_dir.iterdir():
        if not stem_file.suffix == ".wav":
            continue
        tokens = _tokenize_stem_filename(stem_file.stem)
        matched = [s for s in expected_4stem if s in tokens]
        if len(matched) > 1:
            logger.warning(
                "File %s matched multiple stems: %s; using first: %s",
                stem_file.name, matched, matched[0],
            )
        if matched:
            stem_name = matched[0]
            if stem_name not in stems:
                stems[stem_name] = stem_file

    # Re-encode stems to float32 WAV (htdemucs_ft outputs 16-bit)
    for stem_name, stem_path in stems.items():
        if stem_path is None:
            continue
        audio_data, sr = sf.read(str(stem_path), dtype="float32")
        sf.write(str(stem_path), audio_data, sr, subtype="FLOAT")

    # 4-stem model: guitar and piano are not separated
    stems.setdefault("guitar", None)
    stems.setdefault("piano", None)

    logger.info(f"Local separation complete: {list(stems.keys())}")
    return stems
