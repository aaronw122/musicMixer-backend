import logging
from pathlib import Path
from typing import Callable

import soundfile as sf

logger = logging.getLogger(__name__)


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

    # Map output files to stem names
    stems = {}
    expected_4stem = ["vocals", "drums", "bass", "other"]
    for stem_file in output_dir.iterdir():
        if not stem_file.suffix == ".wav":
            continue
        name_lower = stem_file.stem.lower()
        for stem_name in expected_4stem:
            if stem_name in name_lower:
                stems[stem_name] = stem_file
                break

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
