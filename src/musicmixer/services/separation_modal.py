import io
import logging
import re

import modal

logger = logging.getLogger(__name__)

_STEM_TOKEN_RE = re.compile(r"[_\-.\s()]+")  # includes parens from audio-separator output


def _tokenize_stem_filename(filename_stem: str) -> list[str]:
    """Split a filename stem into lowercase tokens on common delimiters."""
    return [t for t in _STEM_TOKEN_RE.split(filename_stem.lower()) if t]

app = modal.App("musicmixer-separation")

MODEL_CKPT = "BS-Roformer-SW.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    # Pre-download model weights into image to avoid cold-start download
    .run_commands(
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MODEL_CKPT}')\"",
    )
)


@app.function(image=image, gpu="L40S", timeout=300)
def separate_stems_remote(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
    """Run BS-Roformer-SW 6-stem separation on cloud GPU.

    Accepts raw audio bytes, returns dict mapping stem name to WAV bytes.
    Stems: vocals, drums, bass, guitar, piano, other.
    """
    import tempfile
    import soundfile as sf
    from pathlib import Path
    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / filename
        output_dir = tmpdir / "stems"
        output_dir.mkdir()

        # Write input file
        input_path.write_bytes(audio_bytes)

        # Run separation with 6-stem model
        separator = Separator(output_dir=str(output_dir))
        separator.load_model(MODEL_CKPT)
        separator.separate(str(input_path))

        # Collect output stems
        stems = {}
        expected_stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        for stem_file in output_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            # Map filename to stem name using whole-token matching
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in expected_stems if s in tokens]
            if len(matched) > 1:
                logger.warning(
                    "File %s matched multiple stems: %s; using first: %s",
                    stem_file.name, matched, matched[0],
                )
            if matched:
                stem_name = matched[0]
                if stem_name not in stems:
                    # Re-encode as float32 WAV to preserve precision
                    audio_data, sr = sf.read(str(stem_file), dtype="float32")
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
                    stems[stem_name] = buf.getvalue()

        return stems
