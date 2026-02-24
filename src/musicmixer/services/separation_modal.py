import modal
import io

app = modal.App("musicmixer-separation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    # Bake model weights into image to eliminate cold-start download
    .run_commands(
        "python -c \"from audio_separator.separator import Separator; "
        "s = Separator(); s.load_model('BS-Roformer-Viperx-1297.ckpt')\""
    )
)


@app.function(image=image, gpu="A10G", timeout=300)
def separate_stems_remote(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
    """Run BS-RoFormer 6-stem separation on cloud GPU.

    Accepts raw audio bytes, returns dict mapping stem name to WAV bytes.
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

        # Run separation
        separator = Separator(output_dir=str(output_dir))
        separator.load_model("BS-Roformer-Viperx-1297.ckpt")
        separator.separate(str(input_path))

        # Collect output stems
        stems = {}
        expected_stems = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        for stem_file in output_dir.iterdir():
            if not stem_file.suffix == ".wav":
                continue
            # Map filename to stem name
            name_lower = stem_file.stem.lower()
            for stem_name in expected_stems:
                if stem_name in name_lower:
                    # Re-encode as float32 WAV to preserve precision
                    audio_data, sr = sf.read(str(stem_file), dtype="float32")
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
                    stems[stem_name] = buf.getvalue()
                    break

        return stems
