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

# MelBand Roformer Karaoke model (aufr33/viperx) -- separates lead vocals
# from backing track. Used with a standard vocals/instrumental model to
# produce 3 stems: lead_vocals, backing_vocals, instrumental.
MELBAND_KARAOKE_CKPT = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"

# MelBand Roformer Vocals model (becruily) -- standard vocals/instrumental split.
# Used as the second pass on the karaoke track to separate backing vocals from
# instrumental: run on karaoke_track -> Vocals = backing_vocals, Instrumental = instrumental.
MELBAND_VOCALS_CKPT = "mel_band_roformer_karaoke_becruily.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    # audio-separator handles all models: BS-RoFormer + MelBand Roformer
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    # Pre-download all three models into the image
    .run_commands(
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MODEL_CKPT}')\"",
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MELBAND_KARAOKE_CKPT}')\"",
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MELBAND_VOCALS_CKPT}')\"",
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

        input_path.write_bytes(audio_bytes)

        separator = Separator(output_dir=str(output_dir))
        separator.load_model(MODEL_CKPT)
        separator.separate(str(input_path))

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


@app.function(image=image, gpu="L40S", timeout=600)
def separate_vocal_song_remote(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
    """Separate a vocal-source song into lead vocals, backing vocals, and instrumental.

    Uses a two-pass MelBand Roformer approach via audio-separator:
      Pass 1 (karaoke model): mix -> lead_vocals + karaoke_track
      Pass 2 (vocals model on karaoke_track): karaoke_track -> backing_vocals + instrumental

    Accepts raw audio bytes, returns dict mapping stem name to float32 WAV bytes.
    Stems: lead_vocals, backing_vocals, instrumental.
    """
    import tempfile
    import soundfile as sf
    from pathlib import Path
    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / filename
        pass1_dir = tmpdir / "pass1"
        pass2_dir = tmpdir / "pass2"
        pass1_dir.mkdir()
        pass2_dir.mkdir()

        input_path.write_bytes(audio_bytes)

        # ---- Pass 1: Karaoke model on the mix ----
        # Produces "vocals" (= lead vocals) and "instrumental" (= karaoke track)
        sep1 = Separator(output_dir=str(pass1_dir))
        sep1.load_model(MELBAND_KARAOKE_CKPT)
        sep1.separate(str(input_path))

        pass1_stems = {}
        expected_pass1 = ["vocals", "instrumental"]
        for stem_file in pass1_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in expected_pass1 if s in tokens]
            if matched:
                pass1_stems[matched[0]] = stem_file

        if "vocals" not in pass1_stems or "instrumental" not in pass1_stems:
            raise RuntimeError(
                f"Karaoke model pass 1 did not produce expected stems. "
                f"Got: {list(pass1_stems.keys())}. "
                f"Files: {[f.name for f in pass1_dir.iterdir()]}"
            )

        lead_vocals_path = pass1_stems["vocals"]
        karaoke_track_path = pass1_stems["instrumental"]

        # ---- Pass 2: Vocals model on the karaoke track ----
        # Produces "vocals" (= backing vocals) and "instrumental" (= clean instrumental)
        sep2 = Separator(output_dir=str(pass2_dir))
        sep2.load_model(MELBAND_VOCALS_CKPT)
        sep2.separate(str(karaoke_track_path))

        pass2_stems = {}
        expected_pass2 = ["vocals", "instrumental"]
        for stem_file in pass2_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in expected_pass2 if s in tokens]
            if matched:
                pass2_stems[matched[0]] = stem_file

        if "vocals" not in pass2_stems or "instrumental" not in pass2_stems:
            raise RuntimeError(
                f"Vocals model pass 2 did not produce expected stems. "
                f"Got: {list(pass2_stems.keys())}. "
                f"Files: {[f.name for f in pass2_dir.iterdir()]}"
            )

        backing_vocals_path = pass2_stems["vocals"]
        instrumental_path = pass2_stems["instrumental"]

        result = {}
        for stem_name, stem_path in [
            ("lead_vocals", lead_vocals_path),
            ("backing_vocals", backing_vocals_path),
            ("instrumental", instrumental_path),
        ]:
            audio_data, sr = sf.read(str(stem_path), dtype="float32")
            buf = io.BytesIO()
            sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
            result[stem_name] = buf.getvalue()

        return result
