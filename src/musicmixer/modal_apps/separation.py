import io
import logging
import re

import modal

logger = logging.getLogger(__name__)

_STEM_TOKEN_RE = re.compile(r"[_\-.\s()]+")


def _tokenize_stem_filename(filename_stem: str) -> list[str]:
    """Split a filename stem into lowercase tokens on common delimiters."""
    return [t for t in _STEM_TOKEN_RE.split(filename_stem.lower()) if t]

app = modal.App("musicmixer-separation")

# Models are baked into /models — NOT audio-separator's /tmp default, which
# Modal remounts fresh at runtime (a /tmp bake silently re-downloads ~700MB
# per cold start).
MODEL_DIR = "/models"

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
    .run_commands(
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(model_file_dir='{MODEL_DIR}'); s.load_model('{MODEL_CKPT}')\"",
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(model_file_dir='{MODEL_DIR}'); s.load_model('{MELBAND_KARAOKE_CKPT}')\"",
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(model_file_dir='{MODEL_DIR}'); s.load_model('{MELBAND_VOCALS_CKPT}')\"",
    )
)

# GPU snapshots (experimental_options enable_gpu_snapshot) segfault on restore
# with this stack (exit 139; onnxruntime-gpu CUDA state) — CPU snapshot only.
_CLS_OPTS = dict(
    image=image,
    gpu="L40S",
    scaledown_window=120,
    enable_memory_snapshot=True,
)


def _preimport() -> None:
    """Heavy imports for the snap=True enter stage; must not touch CUDA."""
    import torch  # noqa: F401
    import soundfile  # noqa: F401
    from audio_separator.separator import Separator  # noqa: F401


def _collect_stems(output_dir, expected_stems: list[str]) -> dict[str, bytes]:
    """Map output WAVs to stem names and re-encode as float32 WAV bytes."""
    import soundfile as sf

    stems: dict[str, bytes] = {}
    for stem_file in output_dir.iterdir():
        if stem_file.suffix != ".wav":
            continue
        tokens = _tokenize_stem_filename(stem_file.stem)
        matched = [s for s in expected_stems if s in tokens]
        if len(matched) > 1:
            logger.warning(
                "File %s matched multiple stems: %s; using first: %s",
                stem_file.name, matched, matched[0],
            )
        if matched and matched[0] not in stems:
            audio_data, sr = sf.read(str(stem_file), dtype="float32")
            buf = io.BytesIO()
            sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
            stems[matched[0]] = buf.getvalue()
    return stems


def _wipe(directory) -> None:
    for f in directory.iterdir():
        f.unlink()


@app.cls(timeout=300, **_CLS_OPTS)
class InstrumentalSeparator:
    """BS-Roformer-SW 6-stem separation, model resident across calls."""

    @modal.enter(snap=True)
    def preimport(self):
        _preimport()

    @modal.enter()
    def load(self):
        from pathlib import Path

        from audio_separator.separator import Separator

        self.output_dir = Path("/root/stems-out")
        self.output_dir.mkdir(exist_ok=True)
        self.separator = Separator(
            output_dir=str(self.output_dir), model_file_dir=MODEL_DIR
        )
        self.separator.load_model(MODEL_CKPT)

    @modal.method()
    def separate(self, audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
        """Run 6-stem separation on cloud GPU.

        Accepts raw audio bytes, returns dict mapping stem name to float32
        WAV bytes. Stems: vocals, drums, bass, guitar, piano, other.
        """
        from pathlib import Path

        input_path = Path("/root") / filename
        input_path.write_bytes(audio_bytes)
        try:
            self.separator.separate(str(input_path))
            return _collect_stems(
                self.output_dir,
                ["vocals", "drums", "bass", "guitar", "piano", "other"],
            )
        finally:
            _wipe(self.output_dir)
            input_path.unlink(missing_ok=True)


@app.cls(timeout=600, **_CLS_OPTS)
class VocalSeparator:
    """Two-pass MelBand Roformer vocal separation, models resident across calls.

    Pass 1 (karaoke model): mix -> lead_vocals + karaoke_track
    Pass 2 (vocals model on karaoke_track): -> backing_vocals + instrumental
    """

    @modal.enter(snap=True)
    def preimport(self):
        _preimport()

    @modal.enter()
    def load(self):
        from pathlib import Path

        from audio_separator.separator import Separator

        self.pass1_dir = Path("/root/pass1-out")
        self.pass2_dir = Path("/root/pass2-out")
        self.pass1_dir.mkdir(exist_ok=True)
        self.pass2_dir.mkdir(exist_ok=True)

        self.sep1 = Separator(output_dir=str(self.pass1_dir), model_file_dir=MODEL_DIR)
        self.sep1.load_model(MELBAND_KARAOKE_CKPT)
        self.sep2 = Separator(output_dir=str(self.pass2_dir), model_file_dir=MODEL_DIR)
        self.sep2.load_model(MELBAND_VOCALS_CKPT)

    @modal.method()
    def separate(self, audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
        """Separate a vocal-source song into lead/backing vocals and instrumental.

        Accepts raw audio bytes, returns dict mapping stem name to float32
        WAV bytes. Stems: lead_vocals, backing_vocals, instrumental.
        """
        import soundfile as sf
        from pathlib import Path

        input_path = Path("/root") / filename
        input_path.write_bytes(audio_bytes)
        try:
            self.sep1.separate(str(input_path))

            pass1_stems = self._match_pass_outputs(self.pass1_dir)
            if "vocals" not in pass1_stems or "instrumental" not in pass1_stems:
                raise RuntimeError(
                    f"Karaoke model pass 1 did not produce expected stems. "
                    f"Got: {list(pass1_stems.keys())}. "
                    f"Files: {[f.name for f in self.pass1_dir.iterdir()]}"
                )

            self.sep2.separate(str(pass1_stems["instrumental"]))

            pass2_stems = self._match_pass_outputs(self.pass2_dir)
            if "vocals" not in pass2_stems or "instrumental" not in pass2_stems:
                raise RuntimeError(
                    f"Vocals model pass 2 did not produce expected stems. "
                    f"Got: {list(pass2_stems.keys())}. "
                    f"Files: {[f.name for f in self.pass2_dir.iterdir()]}"
                )

            result = {}
            for stem_name, stem_path in [
                ("lead_vocals", pass1_stems["vocals"]),
                ("backing_vocals", pass2_stems["vocals"]),
                ("instrumental", pass2_stems["instrumental"]),
            ]:
                audio_data, sr = sf.read(str(stem_path), dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
                result[stem_name] = buf.getvalue()
            return result
        finally:
            _wipe(self.pass1_dir)
            _wipe(self.pass2_dir)
            input_path.unlink(missing_ok=True)

    @staticmethod
    def _match_pass_outputs(pass_dir) -> dict:
        matched_files = {}
        for stem_file in pass_dir.iterdir():
            if stem_file.suffix != ".wav":
                continue
            tokens = _tokenize_stem_filename(stem_file.stem)
            matched = [s for s in ("vocals", "instrumental") if s in tokens]
            if matched:
                matched_files[matched[0]] = stem_file
        return matched_files
