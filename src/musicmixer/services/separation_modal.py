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
MELBAND_KARAOKE_CONFIG = "config_mel_band_roformer_karaoke.yaml"
MELBAND_KARAOKE_HF_REPO = "jarredou/aufr33-viperx-karaoke-melroformer-model"

# MelBand Roformer Vocals model (becruily) -- standard vocals/instrumental split.
# Used as the second pass on the karaoke track to separate backing vocals from
# instrumental: run on karaoke_track -> Vocals = backing_vocals, Instrumental = instrumental.
MELBAND_VOCALS_CKPT = "mel_band_roformer_karaoke_becruily.ckpt"
MELBAND_VOCALS_CONFIG = "config_karaoke_becruily.yaml"
MELBAND_VOCALS_HF_REPO = "becruily/mel-band-roformer-karaoke"

# Directory inside the Modal image where MSST weights + configs are stored
MSST_WEIGHTS_DIR = "/msst_models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    # BS-RoFormer via audio-separator (existing 6-stem pipeline)
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    .run_commands(
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MODEL_CKPT}')\"",
    )
    # MSST for MelBand Roformer inference (separate pip install block)
    .pip_install(
        "music-source-separation-training",
        "librosa",
        "numpy",
        "huggingface_hub",
    )
    # Pre-download MelBand Roformer weights into the image
    .run_commands(
        f"mkdir -p {MSST_WEIGHTS_DIR}",
        # Download karaoke model (aufr33/viperx) -- separates lead vocals
        f'python -c "'
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{MELBAND_KARAOKE_HF_REPO}', "
        f"filename='{MELBAND_KARAOKE_CKPT}', local_dir='{MSST_WEIGHTS_DIR}'); "
        f"hf_hub_download(repo_id='{MELBAND_KARAOKE_HF_REPO}', "
        f"filename='{MELBAND_KARAOKE_CONFIG}', local_dir='{MSST_WEIGHTS_DIR}')"
        f'"',
        # Download vocals model (viperx) -- standard vocals/instrumental split
        f'python -c "'
        f"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download(repo_id='{MELBAND_VOCALS_HF_REPO}', "
        f"filename='{MELBAND_VOCALS_CKPT}', local_dir='{MSST_WEIGHTS_DIR}'); "
        f"hf_hub_download(repo_id='{MELBAND_VOCALS_HF_REPO}', "
        f"filename='{MELBAND_VOCALS_CONFIG}', local_dir='{MSST_WEIGHTS_DIR}')"
        f'"',
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


def _load_msst_model(config_path: str, checkpoint_path: str, device: str = "cuda"):
    """Load an MSST MelBand Roformer model from config + checkpoint.

    Returns (model, config) tuple ready for inference.
    """
    import torch
    import torch.nn as nn
    from utils.settings import get_model_from_config
    from utils.model_utils import load_start_checkpoint

    model, config = get_model_from_config("mel_band_roformer", config_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    # Use a simple namespace for the args that load_start_checkpoint expects
    class _Args:
        start_check_point = checkpoint_path

    load_start_checkpoint(_Args(), model, checkpoint, type_="inference")
    model = model.to(device)
    model.eval()
    return model, config


def _run_msst_inference(model, config, audio_np, device: str = "cuda"):
    """Run MSST inference on a numpy audio array.

    Args:
        model: Loaded MSST model.
        config: MSST config object.
        audio_np: numpy array of shape (channels, samples) at config sample rate.
        device: torch device string.

    Returns:
        dict mapping instrument name to numpy array of shape (channels, samples).
    """
    import numpy as np
    from utils.audio_utils import normalize_audio, denormalize_audio
    from utils.model_utils import bigshifts_wrapper, prefer_target_instrument

    mix = audio_np.copy()

    # Normalize if the config requires it
    norm_params = None
    if hasattr(config, "inference") and hasattr(config.inference, "normalize"):
        if config.inference.normalize:
            mix, norm_params = normalize_audio(mix)

    # Run separation
    waveforms = bigshifts_wrapper(
        config, model, mix, device,
        model_type="mel_band_roformer",
        pbar=False,
        bigshifts=1,
    )

    # Extract instrumental (residual) if not already present
    instruments = prefer_target_instrument(config)[:]
    if "instrumental" not in instruments and "instrumental" not in waveforms:
        # Compute residual: instrumental = original_mix - target_stem
        target = instruments[0]
        waveforms["instrumental"] = audio_np - waveforms[target]

    # Denormalize if we normalized
    if norm_params is not None:
        for key in waveforms:
            waveforms[key] = denormalize_audio(waveforms[key], norm_params)

    return waveforms


def _numpy_to_wav_bytes(audio_np, sr: int) -> bytes:
    """Convert a numpy array (channels, samples) to float32 WAV bytes."""
    import soundfile as sf

    buf = io.BytesIO()
    # soundfile expects (samples, channels)
    sf.write(buf, audio_np.T, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@app.function(image=image, gpu="L40S", timeout=600)
def separate_vocal_song_remote(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, bytes]:
    """Separate a vocal-source song into lead vocals, backing vocals, and instrumental.

    Uses a two-pass MelBand Roformer approach:
      Pass 1 (karaoke model): mix -> karaoke_track + lead_vocals (residual)
      Pass 2 (vocals model on karaoke_track): karaoke_track -> backing_vocals + instrumental

    Accepts raw audio bytes, returns dict mapping stem name to float32 WAV bytes.
    Stems: lead_vocals, backing_vocals, instrumental.
    """
    import os
    import tempfile
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load both models ----
    karaoke_config_path = os.path.join(MSST_WEIGHTS_DIR, MELBAND_KARAOKE_CONFIG)
    karaoke_ckpt_path = os.path.join(MSST_WEIGHTS_DIR, MELBAND_KARAOKE_CKPT)
    vocals_config_path = os.path.join(MSST_WEIGHTS_DIR, MELBAND_VOCALS_CONFIG)
    vocals_ckpt_path = os.path.join(MSST_WEIGHTS_DIR, MELBAND_VOCALS_CKPT)

    karaoke_model, karaoke_config = _load_msst_model(karaoke_config_path, karaoke_ckpt_path, device)
    vocals_model, vocals_config = _load_msst_model(vocals_config_path, vocals_ckpt_path, device)

    # ---- Load audio ----
    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_input = f.name

    try:
        sr = getattr(karaoke_config.audio, "sample_rate", 44100)
        mix, _ = librosa.load(tmp_input, sr=sr, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])  # mono -> stereo
    finally:
        os.unlink(tmp_input)

    # ---- Pass 1: Karaoke model -> karaoke_track + lead_vocals ----
    # The karaoke model's target instrument is "karaoke" (= backing track).
    # lead_vocals = mix - karaoke_track
    karaoke_result = _run_msst_inference(karaoke_model, karaoke_config, mix, device)

    karaoke_track = karaoke_result.get("karaoke")
    if karaoke_track is None:
        raise RuntimeError(
            f"Karaoke model did not produce 'karaoke' stem. "
            f"Got keys: {list(karaoke_result.keys())}"
        )
    lead_vocals = mix - karaoke_track

    # Free karaoke model memory before loading next pass
    del karaoke_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Pass 2: Vocals model on karaoke_track -> backing_vocals + instrumental ----
    # The becruily vocals model separates into Vocals + Instrumental.
    # When run on the karaoke_track (which contains backing vocals + instrumental),
    # the Vocals output = backing_vocals, Instrumental output = instrumental.
    vocals_result = _run_msst_inference(vocals_model, vocals_config, karaoke_track, device)

    # The becruily config uses capitalized instrument names: "Vocals", "Instrumental"
    backing_vocals = vocals_result.get("Vocals") or vocals_result.get("vocals")
    instrumental = vocals_result.get("Instrumental") or vocals_result.get("instrumental")

    if backing_vocals is None or instrumental is None:
        raise RuntimeError(
            f"Vocals model did not produce expected stems. "
            f"Got keys: {list(vocals_result.keys())}"
        )

    del vocals_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Encode as float32 WAV bytes ----
    return {
        "lead_vocals": _numpy_to_wav_bytes(lead_vocals, sr),
        "backing_vocals": _numpy_to_wav_bytes(backing_vocals, sr),
        "instrumental": _numpy_to_wav_bytes(instrumental, sr),
    }
