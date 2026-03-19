"""Modal GPU inference for SongFormer song structure detection.

Defines a Modal app that runs SongFormer on a T4 GPU. Called by
structure_ml.py as the primary inference backend, with local CPU
as fallback.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

import modal

logger = logging.getLogger(__name__)

_PINNED_REVISION = "5ac5227fccf286519464fdf211e15b606898408e"
_HF_REPO = "ASLP-lab/SongFormer"
_MODEL_DIR = "/root/songformer-model"

app = modal.App("musicmixer-songformer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "transformers",
        "huggingface_hub",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "tqdm",
        "muq",
        "ema_pytorch",
        "x-transformers",
        "msaf",
        "loguru",
        "omegaconf",
    )
    # Download full SongFormer repo to a fixed path.
    # Only skip SongFormer.pt (duplicate of safetensors).
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{_HF_REPO}', revision='{_PINNED_REVISION}', "
        f"local_dir='{_MODEL_DIR}', "
        "repo_type='model', local_dir_use_symlinks=False, "
        "ignore_patterns=['SongFormer.pt'])"
        "\"",
    )
    # Add model dir to PYTHONPATH so SongFormer's absolute sibling
    # imports resolve as normal Python imports.
    .env({"PYTHONPATH": _MODEL_DIR, "SONGFORMER_LOCAL_DIR": _MODEL_DIR})
)


@lru_cache(maxsize=1)
def _load_songformer_model():
    """Load SongFormer once per warm Modal container.

    Bypasses both AutoModel (check_imports issue) AND from_pretrained
    (meta tensor issue). Loads the model the same way the official
    README does: direct instantiation + manual weight loading.
    """
    import json
    import os
    import sys

    import torch
    from safetensors.torch import load_file

    if _MODEL_DIR not in sys.path:
        sys.path.insert(0, _MODEL_DIR)

    from configuration_songformer import SongFormerConfig
    from modeling_songformer import SongFormerModel

    logger.info("Loading SongFormer model from %s", _MODEL_DIR)
    t0 = time.monotonic()

    # Load config from disk
    config_path = os.path.join(_MODEL_DIR, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = SongFormerConfig(**config_dict)

    # Instantiate model on CPU (avoids meta tensor context)
    model = SongFormerModel(config)

    # Load weights from safetensors
    weights_path = os.path.join(_MODEL_DIR, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(torch.device("cuda:0"))
    model.eval()

    elapsed = time.monotonic() - t0
    logger.info("SongFormer model loaded in %.1fs", elapsed)
    return model


@app.function(image=image, gpu="T4", timeout=300)
def analyze_structure_remote(audio_bytes: bytes, filename: str = "input.wav") -> list[dict]:
    """Run SongFormer inference on GPU.

    Accepts audio file bytes (Modal containers can't access the caller's
    filesystem). Returns a list of segment dicts with keys ``label``,
    ``start``, and ``end`` (seconds, float).
    """
    import logging as _logging
    import sys
    import tempfile
    from pathlib import Path

    _logger = _logging.getLogger(__name__)

    if _MODEL_DIR not in sys.path:
        sys.path.insert(0, _MODEL_DIR)

    model = _load_songformer_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / filename
        input_path.write_bytes(audio_bytes)

        _logger.info("Running SongFormer inference on %s", filename)
        t0 = time.monotonic()

        raw_segments: list[dict] = model(str(input_path))

        elapsed = time.monotonic() - t0
        _logger.info(
            "Inference completed in %.1fs (%d segments)",
            elapsed,
            len(raw_segments),
        )

    return [
        {
            "label": str(seg["label"]),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
        }
        for seg in raw_segments
    ]
