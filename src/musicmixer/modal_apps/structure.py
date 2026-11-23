"""Modal GPU inference for SongFormer song structure detection.

Defines a Modal app that runs SongFormer on an L40S GPU. Called by
structure_ml.py; on failure the pipeline falls back to heuristics.
"""

from __future__ import annotations

import logging
import time

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


def _build_songformer_model():
    """Build SongFormer on CPU (snapshot-safe: must not touch CUDA).

    Bypasses both AutoModel (check_imports issue) AND from_pretrained
    (meta tensor issue). Loads the model the same way the official
    README does: direct instantiation + manual weight loading.
    """
    import json
    import os
    import sys

    from safetensors.torch import load_file

    if _MODEL_DIR not in sys.path:
        sys.path.insert(0, _MODEL_DIR)

    from configuration_songformer import SongFormerConfig
    from modeling_songformer import SongFormerModel

    logger.info("Loading SongFormer model from %s", _MODEL_DIR)
    t0 = time.monotonic()

    config_path = os.path.join(_MODEL_DIR, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = SongFormerConfig(**config_dict)

    model = SongFormerModel(config)

    weights_path = os.path.join(_MODEL_DIR, "model.safetensors")
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logger.info("SongFormer model built in %.1fs", time.monotonic() - t0)
    return model


# CPU snapshot captures the 18s model build; only the CUDA move runs per cold start.
@app.cls(
    image=image,
    gpu="L40S",
    timeout=300,
    scaledown_window=120,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
)
class SongFormer:
    @modal.enter(snap=True)
    def build(self):
        self.model = _build_songformer_model()

    @modal.enter()
    def to_gpu(self):
        import torch

        self.model.to(torch.device("cuda:0"))

    @modal.method()
    def analyze(self, audio_bytes: bytes, filename: str = "input.wav") -> list[dict]:
        """Run SongFormer inference on GPU.

        Accepts audio file bytes (Modal containers can't access the caller's
        filesystem). Returns a list of segment dicts with keys ``label``,
        ``start``, and ``end`` (seconds, float).
        """
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / filename
            input_path.write_bytes(audio_bytes)

            logger.info("Running SongFormer inference on %s", filename)
            t0 = time.monotonic()
            raw_segments: list[dict] = self.model(str(input_path))
            logger.info(
                "Inference completed in %.1fs (%d segments)",
                time.monotonic() - t0,
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
