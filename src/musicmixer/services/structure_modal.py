"""Modal GPU inference for SongFormer song structure detection.

Defines a Modal app that runs SongFormer on a T4 GPU. Called by
structure_ml.py as the primary inference backend, with local CPU
as fallback.
"""

import logging
import time

import modal

logger = logging.getLogger(__name__)

# Pin to a known-good commit to avoid silent model changes.
# Fill in after validating the model locally (Task 1.1/1.2).
_PINNED_REVISION = "PINNED_COMMIT_SHA"

_HF_REPO = "ASLP-lab/SongFormer"

app = modal.App("musicmixer-songformer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "transformers",
        "librosa",
        "soundfile",
        "numpy",
        "muq",
    )
    # Pre-download model weights into the image so runtime loads are local-only.
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{_HF_REPO}', revision='{_PINNED_REVISION}')"
        "\"",
    )
)


@app.function(image=image, gpu="T4", timeout=300)
def analyze_structure_remote(audio_path: str) -> list[dict]:
    """Run SongFormer inference on GPU.

    Accepts a path to an audio file, returns a list of segment dicts
    with keys ``label``, ``start``, and ``end`` (seconds, float).

    The exact SongFormer inference API may need adjustment after local
    validation (Task 1.1/1.2). The inference call is isolated below
    to make updates straightforward.
    """
    import logging as _logging

    from transformers import AutoModel

    _logger = _logging.getLogger(__name__)

    _logger.info("Loading SongFormer model from cached weights")
    t0 = time.monotonic()

    model = AutoModel.from_pretrained(
        _HF_REPO,
        trust_remote_code=True,
        revision=_PINNED_REVISION,
        local_files_only=True,
    )

    load_elapsed = time.monotonic() - t0
    _logger.info("Model loaded in %.1fs", load_elapsed)

    _logger.info("Running inference on %s", audio_path)
    t1 = time.monotonic()

    # SongFormer exposes a predict() method via trust_remote_code.
    # Returns list of dicts with label/start/end keys.
    raw_segments: list[dict] = model.predict(audio_path)

    infer_elapsed = time.monotonic() - t1
    _logger.info(
        "Inference completed in %.1fs (%d segments)",
        infer_elapsed,
        len(raw_segments),
    )

    # Normalize to ensure consistent output format.
    segments = [
        {
            "label": str(seg["label"]),
            "start": float(seg["start"]),
            "end": float(seg["end"]),
        }
        for seg in raw_segments
    ]

    return segments
