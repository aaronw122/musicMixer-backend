"""ML-based song structure detection using SongFormer.

Wraps SongFormer inference for section boundary detection. Tries Modal
GPU first, falls back to local CPU inference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_PINNED_REVISION = "5ac5227fccf286519464fdf211e15b606898408e"
_HF_REPO = "ASLP-lab/SongFormer"

# Maps SongFormer labels to musicMixer vocabulary.
LABEL_MAP: dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "breakdown",
    "pre-chorus": "build",
    "outro": "outro",
    "instrumental": "instrumental",
    "inst": "instrumental",
    "solo": "instrumental",
    "interlude": "breakdown",
    "silence": "intro",
}

_DEFAULT_LABEL = "verse"


def _map_labels(segments: list[dict]) -> list[dict]:
    """Map SongFormer labels to the app's vocabulary.

    Unknown labels fall back to the default rather than raising.
    """
    mapped = []
    for seg in segments:
        raw_label = seg["label"]
        mapped_label = LABEL_MAP.get(raw_label.lower(), _DEFAULT_LABEL)
        if raw_label.lower() not in LABEL_MAP:
            logger.warning(
                "Unknown SongFormer label %r, defaulting to %r",
                raw_label,
                _DEFAULT_LABEL,
            )
        else:
            logger.debug("Mapped label %r -> %r", raw_label, mapped_label)
        mapped.append({
            "label": mapped_label,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
        })
    return mapped


def _analyze_local(audio_path: Path) -> list[dict]:
    """Run SongFormer inference on local CPU.

    Requires transformers, torch, and SongFormer's dependencies to be
    installed locally. This is a dev/validation fallback — production
    uses Modal GPU.
    """
    import os
    import sys

    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    logger.info("Running SongFormer locally on CPU for %s", audio_path.name)
    t0 = time.monotonic()

    # Download (or locate cached) model repo with all sibling modules.
    local_dir = snapshot_download(
        repo_id=_HF_REPO,
        revision=_PINNED_REVISION,
        repo_type="model",
        local_dir_use_symlinks=False,
        ignore_patterns=["SongFormer.pt", "SongFormer.safetensors"],
    )

    # SongFormer's custom code imports sibling modules from the repo dir.
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    os.environ["SONGFORMER_LOCAL_DIR"] = local_dir

    model = AutoModel.from_pretrained(
        local_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.eval()

    raw_segments: list[dict] = model(str(audio_path))

    elapsed = time.monotonic() - t0
    logger.info(
        "Local SongFormer inference completed in %.1fs (%d segments)",
        elapsed,
        len(raw_segments),
    )
    return raw_segments


def _analyze_modal(audio_path: Path) -> list[dict]:
    """Run SongFormer inference on Modal GPU.

    Sends audio bytes to the remote container (Modal can't access
    local files).
    """
    import modal

    logger.info("Running SongFormer on Modal GPU for %s", audio_path.name)
    t0 = time.monotonic()

    audio_bytes = audio_path.read_bytes()
    analyze_fn = modal.Function.from_name(
        "musicmixer-songformer", "analyze_structure_remote"
    )
    raw_segments: list[dict] = analyze_fn.remote(audio_bytes, audio_path.name)

    elapsed = time.monotonic() - t0
    logger.info(
        "Modal SongFormer inference completed in %.1fs (%d segments)",
        elapsed,
        len(raw_segments),
    )
    return raw_segments


def analyze_structure_ml(audio_path: Path) -> list[dict]:
    """Detect song sections using SongFormer.

    Returns a list of segment dicts with keys ``label``, ``start``, and
    ``end`` (seconds, float). Tries Modal GPU first; falls back to local
    CPU inference on any failure.
    """
    raw_segments: list[dict] | None = None

    # Try Modal first
    try:
        raw_segments = _analyze_modal(audio_path)
    except Exception:
        logger.warning(
            "Modal SongFormer inference failed, falling back to local CPU",
            exc_info=True,
        )

    # Fallback to local CPU
    if raw_segments is None:
        raw_segments = _analyze_local(audio_path)

    logger.info(
        "Raw SongFormer segments: %s",
        [(s.get("label"), round(s.get("start", 0), 2), round(s.get("end", 0), 2)) for s in raw_segments],
    )

    mapped = _map_labels(raw_segments)

    logger.info(
        "Mapped segments: %s",
        [(s["label"], round(s["start"], 2), round(s["end"], 2)) for s in mapped],
    )

    return mapped
