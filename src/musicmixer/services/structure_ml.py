"""ML-based song structure detection using SongFormer.

Wraps SongFormer inference for section boundary detection. Tries Modal
GPU first, falls back to local CPU inference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps SongFormer labels to musicMixer vocabulary.
LABEL_MAP: dict[str, str] = {
    "intro": "intro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "breakdown",
    "pre-chorus": "build",
    "outro": "outro",
    "instrumental": "instrumental",
    "solo": "instrumental",
    "interlude": "breakdown",
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
    """Run SongFormer inference on local CPU."""
    from transformers import AutoModel

    logger.info("Running SongFormer locally on CPU for %s", audio_path.name)
    t0 = time.monotonic()

    model = AutoModel.from_pretrained(
        "ASLP-lab/SongFormer",
        trust_remote_code=True,
        local_files_only=True,
        # Pin to a known-good commit to avoid silent model changes.
        revision="PINNED_COMMIT_SHA",  # TODO: fill after validation
    )

    raw_segments: list[dict] = model.predict(str(audio_path))

    elapsed = time.monotonic() - t0
    logger.info(
        "Local SongFormer inference completed in %.1fs (%d segments)",
        elapsed,
        len(raw_segments),
    )
    return raw_segments


def _analyze_modal(audio_path: Path) -> list[dict]:
    """Run SongFormer inference on Modal GPU with a 120s timeout."""
    from musicmixer.services.structure_modal import analyze_structure_remote

    logger.info("Running SongFormer on Modal GPU for %s", audio_path.name)
    t0 = time.monotonic()

    raw_segments: list[dict] = analyze_structure_remote(str(audio_path))

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
