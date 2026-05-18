"""ML-based song structure detection using SongFormer.

Wraps SongFormer inference for section boundary detection. Uses Modal
GPU; returns empty on failure so the pipeline falls back to heuristics.
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
    "inst": "instrumental",
    "solo": "instrumental",
    "interlude": "breakdown",
    "silence": "intro",
}

_DEFAULT_LABEL = "verse"


def analyze_structure_ml(audio_path: Path) -> list[dict]:
    """Detect song sections using SongFormer.

    Returns a list of segment dicts with keys ``label``, ``start``, and
    ``end`` (seconds, float). Uses Modal GPU; returns ``[]`` on failure
    so the caller can fall back to heuristic detection.
    """
    try:
        raw_segments = _analyze_modal(audio_path)
    except Exception:
        logger.warning(
            "Modal SongFormer inference failed; skipping ML structure "
            "detection so the pipeline can use heuristic fallback",
            exc_info=True,
        )
        return []

    logger.info("Raw SongFormer segments: %s", _summarize_segments(raw_segments))

    mapped_segments = _map_labels(raw_segments)
    logger.info("Mapped segments: %s", _summarize_segments(mapped_segments))

    return mapped_segments


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


def _summarize_segments(segments: list[dict]) -> list[tuple[str | None, float, float]]:
    return [
        (
            segment.get("label"),
            round(segment.get("start", 0), 2),
            round(segment.get("end", 0), 2),
        )
        for segment in segments
    ]
