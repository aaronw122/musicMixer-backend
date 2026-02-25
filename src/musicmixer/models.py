"""Data models for Day 2 pipeline.

All dataclasses used across services are defined here to avoid circular imports.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Session management (Step 1)
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """In-memory state for a single remix session."""
    status: str = "queued"                      # "queued" | "processing" | "complete" | "error"
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=100))
    created_at_mono: float = field(default_factory=time.monotonic)
    remix_path: str | None = None
    explanation: str | None = None
    last_event: dict | None = None              # Most recent event (for reconnecting SSE clients)


@dataclass
class ProgressEvent:
    """A single SSE progress event."""
    step: str           # "separating" | "analyzing" | "processing" | "rendering" | "complete" | "error"
    detail: str
    progress: float     # 0.0 - 1.0


# ---------------------------------------------------------------------------
# Audio analysis (Step 2)
# ---------------------------------------------------------------------------

@dataclass
class AudioMetadata:
    """Audio analysis results for a single song."""
    bpm: float
    bpm_confidence: float
    beat_frames: np.ndarray        # Beat frame positions from librosa.beat.beat_track
    duration_seconds: float
    total_beats: int               # Total beats (rounded to nearest bar boundary)
    # Day 3+: key, scale, key_confidence, energy_regions, groove_type, vocal_prominence_db


# ---------------------------------------------------------------------------
# Remix plan (Steps 5 + 6)
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A section of the remix arrangement."""
    label: str                      # "intro" | "build" | "main" | "breakdown" | "outro"
    start_beat: int                 # Beat-aligned (snapped to grid)
    end_beat: int
    stem_gains: dict[str, float]   # {"vocals": 1.0, "drums": 0.7, "bass": 0.8, ...}
    transition_in: str              # "fade" | "crossfade" | "cut"
    transition_beats: int           # Length of transition envelope


@dataclass
class RemixPlan:
    """Complete remix plan — produced by LLM (Day 3) or deterministic fallback."""
    vocal_source: str                           # "song_a" | "song_b"
    start_time_vocal: float                     # Seconds, original tempo
    end_time_vocal: float
    start_time_instrumental: float
    end_time_instrumental: float
    sections: list[Section]                     # Beat-aligned arrangement
    tempo_source: str                           # "song_a" | "song_b" | "average" | "weighted_midpoint"
    key_source: str                             # "song_a" | "song_b" | "none"
    explanation: str
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False
