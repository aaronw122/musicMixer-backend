"""Data models for the musicMixer pipeline.

All dataclasses used across services are defined here to avoid circular imports.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Fixed convention: Song A always provides vocals, Song B always provides
# instrumentals. Both analysis.py and interpreter.py reference this constant
# to prevent drift.
# ---------------------------------------------------------------------------
VOCAL_SOURCE: str = "song_a"

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
    step: str           # "downloading" | "separating" | "analyzing" | "processing" | "rendering" | "complete" | "error"
    detail: str
    progress: float     # 0.0 - 1.0


# ---------------------------------------------------------------------------
# Song structure analysis (Step 3.5)
# ---------------------------------------------------------------------------

@dataclass
class VocalGap:
    """A contiguous run of bars where vocals are inactive."""
    start_bar: int
    end_bar: int
    length_bars: int


@dataclass
class EnergyBuckets:
    """Adaptive percentile-based energy thresholds for section classification.

    Thresholds computed from normalized combined energy, filtering bars below
    noise_floor. Classification: silent=<noise_floor, low=<p10, medium=p10-p50,
    high=p50-p85, peak=>p85.
    """
    noise_floor: float  # default 0.02; bars below = "silent"
    p10: float
    p50: float
    p85: float


@dataclass
class StemAnalysis:
    """Per-bar stem energy analysis and vocal activity detection."""
    bar_rms: dict[str, np.ndarray]    # stem_name -> per-bar RMS (raw, NOT normalized)
    combined_energy: np.ndarray       # per-bar combined energy (normalized to p99=1.0)
    vocal_active: np.ndarray          # per-bar bool
    vocal_gaps: list[VocalGap]
    bucket_thresholds: EnergyBuckets


@dataclass
class SectionInfo:
    """A detected section of a song's structure."""
    start_bar: int
    end_bar: int
    bar_count: int
    start_time: float               # seconds
    end_time: float                 # seconds
    label: str                      # intro|verse|chorus|instrumental|breakdown|build|outro
    energy_level: str               # low|medium|high|peak
    energy_trajectory: str          # e.g. "medium->high" (thirds, deduped)
    density: str                    # sparse|mid|full|full+extra
    vocal_status: str               # vox:yes|vox:no|vox:fading
    vocal_prominence_db: Optional[float] = None  # dB above accompaniment, None if no vocals
    annotations: list[str] = field(default_factory=list)  # ["DROP","BUILD","GOOD INSTRUMENTAL SOURCE"]


@dataclass
class SongStructure:
    """Complete song structure analysis: sections, vocal gaps, and bar count."""
    sections: list[SectionInfo]
    vocal_gaps: list[VocalGap]
    total_bars: int


@dataclass
class CrossSongRelationships:
    """Cross-song comparison metrics for remix planning."""
    loudness_diff_db: float         # 20*log10(rms_a/rms_b), positive=A louder
    energy_profile_a: str           # "consistent high" / "wide dynamic range"
    energy_profile_b: str
    vocal_source: str               # Always "song_a" — fixed convention, Song A provides vocals
    vocal_prominence_a_db: float    # dB above accompaniment
    vocal_prominence_b_db: float
    instrumental_sections: list[str]  # bar ranges from recommended source
    frequency_conflicts: str        # warning text or empty
    stretch_pct: float


# ---------------------------------------------------------------------------
# Lyrics lookup (Day 3 — lyrics intelligence)
# ---------------------------------------------------------------------------

@dataclass
class LyricLine:
    """A single line of lyrics, optionally synced to a timestamp and bar."""
    text: str
    timestamp_seconds: float | None = None
    bar_number: int | None = None


@dataclass
class LyricsData:
    """Complete lyrics data for a song, with optional bar-level sync."""
    artist: str
    title: str
    source: str              # "lrclib" | "musixmatch" | "filename" | "id3"
    is_synced: bool
    lines: list[LyricLine]
    raw_text: str
    lookup_duration_ms: float = 0.0


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
    # Key detection (Day 3)
    key: Optional[str] = None
    scale: Optional[str] = None
    key_confidence: Optional[float] = None
    has_modulation: bool = False
    # Source quality (YouTube inputs)
    source_quality: Optional[str] = None   # e.g. "youtube-opus-128kbps" or None for file uploads
    # Energy analysis (Day 3)
    mean_rms: Optional[float] = None       # From original mix audio (NOT summed stems)
    stem_analysis: Optional[StemAnalysis] = None
    song_structure: Optional[SongStructure] = None


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


# ---------------------------------------------------------------------------
# Intent-based remix planning (LLM outputs roles, not gains)
# ---------------------------------------------------------------------------

STEM_ROLES = ("lead", "support", "background", "texture", "silent")


@dataclass
class IntentSection:
    """A remix section described by musical intent, not exact gains."""
    label: str                    # "intro" | "verse" | "chorus" | "breakdown" | "drop" | "outro" | "bridge"
    start_beat: int
    end_beat: int
    energy: str                   # "low" | "medium" | "high" | "peak"
    stem_roles: dict[str, str]    # {"vocals": "lead", "drums": "support", ...}
    transition_in: str            # "fade" | "crossfade" | "cut"
    transition_beats: int


@dataclass
class IntentPlan:
    """Musical intent plan — LLM's creative output before gain mapping."""
    start_time_vocal: float
    end_time_vocal: float
    start_time_instrumental: float
    end_time_instrumental: float
    sections: list[IntentSection]
    explanation: str
    vocal_type: str = "sung"      # "sung" | "rap"
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spectral analysis (Adaptive EQ)
# ---------------------------------------------------------------------------

@dataclass
class SpectralProfile:
    """Frequency-domain profile for a single stem.

    Stores 1/3-octave band energies (31 ISO 266 centers from 20 Hz to 20 kHz)
    and detected spectral peaks.  Used by the adaptive EQ system to detect
    per-stem anomalies and cross-stem masking conflicts.
    """
    stem_type: str
    band_centers_hz: np.ndarray       # (31,) ISO 266 1/3-octave centers
    band_energies_db: np.ndarray      # (31,) smoothed energy per band (dB, relative)
    peak_frequencies_hz: np.ndarray   # detected spectral peaks (Hz)
    peak_magnitudes_db: np.ndarray    # magnitude at each peak (dB)


@dataclass
class FrequencyConflict:
    """A detected masking conflict between two stems in a specific band.

    Generated when both stems exceed the anomaly threshold (+6 dB) in the
    same 1/3-octave band.  The recommended cut is applied to the lower-
    priority stem (vocals > bass > drums > guitar/piano/other).
    """
    stem_a: str
    stem_b: str
    center_hz: float
    severity_db: float
    recommended_cut_stem: str
    recommended_cut_db: float
    recommended_q: float


# ---------------------------------------------------------------------------
# Remix plan (Steps 5 + 6)
# ---------------------------------------------------------------------------

@dataclass
class RemixPlan:
    """Complete remix plan — produced by LLM (Day 3) or deterministic fallback."""
    vocal_source: str                           # Always "song_a" — Song A is the fixed vocal source
    start_time_vocal: float                     # Seconds, original tempo
    end_time_vocal: float
    start_time_instrumental: float
    end_time_instrumental: float
    sections: list[Section]                     # Beat-aligned arrangement
    tempo_source: str                           # "song_a" | "song_b" | "average" | "weighted_midpoint"
    explanation: str
    warnings: list[str] = field(default_factory=list)
    used_fallback: bool = False

