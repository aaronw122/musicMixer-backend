"""Pipeline instrumentation: structured metrics and red-flag detection.

Accumulates structured fields throughout the remix pipeline and emits
per-step structured log entries plus a completion summary with red flags.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Red-flag thresholds
# ---------------------------------------------------------------------------

_HIGH_STRETCH_PCT = 25.0
_DURATION_MISMATCH_PCT = 10.0
_BORDERLINE_LUFS_LOW = -50.0
_BORDERLINE_LUFS_HIGH = -35.0
_LONG_INTRO_BEATS = 16
_LONG_SONG_DURATION_S = 120.0
_HIGH_KEY_SHIFT_SEMITONES = 3
_GHOST_STEM_RMS_THRESHOLD = 0.01


@dataclass
class PipelineMetrics:
    """Accumulates structured metrics across the entire pipeline.

    Create one instance per session at pipeline start, pass it through
    each step, and call ``log_completion()`` at the end.
    """

    session_id: str

    # Pipeline timing
    _pipeline_start: float = field(default_factory=time.monotonic)

    # Input fields
    song_a_title: str = ""
    song_b_title: str = ""
    song_a_video_id: str = ""
    song_b_video_id: str = ""
    song_a_cache_hit: bool = False
    song_b_cache_hit: bool = False

    # Modal fields
    separation_time_a_s: float = 0.0
    separation_time_b_s: float = 0.0
    modal_cold_start: bool = False
    stem_backend: str = ""
    stem_count: int = 0

    # Analysis fields
    bpm_a: float = 0.0
    bpm_b: float = 0.0
    key_a: str = ""
    scale_a: str = ""
    key_b: str = ""
    scale_b: str = ""
    key_confidence_a: float = 0.0
    key_confidence_b: float = 0.0
    duration_a_s: float = 0.0
    duration_b_s: float = 0.0

    # LLM plan fields
    start_time_vocal: float = 0.0
    start_time_instrumental: float = 0.0
    section_count: int = 0
    sections_summary: str = ""
    intro_beats: int = 0
    vocal_type: str = ""
    llm_plan_failed: bool = False

    # Tempo/key fields
    target_bpm: float = 0.0
    stretch_pct: float = 0.0
    vocal_semitones: float = 0.0
    inst_semitones: float = 0.0
    tempo_key_warnings: list[str] = field(default_factory=list)

    # Beat grid fields
    beat_grid_source: str = ""
    post_stretch_beat_count: int = 0

    # Stem health fields
    stems_active: list[str] = field(default_factory=list)
    stems_inactive: list[str] = field(default_factory=list)
    per_stem_lufs: dict[str, float] = field(default_factory=dict)

    # Processing fields
    level_match_gain_db: float = 0.0

    # Render fields
    render_duration_s: float = 0.0
    per_step_times: dict[str, float] = field(default_factory=dict)

    # Output fields
    final_lufs: float = 0.0
    output_size_mb: float = 0.0
    total_pipeline_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # Red flags
    red_flags: list[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Step-level logging helpers
    # -----------------------------------------------------------------------

    def log_input(self) -> None:
        """Log input fields when songs are identified."""
        logger.info(
            "Pipeline input",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "input",
                "song_a_title": self.song_a_title,
                "song_b_title": self.song_b_title,
                "song_a_video_id": self.song_a_video_id,
                "song_b_video_id": self.song_b_video_id,
                "song_a_cache_hit": self.song_a_cache_hit,
                "song_b_cache_hit": self.song_b_cache_hit,
            },
        )

    def log_separation(self) -> None:
        """Log separation results and check for red flags."""
        logger.info(
            "Stem separation complete",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "separation",
                "separation_time_a_s": round(self.separation_time_a_s, 2),
                "separation_time_b_s": round(self.separation_time_b_s, 2),
                "modal_cold_start": self.modal_cold_start,
                "stem_backend": self.stem_backend,
                "stem_count": self.stem_count,
            },
        )

    def log_analysis(self) -> None:
        """Log analysis results."""
        logger.info(
            "Audio analysis complete",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "analysis",
                "bpm_a": round(self.bpm_a, 1),
                "bpm_b": round(self.bpm_b, 1),
                "key_a": self.key_a,
                "scale_a": self.scale_a,
                "key_b": self.key_b,
                "scale_b": self.scale_b,
                "key_confidence_a": round(self.key_confidence_a, 2),
                "key_confidence_b": round(self.key_confidence_b, 2),
                "duration_a_s": round(self.duration_a_s, 1),
                "duration_b_s": round(self.duration_b_s, 1),
            },
        )

    def log_llm_plan(self) -> None:
        """Log LLM plan results and check for red flags."""
        logger.info(
            "LLM plan ready",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "llm_plan",
                "start_time_vocal": round(self.start_time_vocal, 2),
                "start_time_instrumental": round(self.start_time_instrumental, 2),
                "section_count": self.section_count,
                "sections_summary": self.sections_summary,
                "intro_beats": self.intro_beats,
                "vocal_type": self.vocal_type,
                "llm_plan_failed": self.llm_plan_failed,
            },
        )
        self._check_llm_plan_flags()

    def log_tempo_key(self) -> None:
        """Log tempo/key plan and check for red flags."""
        logger.info(
            "Tempo and key plan computed",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "tempo_key",
                "target_bpm": round(self.target_bpm, 1),
                "stretch_pct": round(self.stretch_pct, 1),
                "vocal_semitones": round(self.vocal_semitones, 1),
                "inst_semitones": round(self.inst_semitones, 1),
                "tempo_key_warnings": self.tempo_key_warnings,
            },
        )
        self._check_tempo_key_flags()

    def log_beat_grid(self) -> None:
        """Log beat grid results and check for red flags."""
        logger.info(
            "Beat grid established",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "beat_grid",
                "beat_grid_source": self.beat_grid_source,
                "post_stretch_beat_count": self.post_stretch_beat_count,
            },
        )
        self._check_beat_grid_flags()

    def log_stem_health(self) -> None:
        """Log stem health and check for red flags."""
        logger.info(
            "Stem health assessed",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "stem_health",
                "stems_active": self.stems_active,
                "stems_inactive": self.stems_inactive,
                "per_stem_lufs": {k: round(v, 1) for k, v in self.per_stem_lufs.items()},
            },
        )
        self._check_stem_health_flags()

    def log_processing(self) -> None:
        """Log processing details."""
        logger.info(
            "Level matching applied",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "processing",
                "level_match_gain_db": round(self.level_match_gain_db, 1),
            },
        )

    def log_render(self) -> None:
        """Log render results and check for red flags."""
        logger.info(
            "Render complete",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "render",
                "render_duration_s": round(self.render_duration_s, 1),
                "per_step_times": {k: round(v, 2) for k, v in self.per_step_times.items()},
            },
        )
    def log_output(self) -> None:
        """Log output details."""
        logger.info(
            "Pipeline output",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "output",
                "final_lufs": round(self.final_lufs, 1),
                "output_size_mb": round(self.output_size_mb, 2),
                "total_pipeline_time_s": round(self.total_pipeline_time_s, 1),
                "warnings": self.warnings,
            },
        )

    # -----------------------------------------------------------------------
    # Red-flag checks
    # -----------------------------------------------------------------------

    def _add_flag(self, tag: str) -> None:
        """Add a red flag and log at WARNING level."""
        if tag not in self.red_flags:
            self.red_flags.append(tag)
            logger.warning(
                "Red flag: %s",
                tag,
                extra={
                    "session_id": self.session_id,
                    "red_flag": tag,
                },
            )

    def _check_llm_plan_flags(self) -> None:
        """Check LLM plan for red flags."""
        # long_intro: LLM intro section > 16 beats
        if self.intro_beats > _LONG_INTRO_BEATS:
            self._add_flag("long_intro")

        # zero_start_instrumental: start_time_instrumental = 0 with song > 120s
        if self.start_time_instrumental == 0 and self.duration_b_s > _LONG_SONG_DURATION_S:
            self._add_flag("zero_start_instrumental")

    def _check_tempo_key_flags(self) -> None:
        """Check tempo/key plan for red flags."""
        # high_stretch: stretch > 25% on any stem
        if self.stretch_pct > _HIGH_STRETCH_PCT:
            self._add_flag(f"high_stretch_{self.stretch_pct:.0f}%")

        # high_key_shift: key shift > 3 semitones on instrumental stems
        if abs(self.inst_semitones) > _HIGH_KEY_SHIFT_SEMITONES:
            self._add_flag("high_key_shift")

    def _check_beat_grid_flags(self) -> None:
        """Check beat grid for red flags."""
        # beat_grid_fallback: re-detection failed, using scaled grid
        if self.beat_grid_source == "scaled_fallback":
            self._add_flag("beat_grid_fallback")

    def _check_stem_health_flags(self) -> None:
        """Check stem health for red flags."""
        for stem_name, lufs_val in self.per_stem_lufs.items():
            # borderline_stem: LUFS between -50 and -35
            if _BORDERLINE_LUFS_LOW <= lufs_val <= _BORDERLINE_LUFS_HIGH:
                self._add_flag("borderline_stem")
                break  # One flag is enough

    def check_ghost_stems(
        self,
        inst_audio: dict[str, "np.ndarray"],
    ) -> None:
        """Check for ghost stems (guitar/piano with very low RMS).

        Called after stems are loaded but before they are filtered out.
        Needs the raw audio arrays to compute RMS.
        """
        import numpy as np

        for stem_name in ("guitar", "piano"):
            audio = inst_audio.get(stem_name)
            if audio is not None:
                rms = float(np.sqrt(np.mean(audio ** 2)))
                if rms < _GHOST_STEM_RMS_THRESHOLD and stem_name in self.stems_active:
                    self._add_flag("ghost_stem")
                    return  # One flag is enough

    def check_duration_mismatch(self, estimated_duration_s: float) -> None:
        """Check if post-render duration differs from estimated by > 10%.

        Called from the render step where we have both the actual and
        estimated durations.
        """
        if estimated_duration_s > 0 and self.render_duration_s > 0:
            pct_diff = abs(self.render_duration_s - estimated_duration_s) / estimated_duration_s * 100
            if pct_diff > _DURATION_MISMATCH_PCT:
                self._add_flag("duration_mismatch")

    # -----------------------------------------------------------------------
    # Completion summary
    # -----------------------------------------------------------------------

    def log_completion(self) -> None:
        """Emit the pipeline completion summary line."""
        self.total_pipeline_time_s = time.monotonic() - self._pipeline_start

        flag_tags = self.red_flags if self.red_flags else []

        # Human-readable summary at INFO level
        logger.info(
            "Session %s: pipeline complete | red_flags: %d | flags: %s | duration: %.0fs",
            self.session_id,
            len(flag_tags),
            flag_tags,
            self.total_pipeline_time_s,
        )

        # Full structured summary
        logger.info(
            "Pipeline summary",
            extra={
                "session_id": self.session_id,
                "pipeline_step": "summary",
                # Input
                "song_a_title": self.song_a_title,
                "song_b_title": self.song_b_title,
                "song_a_video_id": self.song_a_video_id,
                "song_b_video_id": self.song_b_video_id,
                "song_a_cache_hit": self.song_a_cache_hit,
                "song_b_cache_hit": self.song_b_cache_hit,
                # Modal
                "separation_time_a_s": round(self.separation_time_a_s, 2),
                "separation_time_b_s": round(self.separation_time_b_s, 2),
                "modal_cold_start": self.modal_cold_start,
                "stem_backend": self.stem_backend,
                "stem_count": self.stem_count,
                # Analysis
                "bpm_a": round(self.bpm_a, 1),
                "bpm_b": round(self.bpm_b, 1),
                "key_a": self.key_a,
                "scale_a": self.scale_a,
                "key_b": self.key_b,
                "scale_b": self.scale_b,
                "key_confidence_a": round(self.key_confidence_a, 2),
                "key_confidence_b": round(self.key_confidence_b, 2),
                "duration_a_s": round(self.duration_a_s, 1),
                "duration_b_s": round(self.duration_b_s, 1),
                # LLM plan
                "start_time_vocal": round(self.start_time_vocal, 2),
                "start_time_instrumental": round(self.start_time_instrumental, 2),
                "section_count": self.section_count,
                "sections_summary": self.sections_summary,
                "intro_beats": self.intro_beats,
                "vocal_type": self.vocal_type,
                "llm_plan_failed": self.llm_plan_failed,
                # Tempo/key
                "target_bpm": round(self.target_bpm, 1),
                "stretch_pct": round(self.stretch_pct, 1),
                "vocal_semitones": round(self.vocal_semitones, 1),
                "inst_semitones": round(self.inst_semitones, 1),
                "tempo_key_warnings": self.tempo_key_warnings,
                # Beat grid
                "beat_grid_source": self.beat_grid_source,
                "post_stretch_beat_count": self.post_stretch_beat_count,
                # Stem health
                "stems_active": self.stems_active,
                "stems_inactive": self.stems_inactive,
                "per_stem_lufs": {k: round(v, 1) for k, v in self.per_stem_lufs.items()},
                # Processing
                "level_match_gain_db": round(self.level_match_gain_db, 1),
                # Render
                "render_duration_s": round(self.render_duration_s, 1),
                "per_step_times": {k: round(v, 2) for k, v in self.per_step_times.items()},
                # Output
                "final_lufs": round(self.final_lufs, 1),
                "output_size_mb": round(self.output_size_mb, 2),
                "total_pipeline_time_s": round(self.total_pipeline_time_s, 1),
                "warnings": self.warnings,
                # Red flags
                "red_flags": self.red_flags,
                "red_flags_count": len(self.red_flags),
            },
        )
