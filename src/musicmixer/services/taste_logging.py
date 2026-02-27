"""Logging schema for the taste training stage (plan section 5.3).

Pydantic model for structured per-request logging of the taste training
pipeline. Captures candidates, timing, selection method, and flag config
for training data collection and debugging.
"""

from pydantic import BaseModel


class TasteStageLog(BaseModel):
    """Per-request logging for the taste training stage."""
    request_id: str
    prompt: str
    feature_version: str | None = None
    model_version: str | None = None
    flag_config: dict[str, bool] = {}  # all ab_* flags
    candidates_generated: int = 0
    candidates_after_filter: int = 0
    selected_candidate_index: int | None = None
    selection_method: str = "fallback"
    generation_latency_ms: float = 0.0
    scoring_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    fallback_triggered: bool = False
    fallback_reason: str | None = None
