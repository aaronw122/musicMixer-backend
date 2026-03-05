"""Taste training pipeline stage: candidate generation, filtering, and scoring.

Orchestrates the full taste training flow as a single pipeline step:
  4a. Generate 8-12 candidate plans
  4b. Hard-filter invalid candidates (constraint violations)
  4c. Score surviving candidates and select best

Wrapped in a 400ms timeout with circuit breaker. On any error/timeout,
falls back to the provided fallback plan.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Optional

from musicmixer.models import AudioMetadata, RemixPlan

logger = logging.getLogger(__name__)

# Timeout and circuit breaker settings
TASTE_STAGE_TIMEOUT_MS = 400
CIRCUIT_BREAKER_THRESHOLD = 5  # consecutive fallbacks to disable
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 600  # 10 minutes

# Module-level circuit breaker state
_consecutive_fallbacks: int = 0
_circuit_open_since: float | None = None
_circuit_breaker_lock = threading.Lock()


def _reset_circuit_breaker() -> None:
    """Reset circuit breaker state after a successful run."""
    global _consecutive_fallbacks, _circuit_open_since
    with _circuit_breaker_lock:
        _consecutive_fallbacks = 0
        _circuit_open_since = None


def _record_fallback() -> None:
    """Record a fallback and potentially open the circuit breaker."""
    global _consecutive_fallbacks, _circuit_open_since
    should_log_open = False
    fallback_count = 0
    with _circuit_breaker_lock:
        _consecutive_fallbacks += 1
        fallback_count = _consecutive_fallbacks
        if fallback_count >= CIRCUIT_BREAKER_THRESHOLD and _circuit_open_since is None:
            _circuit_open_since = time.monotonic()
            should_log_open = True

    if should_log_open:
        logger.warning(
            "Taste stage circuit breaker OPEN after %d consecutive fallbacks. "
            "Will disable for %ds.",
            fallback_count,
            CIRCUIT_BREAKER_COOLDOWN_SECONDS,
        )


def _is_circuit_open() -> bool:
    """Check if circuit breaker is open (taste stage disabled)."""
    global _consecutive_fallbacks, _circuit_open_since
    cooldown_elapsed = False
    elapsed = 0.0
    with _circuit_breaker_lock:
        if _circuit_open_since is None:
            return False
        elapsed = time.monotonic() - _circuit_open_since
        if elapsed >= CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            _consecutive_fallbacks = 0
            _circuit_open_since = None
            cooldown_elapsed = True
        else:
            return True

    if cooldown_elapsed:
        # Cooldown elapsed, reset and allow retry
        logger.info(
            "Taste stage circuit breaker cooldown elapsed (%.0fs). Resetting.",
            elapsed,
        )
    return False


def _get_circuit_breaker_state() -> tuple[int, float | None]:
    """Snapshot current circuit breaker state for assertions."""
    with _circuit_breaker_lock:
        return _consecutive_fallbacks, _circuit_open_since


def _set_circuit_open_since(value: float | None) -> None:
    """Set open timestamp with lock protection (test helper)."""
    global _circuit_open_since
    with _circuit_breaker_lock:
        _circuit_open_since = value


@dataclass
class TasteStageResult:
    """Result from the taste training stage."""
    selected_plan: RemixPlan
    candidates_generated: int
    candidates_after_filter: int
    selection_method: str  # "heuristic" | "model" | "fallback"
    generation_latency_ms: float
    scoring_latency_ms: float
    total_latency_ms: float
    fallback_triggered: bool
    fallback_reason: str | None = None


def _make_fallback_result(
    fallback_plan: RemixPlan,
    reason: str,
    total_latency_ms: float = 0.0,
) -> TasteStageResult:
    """Create a TasteStageResult representing a fallback."""
    return TasteStageResult(
        selected_plan=fallback_plan,
        candidates_generated=0,
        candidates_after_filter=0,
        selection_method="fallback",
        generation_latency_ms=0.0,
        scoring_latency_ms=0.0,
        total_latency_ms=total_latency_ms,
        fallback_triggered=True,
        fallback_reason=reason,
    )


def _run_taste_pipeline(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    prompt: str,
    fallback_plan: RemixPlan,
) -> TasteStageResult:
    """Inner taste pipeline logic (runs inside timeout wrapper).

    Steps:
      1. Generate candidates via candidate_planner
      2. Validate each candidate via taste_constraints
      3. Filter out invalid candidates
      4. Score surviving candidates via taste_model
      5. Return best candidate with timing metrics
    """
    # Step 4a: Generate candidates
    gen_start = time.monotonic()
    try:
        from musicmixer.services.candidate_planner import generate_candidates
    except ImportError:
        logger.warning("candidate_planner module not available, falling back")
        return _make_fallback_result(fallback_plan, "candidate_planner not available")

    candidates = generate_candidates(meta_a, meta_b, prompt)
    gen_latency_ms = (time.monotonic() - gen_start) * 1000
    candidates_generated = len(candidates)

    if not candidates:
        return _make_fallback_result(
            fallback_plan,
            "no candidates generated",
            total_latency_ms=gen_latency_ms,
        )

    # Step 4b: Hard-filter invalid candidates
    try:
        from musicmixer.services.taste_constraints import validate_candidate
    except ImportError:
        logger.warning("taste_constraints module not available, falling back")
        return _make_fallback_result(
            fallback_plan,
            "taste_constraints not available",
            total_latency_ms=gen_latency_ms,
        )

    valid_candidates = []
    for candidate in candidates:
        is_valid, _violations = validate_candidate(candidate, meta_a, meta_b)
        if is_valid:
            valid_candidates.append(candidate)

    candidates_after_filter = len(valid_candidates)

    if not valid_candidates:
        return _make_fallback_result(
            fallback_plan,
            "all candidates failed constraint validation",
            total_latency_ms=gen_latency_ms,
        )

    # Step 4c: Score surviving candidates and select best
    score_start = time.monotonic()
    try:
        from musicmixer.services.taste_model import select_best
    except ImportError:
        logger.warning("taste_model module not available, falling back")
        return _make_fallback_result(
            fallback_plan,
            "taste_model not available",
            total_latency_ms=gen_latency_ms,
        )

    selected_plan, _scored = select_best(valid_candidates, meta_a, meta_b)
    selection_method = "heuristic"  # hardcoded: heuristic scorer is active in Phase 0-1
    score_latency_ms = (time.monotonic() - score_start) * 1000
    total_latency_ms = gen_latency_ms + score_latency_ms

    return TasteStageResult(
        selected_plan=selected_plan,
        candidates_generated=candidates_generated,
        candidates_after_filter=candidates_after_filter,
        selection_method=selection_method,
        generation_latency_ms=gen_latency_ms,
        scoring_latency_ms=score_latency_ms,
        total_latency_ms=total_latency_ms,
        fallback_triggered=False,
    )


def run_taste_stage(
    meta_a: AudioMetadata,
    meta_b: AudioMetadata,
    prompt: str = "",
    fallback_plan: RemixPlan | None = None,
) -> TasteStageResult:
    """Run the full taste training pipeline stage.

    Steps:
      4a. Generate 8-12 candidate plans
      4b. Hard-filter invalid candidates (constraint violations)
      4c. Score surviving candidates and select best

    Wrapped in a 400ms timeout. On any error/timeout, falls back
    to the provided fallback_plan (or generates one).

    Circuit breaker: After 5 consecutive fallbacks, disables the
    taste stage for 10 minutes.
    """
    if fallback_plan is None:
        # Should not happen in normal pipeline flow, but guard against it
        raise ValueError("fallback_plan is required for taste stage")

    # Circuit breaker check
    if _is_circuit_open():
        logger.info("Taste stage skipped: circuit breaker open")
        return _make_fallback_result(fallback_plan, "circuit breaker open")

    start = time.monotonic()
    timeout_seconds = TASTE_STAGE_TIMEOUT_MS / 1000.0

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_taste_pipeline, meta_a, meta_b, prompt, fallback_plan,
            )
            result = future.result(timeout=timeout_seconds)

        elapsed_ms = (time.monotonic() - start) * 1000

        if result.fallback_triggered:
            _record_fallback()
            logger.warning(
                "Taste stage fallback: reason=%s, latency=%.0fms",
                result.fallback_reason, elapsed_ms,
            )
        else:
            _reset_circuit_breaker()
            logger.info(
                "Taste stage success: %d candidates, %d after filter, "
                "method=%s, latency=%.0fms",
                result.candidates_generated,
                result.candidates_after_filter,
                result.selection_method,
                elapsed_ms,
            )

        # Update total latency to wall-clock time
        result.total_latency_ms = elapsed_ms
        return result

    except FuturesTimeoutError:
        elapsed_ms = (time.monotonic() - start) * 1000
        _record_fallback()
        logger.warning(
            "Taste stage timed out after %.0fms (limit=%dms)",
            elapsed_ms, TASTE_STAGE_TIMEOUT_MS,
        )
        return _make_fallback_result(
            fallback_plan, "timeout", total_latency_ms=elapsed_ms,
        )

    except Exception:
        elapsed_ms = (time.monotonic() - start) * 1000
        _record_fallback()
        logger.error(
            "Taste stage error after %.0fms", elapsed_ms, exc_info=True,
        )
        return _make_fallback_result(
            fallback_plan, "exception", total_latency_ms=elapsed_ms,
        )
