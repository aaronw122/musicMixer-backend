"""Auto-handled adjustments (key/tempo) are folded into the description rather
than surfaced as separate warning chips; only unrecoverable gaps stay as warnings.
"""

from __future__ import annotations

from types import SimpleNamespace

from musicmixer.services.pipeline import _step_compute_tempo_and_key_plan


def _meta(bpm: float, key: str, scale: str, conf: float = 0.8):
    return SimpleNamespace(
        bpm=bpm, key=key, scale=scale, key_confidence=conf, has_modulation=False
    )


def _plan(explanation: str = "Base description."):
    return SimpleNamespace(
        tempo_source="weighted_midpoint", warnings=[], explanation=explanation
    )


def test_key_shift_folds_positive_note_no_warning():
    """Uptown Funk case: C major vs E minor — keys converge via pitch shift."""
    plan = _plan()
    session = SimpleNamespace(key_warning=None)

    _step_compute_tempo_and_key_plan(
        "sess", _meta(115.0, "C", "major"), _meta(111.0, "E", "minor"),
        plan, vocal_type="sung", session=session,
    )

    assert "tuned to a shared key" in plan.explanation
    assert session.key_warning is None
    assert not any("minor distortions" in w for w in plan.warnings)
    assert not any("key" in w.lower() for w in plan.warnings)


def test_same_key_adds_no_note():
    """When keys already match, nothing is appended (no shift happened)."""
    plan = _plan()
    session = SimpleNamespace(key_warning=None)

    _step_compute_tempo_and_key_plan(
        "sess", _meta(120.0, "C", "major"), _meta(120.0, "C", "major"),
        plan, vocal_type="sung", session=session,
    )

    assert plan.explanation == "Base description."
    assert plan.warnings == []


def test_large_tempo_gap_keeps_major_warning_drops_minor():
    plan = _plan()
    session = SimpleNamespace(key_warning=None)

    _step_compute_tempo_and_key_plan(
        "sess", _meta(80.0, "C", "major"), _meta(160.0, "C", "major"),
        plan, vocal_type="sung", session=session,
    )

    assert any("original tempos" in w for w in plan.warnings)
    assert not any("minor distortions" in w for w in plan.warnings)
