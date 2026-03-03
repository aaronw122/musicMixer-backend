"""Stub -- real implementation in parallel agent's worktree."""
from musicmixer.models import IntentPlan, RemixPlan


def map_intent_to_gains(
    intent: IntentPlan,
    vocal_stem_lufs: dict[str, float] | None = None,
    inst_stem_lufs: dict[str, float] | None = None,
) -> RemixPlan:
    raise NotImplementedError("Gain mapper not yet merged -- see parallel PR")
