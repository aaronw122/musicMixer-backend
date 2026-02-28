"""Run one remix pipeline phase and print status + output path.

Usage:
  uv run python backend/scripts/run_pipeline_phase.py <phase> <song_a> <song_b> \
      [--source-quality-a <quality>] [--source-quality-b <quality>]
"""

from __future__ import annotations

import argparse
import logging
import queue

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from musicmixer.models import SessionState
from musicmixer.services.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one remix pipeline phase and print status + output path.",
    )
    parser.add_argument("phase", help="Phase/session name")
    parser.add_argument("song_a", help="Path to song A WAV file")
    parser.add_argument("song_b", help="Path to song B WAV file")
    parser.add_argument(
        "--source-quality-a",
        default=None,
        help="Source quality string for song A (e.g., youtube-opus-128kbps)",
    )
    parser.add_argument(
        "--source-quality-b",
        default=None,
        help="Source quality string for song B (e.g., youtube-opus-128kbps)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Remix prompt to pass to the LLM interpreter",
    )
    parser.add_argument(
        "--force-vocal-source",
        default=None,
        choices=["song_a", "song_b"],
        help="Override vocal source assignment (for deterministic A/B tests)",
    )
    args = parser.parse_args()

    session = SessionState()
    events: queue.Queue = queue.Queue(maxsize=200)

    run_pipeline(
        session_id=f"ab-{args.phase}",
        song_a_path=args.song_a,
        song_b_path=args.song_b,
        prompt=args.prompt,
        event_queue=events,
        session=session,
        source_quality_a=args.source_quality_a,
        source_quality_b=args.source_quality_b,
        force_vocal_source=args.force_vocal_source,
    )

    print(session.status)
    print(session.remix_path or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
