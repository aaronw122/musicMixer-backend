"""Run one remix pipeline phase and print status + output path.

Usage:
  uv run python backend/scripts/run_pipeline_phase.py <phase> <song_a> <song_b> \
      [--source-quality-a <quality>] [--source-quality-b <quality>]
"""

from __future__ import annotations

import argparse
import queue

from musicmixer.models import SessionState
from musicmixer.services.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one remix pipeline phase and print status + output path."
    )
    parser.add_argument("phase", help="Phase name (used as session ID prefix)")
    parser.add_argument("song_a", help="Path to song A WAV/MP3 file")
    parser.add_argument("song_b", help="Path to song B WAV/MP3 file")
    parser.add_argument(
        "--source-quality-a",
        default=None,
        help="Source quality descriptor for song A (e.g. 'youtube-opus-128kbps')",
    )
    parser.add_argument(
        "--source-quality-b",
        default=None,
        help="Source quality descriptor for song B (e.g. 'youtube-opus-128kbps')",
    )
    args = parser.parse_args()

    session = SessionState()
    events: queue.Queue = queue.Queue(maxsize=200)

    run_pipeline(
        session_id=f"ab-{args.phase}",
        song_a_path=args.song_a,
        song_b_path=args.song_b,
        prompt="",
        event_queue=events,
        session=session,
        source_quality_a=args.source_quality_a,
        source_quality_b=args.source_quality_b,
    )

    print(session.status)
    print(session.remix_path or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
