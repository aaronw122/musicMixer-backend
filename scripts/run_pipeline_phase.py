"""Run one remix pipeline phase and print status + output path.

Usage:
  uv run python backend/scripts/run_pipeline_phase.py <phase> <song_a> <song_b>
"""

from __future__ import annotations

import queue
import sys

from musicmixer.models import SessionState
from musicmixer.services.pipeline import run_pipeline


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: run_pipeline_phase.py <phase> <song_a> <song_b>", file=sys.stderr)
        return 2

    phase = sys.argv[1]
    song_a = sys.argv[2]
    song_b = sys.argv[3]

    session = SessionState()
    events: queue.Queue = queue.Queue(maxsize=200)

    run_pipeline(
        session_id=f"ab-{phase}",
        song_a_path=song_a,
        song_b_path=song_b,
        prompt="",
        event_queue=events,
        session=session,
    )

    print(session.status)
    print(session.remix_path or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
