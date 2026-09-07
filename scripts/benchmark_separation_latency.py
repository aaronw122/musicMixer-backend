"""Measure cold/warm latency of the deployed musicmixer-separation app.

Run with: uv run python scripts/benchmark_separation_latency.py [--vocal]

Calls the deployed class twice: if no container is live, the first call shows
cold-start latency (snapshot restore + model load + first inference) and the
second shows warm latency. Uses a synthetic 20s clip; real songs add inference
time proportional to duration, not overhead.

Baseline (2026-09-07, pre-fix): cold 195.9s / warm 15.4s.
Post-fix (L40S, 20s clip):      cold ~37-41s / warm ~8s.
"""

import argparse
import io
import time

import modal
import numpy as np
import soundfile as sf


def make_clip(seconds: int = 20) -> bytes:
    sr = 44100
    t = np.linspace(0, seconds, sr * seconds, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, np.stack([audio, audio], axis=1), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal", action="store_true", help="benchmark VocalSeparator instead")
    args = parser.parse_args()

    cls_name = "VocalSeparator" if args.vocal else "InstrumentalSeparator"
    inst = modal.Cls.from_name("musicmixer-separation", cls_name)()
    clip = make_clip()

    for label in ("first (cold if no live container)", "second (warm)"):
        t0 = time.monotonic()
        stems = inst.separate.remote(clip)
        print(f"[{cls_name}] {label}: {time.monotonic() - t0:.1f}s, stems={sorted(stems)}")


if __name__ == "__main__":
    main()
