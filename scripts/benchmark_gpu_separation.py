"""Benchmark BS-RoFormer stem separation across GPU types on Modal.

Usage:
    cd backend
    uv run python scripts/benchmark_gpu_separation.py [audio_file]

If no audio file is given, uses the Hypnotize example from examples/.

Runs the same separation on A10G, L40S, and A100-40GB, reports wall-clock
time and estimated cost for each.  Each GPU gets 3 runs (1 warm-up + 2 timed)
to remove cold-start noise from the comparison.
"""

import io
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app + shared image
# ---------------------------------------------------------------------------

app = modal.App("musicmixer-gpu-benchmark")

MODEL_CKPT = "BS-Roformer-SW.ckpt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("audio-separator[gpu]", "torch", "soundfile")
    .run_commands(
        f'python -c "from audio_separator.separator import Separator; '
        f"s = Separator(); s.load_model('{MODEL_CKPT}')\"",
    )
)

# ---------------------------------------------------------------------------
# One function per GPU type (Modal requires gpu= at decoration time)
# ---------------------------------------------------------------------------

def _separate(audio_bytes: bytes, filename: str) -> dict[str, int]:
    """Shared separation logic.  Returns {stem_name: byte_count}."""
    import re
    import tempfile
    import soundfile as sf
    from pathlib import Path as _Path
    from audio_separator.separator import Separator

    _STEM_RE = re.compile(r"[_\-.\s()]+")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = _Path(tmpdir)
        input_path = tmpdir / filename
        output_dir = tmpdir / "stems"
        output_dir.mkdir()

        input_path.write_bytes(audio_bytes)

        separator = Separator(output_dir=str(output_dir))
        separator.load_model(MODEL_CKPT)
        separator.separate(str(input_path))

        stems = {}
        expected = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        for f in output_dir.iterdir():
            if f.suffix != ".wav":
                continue
            tokens = [t for t in _STEM_RE.split(f.stem.lower()) if t]
            matched = [s for s in expected if s in tokens]
            if matched and matched[0] not in stems:
                audio_data, sr = sf.read(str(f), dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, audio_data, sr, format="WAV", subtype="FLOAT")
                stems[matched[0]] = len(buf.getvalue())

    return stems


@app.function(image=image, gpu="A10G", timeout=600)
def separate_a10g(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, int]:
    return _separate(audio_bytes, filename)


@app.function(image=image, gpu="L40S", timeout=600)
def separate_l40s(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, int]:
    return _separate(audio_bytes, filename)


@app.function(image=image, gpu="A100", timeout=600)
def separate_a100(audio_bytes: bytes, filename: str = "input.wav") -> dict[str, int]:
    return _separate(audio_bytes, filename)


# ---------------------------------------------------------------------------
# Modal per-second pricing (as of 2025-05 — check modal.com/pricing)
# ---------------------------------------------------------------------------

GPU_COST_PER_SEC = {
    "A10G":  1.10 / 3600,   # $1.10/hr
    "L40S":  1.70 / 3600,   # $1.70/hr
    "A100":  3.73 / 3600,   # $3.73/hr (40GB)
}

TIMED_RUNS = 2  # runs after warm-up


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _find_default_audio() -> Path:
    """Find an example audio file to benchmark with."""
    examples = Path(__file__).resolve().parent.parent.parent / "examples"
    if examples.is_dir():
        for f in sorted(examples.iterdir()):
            if f.suffix in (".mp3", ".wav"):
                return f
    print("ERROR: No audio file found. Pass a path as argument.", file=sys.stderr)
    sys.exit(1)


def _run_benchmark(fn, audio_bytes: bytes, filename: str, gpu_name: str) -> list[float]:
    """Run warm-up + timed runs, return list of wall-clock seconds."""
    print(f"\n{'='*60}")
    print(f"  {gpu_name}")
    print(f"{'='*60}")

    # Warm-up run (pays cold-start cost)
    print(f"  [warm-up] running...", end="", flush=True)
    t0 = time.monotonic()
    result = fn.remote(audio_bytes, filename)
    warm_elapsed = time.monotonic() - t0
    stems = list(result.keys())
    print(f" {warm_elapsed:.1f}s  (stems: {stems})")

    # Timed runs
    times = []
    for i in range(TIMED_RUNS):
        print(f"  [run {i+1}/{TIMED_RUNS}] running...", end="", flush=True)
        t0 = time.monotonic()
        fn.remote(audio_bytes, filename)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        print(f" {elapsed:.1f}s")

    return times


@app.local_entrypoint()
def main(audio_file: str = ""):
    # Resolve audio file — Modal eats sys.argv, so use function param instead.
    # Usage: uv run modal run scripts/benchmark_gpu_separation.py --audio-file /path/to/song.mp3
    if audio_file:
        audio_path = Path(audio_file).resolve()
    else:
        audio_path = _find_default_audio()

    print(f"Audio file: {audio_path}")
    print(f"File size:  {audio_path.stat().st_size / 1024 / 1024:.1f} MB")

    audio_bytes = audio_path.read_bytes()
    filename = audio_path.name

    gpu_fns = [
        ("A10G", separate_a10g),
        ("L40S", separate_l40s),
        ("A100", separate_a100),
    ]

    results = {}
    for gpu_name, fn in gpu_fns:
        try:
            times = _run_benchmark(fn, audio_bytes, filename, gpu_name)
            avg = sum(times) / len(times)
            results[gpu_name] = {
                "times": times,
                "avg_s": avg,
                "cost_per_run": avg * GPU_COST_PER_SEC[gpu_name],
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            results[gpu_name] = None

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  File: {audio_path.name}")
    print(f"  Timed runs per GPU: {TIMED_RUNS} (after 1 warm-up)\n")
    print(f"  {'GPU':<8} {'Avg Time':>10} {'Cost/Run':>10} {'Cost/2 Songs':>14} {'$/hr':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*14} {'-'*8}")

    for gpu_name in ["A10G", "L40S", "A100"]:
        r = results.get(gpu_name)
        if r is None:
            print(f"  {gpu_name:<8} {'FAILED':>10}")
            continue
        avg = r["avg_s"]
        cost = r["cost_per_run"]
        cost2 = cost * 2  # two songs per remix
        rate = GPU_COST_PER_SEC[gpu_name] * 3600
        print(f"  {gpu_name:<8} {avg:>8.1f}s  ${cost:>8.4f}  ${cost2:>12.4f}  ${rate:>6.2f}")

    # Recommendation
    print()
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        fastest = min(valid, key=lambda k: valid[k]["avg_s"])
        cheapest = min(valid, key=lambda k: valid[k]["cost_per_run"])
        best_value = min(valid, key=lambda k: valid[k]["avg_s"] * valid[k]["cost_per_run"])

        print(f"  Fastest:        {fastest} ({valid[fastest]['avg_s']:.1f}s)")
        print(f"  Cheapest:       {cheapest} (${valid[cheapest]['cost_per_run']:.4f}/run)")
        print(f"  Best value:     {best_value} (time x cost)")
