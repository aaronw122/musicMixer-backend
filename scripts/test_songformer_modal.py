"""Diagnostic script that validates the entire SongFormer Modal chain.

Run with: uv run modal run scripts/test_songformer_modal.py

This tests every step inside the actual Modal container so we catch
ALL failures in one shot instead of deploy-test-fail-fix loops.
"""

from __future__ import annotations

import modal

_PINNED_REVISION = "5ac5227fccf286519464fdf211e15b606898408e"
_HF_REPO = "ASLP-lab/SongFormer"
_MODEL_DIR = "/root/songformer-model"

app = modal.App("songformer-diagnostic")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch",
        "transformers",
        "huggingface_hub",
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "tqdm",
        "muq",
        "ema_pytorch",
        "x-transformers",
        "msaf",
        "loguru",
        "omegaconf",
    )
    .run_commands(
        "python -c \""
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download('{_HF_REPO}', revision='{_PINNED_REVISION}', "
        f"local_dir='{_MODEL_DIR}', "
        "repo_type='model', local_dir_use_symlinks=False, "
        "ignore_patterns=['SongFormer.pt'])"
        "\"",
    )
    .env({"PYTHONPATH": _MODEL_DIR, "SONGFORMER_LOCAL_DIR": _MODEL_DIR})
)


@app.function(image=image, gpu="T4", timeout=600)
def run_diagnostic() -> str:
    """Run every step of SongFormer loading and inference, reporting
    pass/fail for each stage."""
    import os
    import sys
    import traceback

    results = []

    def check(name: str, fn):
        try:
            result = fn()
            results.append(f"PASS: {name}")
            return result
        except Exception as e:
            tb = traceback.format_exc()
            results.append(f"FAIL: {name}\n  Error: {e}\n  Traceback:\n{tb}")
            return None

    # Stage 1: Environment
    def check_env():
        assert os.environ.get("SONGFORMER_LOCAL_DIR") == _MODEL_DIR, \
            f"SONGFORMER_LOCAL_DIR={os.environ.get('SONGFORMER_LOCAL_DIR')}"
        assert _MODEL_DIR in os.environ.get("PYTHONPATH", ""), \
            f"PYTHONPATH={os.environ.get('PYTHONPATH')}"
        return True
    check("Environment vars set", check_env)

    # Stage 2: Model files exist
    def check_files():
        required = [
            "modeling_songformer.py",
            "configuration_songformer.py",
            "model.py",
            "model_config.py",
            "config.json",
            "muq_config2.json",
            "msd_stats.json",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(_MODEL_DIR, f))]
        assert not missing, f"Missing files: {missing}"
        # List all files for diagnostics
        all_files = []
        for root, dirs, files in os.walk(_MODEL_DIR):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), _MODEL_DIR)
                all_files.append(rel)
        results.append(f"  INFO: Model dir has {len(all_files)} files: {sorted(all_files)[:30]}...")
        return True
    check("Model files present", check_files)

    # Stage 3: sys.path setup
    def check_syspath():
        if _MODEL_DIR not in sys.path:
            sys.path.insert(0, _MODEL_DIR)
        return True
    check("sys.path configured", check_syspath)

    # Stage 4: Import SongFormerModel directly
    def check_import():
        from modeling_songformer import SongFormerModel
        return SongFormerModel
    model_cls = check("Import SongFormerModel", check_import)

    # Stage 5: Import all sibling modules
    def check_siblings():
        import configuration_songformer
        import model
        import model_config
        from dataset import label2id
        from postprocessing import functional
        return True
    check("Import sibling modules", check_siblings)

    # Stage 6: Load config
    config = None
    def check_config():
        import json
        from configuration_songformer import SongFormerConfig
        config_path = os.path.join(_MODEL_DIR, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        return SongFormerConfig(**config_dict)
    config = check("Load SongFormerConfig", check_config)

    # Stage 7: Instantiate model (on CPU, no meta tensors)
    model = None
    if model_cls and config:
        def check_init():
            m = model_cls(config)
            return m
        model = check("Instantiate SongFormerModel", check_init)

    # Stage 8: Load weights from safetensors
    if model:
        def check_weights():
            from safetensors.torch import load_file
            weights_path = os.path.join(_MODEL_DIR, "model.safetensors")
            state_dict = load_file(weights_path)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            results.append(f"  INFO: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
            if missing:
                results.append(f"  INFO: Missing keys (first 5): {missing[:5]}")
            return True
        check("Load safetensors weights", check_weights)

    # Stage 9: Move to GPU
    if model:
        def check_gpu():
            import torch
            model.to(torch.device("cuda:0"))
            model.eval()
            return True
        check("Move to GPU + eval", check_gpu)

    # Stage 10: Generate synthetic audio and run inference
    if model:
        def check_inference():
            import tempfile
            from pathlib import Path

            import numpy as np
            import soundfile as sf

            # Generate 10 seconds of synthetic audio at 24kHz
            sr = 24000
            duration = 10
            audio = np.random.randn(sr * duration).astype(np.float32) * 0.1

            with tempfile.TemporaryDirectory() as tmpdir:
                test_path = Path(tmpdir) / "test.wav"
                sf.write(str(test_path), audio, sr)

                segments = model(str(test_path))

                assert isinstance(segments, list), f"Expected list, got {type(segments)}"
                results.append(f"  INFO: Got {len(segments)} segments from synthetic audio")
                if segments:
                    results.append(f"  INFO: First segment: {segments[0]}")
                    # Verify format
                    for seg in segments:
                        assert "label" in seg, f"Missing 'label' key: {seg}"
                        assert "start" in seg, f"Missing 'start' key: {seg}"
                        assert "end" in seg, f"Missing 'end' key: {seg}"
                return segments
        check("Inference on synthetic audio", check_inference)

    # Summary
    report = "\n".join(results)
    passed = sum(1 for r in results if r.startswith("PASS"))
    failed = sum(1 for r in results if r.startswith("FAIL"))
    summary = f"\n\n{'='*60}\nSUMMARY: {passed} passed, {failed} failed\n{'='*60}"

    return report + summary


@app.local_entrypoint()
def main():
    print("Running SongFormer diagnostic in Modal container...")
    print("=" * 60)
    report = run_diagnostic.remote()
    print(report)
