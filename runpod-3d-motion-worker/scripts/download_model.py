#!/usr/bin/env python3
"""Pre-warm a RunPod network volume with the HY-Motion-1.0 weights so the
serverless worker doesn't eat ~22GB of downloads (and a 20GB container
disk overflow) on every cold start.

Run from a RunPod Pod (not the serverless worker) with the same network
volume attached that the `HY-Motion` endpoint uses -- volumes mount at
/workspace on both Pods and Serverless workers. Needs ~25GB free.

Three downloads, matching what the worker loads (USE_HF_MODELS=1):

  1. DiT checkpoint: tencent/HY-Motion-1.0, <variant>/{config.yml,latest.ckpt}
     (4.2GB Standard, 1.8GB Lite) -> {HY_MOTION_CKPTS}/tencent/<variant>,
     the layout T2MRuntime expects (handler.get_runtime reads the same path).
  2. Qwen/Qwen3-8B (16.4GB) text encoder -> standard HF cache (HF_HOME).
  3. openai/clip-vit-large-patch14 (1.7GB, safetensors only) -> HF cache.

    python download_model.py [--variant HY-Motion-1.0|HY-Motion-1.0-Lite|both]
    HF_TOKEN=... python download_model.py      # optional, avoids rate limits
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Must happen before huggingface_hub is imported -- it reads HF_HOME at
# import time. Run on a bare Pod (not inside the worker image, which sets
# these as Dockerfile ENV), this would otherwise silently land in
# ~/.cache on the Pod's container disk and be lost on stop.
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf-cache")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download  # noqa: E402

CKPTS = os.environ.get("HY_MOTION_CKPTS", "/workspace/models/hy-motion")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None
VARIANTS = ("HY-Motion-1.0", "HY-Motion-1.0-Lite")


def download_dit(variant: str) -> None:
    root = Path(CKPTS) / "tencent"
    target = root / variant
    if (target / "latest.ckpt").exists() and (target / "config.yml").exists():
        print(f"[SKIP] {variant} already at {target}", flush=True)
        return
    print(f"[START] tencent/HY-Motion-1.0/{variant} -> {target}", flush=True)
    snapshot_download("tencent/HY-Motion-1.0", allow_patterns=[f"{variant}/*"],
                      local_dir=str(root), token=HF_TOKEN)
    for name in ("latest.ckpt", "config.yml"):
        if not (target / name).exists():
            raise SystemExit(f"[FAIL] {target / name} missing after download")
    print(f"[DONE] {variant}", flush=True)


def download_text_encoders() -> None:
    print("[START] Qwen/Qwen3-8B (16GB)", flush=True)
    snapshot_download("Qwen/Qwen3-8B", token=HF_TOKEN)
    print("[START] openai/clip-vit-large-patch14", flush=True)
    # Worker loads via transformers, which prefers safetensors; the .bin,
    # TF and flax copies are the same 1.7GB each.
    snapshot_download("openai/clip-vit-large-patch14",
                      ignore_patterns=["*.bin", "*.msgpack", "*.h5"], token=HF_TOKEN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="HY-Motion-1.0", choices=[*VARIANTS, "both"])
    args = parser.parse_args()

    for v in (VARIANTS if args.variant == "both" else [args.variant]):
        download_dit(v)
    download_text_encoders()
    print(f"[DONE] weights ready. DiT under {CKPTS}, HF cache under {os.environ['HF_HOME']}", flush=True)


if __name__ == "__main__":
    main()
