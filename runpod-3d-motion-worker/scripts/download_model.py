#!/usr/bin/env python3
"""Pre-warm a RunPod network volume with the HY-Motion-1.0 weights so the
serverless worker doesn't eat ~20GB of downloads on every cold start.

Run from a RunPod Pod with the same network volume attached that the
`3d-motion-hymotion` endpoint uses (mounts at /workspace). Three pieces:

  1. DiT checkpoint (tencent/HY-Motion-1.0, HY-Motion-1.0[-Lite]/ subfolder)
     -> {HY_MOTION_CKPTS}/tencent/<variant>, the layout T2MRuntime expects
     (config.yml + latest.ckpt).
  2. openai/clip-vit-large-patch14 and Qwen/Qwen3-8B -> standard HF cache
     under HF_HOME (the worker runs with USE_HF_MODELS=1, so
     text_encoder.py loads them by repo id).

    python download_model.py [--variant HY-Motion-1.0|HY-Motion-1.0-Lite]
"""
import argparse
import os

# Must happen before huggingface_hub is imported (it reads HF_HOME at import).
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf-cache")

from huggingface_hub import snapshot_download  # noqa: E402

CKPTS = os.environ.get("HY_MOTION_CKPTS", "/workspace/models/hy-motion")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="HY-Motion-1.0",
                        choices=["HY-Motion-1.0", "HY-Motion-1.0-Lite"])
    args = parser.parse_args()

    local_dir = os.path.join(CKPTS, "tencent")
    print(f"[START] tencent/HY-Motion-1.0/{args.variant} -> {local_dir}", flush=True)
    snapshot_download("tencent/HY-Motion-1.0", allow_patterns=[f"{args.variant}/*"],
                      local_dir=local_dir, token=HF_TOKEN)

    print("[START] openai/clip-vit-large-patch14", flush=True)
    # safetensors/pytorch duplicates and TF/flax formats are dead weight
    snapshot_download("openai/clip-vit-large-patch14",
                      ignore_patterns=["*.msgpack", "*.h5", "tf_model*", "flax_model*"],
                      token=HF_TOKEN)

    print("[START] Qwen/Qwen3-8B", flush=True)
    snapshot_download("Qwen/Qwen3-8B", token=HF_TOKEN)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
