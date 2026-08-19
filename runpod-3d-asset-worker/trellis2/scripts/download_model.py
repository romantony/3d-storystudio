#!/usr/bin/env python3
"""Pre-warm a RunPod network volume with the TRELLIS.2-4B checkpoint (~15GB)
so the serverless worker doesn't eat that download on every cold start.

Run this from a RunPod Pod (not the serverless worker itself) with the same
network volume attached that the `3d-asset-trellis2` endpoint uses — this
account's volumes mount at /workspace on both Pods and Serverless workers.
Must land under /workspace/models/Trellis2 specifically (see MODEL_PATH in
../src/handler.py) — anywhere else on a Pod (e.g. the default HF cache
under $HOME) is the pod's local container disk, not the network volume,
and gets thrown away when the pod is stopped.
Matches the pattern already used for the Qwen workers
(qwen-image-gen/scripts/download_models.py).

    pip install -q huggingface_hub
    python download_model.py
"""
import os
import subprocess
import sys
from pathlib import Path

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download

REPO_ID = "microsoft/TRELLIS.2-4B"
LOCAL_DIR = os.environ.get("TRELLIS2_LOCAL_DIR", "/workspace/models/Trellis2")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# The repo is just ckpts/*.safetensors + *.json — no flax/tf/rust duplicates
# to skip, unlike the diffusers-hub models the ignore list was written for
# elsewhere. Kept anyway so this stays correct if upstream ever adds them.
IGNORE = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.ot"]


def main() -> None:
    dest = Path(LOCAL_DIR)
    marker = dest / ".download-complete"

    if marker.exists():
        size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024**3
        print(f"[SKIP] {REPO_ID} already downloaded to {dest} ({size_gb:.1f} GB)")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"[START] {REPO_ID} -> {dest} (~15 GB)", flush=True)

    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        ignore_patterns=IGNORE,
        token=HF_TOKEN,
    )

    marker.touch()
    size_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024**3
    print(f"[DONE] {REPO_ID} -- {size_gb:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
