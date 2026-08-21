#!/usr/bin/env python3
"""Pre-warm a RunPod network volume with the Hunyuan3D-2.1 checkpoints so
the serverless worker doesn't eat these downloads on every cold start.

Run this from a RunPod Pod (not the serverless worker itself) with the
same network volume attached that the `3d-asset-hunyuan3d` endpoint uses
-- this account's volumes mount at /workspace on both Pods and Serverless
workers. Matches the pattern in ../../trellis2/scripts/download_model.py,
but Hunyuan3D-2.1 needs THREE separate downloads instead of one, because
its two pipelines cache to two genuinely different places:

  1. Shape checkpoint (hy3dshape's Hunyuan3DDiTFlowMatchingPipeline) --
     loaded via a custom `smart_load_model` helper that does NOT use
     standard HF caching. It reads HY3DGEN_MODELS (set in the Dockerfile
     to /workspace/models/hy3dgen) and passes
     `{HY3DGEN_MODELS}/tencent/Hunyuan3D-2.1` as snapshot_download's
     local_dir -- confirmed against the actual upstream source
     (hy3dshape/hy3dshape/utils/utils.py). This script replicates that
     exact call so the pre-warmed files land exactly where
     smart_load_model will look for them.
  2. Paint multiview diffusion weights (hunyuan3d-paintpbr-v2-1 subfolder
     of the same HF repo) -- loaded via a plain snapshot_download with no
     local_dir, so it uses the standard HF cache and respects HF_HOME
     (already /workspace/hf-cache in the Dockerfile) automatically.
  3. facebook/dinov2-giant -- used by the paint pipeline's view
     processor, also plain HF caching via HF_HOME.

    pip install -q huggingface_hub
    python download_model.py
"""
import os
import subprocess
import sys
from pathlib import Path

# Must happen before huggingface_hub is imported -- it reads HF_HOME at
# import time to compute its cache path constants. Run on a bare RunPod Pod
# (not inside the worker image, which sets these as Dockerfile ENV), this
# silently fell back to ~/.cache/huggingface on the pod's local container
# disk instead of the network volume -- confirmed against a real run where
# the paint + dinov2 downloads (~16GB) landed there instead of /workspace
# and would've been lost on pod stop.
os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/workspace/hf-cache")

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])

from huggingface_hub import snapshot_download

REPO_ID = "tencent/Hunyuan3D-2.1"
HY3DGEN_MODELS = os.environ.get("HY3DGEN_MODELS", "/workspace/models/hy3dgen")
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None


def already_done(marker: Path) -> bool:
    return marker.exists()


def download_shape_checkpoint():
    """Mirrors smart_load_model's own local_dir computation exactly:
    {HY3DGEN_MODELS}/{repo_id} as local_dir, hunyuan3d-dit-v2-1/* as the
    allow_patterns filter."""
    subfolder = "hunyuan3d-dit-v2-1"
    local_dir = Path(HY3DGEN_MODELS) / REPO_ID
    marker = local_dir / ".download-complete"
    if already_done(marker):
        print(f"[SKIP] shape checkpoint already at {local_dir}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[START] {REPO_ID}/{subfolder} -> {local_dir}", flush=True)
    snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(local_dir),
        token=HF_TOKEN,
    )
    marker.touch()
    print(f"[DONE] shape checkpoint -> {local_dir}", flush=True)


def download_paint_multiview():
    """No local_dir -- lands in the standard HF cache under HF_HOME."""
    subfolder = "hunyuan3d-paintpbr-v2-1"
    print(f"[START] {REPO_ID}/{subfolder} -> standard HF cache (HF_HOME={os.environ.get('HF_HOME')})", flush=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[f"{subfolder}/*"],
        token=HF_TOKEN,
    )
    print(f"[DONE] paint multiview weights -> {path}", flush=True)


def download_dinov2():
    """ignore_patterns skips pytorch_model.bin -- the paint pipeline's view
    processor loads facebook/dinov2-giant via transformers, which prefers
    safetensors when both are present, so the .bin duplicate (same ~4.5GB
    of weights in a second format) was pure waste. Confirmed against a real
    run: this repo downloaded 9.09GB before the filter, vs the ~4.5GB the
    safetensors file alone accounts for."""
    print(f"[START] facebook/dinov2-giant -> standard HF cache (HF_HOME={os.environ.get('HF_HOME')})", flush=True)
    path = snapshot_download(
        repo_id="facebook/dinov2-giant",
        ignore_patterns=["*.bin"],
        token=HF_TOKEN,
    )
    print(f"[DONE] dinov2-giant -> {path}", flush=True)


def main() -> None:
    download_shape_checkpoint()
    download_paint_multiview()
    download_dinov2()


if __name__ == "__main__":
    main()
