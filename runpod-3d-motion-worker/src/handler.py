"""RunPod handler for the motion-gpu worker (Tencent-Hunyuan/HY-Motion-1.0,
tencent-hunyuan-community license). Wraps hymotion's T2MRuntime -- the same
entrypoint local_infer.py / gradio_app.py use -- so job errors surface as
Python exceptions, not parsed subprocess output.

Runs from /app (see Dockerfile) so upstream's relative paths resolve.

The optional LLM prompt rewriter / duration estimator (Text2MotionPrompter
via vllm) is NOT run here: the caller supplies the prompt and duration, and
the runtime is built with prompt engineering disabled.
"""
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import runpod
from botocore.config import Config

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"3d-motion-hymotion","message":"%(message)s"}',
)
LOGGER = logging.getLogger(__name__)

R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL", "")

CKPTS_DIR = os.environ.get("HY_MOTION_CKPTS", "/workspace/models/hy-motion")
# Standard needs ~26GB VRAM, Lite ~24GB (model card) -- plus the Qwen3-8B
# text encoder, so budget a 48GB card (L40S/A6000) or 80GB A100/H100.
DEFAULT_VARIANT = os.environ.get("MOTION_VARIANT", "HY-Motion-1.0")
VARIANTS = ("HY-Motion-1.0", "HY-Motion-1.0-Lite")
FPS = 30  # upstream converts frames->seconds with /30.0
MAX_DURATION_S = float(os.environ.get("MAX_DURATION_S", "10"))
MAX_SEEDS = int(os.environ.get("MAX_SEEDS", "4"))

_runtimes: Dict[str, Any] = {}
_r2 = None


class InputValidationError(ValueError):
    """Maps to 3D_ASSET_INVALID_INPUT, same convention as the other 3D workers."""


class MotionOutputError(RuntimeError):
    """Maps to 3D_ASSET_INVALID_GEOMETRY -- output exists but is unusable."""


def r2():
    global _r2
    if _r2 is None:
        _r2 = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
    return _r2


def upload_bytes(data: bytes, key: str, content_type: str) -> str:
    r2().put_object(Bucket=R2_BUCKET_NAME, Key=key, Body=data, ContentType=content_type)
    return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"


def build_asset_key(asset_type: str, ext: str, project_id: Optional[str],
                    frame_id: Optional[str], job_id: Optional[str], suffix: str = "") -> str:
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{time.strftime('%Y%m%d%H%M%S')}_{job_id or uuid.uuid4().hex[:8]}_{asset_type}"
    return f"storystudio/motion_clips/{name}{suffix}.{ext}"


def get_runtime(variant: str):
    """Lazy singleton per variant. Imports are deferred so a bad input is
    rejected before paying for torch/transformers import time."""
    if variant in _runtimes:
        return _runtimes[variant]

    model_dir = Path(CKPTS_DIR) / "tencent" / variant
    if not (model_dir / "latest.ckpt").exists():
        from huggingface_hub import snapshot_download
        LOGGER.info("DiT checkpoint missing at %s, downloading (pre-warm the volume to avoid this)", model_dir)
        snapshot_download("tencent/HY-Motion-1.0", allow_patterns=[f"{variant}/*"],
                          local_dir=str(Path(CKPTS_DIR) / "tencent"))

    from hymotion.utils.t2m_runtime import T2MRuntime
    runtime = T2MRuntime(
        config_path=str(model_dir / "config.yml"),
        ckpt_name=str(model_dir / "latest.ckpt"),
        disable_prompt_engineering=True,
    )
    _runtimes[variant] = runtime
    return runtime


def classify_error(exc: Exception) -> str:
    if isinstance(exc, InputValidationError):
        return "3D_ASSET_INVALID_INPUT"
    if isinstance(exc, MotionOutputError):
        return "3D_ASSET_INVALID_GEOMETRY"
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "3D_ASSET_PROVIDER_OOM"
    return "3D_ASSET_GENERATION_FAILED"


def validate_input(inp: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        raise InputValidationError("input.prompt is required")
    # Model card: English prompts, ideally < 60 words (< 30 to save VRAM).
    if len(prompt.split()) > 120:
        raise InputValidationError("input.prompt is too long (model expects English, ideally < 60 words)")
    duration = float(inp.get("durationSec", 5.0))
    if not 0.5 <= duration <= MAX_DURATION_S:
        raise InputValidationError(f"input.durationSec must be between 0.5 and {MAX_DURATION_S}")
    seeds = inp.get("seeds")
    if seeds is None:
        seeds = [int(inp.get("seed", 42)) + i for i in range(int(inp.get("numSeeds", 1)))]
    seeds = [int(s) for s in seeds]
    if not 1 <= len(seeds) <= MAX_SEEDS:
        raise InputValidationError(f"between 1 and {MAX_SEEDS} seeds are supported")
    variant = inp.get("variant", DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise InputValidationError(f"input.variant must be one of {VARIANTS}")
    return {
        "prompt": prompt,
        "duration": duration,
        "seeds": seeds,
        "cfg_scale": float(inp.get("cfgScale", 5.0)),  # upstream local_infer default
        "variant": variant,
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cold_start = not _runtimes
        args = validate_input(job.get("input", {}))
        inp = job.get("input", {})

        runtime = get_runtime(args["variant"])
        # Without the FBX SDK upstream silently degrades to "dict" format and
        # returns no files -- fail loudly instead of reporting an empty success.
        if not runtime.fbx_available:
            raise MotionOutputError("fbx module unavailable in this image; refusing to return no clips")

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            LOGGER.info("Generating %.1fs motion x%d seeds: %s", args["duration"], len(args["seeds"]), args["prompt"])
            runtime.generate_motion(
                text=args["prompt"],
                seeds_csv=",".join(map(str, args["seeds"])),
                duration=args["duration"],
                cfg_scale=args["cfg_scale"],
                output_format="fbx",
                original_text=args["prompt"],
                output_dir=str(out_dir),
                output_filename="clip",
            )
            gen_time_s = round(time.time() - t0, 1)

            clips: List[Dict[str, Any]] = []
            for i, seed in enumerate(args["seeds"]):
                fbx_path = out_dir / f"clip_{i:03d}.fbx"
                npz_path = out_dir / f"clip_{i:03d}.npz"
                if not fbx_path.exists() or fbx_path.stat().st_size == 0:
                    raise MotionOutputError(f"expected FBX for seed {seed} was not produced")
                suffix = f"_s{seed}"
                clip: Dict[str, Any] = {"seed": seed}
                fbx_key = build_asset_key("motion_generate", "fbx", inp.get("project_id"),
                                          inp.get("frame_id"), job.get("id"), suffix)
                clip["fbxKey"] = fbx_key
                clip["fbxUrl"] = upload_bytes(fbx_path.read_bytes(), fbx_key, "application/octet-stream")
                if npz_path.exists():
                    # Raw SMPL-H params (Rh/trans/poses/betas) for retargeting
                    # without going through FBX.
                    npz_key = fbx_key[:-len(".fbx")] + ".npz"
                    clip["npzKey"] = npz_key
                    clip["npzUrl"] = upload_bytes(npz_path.read_bytes(), npz_key, "application/octet-stream")
                clips.append(clip)

        output: Dict[str, Any] = {
            "clips": clips,
            "durationSec": args["duration"],
            "fps": FPS,
            "skeleton": "smplh",
            "provenance": {
                "provider": "hy-motion",
                "modelVersion": args["variant"],
                "cfgScale": args["cfg_scale"],
                "genTimeS": gen_time_s,
            },
        }
        if cold_start:
            output["cold_start"] = True
        return output

    except Exception as exc:
        LOGGER.error("%s", exc, exc_info=True)
        return {"error": classify_error(exc), "detail": str(exc), "traceback": __import__("traceback").format_exc()}


runpod.serverless.start({"handler": handler})
