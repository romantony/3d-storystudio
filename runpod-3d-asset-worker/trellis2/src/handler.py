import io
import logging
import os
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import runpod
import torch
from botocore.config import Config
from PIL import Image

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"3d-asset-trellis2","message":"%(message)s"}',
)
LOGGER = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/TRELLIS.2-4B")
MODEL_BASE = os.environ.get("MODEL_BASE", "/runpod-volume")
MODEL_PATH = os.path.join(MODEL_BASE, "trellis2-4b")  # populated by scripts/download_model.py

R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL", "")

# Only decimation_target / texture_size are documented, adjustable knobs on
# o_voxel.postprocess.to_glb (see TRELLIS.2's example.py). There is no
# documented steps/cfg control on pipeline.run() to tier on — confirm
# against the pinned commit before adding one.
QUALITY_PROFILES = {
    "draft":    {"decimation_target": 200_000,   "texture_size": 1024},
    "standard": {"decimation_target": 1_000_000, "texture_size": 2048},
    "high":     {"decimation_target": 2_000_000, "texture_size": 4096},
}
NVDIFFRAST_VERTEX_LIMIT = 16_777_216  # per TRELLIS.2 example.py comment

_pipeline = None
_r2 = None


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


def build_asset_key(category: str, asset_type: str, ext: str,
                     project_id: Optional[str], frame_id: Optional[str],
                     job_id: Optional[str]) -> str:
    """storystudio/{category}/{project_id}_{frame_id}_{asset_type}.{ext}, or
    storystudio/{category}/{timestamp}_{job_id}_{asset_type}.{ext} — same
    convention as the image/video/voice workers, extended with a models_3d
    category for generated GLBs."""
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{time.strftime('%Y%m%d%H%M%S')}_{job_id or uuid.uuid4().hex[:8]}_{asset_type}"
    return f"storystudio/{category}/{name}.{ext}"


def load_image(src: str) -> Image.Image:
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "python-3d-asset-trellis2/1.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
    else:
        import base64
        data = base64.b64decode(src)
    return Image.open(io.BytesIO(data)).convert("RGB")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        # Prefer the volume-cached checkpoint (scripts/download_model.py) —
        # falls back to a live HF pull only if the volume wasn't pre-warmed,
        # which works but adds ~15GB to this cold start.
        local_ready = Path(MODEL_PATH).exists() and any(os.scandir(MODEL_PATH))
        source = MODEL_PATH if local_ready else MODEL_ID
        LOGGER.info("Loading Trellis2ImageTo3DPipeline from %s", source)
        t0 = time.time()
        _pipeline = Trellis2ImageTo3DPipeline.from_pretrained(source)
        _pipeline.cuda()
        LOGGER.info("Pipeline ready in %.1fs", time.time() - t0)
    return _pipeline


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cold_start = _pipeline is None
        pipe = get_pipeline()
        inp: Dict[str, Any] = job.get("input", {})

        image_src = inp.get("image_url") or inp.get("image_b64")
        if not image_src:
            return {"error": "input.image_url or input.image_b64 is required"}

        quality = inp.get("quality", "standard")
        profile = QUALITY_PROFILES.get(quality, QUALITY_PROFILES["standard"])
        seed = inp.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))

        t0 = time.time()
        image = load_image(image_src)
        LOGGER.info("Running TRELLIS.2 image-to-3D (quality=%s)", quality)

        # o_voxel is vendored inside the TRELLIS.2 repo (pip-installed from
        # ./o-voxel in the Dockerfile), not a top-level PyPI package.
        from o_voxel import postprocess as o_voxel_postprocess

        with torch.inference_mode():
            mesh = pipe.run(image)[0]
        mesh.simplify(NVDIFFRAST_VERTEX_LIMIT)

        glb = o_voxel_postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=profile["decimation_target"],
            texture_size=profile["texture_size"],
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )

        # to_glb's documented export contract is export(path, ...) — write
        # to a tmp file rather than guessing at file-object support.
        with tempfile.TemporaryDirectory() as tmp:
            glb_path = Path(tmp) / f"{uuid.uuid4().hex}.glb"
            glb.export(str(glb_path), extension_webp=True)
            glb_bytes = glb_path.read_bytes()

        torch.cuda.empty_cache()
        gen_time_s = round(time.time() - t0, 1)
        LOGGER.info("Generation completed in %.1fs", gen_time_s)

        key = inp.get("output_key") or build_asset_key(
            "models_3d", "asset_generate", "glb",
            inp.get("project_id"), inp.get("frame_id"), job.get("id"),
        )
        model_url = upload_bytes(glb_bytes, key, "model/gltf-binary")

        output: Dict[str, Any] = {
            "modelUrl": model_url,
            "provenance": {
                "provider": "trellis2",
                "modelVersion": "TRELLIS.2-4B",
                "quality": quality,
                "seed": seed,
                "sourceImageId": inp.get("sourceImageId"),
                "genTimeS": gen_time_s,
            },
        }
        if cold_start:
            output["cold_start"] = True

        return output

    except Exception as exc:
        LOGGER.error("%s", exc, exc_info=True)
        return {"error": str(exc), "traceback": __import__("traceback").format_exc()}


runpod.serverless.start({"handler": handler})
