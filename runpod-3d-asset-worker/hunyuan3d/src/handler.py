import base64
import io
import logging
import os
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
import runpod
from botocore.config import Config
from PIL import Image

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"3d-asset-hunyuan3d","message":"%(message)s"}',
)
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, "./hy3dshape")
sys.path.insert(0, "./hy3dpaint")

MODEL_ID = os.environ.get("MODEL_ID", "tencent/Hunyuan3D-2.1")

R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL", "")

# README-documented default (Hunyuan3DPaintConfig(max_num_view=6,
# resolution=512)). Per-tier overrides aren't in the documented API — confirm
# against the pinned commit before adding one instead of guessing at kwargs.
DEFAULT_MAX_NUM_VIEW = int(os.environ.get("PAINT_MAX_NUM_VIEW", "6"))
DEFAULT_PAINT_RESOLUTION = int(os.environ.get("PAINT_RESOLUTION", "512"))

_shape_pipeline = None
_paint_pipeline = None
_paint_pipeline_cfg = None
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
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{time.strftime('%Y%m%d%H%M%S')}_{job_id or uuid.uuid4().hex[:8]}_{asset_type}"
    return f"storystudio/{category}/{name}.{ext}"


def fetch_image_to_file(src: str, dest_path: Path) -> None:
    """Both pipelines take an image *path*, not a PIL object or bytes, per
    the documented shape_pipeline(image='assets/demo.png') / paint_pipeline
    image_path= usage — materialize the input to disk rather than guessing
    at in-memory support."""
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "python-3d-asset-hunyuan3d/1.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
    else:
        data = base64.b64decode(src)
    Image.open(io.BytesIO(data)).convert("RGB").save(dest_path, format="PNG")


def get_shape_pipeline():
    global _shape_pipeline
    if _shape_pipeline is None:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        LOGGER.info("Loading Hunyuan3DDiTFlowMatchingPipeline from %s", MODEL_ID)
        t0 = time.time()
        _shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_ID)
        LOGGER.info("Shape pipeline ready in %.1fs", time.time() - t0)
    return _shape_pipeline


def get_paint_pipeline(max_num_view: int, resolution: int):
    global _paint_pipeline, _paint_pipeline_cfg
    cfg_key = (max_num_view, resolution)
    if _paint_pipeline is None or _paint_pipeline_cfg != cfg_key:
        from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

        LOGGER.info("Loading Hunyuan3DPaintPipeline (max_num_view=%d, resolution=%d)", max_num_view, resolution)
        t0 = time.time()
        paint_config = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
        # Hunyuan3DPaintConfig's own default ("ckpt/RealESRGAN_x4plus.pth")
        # is relative to hy3dpaint/, not the repo root we run from -- confirmed
        # against a real job traceback (FileNotFoundError: 'ckpt/RealESRGAN_
        # x4plus.pth') and against upstream's own demo.py/gradio_app.py, which
        # both patch this exact attribute the same way rather than relying on
        # the class default.
        paint_config.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        _paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
        _paint_pipeline_cfg = cfg_key
        LOGGER.info("Paint pipeline ready in %.1fs", time.time() - t0)
    return _paint_pipeline


def _export_mesh(mesh_obj: Any, dest_path: Path) -> Path:
    """Shape-pipeline output is trimesh-like per the repo's dependency set
    (trimesh/pymeshlab in requirements.txt) — export via .export(). If a
    pipeline instead hands back a path string (paint_pipeline's return type
    isn't fully specified in the README), use that file directly."""
    if isinstance(mesh_obj, (str, os.PathLike)):
        return Path(mesh_obj)
    mesh_obj.export(str(dest_path))
    return dest_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cold_start = _shape_pipeline is None
        inp: Dict[str, Any] = job.get("input", {})

        image_src = inp.get("image_url") or inp.get("image_b64")
        if not image_src:
            return {"error": "input.image_url or input.image_b64 is required"}

        quality = inp.get("quality", "standard")  # "shape_only" | "standard" (shape+texture)
        max_num_view = int(inp.get("maxNumView", DEFAULT_MAX_NUM_VIEW))
        paint_resolution = int(inp.get("paintResolution", DEFAULT_PAINT_RESOLUTION))

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            image_path = tmp_dir / "input.png"
            fetch_image_to_file(image_src, image_path)

            shape_pipeline = get_shape_pipeline()
            LOGGER.info("Running Hunyuan3D shape generation (quality=%s)", quality)
            mesh_untextured = shape_pipeline(image=str(image_path))[0]
            shape_glb = _export_mesh(mesh_untextured, tmp_dir / "shape.glb")

            final_path = shape_glb
            textured = quality != "shape_only"
            if textured:
                paint_pipeline = get_paint_pipeline(max_num_view, paint_resolution)
                LOGGER.info("Running Hunyuan3D-Paint texture generation")
                mesh_textured = paint_pipeline(str(shape_glb), image_path=str(image_path))
                final_path = _export_mesh(mesh_textured, tmp_dir / "textured.glb")

            glb_bytes = Path(final_path).read_bytes()

        gen_time_s = round(time.time() - t0, 1)
        LOGGER.info("Generation completed in %.1fs (textured=%s)", gen_time_s, textured)

        key = inp.get("output_key") or build_asset_key(
            "models_3d", "asset_generate", "glb",
            inp.get("project_id"), inp.get("frame_id"), job.get("id"),
        )
        model_url = upload_bytes(glb_bytes, key, "model/gltf-binary")

        output: Dict[str, Any] = {
            "modelUrl": model_url,
            "provenance": {
                "provider": "hunyuan3d",
                "modelVersion": "Hunyuan3D-2.1",
                "quality": quality,
                "textured": textured,
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
