import base64
import hashlib
import io
import json
import logging
import os
import struct
import sys
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Text-to-3D (input.prompt): this worker only ever does image-to-3D itself
# -- a prompt is resolved to an image by calling the existing Qwen
# image-gen RunPod endpoint synchronously, then the normal image_url path
# runs unchanged. Both are optional; a prompt-only request without these
# configured fails with a clear 3D_ASSET_INVALID_INPUT rather than a
# missing-env-var KeyError.
QWEN_ENDPOINT_ID = os.environ.get("QWEN_ENDPOINT_ID")
RUNPOD_API_KEY   = os.environ.get("RUNPOD_API_KEY")
QWEN_POLL_INTERVAL_S = 5
QWEN_POLL_TIMEOUT_S = 300  # leaves headroom under executionTimeoutMs (600s) for the actual 3D generation

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


class InputValidationError(ValueError):
    """Bad/missing request input -- maps to 3D_ASSET_INVALID_INPUT, an
    extension of spec §24.1's (non-exhaustive, per its own heading) error
    code list -- not a provider/geometry failure."""


class GeometryValidationError(ValueError):
    """Maps to spec §24.1's 3D_ASSET_INVALID_GEOMETRY."""


def validate_glb_bytes(data: bytes) -> None:
    """Byte-level glTF geometry sanity check, deliberately stdlib-only
    (struct+json, no trimesh/pymeshlab load) so it adds no new dependency
    and can't itself crash/OOM on a huge mesh. Catches both real failure
    shapes this worker hit and shipped for weeks before anyone noticed by
    eye: (1) a non-glTF file uploaded with a glTF content-type -- the
    exact OBJ/GLB mislabeling bug _export_mesh() below now works around,
    caught here as defense in depth -- and (2) a structurally-valid-but-
    degenerate mesh (the char_kitten torn-sheet bug from the provider
    bake-off: job reports COMPLETED with a populated URL, but the
    geometry itself is a near-empty sheet)."""
    if len(data) < 20 or data[:4] != b"glTF":
        raise GeometryValidationError("output is not a valid GLB (bad magic bytes)")
    _version, total_length = struct.unpack_from("<II", data, 4)
    if total_length > len(data):
        raise GeometryValidationError("GLB header length exceeds actual file size (truncated upload)")

    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        raise GeometryValidationError("GLB's first chunk is not JSON")
    try:
        gltf = json.loads(data[20:20 + chunk_length])
    except json.JSONDecodeError as exc:
        raise GeometryValidationError(f"GLB JSON chunk is not valid JSON: {exc}") from exc

    meshes = gltf.get("meshes", [])
    if not meshes:
        raise GeometryValidationError("GLB has zero meshes")

    accessors = gltf.get("accessors", [])
    total_vertices = 0
    min_bound: Optional[List[float]] = None
    max_bound: Optional[List[float]] = None
    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            pos_idx = prim.get("attributes", {}).get("POSITION")
            if pos_idx is None or pos_idx >= len(accessors):
                continue
            acc = accessors[pos_idx]
            total_vertices += acc.get("count", 0)
            acc_min, acc_max = acc.get("min"), acc.get("max")
            if acc_min and acc_max:
                min_bound = acc_min if min_bound is None else [min(a, b) for a, b in zip(min_bound, acc_min)]
                max_bound = acc_max if max_bound is None else [max(a, b) for a, b in zip(max_bound, acc_max)]

    if total_vertices < 4:
        raise GeometryValidationError(f"GLB has too few vertices ({total_vertices}) to be a real mesh")
    if min_bound and max_bound and max(hi - lo for lo, hi in zip(min_bound, max_bound)) < 1e-6:
        raise GeometryValidationError("GLB geometry bounding box is degenerate (near-zero extent)")


def classify_error(exc: Exception) -> str:
    """Spec §24.1 error code mapping. This worker previously returned raw
    {error: str(exc), traceback} for everything -- per doc/phase1-handoff.md,
    Phase 1's job wrapper was left to guess this mapping from the raw
    Python message. OOM detection is text-based (rather than
    isinstance(exc, torch.cuda.OutOfMemoryError)) since this handler
    doesn't import torch directly and OOMs can surface wrapped in a
    RuntimeError from deep inside a pipeline call anyway."""
    if isinstance(exc, InputValidationError):
        return "3D_ASSET_INVALID_INPUT"
    if isinstance(exc, GeometryValidationError):
        return "3D_ASSET_INVALID_GEOMETRY"
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "3D_ASSET_PROVIDER_OOM"
    return "3D_ASSET_GENERATION_FAILED"


def _runpod_request(endpoint_id: str, path: str, method: str = "GET",
                     body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"https://api.runpod.ai/v2/{endpoint_id}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {RUNPOD_API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _extract_qwen_image_url(output: Dict[str, Any]) -> str:
    """Qwen's exact output field name isn't a documented contract here
    (that worker lives in a different repo) -- try the common shapes and
    fail loudly with the raw payload rather than guessing silently. Same
    logic as fixtures/3d/pipeline-smoke-test/run_full_pipeline_test.py's
    extract_image_url(), which confirmed this against real Qwen output."""
    for key in ("imageUrl", "image_url", "url", "output_url"):
        if isinstance(output.get(key), str):
            return output[key]
    images = output.get("images")
    if isinstance(images, list) and images:
        if isinstance(images[0], str):
            return images[0]
        if isinstance(images[0], dict):
            for key in ("url", "imageUrl", "image_url"):
                if key in images[0]:
                    return images[0][key]
    raise InputValidationError(f"Qwen image-gen returned no recognizable image URL: {json.dumps(output)[:500]}")


def generate_image_url_from_prompt(prompt: str, project_id: Optional[str], frame_id: Optional[str],
                                    aspect_ratio: str) -> str:
    """input.prompt (text-to-3D): chains the existing Qwen image-gen
    RunPod endpoint synchronously ahead of this worker's own image-to-3D
    pipeline -- same three-endpoint chain
    fixtures/3d/pipeline-smoke-test/run_full_pipeline_test.py already
    proved works end-to-end, just done in-worker so a caller can submit
    one job instead of orchestrating three themselves."""
    if not QWEN_ENDPOINT_ID or not RUNPOD_API_KEY:
        raise InputValidationError(
            "input.prompt (text-to-3D) requires QWEN_ENDPOINT_ID and RUNPOD_API_KEY to be configured "
            "on this endpoint -- this worker chains Qwen image-gen -> image-to-3D internally rather "
            "than generating 3D directly from text."
        )
    LOGGER.info("Resolving input.prompt via Qwen text-to-image: %.70s...", prompt)
    submitted = _runpod_request(QWEN_ENDPOINT_ID, "/run", method="POST", body={
        "input": {"model": "qwen", "prompt": prompt, "aspect_ratio": aspect_ratio,
                   "project_id": project_id, "frame_id": frame_id}
    })
    job_id = submitted["id"]
    deadline = time.time() + QWEN_POLL_TIMEOUT_S
    while time.time() < deadline:
        result = _runpod_request(QWEN_ENDPOINT_ID, f"/status/{job_id}")
        status = result.get("status")
        if status == "COMPLETED":
            return _extract_qwen_image_url(result.get("output", {}))
        if status == "FAILED":
            raise InputValidationError(f"Qwen image-gen job failed: {result.get('output')}")
        time.sleep(QWEN_POLL_INTERVAL_S)
    raise InputValidationError(f"Qwen image-gen job {job_id} did not finish within {QWEN_POLL_TIMEOUT_S}s")


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
    (trimesh/pymeshlab in requirements.txt) — export via .export().

    Paint-pipeline output is a path string, but it's an upstream bug, not a
    documented contract: Hunyuan3DPaintPipeline.__call__ (textureGenPipeline.py)
    saves the mesh as OBJ, then -- since save_glb defaults to True, and we
    never pass save_glb=False -- calls convert_obj_to_glb() to also produce
    a real .glb right next to it, but its `return output_mesh_path` line
    returns the *.obj* path regardless, leaving the .glb it just built
    unused. Confirmed against a real render-worker failure: the uploaded
    "textured.glb" was actually plain-text Wavefront OBJ (`file` reported
    "ASCII text"; content started with `mtllib ... v ... f ...`), which
    Blender's gltf importer silently produced zero objects from -- no
    exception, no error, RGB pass just never got written. If the returned
    path is an .obj with a .glb sibling (i.e. this exact case), prefer the
    real glb rather than trusting the extension-less-but-wrong return
    value."""
    if isinstance(mesh_obj, (str, os.PathLike)):
        mesh_path = Path(mesh_obj)
        if mesh_path.suffix == ".obj":
            glb_sibling = mesh_path.with_suffix(".glb")
            if glb_sibling.exists():
                return glb_sibling
        return mesh_path
    mesh_obj.export(str(dest_path))
    return dest_path


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cold_start = _shape_pipeline is None
        inp: Dict[str, Any] = job.get("input", {})

        image_src = inp.get("image_url") or inp.get("image_b64")
        prompt = inp.get("prompt")
        if not image_src and not prompt:
            return {"error": "3D_ASSET_INVALID_INPUT",
                    "detail": "input.image_url, input.image_b64, or input.prompt is required"}
        if not image_src:
            image_src = generate_image_url_from_prompt(
                prompt, inp.get("project_id"), inp.get("frame_id"), inp.get("promptAspectRatio", "1:1"),
            )

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

        # Spec §20.3-equivalent QC for asset generation: defense in depth
        # alongside _export_mesh()'s own OBJ/GLB-sibling fix above, and the
        # actual catch for the char_kitten-class failure (mesh exports
        # clean, geometry itself is garbage) that _export_mesh() can't see.
        validate_glb_bytes(glb_bytes)

        gen_time_s = round(time.time() - t0, 1)
        LOGGER.info("Generation completed in %.1fs (textured=%s)", gen_time_s, textured)

        key = inp.get("output_key") or build_asset_key(
            "models_3d", "asset_generate", "glb",
            inp.get("project_id"), inp.get("frame_id"), job.get("id"),
        )
        model_url = upload_bytes(glb_bytes, key, "model/gltf-binary")

        provenance: Dict[str, Any] = {
            "provider": "hunyuan3d",
            "modelVersion": "Hunyuan3D-2.1",
            "quality": quality,
            "textured": textured,
            "sourceImageId": inp.get("sourceImageId"),
            "genTimeS": gen_time_s,
        }
        if prompt:
            provenance["sourcePromptHash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        output: Dict[str, Any] = {
            "modelUrl": model_url,
            "modelKey": key,
            "provenance": provenance,
        }
        if cold_start:
            output["cold_start"] = True

        return output

    except Exception as exc:
        LOGGER.error("%s", exc, exc_info=True)
        return {"error": classify_error(exc), "detail": str(exc), "traceback": __import__("traceback").format_exc()}


runpod.serverless.start({"handler": handler})
