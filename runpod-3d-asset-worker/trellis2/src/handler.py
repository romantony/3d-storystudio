import hashlib
import io
import json
import logging
import os
import struct
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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
MODEL_BASE = os.environ.get("MODEL_BASE", "/workspace")
MODEL_PATH = os.path.join(MODEL_BASE, "models", "Trellis2")  # populated by scripts/download_model.py

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
    shapes Phase 0 hit and shipped for weeks before anyone noticed by eye
    (see doc/phase1-handoff.md Part 2.4): (1) a non-glTF file uploaded
    with a glTF content-type (magic-byte check), and (2) a structurally-
    valid-but-degenerate mesh -- job reports success, URL is populated,
    but the geometry itself is near-empty (vertex count / bounding-box
    extent checks)."""
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
    """Spec §24.1 error code mapping. The asset workers previously
    returned raw {error: str(exc), traceback} for everything -- per
    doc/phase1-handoff.md, Phase 1's job wrapper was left to guess this
    mapping from the raw Python message. OOM detection is text-based
    (rather than isinstance(exc, torch.cuda.OutOfMemoryError)) since
    OOMs can surface wrapped in a RuntimeError from deep inside a
    pipeline call, not always as the literal OOM exception type."""
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
        prompt = inp.get("prompt")
        if not image_src and not prompt:
            return {"error": "3D_ASSET_INVALID_INPUT",
                    "detail": "input.image_url, input.image_b64, or input.prompt is required"}
        if not image_src:
            image_src = generate_image_url_from_prompt(
                prompt, inp.get("project_id"), inp.get("frame_id"), inp.get("promptAspectRatio", "1:1"),
            )

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
            try:
                mesh = pipe.run(image)[0]
            except ValueError as exc:
                # trellis2_image_to_3d.py's preprocess_image() does
                # `np.min(np.argwhere(alpha > 0.8*255)[:, 1])` after running
                # RMBG-2.0 on the input. If RMBG finds zero foreground
                # pixels (e.g. a full-scene/interior image instead of an
                # isolated single-object shot), that argwhere is empty and
                # np.min raises this exact opaque message -- not a bug in
                # our worker, but worth surfacing as an actionable error
                # instead of a raw numpy stack trace. Confirmed against the
                # vendored source at the pinned TRELLIS2_COMMIT.
                if "zero-size array" in str(exc):
                    raise InputValidationError(
                        "TRELLIS.2 found no foreground subject in the input "
                        "image (background removal produced an all-transparent "
                        "mask). Provide an isolated single-object product shot, "
                        "not a full-scene or interior image."
                    ) from exc
                raise
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

        # Spec §20.3-equivalent QC for asset generation: a job can exit
        # clean with a populated URL and still ship garbage (the
        # char_kitten torn-sheet finding from the provider bake-off) --
        # verify the actual bytes before uploading, not just that export()
        # didn't raise.
        validate_glb_bytes(glb_bytes)

        torch.cuda.empty_cache()
        gen_time_s = round(time.time() - t0, 1)
        LOGGER.info("Generation completed in %.1fs", gen_time_s)

        key = inp.get("output_key") or build_asset_key(
            "models_3d", "asset_generate", "glb",
            inp.get("project_id"), inp.get("frame_id"), job.get("id"),
        )
        model_url = upload_bytes(glb_bytes, key, "model/gltf-binary")

        provenance: Dict[str, Any] = {
            "provider": "trellis2",
            "modelVersion": "TRELLIS.2-4B",
            "quality": quality,
            "seed": seed,
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
