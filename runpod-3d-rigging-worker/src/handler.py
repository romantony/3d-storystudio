"""RunPod handler for the rigging-gpu worker (VAST-AI-Research/SkinTokens,
MIT license, TokenRig model). Wraps demo.py's run_rig() -- the same Python
entrypoint the upstream CLI/Gradio demo calls -- rather than shelling out
to demo.py itself, so job errors surface as real Python exceptions instead
of parsed subprocess output.

Runs from /workspace/SkinTokens (see Dockerfile) so demo.py's relative
paths (experiments/.../grpo_1400.ckpt, bpy_server.py) resolve correctly.
"""
import hashlib
import io
import json
import logging
import os
import struct
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import runpod
from botocore.config import Config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import run_rig, start_bpy_server, wait_for_bpy_server, MODEL_CKPTS  # noqa: E402

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"3d-rigging-skintokens","message":"%(message)s"}',
)
LOGGER = logging.getLogger(__name__)

R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL", "")

_bpy_server_proc = None
_bpy_server_lock = threading.Lock()
_r2 = None


def ensure_bpy_server() -> None:
    """demo.py's run_rig() talks to a background bpy_server.py subprocess
    over localhost HTTP for the actual glTF import/export/skin-transfer
    work (see src/server/bpy_server.py) -- started once and left running
    across jobs, same lazy-singleton pattern as the other 3D workers'
    pipeline caching."""
    global _bpy_server_proc
    with _bpy_server_lock:
        if _bpy_server_proc is None or _bpy_server_proc.poll() is not None:
            _bpy_server_proc = start_bpy_server()
            wait_for_bpy_server(timeout=60)


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


def download_to(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "python-3d-rigging-skintokens/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


class InputValidationError(ValueError):
    """Maps to 3D_ASSET_INVALID_INPUT -- extension of spec §24.1's
    (non-exhaustive) error code list, same convention as the other two
    asset workers in this repo."""


class GeometryValidationError(ValueError):
    """Maps to spec §24.1's 3D_ASSET_INVALID_GEOMETRY."""


def _parse_gltf_json_chunk(data: bytes) -> Dict[str, Any]:
    """Shared byte-level GLB parsing helper -- same approach as the other
    two asset workers' validate_glb_bytes() (stdlib struct+json, no
    trimesh/pymeshlab dependency), extended here to also read `skins`/
    `nodes` for joint-name extraction, not just mesh/accessor validation."""
    if len(data) < 20 or data[:4] != b"glTF":
        raise GeometryValidationError("output is not a valid GLB (bad magic bytes)")
    _version, total_length = struct.unpack_from("<II", data, 4)
    if total_length > len(data):
        raise GeometryValidationError("GLB header length exceeds actual file size (truncated upload)")
    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    if chunk_type != b"JSON":
        raise GeometryValidationError("GLB's first chunk is not JSON")
    try:
        return json.loads(data[20:20 + chunk_length])
    except json.JSONDecodeError as exc:
        raise GeometryValidationError(f"GLB JSON chunk is not valid JSON: {exc}") from exc


def validate_rigged_glb(data: bytes) -> Dict[str, Any]:
    """Sanity-checks the rigged output the same way the other two asset
    workers check generated geometry (real bytes, not job status alone --
    see doc/phase1-handoff.md Part 2.4), plus a rigging-specific check:
    a "rigged" GLB with a mesh but zero skins/joints is exactly the kind
    of silent-degrade failure (job succeeds, output looks plausible, but
    nothing is actually posable) that pattern already caught twice for
    this pipeline's asset-generation workers."""
    gltf = _parse_gltf_json_chunk(data)
    if not gltf.get("meshes"):
        raise GeometryValidationError("rigged GLB has zero meshes")
    skins = gltf.get("skins", [])
    if not skins:
        raise GeometryValidationError(
            "rigged GLB has no skin/skeleton -- rigging silently produced an unrigged mesh"
        )
    nodes = gltf.get("nodes", [])
    joint_indices = skins[0].get("joints", [])
    if not joint_indices:
        raise GeometryValidationError("rigged GLB's skin has zero joints")
    joint_names = [nodes[j].get("name", f"joint_{j}") for j in joint_indices if j < len(nodes)]
    return {"jointCount": len(joint_indices), "jointNames": joint_names}


# Heuristic, not a confirmed taxonomy: the recommended checkpoint
# (articulation_xl_quantization_256_token_4) is trained on the broad,
# non-humanoid-only ArticulationXL2.0 dataset, not one of the fixed
# Mixamo/VRoid bone-name schemes shipped in configs/skeleton/ for a
# different, constrained mode -- so the actual joint-naming convention
# this model predicts hasn't been confirmed against real inference output
# yet. This matches common naming families (Mixamo-style "LeftHand",
# VRoid-style "J_Bip_L_Hand", or plain "hand_l") via loose substring
# matching on a normalized (lowercased, separators stripped) name, so it
# degrades gracefully (fewer anchors filled in) rather than crashing on an
# unexpected scheme. Revisit once real output is inspected -- see
# README.md's verification-status note.
_ANCHOR_PATTERNS: Dict[str, List[str]] = {
    "head":       ["head"],
    "left_hand":  ["lefthand", "handleft", "hand_l", "l_hand"],
    "right_hand": ["righthand", "handright", "hand_r", "r_hand"],
    "left_foot":  ["leftfoot", "footleft", "foot_l", "l_foot", "lefttoe"],
    "right_foot": ["rightfoot", "footright", "foot_r", "r_foot", "righttoe"],
    "root":       ["hips", "pelvis", "root"],
}


def map_semantic_anchors(joint_names: List[str]) -> Dict[str, str]:
    anchors: Dict[str, str] = {}
    for name in joint_names:
        normalized = "".join(ch for ch in name.lower() if ch.isalnum())
        for anchor, patterns in _ANCHOR_PATTERNS.items():
            if anchor in anchors:
                continue
            if any(p in normalized for p in patterns):
                anchors[anchor] = name
    return anchors


def classify_error(exc: Exception) -> str:
    if isinstance(exc, InputValidationError):
        return "3D_ASSET_INVALID_INPUT"
    if isinstance(exc, GeometryValidationError):
        return "3D_ASSET_INVALID_GEOMETRY"
    text = str(exc).lower()
    if "out of memory" in text or "cuda oom" in text:
        return "3D_ASSET_PROVIDER_OOM"
    return "3D_ASSET_GENERATION_FAILED"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        cold_start = _bpy_server_proc is None
        inp: Dict[str, Any] = job.get("input", {})

        asset_url = inp.get("assetUrl") or inp.get("model_url")
        if not asset_url:
            return {"error": "3D_ASSET_INVALID_INPUT", "detail": "input.assetUrl is required"}

        ensure_bpy_server()

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.glb"
            output_path = tmp_dir / "output.glb"
            download_to(asset_url, input_path)

            LOGGER.info("Running SkinTokens auto-rig on %s", asset_url)
            run_rig(
                [input_path],
                float(inp.get("topK", 5)),
                float(inp.get("topP", 0.95)),
                float(inp.get("temperature", 1.0)),
                float(inp.get("repetitionPenalty", 2.0)),
                int(inp.get("numBeams", 10)),
                False,  # use_skeleton -- MVP is full auto-rig (skeleton + skin), not skin-only reuse
                True,   # use_transfer -- preserve the source PBR texture/scale (spec §9's manifest.materials)
                bool(inp.get("usePostprocess", False)),
                [output_path],
                MODEL_CKPTS[0],
                None,
            )
            gen_time_s = round(time.time() - t0, 1)

            glb_bytes = output_path.read_bytes()

        skeleton_info = validate_rigged_glb(glb_bytes)
        anchors = map_semantic_anchors(skeleton_info["jointNames"])

        key = inp.get("output_key") or build_asset_key(
            "models_3d_rigged", "rig_generate", "glb",
            inp.get("project_id"), inp.get("frame_id"), job.get("id"),
        )
        model_url = upload_bytes(glb_bytes, key, "model/gltf-binary")

        output: Dict[str, Any] = {
            "modelUrl": model_url,
            "modelKey": key,
            "skeleton": skeleton_info,
            "anchors": anchors,
            "provenance": {
                "provider": "skintokens",
                "modelVersion": "TokenRig-articulation_xl_quantization_256_token_4",
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
