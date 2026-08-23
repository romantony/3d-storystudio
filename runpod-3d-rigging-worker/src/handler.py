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
    return {
        "jointCount": len(joint_indices),
        "jointNames": joint_names,
        "_gltf": gltf,
        "_skin": skins[0],
    }


# Name-based matching kept as a cheap fallback/supplement, but confirmed
# NOT useful against the actual deployed checkpoint
# (articulation_xl_quantization_256_token_4): a real live job (giraffe.glb,
# 67 joints) came back with every joint named "bone_0".."bone_66" -- pure
# autoregressive index placeholders, not semantic labels. This model
# predicts skeleton *topology* (positions + parent/child structure) via
# Skeleton Tree Tokenization, not a joint classifier, so there is no
# semantic name anywhere in this pipeline to match against. Geometric
# inference (below) is the real mechanism; this only helps if some future
# checkpoint/asset happens to carry real names.
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


# ---- Geometric anchor inference -----------------------------------------
# Since joint names carry no semantics (see above), anchors are inferred
# from the skeleton's own geometry: each joint's world-space position
# (accumulated by walking the glTF node hierarchy's local TRS transforms
# from the scene root down to each joint) plus its topology (parent/child
# structure within the skin). Deliberately stdlib-only -- plain Python
# lists for 4x4 matrices, no numpy -- matching this repo's other workers'
# "no new dependency for a stdlib-sized problem" convention.
#
# Honesty limit: this can identify `root` (topologically) and a
# reasonable `head` guess (highest leaf joint), but NOT left/right --
# telling a character's own left paw from its right purely from joint
# positions needs a facing/forward-direction signal (spec §8.2's
# `semantic.forward`) that lives on the *scene node* wrapping this asset,
# not inside the isolated rigged mesh this worker produces. Instead of
# shipping a coin-flip left/right label, `extremities` returns every
# non-head leaf joint's name + position + distance from root, ranked, so
# a caller with the forward vector (or a human) can assign left/right
# correctly downstream.

def _quat_to_rotation_matrix(q: List[float]) -> List[List[float]]:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]


def _local_matrix(node: Dict[str, Any]) -> List[List[float]]:
    """4x4 row-major local transform from a glTF node's explicit `matrix`
    (glTF stores this column-major) or its translation/rotation/scale."""
    if "matrix" in node:
        raw = node["matrix"]
        return [[raw[c * 4 + r] for c in range(4)] for r in range(4)]
    t = node.get("translation", [0.0, 0.0, 0.0])
    q = node.get("rotation", [0.0, 0.0, 0.0, 1.0])  # glTF order [x,y,z,w]
    s = node.get("scale", [1.0, 1.0, 1.0])
    rot = _quat_to_rotation_matrix(q)
    m = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            m[i][j] = rot[i][j] * s[j]
        m[i][3] = t[i]
    m[3] = [0.0, 0.0, 0.0, 1.0]
    return m


def _mat_mul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _joint_world_positions(nodes: List[Dict[str, Any]], joint_indices: List[int]) -> Dict[int, List[float]]:
    parent_of_node: Dict[int, int] = {}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            parent_of_node[c] = i

    world_cache: Dict[int, List[List[float]]] = {}

    def world_matrix(idx: int) -> List[List[float]]:
        if idx not in world_cache:
            local = _local_matrix(nodes[idx])
            parent = parent_of_node.get(idx)
            world_cache[idx] = _mat_mul(world_matrix(parent), local) if parent is not None else local
        return world_cache[idx]

    return {j: [world_matrix(j)[0][3], world_matrix(j)[1][3], world_matrix(j)[2][3]] for j in joint_indices}


def infer_geometric_anchors(gltf: Dict[str, Any], skin: Dict[str, Any]) -> Dict[str, Any]:
    nodes = gltf.get("nodes", [])
    joint_indices = skin.get("joints", [])
    if not joint_indices:
        return {}

    parent_of_node: Dict[int, int] = {}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            parent_of_node[c] = i

    joint_set = set(joint_indices)
    skin_parent: Dict[int, Optional[int]] = {}
    child_count: Dict[int, int] = {j: 0 for j in joint_indices}
    for j in joint_indices:
        p = parent_of_node.get(j)
        while p is not None and p not in joint_set:
            p = parent_of_node.get(p)
        skin_parent[j] = p
        if p is not None:
            child_count[p] = child_count.get(p, 0) + 1

    skeleton_hint = skin.get("skeleton")
    roots = [j for j in joint_indices if skin_parent.get(j) is None]
    if skeleton_hint in joint_set:
        root = skeleton_hint
    elif roots:
        root = roots[0]
    else:
        root = joint_indices[0]

    positions = _joint_world_positions(nodes, joint_indices)
    root_pos = positions[root]

    def dist_from_root(j: int) -> float:
        p = positions[j]
        return sum((p[k] - root_pos[k]) ** 2 for k in range(3)) ** 0.5

    leaves = [j for j in joint_indices if child_count.get(j, 0) == 0 and j != root]
    # glTF's canonical up-axis is Y -- this is the exporter's own
    # coordinate space, independent of StoryStudio's Z-up scene
    # convention (that conversion, if needed, happens downstream).
    head = max(leaves, key=lambda j: positions[j][1]) if leaves else None

    def joint_name(j: int) -> str:
        return nodes[j].get("name", f"joint_{j}")

    anchors: Dict[str, Any] = {"root": joint_name(root)}
    if head is not None:
        anchors["head"] = joint_name(head)

    extremities = [
        {
            "jointName": joint_name(j),
            "position": [round(v, 5) for v in positions[j]],
            "distanceFromRoot": round(dist_from_root(j), 5),
        }
        for j in leaves if j != head
    ]
    extremities.sort(key=lambda e: -e["distanceFromRoot"])
    if extremities:
        anchors["extremities"] = extremities

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
        gltf, skin = skeleton_info.pop("_gltf"), skeleton_info.pop("_skin")

        # Name-based matching first (cheap; wins if a future checkpoint
        # ever emits real semantic names), geometric fills in whatever
        # name-based didn't find (root/head always; extremities always).
        anchors = map_semantic_anchors(skeleton_info["jointNames"])
        for anchor_key, value in infer_geometric_anchors(gltf, skin).items():
            anchors.setdefault(anchor_key, value)

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
