import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import runpod
from botocore.config import Config

import scene_usd

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"blender-render","message":"%(message)s"}',
)
LOGGER = logging.getLogger(__name__)

BLENDER_BIN = os.environ.get("BLENDER_BIN", "blender")
RENDER_SCRIPT = os.environ.get("RENDER_SCRIPT", "/app/blender/render_scene.py")
RENDER_TIMEOUT_S = int(os.environ.get("RENDER_TIMEOUT_S", "600"))
DEFAULT_ENGINE = os.environ.get("RENDER_ENGINE", "CYCLES")
DEFAULT_SAMPLES = os.environ.get("RENDER_SAMPLES", "64")

R2_ACCOUNT_ID        = os.environ["R2_ACCOUNT_ID"]
R2_ACCESS_KEY_ID     = os.environ["R2_ACCESS_KEY_ID"]
R2_SECRET_ACCESS_KEY = os.environ["R2_SECRET_ACCESS_KEY"]
R2_BUCKET_NAME       = os.environ.get("R2_BUCKET_NAME", "e2e-storystudio")
R2_PUBLIC_URL        = os.environ.get("R2_PUBLIC_URL", "")

CONTENT_TYPES = {
    ".png": "image/png",
    ".exr": "image/x-exr",
    ".json": "application/json",
    ".jpg": "image/jpeg",
}

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


def upload_file(path: Path, key: str) -> str:
    content_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    r2().upload_file(str(path), R2_BUCKET_NAME, key, ExtraArgs={"ContentType": content_type})
    return f"{R2_PUBLIC_URL.rstrip('/')}/{key}"


def build_asset_key(asset_type: str, ext: str, project_id: Optional[str],
                     frame_id: Optional[str], job_id: Optional[str],
                     output_prefix: Optional[str] = None) -> str:
    # outputPrefix (spec §17.1's renders/{renderId}/ layout) lets the
    # caller pin every output of a render to a deterministic, caller-owned
    # key -- without it we fall back to this worker's own flat internal
    # namespace, which is what previously forced callers onto a
    # keyFromWorkerUrl() fallback instead of a real, predictable key.
    if output_prefix:
        return f"{output_prefix.rstrip('/')}/{asset_type}.{ext}"
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{time.strftime('%Y%m%d%H%M%S')}_{job_id or uuid.uuid4().hex[:8]}_{asset_type}"
    return f"storystudio/renders_3d/{name}.{ext}"


def download_to(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "python-blender-render/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def validate_camera(camera: Dict[str, Any]) -> None:
    """Spec §24.1's 3D_SHOT_CAMERA_INVALID -- fail fast on a malformed
    camera before paying for a Blender cold start + scene import, rather
    than letting render_scene.py crash partway through setup_camera()."""
    if not isinstance(camera, dict):
        raise ValueError("camera must be an object")
    if "focalLengthMm" in camera and not isinstance(camera["focalLengthMm"], (int, float)):
        raise ValueError("camera.focalLengthMm must be a number")
    translation = camera.get("translation")
    if translation is not None and (not isinstance(translation, list) or len(translation) != 3):
        raise ValueError("camera.translation must be a 3-element [x,y,z] array")
    rotation_quat = camera.get("rotationQuat")
    if rotation_quat is not None and (not isinstance(rotation_quat, list) or len(rotation_quat) != 4):
        raise ValueError("camera.rotationQuat must be a 4-element [x,y,z,w] array")
    look_at = camera.get("lookAt")
    if look_at is not None:
        if not isinstance(look_at, dict):
            raise ValueError("camera.lookAt must be an object")
        if "nodeId" not in look_at and "_worldPosition" not in look_at:
            raise ValueError("camera.lookAt requires nodeId or _worldPosition")
    dof = camera.get("dof")
    if dof is not None:
        if not isinstance(dof, dict):
            raise ValueError("camera.dof must be an object")
        if dof.get("enabled") and not isinstance(dof.get("focusDistanceM", 3.0), (int, float)):
            raise ValueError("camera.dof.focusDistanceM must be a number")


def content_hash(payload: Dict[str, Any]) -> str:
    """sha256 of a canonical (sorted-key) JSON encoding -- spec AC-12 /
    §20.3's 'content hash and manifest are written before job is marked
    complete', and the reproducibility half of the manifest schema (see
    README): identical scene+camera+render-settings inputs must hash
    identically regardless of key ordering."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "python-blender-render/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp: Dict[str, Any] = job.get("input", {})

        # Scene composition can arrive as flat scene+nodes JSON (original
        # contract), a single USD stage (sceneUsd/sceneUsdUrl), or a base
        # scene.usda + sparse shot.usda override layer
        # (sceneUsd+shotUsd/shotUsdUrl, spec §7.4/§17.1) -- all three are
        # normalized into the same scene+nodes(+camera+lights) shape via
        # scene_usd.py so every downstream step (asset download,
        # scene.json, the Blender subprocess) is unchanged regardless of
        # input encoding. Spec Appendix B Phase 0 DoD: "Validate USD ->
        # Blender import ... round-trip."
        scene_usd_text = inp.get("sceneUsd") or (inp.get("sceneUsdUrl") and _fetch_text(inp["sceneUsdUrl"]))
        shot_usd_text = inp.get("shotUsd") or (inp.get("shotUsdUrl") and _fetch_text(inp["shotUsdUrl"]))

        usd_camera = usd_lights = None
        nodes: List[Dict[str, Any]]
        if scene_usd_text and shot_usd_text:
            composed = scene_usd.parse_composed_stage(scene_usd_text, shot_usd_text)
            scene, nodes, usd_camera, usd_lights = (
                composed["scene"], composed["nodes"], composed["camera"], composed["lights"],
            )
        elif scene_usd_text:
            composed = scene_usd.parse_usda(scene_usd_text)
            scene, nodes, usd_camera, usd_lights = (
                composed["scene"], composed["nodes"], composed["camera"], composed["lights"],
            )
        else:
            scene = inp.get("scene")
            nodes = inp.get("nodes", [])
            if not scene:
                return {"error": "3D_SCENE_REFERENCE_NOT_FOUND",
                        "detail": "input.scene (or sceneUsd/sceneUsdUrl) is required"}

        if not nodes:
            return {"error": "3D_SCENE_REFERENCE_NOT_FOUND", "detail": "no renderable nodes in scene"}

        camera = inp.get("camera") or usd_camera
        if not camera:
            return {"error": "3D_SCENE_REFERENCE_NOT_FOUND",
                     "detail": "input.camera is required (or a Camera prim in sceneUsd/shotUsd)"}
        try:
            validate_camera(camera)
        except ValueError as exc:
            return {"error": "3D_SHOT_CAMERA_INVALID", "detail": str(exc)}

        lights = inp.get("lights") if inp.get("lights") is not None else (usd_lights or [])
        seed = int(inp.get("seed", 0))
        passes = inp.get("passes", ["rgb", "depth", "normal", "object_id", "alpha"])
        resolution = inp.get("resolution", {"width": 1920, "height": 1080})
        engine = inp.get("rendererEngine", DEFAULT_ENGINE)
        samples = int(inp.get("samples", DEFAULT_SAMPLES))
        output_prefix = inp.get("outputPrefix")

        job_dir = Path(tempfile.mkdtemp(prefix="render_"))
        assets_dir = job_dir / "assets"
        out_dir = job_dir / "out"
        assets_dir.mkdir()
        out_dir.mkdir()

        t0 = time.time()
        for i, node in enumerate(nodes):
            asset_url = node.get("assetUrl")
            if not asset_url:
                continue
            ext = Path(asset_url.split("?")[0]).suffix or ".glb"
            local_path = assets_dir / f"{node.get('nodeId', f'node_{i}')}{ext}"
            download_to(asset_url, local_path)
            node["_localPath"] = str(local_path)

        scene_path = job_dir / "scene.json"
        camera_path = job_dir / "camera.json"
        scene_path.write_text(json.dumps({**scene, "nodes": nodes, "lights": lights}))
        camera_path.write_text(json.dumps(camera))

        cmd = [
            BLENDER_BIN, "--background", "--factory-startup", "--python", RENDER_SCRIPT, "--",
            "--scene", str(scene_path),
            "--camera", str(camera_path),
            "--out-dir", str(out_dir),
            "--passes", ",".join(passes),
            "--width", str(resolution["width"]),
            "--height", str(resolution["height"]),
            "--engine", engine,
            "--samples", str(samples),
            "--seed", str(seed),
        ]
        LOGGER.info("Rendering shot=%s engine=%s samples=%d passes=%s",
                    inp.get("shotId"), engine, samples, passes)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=RENDER_TIMEOUT_S)
        if proc.returncode != 0:
            LOGGER.error("Blender exited %d: %s", proc.returncode, proc.stderr[-4000:])
            return {"error": "3D_RENDER_WORKER_FAILED", "detail": proc.stderr[-2000:]}

        render_time_s = round(time.time() - t0, 1)

        rgb_path = out_dir / "rgb.png"
        if not rgb_path.exists() or rgb_path.stat().st_size == 0:
            # Blender can exit 0 and still produce nothing -- e.g. a glTF
            # import that silently yields zero objects from a malformed
            # asset. Without stdout/stderr here this is undiagnosable from
            # the API response alone (confirmed against a real case: the
            # hunyuan3d OBJ-mislabeled-as-glb bug took far longer to track
            # down than necessary because this path returned no detail).
            LOGGER.error("Renderer produced no rgb.png. stdout: %s | stderr: %s",
                         proc.stdout[-2000:], proc.stderr[-2000:])
            return {
                "error": "3D_RENDER_EMPTY_FRAME",
                "detail": "renderer produced no rgb.png",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }

        blender_version = ""
        version_file = out_dir / "version.txt"
        if version_file.exists():
            blender_version = version_file.read_text().strip()
            version_file.unlink()

        # render_stats.json is render_scene.py's internal QC signal (see
        # check_has_subject() there) -- consumed here, never uploaded as an
        # "output" pass.
        has_subject: Optional[bool] = None
        stats_file = out_dir / "render_stats.json"
        if stats_file.exists():
            has_subject = json.loads(stats_file.read_text()).get("hasSubject")
            stats_file.unlink()
        if has_subject is False:
            # Spec §20.3: "RGB image is non-empty and camera sees at least
            # one expected subject" -- distinct from 3D_RENDER_EMPTY_FRAME
            # (no rgb.png at all). This is the "reports-success-ships-
            # garbage" failure shape from doc/phase1-handoff.md Part 2.4:
            # exit 0, a populated rgb.png, but nothing actually visible
            # (e.g. every node failed to import, or the camera points
            # entirely away from the scene content).
            LOGGER.error("Render produced rgb.png but no visible subject (nodes=%d)", len(nodes))
            return {
                "error": "3D_RENDER_NO_SUBJECT",
                "detail": "camera does not see any expected subject -- rendered frame is fully transparent/background",
            }

        # Reproducibility fields (AC-12: re-render from the manifest alone,
        # with no live scene state, must be pixel-equivalent). nodes here
        # excludes _localPath (a job-local temp filesystem detail, not
        # part of the reproducible input).
        repro_nodes = [{k: v for k, v in n.items() if k != "_localPath"} for n in nodes]
        repro_inputs = {
            "scene": scene,
            "nodes": repro_nodes,
            "camera": camera,
            "lights": lights,
            "rendererEngine": engine,
            "samples": samples,
            "seed": seed,
            "resolution": resolution,
            "passes": passes,
        }
        manifest = {
            "shotId": inp.get("shotId"),
            "sceneRevision": inp.get("sceneRevision"),
            "shotRevision": inp.get("shotRevision"),
            "projectId": inp.get("projectId"),
            "frameId": inp.get("frameId"),
            "rendererEngine": engine,
            "samples": samples,
            "seed": seed,
            "blenderVersion": blender_version,
            "renderSeconds": render_time_s,
            "resolution": resolution,
            "passes": passes,
            "camera": camera,
            "scene": scene,
            "nodes": repro_nodes,
            "lights": lights,
            "contentHash": content_hash(repro_inputs),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        project_id, frame_id = inp.get("projectId"), inp.get("frameId")
        outputs: Dict[str, str] = {}
        output_keys: Dict[str, str] = {}
        for f in sorted(out_dir.iterdir()):
            key = build_asset_key(f.stem, f.suffix.lstrip("."), project_id, frame_id, job.get("id"), output_prefix)
            outputs[f.stem] = upload_file(f, key)
            output_keys[f.stem] = key

        return {"renderTimeS": render_time_s, "outputs": outputs, "outputKeys": output_keys}

    except subprocess.TimeoutExpired:
        return {"error": "3D_RENDER_WORKER_FAILED", "detail": f"render exceeded {RENDER_TIMEOUT_S}s"}
    except Exception as exc:
        LOGGER.error("%s", exc, exc_info=True)
        return {"error": str(exc), "traceback": __import__("traceback").format_exc()}


runpod.serverless.start({"handler": handler})
