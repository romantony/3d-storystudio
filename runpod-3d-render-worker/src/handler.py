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
                     frame_id: Optional[str], job_id: Optional[str]) -> str:
    if project_id and frame_id:
        name = f"{project_id}_{frame_id}_{asset_type}"
    else:
        name = f"{time.strftime('%Y%m%d%H%M%S')}_{job_id or uuid.uuid4().hex[:8]}_{asset_type}"
    return f"storystudio/renders_3d/{name}.{ext}"


def download_to(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "python-blender-render/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inp: Dict[str, Any] = job.get("input", {})

        camera = inp.get("camera")
        if not camera:
            return {"error": "3D_SCENE_REFERENCE_NOT_FOUND", "detail": "input.camera is required"}

        # Scene composition can arrive as flat scene+nodes JSON (original
        # contract) or as a USD stage (sceneUsd text, or sceneUsdUrl to
        # fetch one) -- USD is parsed into the exact same scene+nodes
        # shape via scene_usd.parse_usda() so every downstream step (asset
        # download, scene.json, the Blender subprocess) is unchanged
        # either way. Spec Appendix B Phase 0 DoD: "Validate USD -> Blender
        # import ... round-trip."
        scene_usd_text = inp.get("sceneUsd")
        scene_usd_url = inp.get("sceneUsdUrl")
        if scene_usd_url and not scene_usd_text:
            req = urllib.request.Request(scene_usd_url, headers={"User-Agent": "python-blender-render/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                scene_usd_text = resp.read().decode("utf-8")

        nodes: List[Dict[str, Any]]
        if scene_usd_text:
            scene, nodes = scene_usd.parse_usda(scene_usd_text)
        else:
            scene = inp.get("scene")
            nodes = inp.get("nodes", [])
            if not scene:
                return {"error": "3D_SCENE_REFERENCE_NOT_FOUND",
                        "detail": "input.scene (or sceneUsd/sceneUsdUrl) is required"}

        if not nodes:
            return {"error": "3D_SCENE_REFERENCE_NOT_FOUND", "detail": "no renderable nodes in scene"}

        passes = inp.get("passes", ["rgb", "depth", "normal", "object_id", "alpha"])
        resolution = inp.get("resolution", {"width": 1920, "height": 1080})
        engine = inp.get("rendererEngine", DEFAULT_ENGINE)
        samples = int(inp.get("samples", DEFAULT_SAMPLES))

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
        scene_path.write_text(json.dumps({**scene, "nodes": nodes}))
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

        manifest = {
            "shotId": inp.get("shotId"),
            "sceneRevision": inp.get("sceneRevision"),
            "shotRevision": inp.get("shotRevision"),
            "rendererEngine": engine,
            "samples": samples,
            "blenderVersion": blender_version,
            "renderSeconds": render_time_s,
            "resolution": resolution,
            "passes": passes,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        project_id, frame_id = inp.get("projectId"), inp.get("frameId")
        outputs: Dict[str, str] = {}
        for f in sorted(out_dir.iterdir()):
            key = build_asset_key(f.stem, f.suffix.lstrip("."), project_id, frame_id, job.get("id"))
            outputs[f.stem] = upload_file(f, key)

        return {"renderTimeS": render_time_s, "outputs": outputs}

    except subprocess.TimeoutExpired:
        return {"error": "3D_RENDER_WORKER_FAILED", "detail": f"render exceeded {RENDER_TIMEOUT_S}s"}
    except Exception as exc:
        LOGGER.error("%s", exc, exc_info=True)
        return {"error": str(exc), "traceback": __import__("traceback").format_exc()}


runpod.serverless.start({"handler": handler})
