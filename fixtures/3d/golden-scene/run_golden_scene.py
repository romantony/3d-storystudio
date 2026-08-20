#!/usr/bin/env python3
"""Golden-scene regression check for runpod-3d-render-worker.

Uploads assets/*.glb to R2 **once**, then fires all 6 camera.json shots
against the same uploaded assets and asserts every requested pass comes
back non-empty for every shot. This is the concrete test for spec
Appendix B's "same canonical scene rendered from six cameras with no
asset regeneration" line, and for AC-04/AC-05/AC-06.

Reads all secrets from the environment -- never pass API keys/credentials
as arguments or hardcode them here.

Required env vars:
    RUNPOD_API_KEY            RunPod bearer token
    BLENDER_ENDPOINT_ID       e.g. 5cr41g9wtojd1t
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET_NAME            defaults to e2e-storystudio
    R2_PUBLIC_URL             e.g. https://pub-xxx.r2.dev

Usage:
    pip install boto3
    RUNPOD_API_KEY=... BLENDER_ENDPOINT_ID=... R2_ACCOUNT_ID=... \\
      R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_PUBLIC_URL=... \\
      python3 run_golden_scene.py
"""
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import boto3
from botocore.config import Config

HERE = Path(__file__).parent
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900
ASSET_PREFIX = "storystudio/fixtures/golden-scene"


def env(name, default=None, required=True):
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com",
        aws_access_key_id=env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_assets(scene):
    """Uploads every glbFile referenced in scene.json once; returns
    {glbFile: publicUrl}."""
    bucket = env("R2_BUCKET_NAME", "e2e-storystudio", required=False)
    public_base = env("R2_PUBLIC_URL").rstrip("/")
    client = r2_client()

    glb_files = sorted({n["glbFile"] for n in scene["nodes"]})
    url_by_file = {}
    print(f"Uploading {len(glb_files)} assets to R2 (once, reused across all shots)...")
    for rel in glb_files:
        local_path = HERE / rel
        key = f"{ASSET_PREFIX}/{Path(rel).name}"
        client.upload_file(str(local_path), bucket, key, ExtraArgs={"ContentType": "model/gltf-binary"})
        url = f"{public_base}/{key}"
        url_by_file[rel] = url
        print(f"  {rel} -> {url}")
    return url_by_file


def build_input(scene, camera_cfg, url_by_file):
    nodes = []
    for n in scene["nodes"]:
        node = copy.deepcopy(n)
        node["assetUrl"] = url_by_file[node.pop("glbFile")]
        nodes.append(node)

    return {
        "shotId": camera_cfg["shotId"],
        "sceneRevision": scene["revision"],
        "shotRevision": 1,
        "projectId": scene["projectId"],
        "frameId": camera_cfg["shotId"],
        "scene": {"sceneId": scene["sceneId"], "upAxis": scene["upAxis"], "units": scene["units"]},
        "nodes": nodes,
        "camera": camera_cfg["camera"],
        "passes": camera_cfg["passes"],
        "resolution": camera_cfg["resolution"],
        "rendererEngine": camera_cfg["rendererEngine"],
        "samples": camera_cfg["samples"],
    }


def api_request(path, method="GET", body=None):
    url = f"https://api.runpod.ai/v2/{env('BLENDER_ENDPOINT_ID')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {env('RUNPOD_API_KEY')}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def poll_until_done(job_id):
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        result = api_request(f"/status/{job_id}")
        status = result.get("status")
        if status in ("COMPLETED", "FAILED"):
            return result
        time.sleep(POLL_INTERVAL_S)
    sys.exit(f"job {job_id} did not finish within {POLL_TIMEOUT_S}s")


def check_output_nonempty(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = int(resp.headers.get("Content-Length", "0"))
            return length > 0
    except urllib.error.URLError:
        return False


def run_shot(scene, camera_path, url_by_file):
    camera_cfg = json.loads(camera_path.read_text())
    payload = build_input(scene, camera_cfg, url_by_file)

    print(f"\n=== {camera_cfg['shotId']} ({camera_cfg['preset']}) ===")
    submitted = api_request("/run", method="POST", body={"input": payload})
    job_id = submitted["id"]
    print(f"  job {job_id} submitted, polling...")

    result = poll_until_done(job_id)
    if result["status"] != "COMPLETED":
        print(f"  FAILED: {result.get('output')}")
        return False

    outputs = result["output"].get("outputs", {})
    expected = set(camera_cfg["passes"]) | {"masks", "manifest"}
    missing = expected - set(outputs.keys())
    if missing:
        print(f"  FAILED: missing outputs {missing}")
        return False

    all_nonempty = True
    for pass_name, url in outputs.items():
        ok = check_output_nonempty(url)
        all_nonempty &= ok
        print(f"  {pass_name:10s} {'OK ' if ok else 'EMPTY'} {url}")

    render_time = result["output"].get("renderTimeS")
    print(f"  renderTimeS={render_time}  status={'PASS' if all_nonempty else 'FAIL (empty output)'}")
    return all_nonempty


def main():
    scene = json.loads((HERE / "scene.json").read_text())
    camera_files = sorted((HERE / "cameras").glob("*.json"))
    if len(camera_files) != 6:
        print(f"warning: expected 6 camera files, found {len(camera_files)}")

    url_by_file = upload_assets(scene)

    results = {}
    for camera_path in camera_files:
        results[camera_path.name] = run_shot(scene, camera_path, url_by_file)

    print("\n=== Golden scene summary ===")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not all(results.values()):
        sys.exit(1)
    print(f"\nAll {len(results)} shots rendered successfully from the same {len(url_by_file)} "
          f"uploaded assets -- zero asset regeneration across cameras.")


if __name__ == "__main__":
    main()
