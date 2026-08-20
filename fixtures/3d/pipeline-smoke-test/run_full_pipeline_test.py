#!/usr/bin/env python3
"""End-to-end pipeline smoke test: Qwen image-gen -> TRELLIS.2 image-to-3D
-> Blender/Cycles render, three different generated images, three camera
shots each.

Chains three separately-owned RunPod endpoints exactly the way the
"Chaining them in StoryStudio" section of the Trellis & Blender Pipeline
doc describes, but starting one step earlier -- from a text prompt
instead of a pre-existing reference image.

Each TEST_CASES prompt is deliberately phrased as an isolated product
shot (single object, plain background) because TRELLIS.2's own
background-removal step (briaai/RMBG-2.0) and image-conditioning model
both assume one clean foreground subject -- a busy/cluttered generated
scene is far more likely to produce a broken mesh.

Camera framing assumes TRELLIS.2's fixed output convention: every GLB is
normalized to fit inside an aabb of [-0.5,-0.5,-0.5]..[0.5,0.5,0.5]
(handler.py's o_voxel.postprocess.to_glb call), i.e. roughly a 1m cube
centered on the origin -- not measured per-asset, so an unusually shaped
generation may not frame perfectly. That's expected at this stage: there
is no asset normalization/re-centering step yet (spec §9.1), and this
script doesn't try to compensate for it.

Reads secrets from `.env` at the repo root via python-dotenv, same as
run_golden_scene.py. Required env vars: RUNPOD_API_KEY, QWEN_ENDPOINT_ID,
TRELLIS2_ENDPOINT_ID, BLENDER_ENDPOINT_ID.

Usage:
    pip install python-dotenv   # boto3 not needed -- no R2 upload here
    python3 run_full_pipeline_test.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv(HERE.parent.parent.parent / ".env")
    load_dotenv(HERE / ".env")
except ImportError:
    pass

POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900
TRELLIS_QUALITY = "draft"  # keep the smoke test cheap; bump to "standard"/"high" for real evaluation
UA = {"User-Agent": "python-pipeline-smoke-test/1.0"}

TEST_CASES = [
    {
        "name": "treasure_chest",
        "prompt": "A weathered leather and brass treasure chest, product photography, "
                   "isolated on a plain white background, soft studio lighting, single object, no shadow clutter",
        "aspect_ratio": "1:1",
    },
    {
        "name": "toy_robot",
        "prompt": "A small cartoon-style toy robot, front three-quarter view, product photography, "
                   "isolated on a plain white background, soft studio lighting, single object",
        "aspect_ratio": "1:1",
    },
    {
        "name": "ceramic_mug",
        "prompt": "A ceramic coffee mug with a blue floral pattern, product photography, "
                   "isolated on a plain white background, soft studio lighting, single object",
        "aspect_ratio": "1:1",
    },
]

# Three shots per generated asset. Object is assumed to sit roughly within
# a 1m cube centered on the origin (see module docstring).
CAMERA_SHOTS = [
    {
        "name": "three_quarter",
        "camera": {"focalLengthMm": 50, "translation": [1.8, -1.8, 1.0], "lookAt": {"_worldPosition": [0, 0, 0]}},
    },
    {
        "name": "front_closeup",
        "camera": {"focalLengthMm": 85, "translation": [0, -1.2, 0.1],
                   "lookAt": {"_worldPosition": [0, 0, 0]},
                   "dof": {"enabled": True, "focusDistanceM": 1.2}},
    },
    {
        "name": "top_down",
        "camera": {"focalLengthMm": 35, "translation": [0, 0.05, 2.0], "lookAt": {"_worldPosition": [0, 0, 0]}},
    },
]


def env(name, required=True):
    v = os.environ.get(name)
    if required and not v:
        sys.exit(f"Missing required env var: {name}")
    return v


def api_request(endpoint_id, path, method="GET", body=None):
    url = f"https://api.runpod.ai/v2/{endpoint_id}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {**UA, "Content-Type": "application/json", "Authorization": f"Bearer {env('RUNPOD_API_KEY')}"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def poll_until_done(endpoint_id, job_id, label):
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        result = api_request(endpoint_id, f"/status/{job_id}")
        status = result.get("status")
        if status in ("COMPLETED", "FAILED"):
            if status == "FAILED":
                sys.exit(f"{label} job {job_id} FAILED: {result.get('output')}")
            return result["output"]
        time.sleep(POLL_INTERVAL_S)
    sys.exit(f"{label} job {job_id} did not finish within {POLL_TIMEOUT_S}s")


def extract_image_url(output):
    """Qwen's exact output field name isn't confirmed against real handler
    source (that worker isn't in this repo) -- try the common shapes and
    fail loudly with the raw payload rather than guessing silently."""
    for key in ("imageUrl", "image_url", "url", "output_url"):
        if isinstance(output.get(key), str):
            return output[key]
    images = output.get("images")
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
    if isinstance(images, list) and images and isinstance(images[0], dict):
        for key in ("url", "imageUrl", "image_url"):
            if key in images[0]:
                return images[0][key]
    sys.exit(f"Couldn't find an image URL in Qwen's output -- raw payload:\n{json.dumps(output, indent=2)}")


def run_test_case(case):
    print(f"\n{'=' * 60}\n{case['name']}\n{'=' * 60}")
    project_id, frame_id = "pipeline_smoke_test", case["name"]

    print(f"[1/3] Qwen image-gen: \"{case['prompt'][:70]}...\"")
    submitted = api_request(env("QWEN_ENDPOINT_ID"), "/run", method="POST", body={
        "input": {
            "model": "qwen",
            "prompt": case["prompt"],
            "aspect_ratio": case["aspect_ratio"],
            "project_id": project_id,
            "frame_id": frame_id,
        }
    })
    qwen_output = poll_until_done(env("QWEN_ENDPOINT_ID"), submitted["id"], "qwen")
    image_url = extract_image_url(qwen_output)
    print(f"      image -> {image_url}")

    print(f"[2/3] TRELLIS.2 image-to-3D (quality={TRELLIS_QUALITY})")
    submitted = api_request(env("TRELLIS2_ENDPOINT_ID"), "/run", method="POST", body={
        "input": {
            "image_url": image_url,
            "quality": TRELLIS_QUALITY,
            "project_id": project_id,
            "frame_id": frame_id,
        }
    })
    trellis_output = poll_until_done(env("TRELLIS2_ENDPOINT_ID"), submitted["id"], "trellis2")
    model_url = trellis_output["modelUrl"]
    gen_time = trellis_output.get("provenance", {}).get("genTimeS")
    print(f"      model -> {model_url}  (genTimeS={gen_time})")

    print(f"[3/3] Blender render x{len(CAMERA_SHOTS)} camera shots")
    renders = {}
    for shot in CAMERA_SHOTS:
        shot_id = f"{frame_id}_{shot['name']}"
        submitted = api_request(env("BLENDER_ENDPOINT_ID"), "/run", method="POST", body={
            "input": {
                "shotId": shot_id,
                "projectId": project_id,
                "frameId": shot_id,
                "scene": {"sceneId": f"scn_{case['name']}", "upAxis": "Z", "units": "meter"},
                "nodes": [{
                    "nodeId": "node_generated",
                    "assetUrl": model_url,
                    "transform": {"translation": [0, 0, 0], "rotationQuat": [0, 0, 0, 1], "scale": [1, 1, 1]},
                }],
                "camera": shot["camera"],
                "passes": ["rgb"],
                "resolution": {"width": 960, "height": 960},
                "rendererEngine": "CYCLES",
                "samples": 32,
            }
        })
        render_output = poll_until_done(env("BLENDER_ENDPOINT_ID"), submitted["id"], f"blender/{shot['name']}")
        rgb_url = render_output["outputs"]["rgb"]
        renders[shot["name"]] = rgb_url
        print(f"      {shot['name']:15s} -> {rgb_url}  (renderTimeS={render_output.get('renderTimeS')})")

    return {"prompt": case["prompt"], "image": image_url, "model": model_url, "renders": renders}


def main():
    results = {}
    for case in TEST_CASES:
        results[case["name"]] = run_test_case(case)

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    for name, r in results.items():
        print(f"\n{name}")
        print(f"  image : {r['image']}")
        print(f"  model : {r['model']}")
        for shot_name, url in r["renders"].items():
            print(f"  {shot_name:15s}: {url}")

    Path(HERE / "last_run_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {HERE / 'last_run_results.json'}")


if __name__ == "__main__":
    main()
