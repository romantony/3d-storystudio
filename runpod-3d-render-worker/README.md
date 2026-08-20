# blender-render — headless scene compile + reference render

One RunPod Serverless image: Blender 4.5 LTS, headless, GPU-accelerated
Cycles. Implements Appendix B's Definition of Done directly — accepts scene
JSON + GLB assets + camera JSON, produces RGB/depth/normal/object_id/alpha
passes plus a manifest.

## Render engine decision: Cycles (OptiX/CUDA), not Eevee, by default

The spec's own suggestion (§11.2) is an "Eevee-class real-time renderer"
for speed. In practice, **headless Eevee GPU rendering inside Docker is a
known-fragile setup** — it needs a working EGL context in a container with
no display, and there are open, GPU-specific bugs (e.g. reported SSBO
binding failures on A100). Cycles' OptiX/CUDA compute path needs no
display/EGL context at all — it's the well-trodden pattern for headless
server rendering — and at low sample counts with OIDN denoising it's
competitive with Eevee for a *structural* reference render, which is all
the MVP needs (§11.2's "Beauty"/Cycles-path-tracing tier is explicitly
deferred, but nothing stops using Cycles at reference quality *now*).

`BLENDER_EEVEE_NEXT` is still wired up behind `rendererEngine` /
`RENDER_ENGINE` for a team that verifies it on their specific GPU/driver
combo — just don't make it the default without that verification.

## GPU

RTX 6000 Ada / L40 / L40S / H100 — reusing the exact class already live for
the `Wan2-14b-fp8-RTX6000ADA` endpoint (see `~/runpod/API.md`). Blender
rendering isn't VRAM-bound the way asset generation is, so this doesn't
need a dedicated large network volume — the Blender binary (~350MB) ships
baked into the image.

## Request / response contract

```json
POST /run
{"input": {
  "shotId": "shot_05_03",
  "sceneRevision": 12,
  "shotRevision": 3,
  "projectId": "prj_hero_puppy",
  "frameId": "shot_05_03",
  "scene": {"sceneId": "scn_005", "upAxis": "Z", "units": "meter"},
  "nodes": [
    {"nodeId": "node_puppy", "assetUrl": "https://.../ast_puppy/v5/model.glb",
     "transform": {"translation": [0.20, 0.00, 0.82], "rotationQuat": [0,0,0.2164,0.9763], "scale": [1,1,1]}}
  ],
  "camera": {"focalLengthMm": 70, "translation": [2.2, 1.1, 0.95], "rotationQuat": [0.02,-0.10,0.71,0.69],
             "dof": {"enabled": true, "focusDistanceM": 1.8}},
  "passes": ["rgb", "depth", "normal", "object_id", "alpha"],
  "resolution": {"width": 1920, "height": 1080},
  "rendererEngine": "CYCLES",
  "samples": 64
}}
```

`camera.lookAt` is resolved to a `_worldPosition` **before** this worker
sees it — the orchestration layer (which holds the full scene graph) turns
`{"nodeId": "node_puppy", "anchor": "right_paw"}` into a world-space
vector, since this worker only ever sees the subset of nodes actually
being rendered, not the full scene.

```json
{"renderTimeS": 4.2,
 "outputs": {
   "rgb": "https://pub-xxx.r2.dev/storystudio/renders_3d/prj_hero_puppy_shot_05_03_rgb.png",
   "depth": "https://.../..._depth.exr",
   "normal": "https://.../..._normal.exr",
   "object_id": "https://.../..._object_id.png",
   "alpha": "https://.../..._alpha.png",
   "masks": "https://.../..._masks.json",
   "manifest": "https://.../..._manifest.json"
 }}
```

Error codes match spec §24.1 (`3D_RENDER_EMPTY_FRAME`,
`3D_RENDER_WORKER_FAILED`, `3D_SCENE_REFERENCE_NOT_FOUND`).

**Storage note:** this MVP handler uploads every pass to the same public R2
bucket as everything else, for simplicity. Spec §17 calls for the
structural passes (depth/normal/masks/camera/manifest) to live in private
S3 and only RGB + thumbnail to be public — implement that split before this
goes past Phase 0, it's not done here yet.

## Build

```sh
docker build -t storystudio/blender-render:latest .
```

No compiled CUDA extensions — this builds fast relative to the asset-gpu
images (mostly a Blender tarball download).

## Golden test scene

Phase 0 exit criteria calls for a fixture scene (5–10 objects, 6 cameras)
rendered from every camera with zero asset regeneration. Built at
[`fixtures/3d/golden-scene/`](../fixtures/3d/golden-scene/) — 9 nodes
(ground + 8 procedural props), 6 camera presets, and a driver script
(`run_golden_scene.py`) that uploads the assets once and asserts every
pass comes back non-empty across all 6 shots. Run it as the regression
check for this worker before wiring up the Convex/Step Functions layer
around it.
