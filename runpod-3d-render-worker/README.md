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
  "lights": [
    {"lightId": "key_sun", "type": "sun", "energy": 5.0, "color": [1,0.95,0.9],
     "translation": [0,0,3], "rotationQuat": [0,0,0,1]}
  ],
  "camera": {"focalLengthMm": 70, "translation": [2.2, 1.1, 0.95], "rotationQuat": [0.02,-0.10,0.71,0.69],
             "dof": {"enabled": true, "focusDistanceM": 1.8}},
  "passes": ["rgb", "depth", "normal", "object_id", "alpha"],
  "resolution": {"width": 1920, "height": 1080},
  "rendererEngine": "CYCLES",
  "samples": 64,
  "seed": 0,
  "outputPrefix": "projects/prj_hero_puppy/3d/renders/rnd_879"
}}
```

| Field | Required | Notes |
|---|---|---|
| `lights` | no | Real authored lights (spec §7.3's `/Lights` branch): `{lightId, type: "sun"\|"point"\|"area", energy, color?, translation, rotationQuat}`. Omit entirely (or send `[]`) to fall back to the worker's flat default sun + world fill — the original Phase 0 behavior, unchanged. |
| `seed` | no (default `0`) | Cycles' RNG seed — fixed by default (not per-job random) so re-rendering the same manifest is pixel-reproducible (AC-12), not just structurally similar. |
| `outputPrefix` | no | When set, every output file is written to `{outputPrefix}/{passName}.{ext}` (e.g. spec §17.1's `renders/{renderId}/rgb.png`) instead of this worker's own flat internal namespace. Callers that need a predictable, caller-owned key should always set this rather than deriving one from the returned URL. |

`camera.lookAt.nodeId` is resolved to a `_worldPosition` **before** this
worker sees it — the orchestration layer (which holds the full scene
graph) turns `{"nodeId": "node_puppy", "anchor": "right_paw"}` into a
world-space vector, since this worker only ever sees the subset of nodes
actually being rendered, not the full scene. A malformed `camera` (wrong
types/shapes) is rejected with `3D_SHOT_CAMERA_INVALID` before any Blender
process is spawned.

### USD scene composition — camera, lights, semantics, sparse shot layers

`sceneUsd`/`sceneUsdUrl` (a single flat stage) and `sceneUsd` + `shotUsd`/
`shotUsdUrl` (a base scene sublayered by a sparse override shot layer, spec
§7.4/§17.1) are both accepted as alternatives to flat `scene`+`nodes`+
`camera`+`lights` JSON — see `src/scene_usd.py`'s module docstring for the
exact composition/attribute contract. A stage's authored `Camera`/`UsdLux`
prims are used when `input.camera`/`input.lights` aren't also given
directly; explicit JSON fields always win over what's in the USD. Per-node
`semantic.role`/`semantic.forward` round-trip as custom attributes.

```python
# building a shot.usda that only overrides camera + two node transforms,
# sublayering a base scene.usda (see scene_usd.compose_shot_usda docstring
# for why base_scene must be passed even though only upAxis/units get used)
shot_usda = scene_usd.compose_shot_usda(
    {"shotId": "shot_05_03",
     "camera": {"cameraId": "cam_05_03", "focalLengthMm": 70, "translation": [2.2, 1.1, 0.95]},
     "nodeOverrides": [{"nodeId": "node_puppy", "transform": {...}}]},
    base_scene={"upAxis": "Z", "units": "meter"},
)
```

```json
{"renderTimeS": 4.2,
 "outputs": {
   "rgb": "https://pub-xxx.r2.dev/projects/prj_hero_puppy/3d/renders/rnd_879/rgb.png",
   "depth": "https://.../depth.exr",
   "normal": "https://.../normal.exr",
   "object_id": "https://.../object_id.png",
   "alpha": "https://.../alpha.png",
   "masks": "https://.../masks.json",
   "manifest": "https://.../manifest.json"
 },
 "outputKeys": {
   "rgb": "projects/prj_hero_puppy/3d/renders/rnd_879/rgb.png",
   "...": "... (same key set as outputs, R2 object keys instead of full URLs)"
 }}
```

`outputKeys` is additive alongside `outputs` — existing callers using only
`outputs` (URLs) are unaffected; callers that need the raw R2 key (to
persist in a manifest/database row without re-deriving it from a URL) can
read `outputKeys` directly instead of parsing one out.

### masks.json label contract

```json
{"background": 0,
 "nodes": [
   {"nodeId": "node_puppy", "name": "Hero Puppy", "type": "character", "passIndex": 1},
   {"nodeId": "node_kitten", "name": "Kitten", "type": "character", "passIndex": 2}
 ]}
```

`passIndex` is exactly the pixel value written into the `object_id.png`
pass for that node's geometry (Blender's `pass_index`, stamped per-object
in `render_scene.py`'s `import_nodes()`); `0` is always background, never
assigned to a node. A node present in the request but missing from
`masks.json.nodes` failed to import (see `3D_RENDER_NO_SUBJECT` below).

### manifest.json — reproduction fields (AC-12)

`manifest.json` is self-contained: it embeds the *exact* `scene`, `nodes`
(transforms + assetUrls, without the job-local `_localPath`), `camera`,
`lights`, `rendererEngine`, `samples`, and `seed` used for the render,
alongside `contentHash` (sha256 of that same input set, canonical JSON) —
re-submitting those fields verbatim to `/run` must reproduce a
pixel-equivalent frame with no other live scene state required. `shotId`/
`sceneRevision`/`shotRevision`/`projectId`/`frameId`/`blenderVersion`/
`renderSeconds`/`resolution`/`passes` are carried for traceability, not
reproduction (changing them doesn't change `contentHash`).

### Error codes (spec §24.1)

| Code | Meaning |
|---|---|
| `3D_SCENE_REFERENCE_NOT_FOUND` | Missing `camera` (and no Camera prim in `sceneUsd`/`shotUsd`), or no usable scene composition |
| `3D_SHOT_CAMERA_INVALID` | `camera` present but malformed (wrong field types/shapes) — rejected before spawning Blender |
| `3D_RENDER_WORKER_FAILED` | Blender subprocess exited nonzero, or exceeded the render timeout |
| `3D_RENDER_EMPTY_FRAME` | Blender exited 0 but produced no `rgb.png` at all — response includes `stdout`/`stderr` |
| `3D_RENDER_NO_SUBJECT` | Blender exited 0 and wrote a non-empty `rgb.png`, but nothing is actually visible in it (spec §20.3: "camera sees at least one expected subject") — the "reports-success-ships-garbage" failure shape, distinct from `3D_RENDER_EMPTY_FRAME` |

### Webhook passthrough

This worker needs **no code changes** to support async completion
callbacks — RunPod's job-submission API accepts a top-level `webhook` field
sitting alongside `input`, not something a handler implements itself:

```json
POST /run
{"input": {...}, "webhook": "https://storystudio-unified.example/hooks/3d-render"}
```

RunPod POSTs the same `{status, output}` shape `/status/{jobId}` would
return once the job reaches a terminal state. Prefer polling for anything
latency-sensitive during interactive editing; use the webhook for
fire-and-forget batch/background renders to cut poll traffic.

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
rendered from every camera with zero asset regeneration; extended for AC-02
("environment + 10 independently movable instances"). Built at
[`fixtures/3d/golden-scene/`](../fixtures/3d/golden-scene/) — 11 nodes
(ground + 10 procedural props), 6 camera presets, and a driver script
(`run_golden_scene.py`) that uploads the assets once and asserts every
pass comes back non-empty across all 6 shots. Run it as the regression
check for this worker before wiring up the Convex/Step Functions layer
around it.
