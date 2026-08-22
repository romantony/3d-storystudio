# Phase 1 handoff: what storystudio-unified needs to build and test

**For:** the `storystudio-unified` team
**From:** `3d-storystudio` (Phase 0 technical spike — complete as of 2026-08-22)
**Source:** `StoryStudio_3D_Scene_Pipeline_Spec_v1.0.docx`, §9–§26
**Scope:** Phase 1 — MVP persistent 3D scene layer
**Exit bar:** AC-01 through AC-12 (§26)

## How to use this document

Phase 0 proved the render path works and picked a provider. It did **not**
build any of the product — no persistence, no API, no UI, no agent
control. Everything below is what's still unbuilt, organized in two
halves:

- **Part 1 — Implement** (§9, §13–§18): the build list, grouped the way
  the work actually splits (data & API / orchestration / assets & USD /
  frontend & agent).
- **Part 2 — Test** (§20, §26): how to know each piece actually works,
  including three real bugs Phase 0 hit where the job reported success
  and the output was still wrong — worth reading before writing any test
  that only checks HTTP status or job state.

Part 0 lists what already exists and should be consumed, not
re-implemented.

---

## Part 0 — Already built, don't re-implement

Everything here lives in `3d-storystudio` and is live/verified as of
2026-08-22.

| Asset | What it is | Where |
|---|---|---|
| Object generation | Hunyuan3D 2.1 primary, TRELLIS.2 fallback — decided by a 25-image bake-off (geometry/texture tied 4.38/5 both; Hunyuan3D won on latency, ~2x faster at the median, and license fits current target markets) | RunPod endpoint `i703r1118k06ot` (Hunyuan3D), `rr133cjx94lkwt` (TRELLIS.2) |
| Reference render | Headless Blender 4.5 LTS, Cycles (OptiX), produces RGB/depth/normal/object_id/alpha | RunPod endpoint `5cr41g9wtojd1t` |
| USD scene composition | `compose_usda()` / `parse_usda()` — one Xform prim per scene node (translate/orient/scale ops + a custom `storystudio:assetUrl` reference attribute) under a `/World` default prim. Verified end-to-end: local round-trip against real fixture data, then a live 6-camera render through the actual RunPod endpoint | `runpod-3d-render-worker/src/scene_usd.py` |
| Golden scene fixture | 9-node / 6-camera regression scene proving zero-asset-regeneration multi-shot reuse, exercised through both flat-JSON and USD input paths | `fixtures/3d/golden-scene/` |
| Render wire contract | Exact request/response JSON shape both workers accept — reuse verbatim in the Step Functions Lambda handlers rather than re-deriving it | see `runpod-3d-render-worker/README.md` and the render worker's `src/handler.py` |

Phase 1's job is the layer **around** these two workers — persistence,
revisions, API, UI, agent control — not new GPU workers. The asset
normalization pipeline (§9.1) and full scene/shot USD authoring
(extending the pattern in `scene_usd.py`) are the only pieces of "core
3D logic" still owed; everything else below is application layer.

### Calling the endpoints

All three workers are RunPod Serverless endpoints, called the same way.
`RUNPOD_API_KEY` is the **only** credential Step Functions/Lambda needs
to call them — R2 upload credentials live inside the workers, never in
the caller.

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run
Authorization: Bearer $RUNPOD_API_KEY
Content-Type: application/json

{ "input": { ...worker-specific fields below... } }

→ { "id": "<jobId>", "status": "IN_QUEUE" }
```

```
GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{jobId}
Authorization: Bearer $RUNPOD_API_KEY

→ { "status": "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED", "output": {...} }
```

Poll `/status/{jobId}` until `COMPLETED`/`FAILED`. All three endpoints
scale to zero (`workersMin: 0`) with a 5s idle timeout, so the first
request after a quiet period pays a cold-start (image pull + model
load), typically the largest chunk of `delayTime`.

#### Object generation — Hunyuan3D 2.1 (primary)

`ENDPOINT_ID = i703r1118k06ot` · `workersMax: 2`, `workersStandby: 2` ·
GPU pool: RTX 6000 Ada / L40S / RTX A6000 · `executionTimeoutMs: 600000`

| `input` field | Required | Notes |
|---|---|---|
| `image_url` or `image_b64` | one of these | Isolated single-object shot — same input class TRELLIS.2 needs (see below) |
| `quality` | no (default `"standard"`) | `"standard"` = shape + texture, `"shape_only"` = untextured mesh only |
| `maxNumView` | no (default `6`) | Texture-pass view count; rarely needs overriding |
| `paintResolution` | no (default `512`) | Texture-pass resolution; rarely needs overriding |
| `project_id`, `frame_id` | no | If both given, asset key is deterministic (`{project_id}_{frame_id}_asset_generate.glb`); otherwise a timestamp+jobId key is used |
| `output_key` | no | Explicit override for the key above |
| `sourceImageId` | no | Passed through into the response's `provenance`, not used for logic |

```json
// response (success)
{ "modelUrl": "https://.../model.glb",
  "provenance": { "provider": "hunyuan3d", "modelVersion": "Hunyuan3D-2.1",
                   "quality": "standard", "textured": true,
                   "sourceImageId": "img_123", "genTimeS": 104.2 },
  "cold_start": true }
```
`cold_start` is present (and `true`) only on the first job after a cold
start — useful for excluding warm-up latency from p50/p95 metrics.

On failure: `{"error": "<message>", "traceback": "<python traceback>"}`.
There's no fixed error-code taxonomy on the asset workers yet — Phase
1's job wrapper should map these onto spec §24.1's codes
(`3D_ASSET_PROVIDER_OOM`, `3D_ASSET_INVALID_GEOMETRY`, ...) rather than
surface the raw Python message to the UI.

#### Object generation — TRELLIS.2 (fallback)

`ENDPOINT_ID = rr133cjx94lkwt` · `workersMax: 4`, `workersStandby: 4` ·
GPU pool: A40 / RTX A6000 · `executionTimeoutMs: 600000`

| `input` field | Required | Notes |
|---|---|---|
| `image_url` or `image_b64` | one of these | Same isolated-object requirement as Hunyuan3D |
| `quality` | no (default `"standard"`) | `"draft"` \| `"standard"` \| `"high"` — **a different tiering scheme than Hunyuan3D** (decimation target + texture size, not a shape/texture toggle). Don't assume the two providers share a quality vocabulary at the API layer |
| `seed` | no | Passed to `torch.manual_seed` |
| `project_id`, `frame_id`, `output_key`, `sourceImageId` | no | Same as Hunyuan3D |

```json
// response (success)
{ "modelUrl": "https://.../model.glb",
  "provenance": { "provider": "trellis2", "modelVersion": "TRELLIS.2-4B",
                   "quality": "standard", "seed": null,
                   "sourceImageId": "img_123", "genTimeS": 190.3 },
  "cold_start": true }
```

Known input-validation case (fixed 2026-08-22): if RMBG-2.0 finds zero
foreground pixels — e.g. a full-scene/interior image instead of an
isolated single-object shot — this returns a clear
`{"error": "TRELLIS.2 found no foreground subject in the input
image..."}` rather than a crash. Treat this as a **user-facing input
validation failure**, not a transient/retry-worthy error.

#### Reference render — Blender worker

`ENDPOINT_ID = 5cr41g9wtojd1t` · `workersMax: 2`, `workersStandby: 2` ·
GPU pool: RTX A4000 / A4500 / RTX 4000 Ada / RTX 2000 Ada ·
`executionTimeoutMs: 600000` (also the hard render timeout, configurable
worker-side via `RENDER_TIMEOUT_S`)

| `input` field | Required | Notes |
|---|---|---|
| `camera` | **yes** | `{focalLengthMm, translation:[x,y,z], lookAt:{"_worldPosition":[x,y,z]}` or `rotationQuat:[x,y,z,w], dof:{enabled, focusDistanceM}}` |
| `scene` + `nodes` | one of these two scene-composition forms is required | `scene:{sceneId, upAxis, units}` plus `nodes:[{nodeId, assetUrl, transform:{translation, rotationQuat, scale}}, ...]` |
| `sceneUsd` or `sceneUsdUrl` | (alternative to `scene`+`nodes`) | Raw `.usda` text, or a URL to fetch one — parsed via `scene_usd.py` into the identical node shape above. **Every `assetUrl` (flat or inside the USD's `storystudio:assetUrl` attributes) must be a publicly fetchable URL** — the worker downloads it directly with no auth, so assets need to be on R2 public, not S3 private, before rendering |
| `passes` | no (default all 5) | `["rgb","depth","normal","object_id","alpha"]` |
| `resolution` | no (default `{width:1920,height:1080}`) | |
| `rendererEngine` | no (default `"CYCLES"`) | |
| `samples` | no (default `64`) | |
| `shotId`, `sceneRevision`, `shotRevision`, `projectId`, `frameId` | no | Metadata passthrough into the render manifest, not used for render logic |

```json
// response (success)
{ "renderTimeS": 18.2,
  "outputs": { "rgb": "https://.../rgb.png", "depth": "https://.../depth.exr",
               "normal": "https://.../normal.exr", "object_id": "https://.../object_id.png",
               "alpha": "https://.../alpha.png", "masks": "https://.../masks.json",
               "manifest": "https://.../manifest.json" } }
```

This is the one worker that already uses spec §24.1's real error codes —
propagate them as-is rather than re-mapping:

| `error` | Meaning |
|---|---|
| `3D_SCENE_REFERENCE_NOT_FOUND` | Missing `camera`, or no usable scene composition (`scene`/`nodes` empty and no `sceneUsd`/`sceneUsdUrl`) |
| `3D_RENDER_WORKER_FAILED` | Blender subprocess exited nonzero, or exceeded the render timeout |
| `3D_RENDER_EMPTY_FRAME` | Blender exited 0 but produced no `rgb.png` — response includes `stdout`/`stderr` for diagnosis; see Part 2.4's bug 3 for why this check exists |

**Capacity note:** all three endpoints currently scale to a small worker
cap (`workersMax` 2/4/2). Phase 1 traffic beyond that queues rather than
autoscaling further — size QuarterMaster's shared GPU budget with this
in mind before assuming render/generate throughput is elastic (see
Risks, below).

---

## Part 1 — Implement

### 1.1 Data model (spec §16)

Convex tables, immutable at the version boundary — **every edit forks a
new revision; nothing mutates a published revision in place** (§16.1).
This is required so an old shot keeps referencing the exact asset/scene
version it was approved against, even after later edits — necessary to
reproduce a published video months later.

| Table | Key fields |
|---|---|
| `threeDAssets` | `assetId`, `ownerId`, `assetClass`, `activeVersion`, `permissions`, `createdAt` |
| `threeDAssetVersions` | `assetId`, `version`, `canonicalUri`, `previewUri`, `source`, `dimensions`, `bounds`, `validation`, `hashes` |
| `threeDScenes` | `sceneId`, `projectId`, `storyboardSceneId`, `activeRevision`, `status` |
| `threeDSceneRevisions` | `sceneId`, `revision`, `usdUri`, `nodes`, `assetVersionRefs`, `hash`, `createdBy` |
| `threeDShots` | `shotId`, `sceneId`, `activeRevision`, `storyboardShotId`, `duration` |
| `threeDShotRevisions` | `shotId`, `revision`, `baseSceneRevision`, `camera`, `overrides`, `motionIntent` |
| `threeDRenders` | `renderId`, `shotRevision`, `rendererProfile`, `passUris`, `status`, `timings`, `hash` |
| `threeDJobs` | `jobId`, `type`, `state`, `provider`, `retries`, `errorCode`, `timing`, `cost` |
| `threeDProviderConfigs` | `profile`, `provider`, `model`/`version`, `endpoint`, `limits`, `enabled` |

### 1.2 API — `/v1/3d/*` (spec §14)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/v1/3d/assets/generate` | Generate asset from reference image/text |
| POST | `/v1/3d/assets/import` | Register uploaded GLB/USD asset |
| GET | `/v1/3d/assets/{assetId}` | Asset/version/preview/provenance |
| POST | `/v1/3d/scenes` | Create scene |
| GET | `/v1/3d/scenes/{sceneId}` | Get scene graph + active revision |
| POST | `/v1/3d/scenes/{id}/nodes` | Add asset instance/group/anchor |
| PATCH | `/v1/3d/scenes/{id}/nodes/{nodeId}` | Transform/visibility/metadata edit |
| DELETE | `/v1/3d/scenes/{id}/nodes/{nodeId}` | Remove instance (asset stays in library) |
| POST | `/v1/3d/scenes/{id}/shots` | Create shot/camera |
| PATCH | `/v1/3d/shots/{id}` | Update camera/shot overrides |
| POST | `/v1/3d/shots/{id}/render` | Deterministic reference-render job |
| POST | `/v1/3d/shots/{id}/generate-video` | Handoff to existing video pipeline |
| POST | `/v1/3d/scenes/{id}/commands` | Natural-language scene command (§14.3) |
| GET | `/v1/jobs/{jobId}` | Common async job status |

**Optimistic concurrency is not optional.** Every node PATCH requires
`If-Match: scene-revision-N`. A request against a stale revision must
return a conflict, never a silent overwrite — this is the one API rule
with its own dedicated acceptance criterion (AC-11).

**NL command safety rule (§14.3):** the LLM never emits raw transforms
or Blender Python. It emits operations from a constrained DSL
(`translateToward`, `lookAt`, ...); the Scene Service validates and
applies them. Keep this boundary at the MCP tool layer too (§1.4) —
tools accept structured operations, not free-form scripts.

```
POST /v1/3d/scenes/scn_005/commands
{ "text": "Move the puppy 30cm closer to the drain and turn him toward the kitten",
  "previewOnly": true }

→ { "operations": [
      {"op":"translateToward","node":"node_puppy","target":"node_drain","distance":0.30},
      {"op":"lookAt","node":"node_puppy","target":"node_kitten"}
    ],
    "predictedRevision": 13 }
```

### 1.3 Orchestration & job workflows (spec §13)

Three Step Functions workflows:

**E2E-3D-Asset-Generate**
1. Validate request + token/cost authorization
2. Resolve provider profile
3. Run provider generation
4. Store provider-native result in private source storage
5. Normalize geometry/materials/origin/units (§1.5 below)
6. Generate canonical GLB + preview LOD + thumbnail
7. Run asset validation/QC (§2.2)
8. Persist immutable asset version, publish preview
9. Record provider cost/timing/provenance

**E2E-3D-Scene-Build**
1. Create scene revision in pending state
2. Resolve all referenced asset versions
3. Generate missing assets in parallel, if explicitly allowed
4. Compose/update canonical USD scene
5. Generate viewport GLB derivative
6. Validate scene bounds/references/transforms/materials (§2.2)
7. Persist scene revision + preview
8. Mark scene ready

**E2E-3D-Shot-Render**
1. Lock `sceneRevision` + `shotRevision` for render
2. Resolve/copy required assets to render worker cache
3. Compile USD/GLB scene into Blender working file or load directly
4. Apply shot overrides, camera, lighting
5. Render RGB + configured structural passes
6. Validate render dimensions, non-empty frame, mask/depth outputs (§2.3)
7. Publish render package
8. Optionally invoke image-generation/refinement workflow
9. Optionally invoke video generation
10. Persist final job metrics + artifacts

**Idempotency keys (§13.4) — this is what makes AC-06 true:**

| Key | Hashed from |
|---|---|
| `assetGenerationKey` | `providerProfile + sourceImageHash + prompt + seed + settings` |
| `sceneCompileKey` | `sceneRevision + referencedAssetVersionHashes` |
| `shotRenderKey` | `sceneRevision + shotRevision + rendererProfile + resolution` |
| `videoKey` | `referenceRenderHash + modelProfile + prompt + duration + seed` |

A camera-only or transform-only edit must change `shotRenderKey`
**without** touching `assetGenerationKey`. If a transform edit ever
recomputes the asset key, AC-06 breaks silently even though every
individual job still reports success — this is exactly the kind of bug
that won't show up in a status check (see Part 2.3).

### 1.4 Frontend viewport & agent surface (spec §15, §18)

**React Three Fiber viewport:**
- Scene tree (environment / characters / props / cameras), inspector, transform gizmo
- Loads preview GLB derivatives + transform metadata — never full production source files (§18.2); the browser doesn't need to understand every OpenUSD feature
- Shot strip + camera panel (preset → focal length / height / look-at)
- Undo/redo implemented as recorded scene operations, not raw mesh mutation (§18.3) — UI applies optimistically, then persists as a new/batched revision

**MCP tool group (§15):**

| Tool | Key arguments |
|---|---|
| `create_3d_scene` | `projectId`, `storyboardSceneId`, `environmentAssetId` |
| `generate_3d_asset` | `sourceImage`, `assetClass`, `providerProfile`, `quality` |
| `add_3d_asset_to_scene` | `sceneId`, `assetId`/version, `transform`, `semanticRole` |
| `transform_3d_object` | `sceneId`, `nodeId`, `transform` or constrained operation |
| `set_3d_object_visibility` | `sceneId`/`shotId`, `nodeId`, `visible` |
| `create_3d_camera` / `set_3d_camera` | `sceneId`, `preset`, `focalLength`, `target` |
| `render_3d_shot` | `shotId`, `rendererProfile`, `resolution`, `passes` |
| `generate_video_from_3d_shot` | `shotId`, `modelProfile`, `prompt`, `duration` |
| `get_3d_scene_state` | `sceneId`, `revision` |

**Storyboard schema addition (§12.3):**
```
storyboardShot {
  ...existingFields,
  "visualizationMode": "2d" | "3d" | "hybrid",
  "scene3dId": "scn_005",
  "shot3dId": "shot_05_03",
  "referenceRenderId": "rnd_879",
  "useReferenceForImageGen": true
}
```
Setting `visualizationMode=2d` must route straight to the existing
pipeline **without touching any 3D state** (AC-08) — this is the fallback
switch, and it needs to be a true no-op on the 3D tables, not just a UI
branch.

### 1.5 Asset normalization & USD authoring (spec §9.1, §17.1)

The one piece of core 3D logic still owed beyond the two GPU workers
already running.

**Normalization pipeline (§9.1):**
1. Download generated/uploaded source into isolated worker storage
2. Validate file type, max size, mesh counts, texture references
3. Import with a controlled converter — never execute embedded scripts (§21.1)
4. Convert units/orientation to StoryStudio's canonical coordinate system
5. Center/ground per asset-class policy
6. Generate bounds, dimensions, thumbnail, turntable preview
7. Normalize PBR material inputs and texture sizes
8. Generate viewport LOD derivative(s)
9. Generate canonical GLB and optional USD representation
10. Write immutable manifest and content hashes

**USD scene/shot authoring:** extend `scene_usd.py`'s pattern (already
proven end-to-end in Phase 0 — see Part 0) from single-scene Xform +
reference composition to full assembly per spec §17.1's layout: a base
`scene.usda` plus a **sparse** `shot.usda` override layer per shot,
rather than flattening every shot into a full scene copy. This is what
spec §10 means by shots storing only camera selection, transform
overrides, and visibility/lighting tweaks, not a whole-scene duplicate.

**Storage layout (§17.1):**
```
s3://storystudio-private/
  projects/{projectId}/3d/
    assets/{assetId}/v{version}/
      source/              # provider-native / uploads
      model.glb            # canonical mesh
      asset.usda           # optional canonical USD wrapper
      manifest.json
      textures/...
    scenes/{sceneId}/r{revision}/
      scene.usda
      scene.json
    shots/{shotId}/r{revision}/
      shot.usda             # sparse overrides where used
      shot.json
    renders/{renderId}/
      rgb.png  depth.exr  normal.exr  object_id.png
      masks.json  camera.json  manifest.json

r2://storystudio-public/3d/...
  thumbnails/  previews/*.glb  approved-reference-renders/...
```
Provider-native source stays private unless the user explicitly exports
it; public R2 holds only delivery-safe previews and approved artifacts.

---

## Part 2 — Test

### 2.1 Acceptance criteria (spec §26) — the Phase 1 exit bar

| ID | Criterion | What actually proves it |
|---|---|---|
| AC-01 | Import or generate a GLB asset, see it in the viewport | Round-trip through `/assets/import` + `/assets/generate`; confirm the viewport GLB's hash matches the manifest |
| AC-02 | Scene holds an environment + 10 independently movable instances | Extend the golden-scene fixture past 9 nodes; each node selectable/transformable in isolation |
| AC-03 | Move/rotate/scale persist as reproducible scene revisions | Transform → reload the scene at that revision → transform values come back byte-identical |
| AC-04 | Same base scene → 6 distinct camera shots, zero asset regeneration | The golden-scene check already passing in `3d-storystudio`, re-run through the real `/v1/3d/*` API instead of calling the render worker directly |
| AC-05 | Render job produces RGB, depth, object ID/mask, camera metadata | Assert non-empty, format-valid bytes per §2.3 — not just HTTP 200 (see §2.4) |
| AC-06 | Camera-only / transform-only edits never invoke 3D asset generation | Assert `assetGenerationKey` is unchanged across a transform-only edit + re-render |
| AC-07 | Approved 3D reference hands into existing image/video generation | `/generate-video` call produces a job the existing WAN/Seedance path accepts unmodified |
| AC-08 | Shot falls back to 2D pipeline without corrupting project state | Set `visualizationMode=2d` mid-flow; assert zero writes to any 3D table |
| AC-09 | All generated/imported assets have provenance, version, hash, validation state | Schema-validate the §21.3 provenance block on every persisted asset version |
| AC-10 | MCP can create a scene, add an object, transform it, create a camera, request a render — using IDs/revisions | Scripted MCP session with no direct DB access; assert every returned ID resolves through the API |
| AC-11 | Stale scene edits produce a revision conflict, not a silent overwrite | Two concurrent PATCHes against the same `If-Match` revision — the second must `409`, never merge silently |
| AC-12 | Render manifests capture enough to reproduce an approved reference frame | Re-render from the manifest alone (no live scene state) → pixel-equivalent output |

### 2.2 Asset validation (spec §20.1)

| Check | Action on failure |
|---|---|
| Readable geometry | Reject asset version |
| No external/missing texture references | Attempt package repair; otherwise reject/warn |
| Finite bounds / no NaN transforms | Reject |
| Reasonable scale for asset class | Auto-normalize if confidence is high; otherwise require review |
| Ground origin | Auto-adjust for grounded classes |
| Texture budget | Downscale preview derivative; preserve source if allowed |
| Polygon/vertex budget | Generate LOD/decimated preview; retain production mesh separately |
| Material compatibility | Bake/convert unsupported nodes to supported PBR maps where feasible |

### 2.3 Scene validation & render QC (spec §20.2, §20.3)

**Scene validation:**
- All asset references resolve to immutable versions
- No camera inside invalid geometry unless explicitly allowed
- Grounded nodes not floating beyond tolerance
- Hero characters / required props inside camera frustum for the shot
- Object intersections above configured severity are flagged, not always blocked
- Scene bounds within configured maximum size
- No missing anchor required by a shot or agent command
- Shot references the intended scene revision

**Render QC:**
- Output dimensions/aspect ratio match the requested format (16:9, 9:16, 1:1, ...)
- RGB image is non-empty **and** the camera sees at least one expected subject — not just non-zero file size
- Depth and mask passes have valid range/labels
- Camera metadata exactly matches the render camera used
- No unresolved texture-error material in an approved reference render
- Content hash and manifest are written **before** the job is marked complete

### 2.4 Verification lessons Phase 1 inherits

Three real Phase 0 bugs that all "looked done" — job status COMPLETED,
output URL populated — until someone actually opened the output file.
Phase 1's own test suite needs to check for exactly this failure shape,
not just repeat the Phase 0 fixes.

1. **Mislabeled export.** Hunyuan3D's paint pipeline built a real `.glb`
   internally, but its return value pointed at the source `.obj` file.
   The worker uploaded Wavefront OBJ text with a `model/gltf-binary`
   content-type. Every job reported COMPLETED with a populated
   `modelUrl`, for weeks, before this was caught — only found because the
   render worker tried to import the "glTF" and got zero objects.

2. **Reports-success-ships-garbage.** A fine-fur character asset
   (`char_kitten`) came back from Hunyuan3D as COMPLETED with a
   populated URL — but the mesh itself was a thin torn sheet, not a
   volumetric character. Same failure shape as bug 1, different root
   cause: some inputs make a model degrade silently instead of erroring.

3. **Silent empty frame.** Blender can exit code 0 having rendered
   nothing at all — e.g. a glTF import that silently yields zero
   objects from a malformed asset. Exit code alone said success.

**Concrete rule for Phase 1's test suite:** every asset-generation and
render test must assert on the **artifact itself** — real byte-level
format checks (glTF magic bytes, not the extension or content-type the
worker claims), non-trivial file size, and for renders, an actual
pixel/geometry sanity check — never on job status or HTTP code alone.
"Job succeeded" is necessary, never sufficient, for both
E2E-3D-Asset-Generate and E2E-3D-Shot-Render (§1.3).

---

## Risks carried into Phase 1

| Risk | Level | Mitigation |
|---|---|---|
| Shared RunPod GPU cap contended by 3D jobs during video-pipeline spikes | watch | Size QuarterMaster's shared cap explicitly for the addition; alert on `asset-gpu`/`blender-render` queue depth separately from video queues |
| Unrigged generated mesh treated as a posable character before Phase 2 | watch | Enforce at validation: unrigged characters are static-pose only until rigging ships (§19.3, §24) |
| Scene truth drifts into a working `.blend` file | guard rail | Spec's own critical rule — `.blend` is render cache only; never round-trip edits back out of it |
| License review skipped under release pressure | guard rail | `threeDProviderConfigs` requires a populated `licenseProfileId` before a profile can go live |
| Hunyuan3D's Tencent Community License excludes EU/UK/South Korea | resolved | Not a blocker today — target markets are Americas/India/Middle East/Africa. Revisit if that ever changes; TRELLIS.2 (MIT) is the configured fallback |

---

*Companion artifact (same content, browsable):
[Phase 1 Build & Test Ledger](https://claude.ai/code/artifact/70104ccb-35eb-4dff-811b-e9e5d238bd0c)*
