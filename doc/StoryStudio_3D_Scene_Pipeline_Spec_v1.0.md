# StoryStudio — 3D Scene Pipeline

### Technical Design & Implementation Specification

> **Primary objective.** Add persistent, editable 3D scene composition to StoryStudio so environments, characters, props, cameras and lights can be positioned once, reused across shots, and rendered into deterministic reference frames for the existing cinematic image/video generation pipeline.

**Version:** 1.0  
**Status:** Proposed implementation baseline  
**Date:** 19 August 2026  
**Product:** StoryStudio  

> **Design principle.** 3D controls composition. Generative AI controls final cinematic appearance.

## Document Control

| Item | Definition |
| --- | --- |
| Purpose | Engineering specification for introducing persistent 3D scenes and reusable 3D assets into StoryStudio. |
| Audience | StoryStudio product engineering, backend, frontend, GPU infrastructure, AI pipeline and QA teams. |
| Decision level | Architecture baseline. Model providers remain pluggable and can be replaced without changing the StoryStudio scene contract. |
| MVP boundary | Static/posed assets, editable transforms, cameras, lights, deterministic reference rendering and handoff to existing image/video generation. |
| Out of MVP | Full character animation system, advanced cloth/hair simulation, general-purpose physics authoring, real-time multiplayer scene editing. |

### Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Product Goals, Scope and Requirements](#2-product-goals-scope-and-requirements)
- [3. Architecture Decisions](#3-architecture-decisions)
- [4. Target User Experience](#4-target-user-experience)
- [5. Reference Architecture](#5-reference-architecture)
- [6. 3D Asset and Model Strategy](#6-3d-asset-and-model-strategy)
- [7. Canonical Scene Representation](#7-canonical-scene-representation)
- [8. Scene Graph Contract](#8-scene-graph-contract)
- [9. Asset Manifest Contract](#9-asset-manifest-contract)
- [10. Shot and Camera Contract](#10-shot-and-camera-contract)
- [11. Render Outputs and AI Conditioning](#11-render-outputs-and-ai-conditioning)
- [12. StoryStudio Pipeline Integration](#12-storystudio-pipeline-integration)
- [13. Orchestration and Job Workflows](#13-orchestration-and-job-workflows)
- [14. API Specification](#14-api-specification)
- [15. StoryStudio MCP Tools](#15-storystudio-mcp-tools)
- [16. Backend Data Model](#16-backend-data-model)
- [17. Storage and Versioning](#17-storage-and-versioning)
- [18. 3D Editor UX](#18-3d-editor-ux)
- [19. Character, Rigging and Animation Strategy](#19-character-rigging-and-animation-strategy)
- [20. Validation and Quality Control](#20-validation-and-quality-control)
- [21. Security, Licensing and Provenance](#21-security-licensing-and-provenance)
- [22. Observability, Cost and Token Accounting](#22-observability-cost-and-token-accounting)
- [23. Performance and Scaling](#23-performance-and-scaling)
- [24. Failure Handling and Fallbacks](#24-failure-handling-and-fallbacks)
- [25. Implementation Phases](#25-implementation-phases)
- [26. MVP Acceptance Criteria](#26-mvp-acceptance-criteria)
- [27. End-to-End Example](#27-end-to-end-example-hero-puppy-storm-drain-rescue)
- [28. Architecture Decisions Pending](#28-architecture-decisions-pending)
- [29. References](#29-references)

## 1. Executive Summary

StoryStudio should introduce a persistent 3D scene layer between script/storyboard planning and cinematic image/video generation. The 3D layer is not intended to replace WAN, Seedance, image generators or dialogue/video models. Its job is to make spatial decisions deterministic: where objects are, where characters stand, what the camera sees, which direction subjects face, and how a shot relates spatially to the previous shot.
The canonical StoryStudio scene should be stored as OpenUSD-compatible scene data, with portable GLB/glTF assets referenced into the scene and Blender used as the deterministic scene compiler and renderer. OpenUSD is selected because its composition model is designed for assembling referenced assets into hierarchical scenes with sparse overrides, while glTF/GLB is optimized for portable runtime asset delivery and contains scene nodes, meshes, materials, cameras, skins and animations. [R5][R6]
AI 3D generators are treated as interchangeable upstream asset providers. TRELLIS.2 and Hunyuan3D 2.1 are suitable candidates for individual props/objects from reference images; HunyuanWorld 1.0 and HY-World 2.x are candidates for scene-scale environments/worlds. Current official repositories describe TRELLIS.2 as a high-fidelity image-to-3D model with PBR material attributes, HunyuanWorld 1.0 as a semantically layered mesh world generator with mesh export and disentangled object representations, and HY-World 2.0 as producing persistent meshes/3D Gaussian Splatting worlds from text/images and reconstructions from multi-view images/video. [R1][R2][R3][R4]

> **Recommended MVP.** Build the scene system and Blender rendering layer first. Use manually uploaded GLB assets plus one object-generation provider. Add full world generation only after scene editing, camera reuse, render passes, caching and fallback paths are stable.

### 1.1 Desired outcome

```
SCRIPT / STORYBOARD
        │
        ▼
  3D SCENE LAYOUT  ── persistent objects + transforms + cameras
        │
        ├── RGB reference frame
        ├── depth / normal / object-mask passes
        └── camera metadata
        │
        ▼
IMAGE REFINEMENT / IMAGE GENERATION
        │
        ▼
WAN / SEEDANCE / DIALOGUE VIDEO MODEL
        │
        ▼
FINAL CINEMATIC SHOT
```

### 1.2 Why this matters for StoryStudio

- Persistent spatial continuity across many shots in the same scene.
- Ability to move a prop or character without regenerating the entire composition from scratch.
- Deterministic camera framing, lens, height, look-at target and shot-to-shot screen direction.
- Reusable environments and character/prop assets across episodes.
- Reference outputs that reduce composition drift in downstream image-to-video generation.
- An explicit scene graph that an AI Director Agent can manipulate with commands instead of only natural-language prompting.
- Graceful fallback: scenes can still use the existing 2D pipeline when 3D is unnecessary or fails.

## 2. Product Goals, Scope and Requirements

### 2.1 Goals

| ID | Goal |
| --- | --- |
| G-01 | Represent each 3D-enabled StoryStudio scene as persistent structured data rather than a single image. |
| G-02 | Allow independent movement, rotation, scaling, visibility and grouping of objects. |
| G-03 | Create multiple camera shots from the same scene without rebuilding assets. |
| G-04 | Produce deterministic reference renders for existing image/video pipelines. |
| G-05 | Support generated, uploaded and library assets through the same asset contract. |
| G-06 | Expose scene operations to StoryStudio agents through API and MCP tools. |
| G-07 | Track asset/model provenance, cost, versions and generation parameters. |
| G-08 | Preserve the existing StoryStudio generation path as a fallback and as the final cinematic stage. |

### 2.2 Non-goals for MVP

- Building a full Blender replacement in the StoryStudio UI.
- Requiring every project or every shot to use 3D.
- Photorealistic final-frame rendering directly from 3D; the 3D render may intentionally be a structural/conditioning image.
- General game-engine runtime, multiplayer world simulation or user-authored scripting.
- Automatic high-quality rigging for every AI-generated character in the first release.
- Guaranteeing semantic editability of arbitrary Gaussian-splat world outputs.

### 2.3 Functional requirements

| ID | Capability | Requirement |
| --- | --- | --- |
| FR-001 | Asset import | Import GLB/glTF and approved USD assets; validate geometry/materials and normalize scale/origin. |
| FR-002 | Asset generation | Generate a 3D asset from a reference image through a provider adapter. |
| FR-003 | Scene creation | Create a scene from environment asset(s), character/prop assets and scene metadata. |
| FR-004 | Scene graph | Maintain hierarchy, transforms, semantic roles, anchors, visibility and lock state. |
| FR-005 | Manipulation | Move/rotate/scale objects numerically, through UI gizmos or agent commands. |
| FR-006 | Camera | Create, duplicate and edit cameras with lens, sensor/framing, position, target and DOF metadata. |
| FR-007 | Shot | Create multiple shots referencing the same base scene using non-destructive shot overrides. |
| FR-008 | Lighting | Support simple sun/area/point/world lighting presets and per-shot overrides. |
| FR-009 | Render | Render RGB and structural passes from a specific immutable scene revision + shot revision. |
| FR-010 | AI handoff | Publish reference image, prompt package and metadata into current image/video generation jobs. |
| FR-011 | Versioning | Version assets and scenes so prior shots remain reproducible after edits. |
| FR-012 | Fallback | If 3D generation/rendering fails, allow scene/shot to revert to current 2D storyboard workflow. |
| FR-013 | Library | Reuse a normalized asset across projects subject to ownership and permissions. |
| FR-014 | Agent tools | Expose create/add/move/camera/render operations through StoryStudio MCP. |
| FR-015 | Preview | Provide web viewport preview using a lightweight GLB derivative of the scene or referenced assets. |

### 2.4 Non-functional requirements

| ID | Requirement | Target / rule |
| --- | --- | --- |
| NFR-01 | Reproducibility | Every render records exact scene revision, asset versions, camera, renderer version, seed and render settings. |
| NFR-02 | Idempotency | Generation/render jobs use idempotency keys and content hashes; retries must not duplicate billable output. |
| NFR-03 | Isolation | User-supplied 3D files are treated as data; arbitrary embedded scripts are never executed. |
| NFR-04 | Scalability | Heavy 3D generation and Blender rendering run outside the web/API process as queued workers. |
| NFR-05 | Compatibility | Canonical coordinates and material conventions are normalized at ingestion boundaries. |
| NFR-06 | Observability | Per-stage timing, GPU/CPU usage, model/provider, failures and cost are captured. |
| NFR-07 | Progressive loading | Viewport uses optimized derivatives/LODs rather than production-resolution meshes where required. |
| NFR-08 | Backward compatibility | Existing Narration/Movie/Documentary/Dialogue projects remain valid without 3D data. |

## 3. Architecture Decisions

| Decision | Choice | Reason |
| --- | --- | --- |
| Canonical scene format | OpenUSD (.usda/.usd) + StoryStudio scene metadata | Best fit for hierarchical scene assembly, referenced assets, non-destructive layers/overrides and future animation/variants. [R6] |
| Portable object format | GLB/glTF 2.0 | Efficient asset interchange/runtime preview; represents nodes, transforms, hierarchy, PBR materials, cameras, skins and animations. [R5] |
| Render/compile engine | Blender headless + Python | Scriptable scene import/manipulation and UI-less command-line rendering are officially supported. [R7] |
| 3D generation | Provider abstraction | TRELLIS.2, Hunyuan3D and Hunyuan world models can evolve independently of StoryStudio scene contracts. |
| World representation | Mesh preferred for movable foreground; 3DGS allowed for background/world | Mesh is directly transformable as semantic objects. 3DGS is excellent for persistent environment appearance but semantic object editing is not guaranteed. |
| Coordinate system | Canonical StoryStudio world = meters, right-handed, Z-up | Aligns naturally with Blender authoring; convert at GLB boundaries. |
| Transform storage | Translation + quaternion rotation + scale | Stable persistence and interpolation; UI may display Euler angles. |
| Shot changes | Sparse shot override layer | Avoid duplicating base scene; preserves edits and enables many camera setups. |
| Final look | Generative image/video pipeline | 3D provides geometry/composition; current cinematic models supply style, motion and realism. |

> **Critical rule.** Do not store StoryStudio scene truth only in .blend. A .blend file is a generated render cache/working artifact. Persist scene semantics in StoryStudio metadata + USD-compatible representation so the scene is portable and versionable.

## 4. Target User Experience

### 4.1 Scene-level workflow

1. User or StoryStudio Agent opens a storyboard scene and enables “3D Scene”.
2. StoryStudio resolves reusable assets from the project/library; missing assets can be generated from reference images.
3. The environment loads in a 3D viewport with scene-tree nodes for characters, props, set pieces, cameras and lights.
4. User drags objects or enters a natural-language command such as “move the puppy 30 cm closer to the drain and rotate it toward the kitten”.
5. User creates Shot 05.03, selects a camera preset, adjusts focal length/height/look-at, then previews the frame.
6. StoryStudio renders the selected shot and produces RGB + structural conditioning outputs.
7. The resulting reference frame is sent into the normal StoryStudio image/video generation pipeline.
8. User creates Shot 05.04 by duplicating the camera or adding another camera; the world and objects remain persistent.

### 4.2 UI controls required for MVP

| Area | Controls |
| --- | --- |
| Viewport | Orbit/pan/zoom, object selection, transform gizmo, grid, ground plane, camera view. |
| Scene Tree | Visibility, lock, parent/group, semantic role, rename, select. |
| Transform Panel | Position (m), rotation, scale, snap-to-ground, reset. |
| Camera Panel | Shot size preset, focal length, position, look-at target, height, roll, DOF target metadata. |
| Asset Panel | Project/library assets, generated variants, thumbnail, replace asset, asset version. |
| Shot Strip | Shot ID, camera, thumbnail, duration, render status, AI generation status. |
| Agent Command | Natural-language edit box translated to explicit scene operations with preview/undo. |

## 5. Reference Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       StoryStudio Web UI                         │
│ Storyboard | 3D Viewport | Scene Tree | Camera | Shot Strip      │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                         Convex / API layer
                               │
              ┌────────────────┼──────────────────┐
              │                │                  │
        Scene metadata     Job commands       MCP tools
              │                │                  │
              └────────────────┼──────────────────┘
                               ▼
                    Step Functions / Queue
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
       3D Asset Worker    Blender Worker    Existing AI Workers
   TRELLIS/Hunyuan/etc.   USD/GLB compile    Image/WAN/etc.
              │                │                  │
              └────────────────┼──────────────────┘
                               ▼
                   S3 private source assets
                   R2/public delivery assets
                               │
                               ▼
                      Final StoryStudio shot
```

### 5.1 Component responsibilities

| Component | Responsibility |
| --- | --- |
| StoryStudio Web | Edit scene graph and cameras; display optimized 3D preview; submit generation/render jobs. |
| Convex/API | Persist project-scoped metadata, revisions, status, permissions, job references and UI state. |
| Step Functions | Coordinate asset generation, normalization, scene compilation, rendering and AI handoff. |
| 3D Generation Adapter | Uniform provider interface for TRELLIS.2, Hunyuan3D, HunyuanWorld/HY-World or future providers. |
| Asset Normalizer | Convert to canonical units/origin/material policy; decimate/LOD where required; generate preview GLB and thumbnails. |
| USD Scene Composer | Build/revise canonical scene references and sparse shot overrides. |
| Blender Worker | Load scene, apply shot override, configure camera/light/render passes, render deterministic outputs. |
| AI Generation Pipeline | Refine 3D reference and generate final video using existing StoryStudio models. |
| S3/R2 | Store source, canonical, preview and published derivatives with immutable revisions. |

## 6. 3D Asset and Model Strategy

### 6.1 Provider roles

| Provider / model | Recommended role | Current verified capability | StoryStudio treatment |
| --- | --- | --- | --- |
| TRELLIS.2 | Foreground objects/props from reference images | Official repo: 4B image-to-3D; arbitrary topology; PBR surface attributes; at least 24 GB NVIDIA GPU; MIT model/code license. [R3] | Strong MVP candidate for props. Normalize output to GLB + canonical manifest. |
| Hunyuan3D 2.1 | Foreground objects/props from reference images | Official repo: image-to-shape + PBR texture pipeline; published VRAM figures of ~10 GB shape, ~21 GB texture, ~29 GB combined. [R4] | Alternative/secondary asset provider; benchmark quality/cost per asset class. |
| HunyuanWorld 1.0 | Scene-scale environment generation | Official repo: text/image → 360° 3D worlds; mesh export; semantically layered mesh and disentangled object representations. [R1] | Environment/world provider; still normalize semantic elements before enabling object edits. |
| HY-World 2.x | Persistent world generation/reconstruction | Official repo: text/single image world generation; multi-view/video reconstruction; outputs meshes/3DGS; July 2026 update announced HY World 2.1. [R2] | Phase 3 world provider; background 3DGS + extracted/movable foreground mesh is preferred. |
| Uploaded/library GLB | Known reusable assets | Existing authored asset with controlled topology/materials/rig. | Preferred whenever available; zero generation cost and best repeatability. |

### 6.2 Asset-class policy

| Asset class | Preferred representation | Reason |
| --- | --- | --- |
| Environment shell / terrain | USD/mesh; optional 3DGS background | Stable camera exploration and collision/grounding. |
| Movable prop | GLB mesh with PBR materials | Independent transform and reuse. |
| Hero character | Rigged mesh (GLB/FBX ingress → canonical scene reference) | Needs skeleton/pose anchors; generated unrigged mesh alone is insufficient for controlled posing. |
| Background crowd | Instanced low-LOD mesh/cards | Performance; fine-grained editability is usually unnecessary. |
| Vegetation/decor | Instanced mesh collection | Reuse and memory efficiency. |
| FX: rain/smoke/fire | Procedural/particle or downstream generative video | Do not require expensive semantic 3D asset generation for transient effects. |

### 6.3 Required provider abstraction

```ts
interface ThreeDProvider {
  generateAsset(request: AssetGenerationRequest): JobHandle
  generateWorld?(request: WorldGenerationRequest): JobHandle
  getStatus(jobId): ProviderJobStatus
  fetchResult(jobId): Generated3DResult
}

Generated3DResult {
  sourceFiles[]        // provider-native outputs
  primaryMesh?         // glb/obj/ply/etc.
  gaussianSplat?       // optional
  previewImages[]
  providerMetadata     // model, version, seed, prompt, timings
}
```

### 6.4 Model selection policy

The model/provider is selected by asset class and quality tier, not hard-coded in project logic. For example, a “prop-high” profile may choose TRELLIS.2 while an “environment-world” profile may choose HY-World. The StoryStudio API must accept a logical profile and resolve it to an active model configuration server-side.

## 7. Canonical Scene Representation

### 7.1 Format hierarchy

```
StoryStudio scene metadata (JSON / database)
        │
        ├── scene_id / revision / semantic metadata / job state
        │
        ▼
Canonical scene stage (USD / USDA)
        │
        ├── references → canonical asset GLB/USD files
        ├── base transforms
        ├── cameras / lights
        └── shot override layers

Generated derivatives (not source of truth)
        ├── scene_preview.glb        web viewport
        ├── scene_compile.blend      render cache
        ├── thumbnails/*.jpg
        └── renders/<shot>/<pass>
```

### 7.2 Coordinate and unit contract

| Property | Canonical rule |
| --- | --- |
| World units | 1 unit = 1 meter. |
| Up axis | Z-up in canonical StoryStudio scene. |
| Transform rotation | Quaternion [x, y, z, w] in persisted API. |
| Scale | [1,1,1] after normalization unless intentional. |
| Asset origin | Grounded assets use an origin at/near ground contact and horizontal footprint center. |
| Forward direction | Each asset stores an explicit semantic forward vector; do not infer from mesh orientation after ingestion. |
| Bounding data | Store axis-aligned bounds, dimensions and ground-contact height. |
| Conversion | Importer/exporter handles external format axis/unit conversion exactly once at ingestion/export. |

### 7.3 Scene hierarchy convention

```
/StoryStudioScene
  /Environment
    /Set
    /Terrain
  /Characters
    /hero_puppy
    /kitten
  /Props
    /storm_drain
    /trash_can
  /FX
  /Lights
  /Cameras
    /shot_05_01
    /shot_05_02
  /Anchors
  /Metadata
```

### 7.4 Non-destructive shot layers

A shot should not duplicate the whole scene. Base scene content remains in the scene stage, while a shot revision stores only differences: camera selection/transform, object transform overrides, temporary visibility changes, lighting tweaks and optional pose references. This maps naturally to USD composition/layering concepts, where stronger layers can override weaker scene descriptions without modifying the underlying asset. [R6]

## 8. Scene Graph Contract

### 8.1 Scene JSON

```json
{
  "sceneId": "scn_005",
  "revision": 12,
  "projectId": "prj_hero_puppy",
  "units": "meter",
  "upAxis": "Z",
  "environment": {"assetId": "ast_street_01", "version": 3},
  "nodes": [
    {
      "nodeId": "node_puppy",
      "name": "Hero Puppy",
      "type": "character",
      "assetRef": {"assetId": "ast_puppy", "version": 5},
      "transform": {
        "translation": [0.20, 0.00, 0.82],
        "rotationQuat": [0.0, 0.0, 0.2164, 0.9763],
        "scale": [1.0, 1.0, 1.0]
      },
      "semantic": {"role": "hero", "forward": [0,1,0]},
      "constraints": {"grounded": true, "locked": false}
    }
  ],
  "cameraIds": ["cam_05_01", "cam_05_02"],
  "activeShotId": "shot_05_03"
}
```

### 8.2 Node fields

| Field | Required | Definition |
| --- | --- | --- |
| nodeId | Yes | Stable unique ID within scene; never use display name as identity. |
| type | Yes | environment \| character \| prop \| fx \| light \| camera \| group \| anchor. |
| assetRef | For renderable assets | Immutable assetId + version. |
| parentNodeId | No | Hierarchy/grouping. |
| transform | Yes | Translation, quaternion rotation and scale. |
| semantic.role | Recommended | hero, supporting, vehicle, set-piece, background, etc. |
| semantic.forward | Recommended | Asset facing vector used by “look at” and blocking operations. |
| anchors | No | Named local points such as left_hand, right_hand, eyes, feet, door_handle. |
| constraints.grounded | No | Snap/bind to ground. |
| constraints.locked | No | Prevents unintended agent/UI transform. |
| visibility | Yes | Viewport/render visibility. |
| metadata | No | Provider, user tags, safety/provenance annotations. |

### 8.3 Semantic anchors

Anchors make agent operations reliable. Instead of guessing mesh coordinates, the Director Agent can target explicit points such as hero_puppy.right_paw, kitten.left_paw or storm_drain.opening_center. Anchors may be authored manually, derived from a skeleton, or detected during asset normalization.

```
"anchors": {
  "eyes":        {"position": [0.0, 0.18, 0.42]},
  "right_paw":   {"position": [0.21, 0.30, 0.08]},
  "ground":      {"position": [0.0, 0.0, 0.0]}
}
```

## 9. Asset Manifest Contract

```json
{
  "assetId": "ast_storm_drain",
  "version": 4,
  "name": "Iron Storm Drain",
  "assetClass": "prop",
  "canonicalUri": "s3://.../ast_storm_drain/v4/model.glb",
  "previewUri": "r2://.../ast_storm_drain/v4/preview.glb",
  "units": "meter",
  "dimensions": [1.05, 0.72, 0.18],
  "originPolicy": "ground_center",
  "forward": [0, 1, 0],
  "materials": {"workflow": "PBR", "textureMax": 2048},
  "lods": ["lod0", "lod1", "lod2"],
  "source": {
    "type": "generated",
    "provider": "trellis2",
    "modelVersion": "4B",
    "sourceImageId": "img_123",
    "promptHash": "...",
    "seed": 19203
  },
  "validation": {"status": "passed", "warnings": []}
}
```

### 9.1 Asset normalization pipeline

1. Download generated/uploaded source into isolated worker storage.
2. Validate file type, maximum size, mesh counts and texture references.
3. Import with controlled converter; never execute embedded scripts.
4. Convert units/orientation to StoryStudio canonical coordinate system.
5. Center/ground asset according to asset class policy.
6. Generate bounds, dimensions, thumbnail and turntable previews.
7. Normalize PBR material inputs and texture sizes.
8. Generate viewport LOD derivative(s).
9. Generate canonical GLB and optional USD representation.
10. Write immutable manifest and content hashes.

## 10. Shot and Camera Contract

### 10.1 Shot data model

```json
{
  "shotId": "shot_05_03",
  "sceneId": "scn_005",
  "sceneRevision": 12,
  "durationSeconds": 5,
  "camera": {
    "cameraId": "cam_05_03",
    "focalLengthMm": 70,
    "translation": [2.2, 1.1, 0.95],
    "rotationQuat": [0.02, -0.10, 0.71, 0.69],
    "lookAt": {"nodeId": "node_puppy", "anchor": "right_paw"},
    "dof": {"enabled": true, "focusTarget": "node_puppy:right_paw"}
  },
  "overrides": [
    {"nodeId": "node_puppy", "path": "transform.translation", "value": [0.18,0.07,0.82]}
  ],
  "motionIntent": "slow_side_push_in",
  "status": "approved"
}
```

### 10.2 Camera preset metadata

| Preset | Initial behavior |
| --- | --- |
| Extreme wide / establishing | Wide lens, environment dominates; used for geography and scale. |
| Wide | Full character and environment context. |
| Medium | Waist/chest framing for dialogue/action. |
| Close-up | Face or primary action detail. |
| Extreme close-up / macro | Hands, paws, object detail, eyes, mechanisms. |
| Low angle | Camera below subject eye level; target and lens remain explicit. |
| High angle / top-down | Camera elevated; can target a named anchor or group center. |
| Over-the-shoulder | Camera position derived from source character shoulder anchor and target character. |
| POV | Camera bound to subject eye/head anchor with optional offset. |

### 10.3 Camera motion

For MVP, camera motion is stored as intent metadata plus optional start/end transforms. It is used primarily to inform downstream video prompting. Phase 2 can render actual animated camera trajectories to depth/RGB sequences when a video model benefits from them.

## 11. Render Outputs and AI Conditioning

### 11.1 Required output package per shot

| Output | Format | Use |
| --- | --- | --- |
| RGB reference | PNG/JPEG | Primary composition/reference input for image refinement or I2V. |
| Depth | 16-bit PNG or EXR | Structural conditioning, QA, occlusion reasoning. |
| Normals | PNG or EXR | Optional geometry conditioning and QA. |
| Object ID / masks | PNG(s) + label JSON | Object-specific editing, segmentation and replacement. |
| Alpha | PNG/EXR | Compositing where background replacement is required. |
| Camera metadata | JSON | Focal length, transform, target, clipping, aspect ratio. |
| Scene metadata | JSON | Immutable scene + shot revisions and hashes used for render. |
| Preview thumbnail | JPEG/WebP | UI shot strip and approval. |

### 11.2 Renderer modes

| Mode | Purpose | Recommended engine |
| --- | --- | --- |
| Viewport | Fast interactive preview in browser | WebGL/Three.js derivative; not final. |
| Structure | Fast geometry render for layout QA | Blender Eevee-class real-time renderer or equivalent. |
| Reference | Clean reference frame for downstream generation | Blender render with normalized lights/materials. |
| Beauty (optional) | When final 3D look itself is desired | Cycles-class path tracing; not required for core MVP. |

### 11.3 AI handoff package

```json
{
  "shotId": "shot_05_03",
  "referenceImage": ".../rgb.png",
  "depthMap": ".../depth.exr",
  "objectMaskManifest": ".../masks.json",
  "prompt": "Hero puppy reaches through the storm drain...",
  "cameraPrompt": "slow side push-in, 70mm close-up",
  "stylePrompt": "Pixar-like 3D animation, cinematic rainy night",
  "negativePrompt": "duplicate limbs, warped grate, changed costume",
  "durationSeconds": 5
}
```

> **Compatibility note.** Downstream models should receive only the controls they actually support. The package can contain depth/masks for future or model-specific paths while an image-only I2V model uses the RGB reference + textual camera/motion prompt.

## 12. StoryStudio Pipeline Integration

### 12.1 Proposed pipeline position

```
Movie Script / Narration Plan
        │
        ▼
Storyboard + Voice-over segmentation
        │
        ├── 2D-only shot ───────────────► Existing pipeline
        │
        └── 3D-enabled scene
               │
               ▼
        Asset Resolver / Generator
               │
               ▼
        Persistent 3D Scene
               │
               ▼
        Shot Planner + Camera
               │
               ▼
        Blender Reference Render
               │
               ▼
        Image refinement (optional)
               │
               ▼
        WAN / Seedance / Dialogue pipeline
               │
               ▼
        Existing assembly / captions / audio
```

### 12.2 When StoryStudio should choose 3D

| Condition | Recommendation |
| --- | --- |
| Same location appears in 3+ shots | Prefer 3D for camera continuity and scene reuse. |
| Precise object interaction or blocking | Prefer 3D if anchors/rigged assets are available. |
| Complex camera angle changes around same subjects | Prefer 3D. |
| One-off abstract/illustrative shot | Use existing 2D generation; 3D may add unnecessary cost. |
| Historical/space/mechanical explainer needing spatial clarity | 3D can be valuable even if final output is stylized. |
| Fast talking-head/dialogue with fixed composition | 3D optional; likely not needed. |

### 12.3 Storyboard schema additions

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

## 13. Orchestration and Job Workflows

### 13.1 Workflow A — E2E-3D-Asset-Generate

1. Validate request and token/cost authorization.
2. Resolve provider profile.
3. Run provider generation.
4. Store provider-native result in private source storage.
5. Normalize geometry/materials/origin/units.
6. Generate canonical GLB + preview LOD + thumbnail.
7. Run asset validation/QC.
8. Persist immutable asset version and publish preview.
9. Record provider cost/timing/provenance.

### 13.2 Workflow B — E2E-3D-Scene-Build

1. Create scene revision in pending state.
2. Resolve all referenced asset versions.
3. Generate missing assets in parallel when explicitly allowed.
4. Compose/update canonical USD scene.
5. Generate viewport derivative.
6. Validate scene bounds, references, transforms and missing materials.
7. Persist scene revision and preview.
8. Mark scene ready.

### 13.3 Workflow C — E2E-3D-Shot-Render

1. Lock sceneRevision + shotRevision for render.
2. Resolve/copy required assets to render worker cache.
3. Compile USD/GLB scene into Blender working file or load directly.
4. Apply shot overrides, camera and lighting.
5. Render RGB + configured structural passes.
6. Validate render dimensions, non-empty frame and mask/depth outputs.
7. Publish render package.
8. Optionally invoke existing image-generation/refinement workflow.
9. Optionally invoke video generation.
10. Persist final job metrics and artifacts.

### 13.4 Idempotency keys

```
assetGenerationKey = hash(providerProfile + sourceImageHash + prompt + seed + settings)
sceneCompileKey    = hash(sceneRevision + referencedAssetVersionHashes)
shotRenderKey      = hash(sceneRevision + shotRevision + rendererProfile + resolution)
videoKey           = hash(referenceRenderHash + modelProfile + prompt + duration + seed)
```

## 14. API Specification

### 14.1 Core endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | /v1/3d/assets/generate | Generate an asset from reference image/text metadata. |
| POST | /v1/3d/assets/import | Register uploaded GLB/USD asset. |
| GET | /v1/3d/assets/{assetId} | Get asset/version/preview/provenance. |
| POST | /v1/3d/scenes | Create 3D scene. |
| GET | /v1/3d/scenes/{sceneId} | Get scene graph + active revision. |
| PATCH | /v1/3d/scenes/{sceneId}/nodes/{nodeId} | Apply transform/visibility/metadata edit. |
| POST | /v1/3d/scenes/{sceneId}/nodes | Add asset instance/group/anchor. |
| DELETE | /v1/3d/scenes/{sceneId}/nodes/{nodeId} | Remove scene instance (asset remains in library). |
| POST | /v1/3d/scenes/{sceneId}/shots | Create shot/camera. |
| PATCH | /v1/3d/shots/{shotId} | Update camera/shot overrides. |
| POST | /v1/3d/shots/{shotId}/render | Create deterministic reference-render job. |
| POST | /v1/3d/shots/{shotId}/generate-video | Handoff approved reference to video pipeline. |
| GET | /v1/jobs/{jobId} | Common async job status. |

### 14.2 Transform update request

```http
PATCH /v1/3d/scenes/scn_005/nodes/node_puppy
If-Match: scene-revision-12

{
  "operation": "transform",
  "translation": [0.18, 0.07, 0.82],
  "rotationQuat": [0,0,0.2164,0.9763],
  "reason": "move closer to storm drain"
}
```

The API should use optimistic concurrency. A transform request against a stale scene revision returns a conflict instead of silently overwriting another edit.

### 14.3 Natural-language scene command

```http
POST /v1/3d/scenes/scn_005/commands
{
  "text": "Move the puppy 30 cm closer to the drain and turn him toward the kitten",
  "previewOnly": true
}

Response:
{
  "operations": [
    {"op":"translateToward","node":"node_puppy","target":"node_drain","distance":0.30},
    {"op":"lookAt","node":"node_puppy","target":"node_kitten"}
  ],
  "predictedRevision": 13
}
```

> **Safety/control rule.** The LLM does not write arbitrary Blender Python. It produces a constrained scene-operation DSL; the Scene Service validates that DSL and applies only allowed transforms/relationships.

## 15. StoryStudio MCP Tools

The 3D scene capabilities should be accessible to the same orchestration agent that plans StoryStudio content. MCP calls return explicit IDs and revisions so the agent can reason over deterministic state.

| Tool | Purpose | Key arguments |
| --- | --- | --- |
| create_3d_scene | Create scene and optionally attach an environment. | projectId, storyboardSceneId, environmentAssetId |
| generate_3d_asset | Generate reusable asset. | sourceImage, assetClass, providerProfile, quality |
| add_3d_asset_to_scene | Instance asset. | sceneId, assetId/version, transform, semanticRole |
| transform_3d_object | Move/rotate/scale node. | sceneId, nodeId, transform or constrained operation |
| set_3d_object_visibility | Show/hide object per base scene or shot. | sceneId/shotId, nodeId, visible |
| create_3d_camera | Create camera/shot. | sceneId, preset, focalLength, target |
| set_3d_camera | Edit position/lens/look-at. | shotId, camera params |
| render_3d_shot | Generate reference package. | shotId, rendererProfile, resolution, passes |
| generate_video_from_3d_shot | Invoke existing video pipeline. | shotId, modelProfile, prompt, duration |
| get_3d_scene_state | Return semantic scene state for planning. | sceneId, revision |

### 15.1 Agent command examples

```
transform_3d_object(
  sceneId="scn_005",
  nodeId="node_puppy",
  operation={"type":"move_toward","targetNodeId":"node_drain","distanceMeters":0.30}
)

create_3d_camera(
  sceneId="scn_005",
  preset="close_up",
  focalLengthMm=70,
  target={"nodeId":"node_puppy","anchor":"right_paw"}
)
```

## 16. Backend Data Model

| Entity | Important fields |
| --- | --- |
| threeDAssets | assetId, ownerId, assetClass, activeVersion, permissions, createdAt. |
| threeDAssetVersions | assetId, version, canonicalUri, previewUri, source, dimensions, bounds, validation, hashes. |
| threeDScenes | sceneId, projectId, storyboardSceneId, activeRevision, status. |
| threeDSceneRevisions | sceneId, revision, usdUri, nodes, assetVersionRefs, hash, createdBy. |
| threeDShots | shotId, sceneId, activeRevision, storyboardShotId, duration. |
| threeDShotRevisions | shotId, revision, baseSceneRevision, camera, overrides, motionIntent. |
| threeDRenders | renderId, shotRevision, rendererProfile, passUris, status, timings, hash. |
| threeDJobs | jobId, type, state, provider, retries, errorCode, timing, cost. |
| threeDProviderConfigs | profile, provider, model/version, endpoint, limits, enabled. |

### 16.1 Revision rule

Scene and asset edits are immutable at the version boundary. “Updating” an asset or scene creates a new version/revision. Old StoryStudio shots continue to reference the version used when they were approved. This is necessary to reproduce published videos.

## 17. Storage and Versioning

### 17.1 Suggested object layout

```
s3://storystudio-private/
  projects/{projectId}/3d/
    assets/{assetId}/v{version}/
      source/                  # provider-native / uploads
      model.glb                # canonical mesh
      asset.usda               # optional canonical USD wrapper
      manifest.json
      textures/...
    scenes/{sceneId}/r{revision}/
      scene.usda
      scene.json
    shots/{shotId}/r{revision}/
      shot.usda                # sparse overrides where used
      shot.json
    renders/{renderId}/
      rgb.png
      depth.exr
      normal.exr
      object_id.png
      masks.json
      camera.json
      manifest.json

r2://storystudio-public/3d/...
  thumbnails/
  previews/*.glb
  approved-reference-renders/...
```

### 17.2 Storage policy

- Provider-native source stays private unless explicitly exported by the user.
- Public R2 contains only delivery-safe previews and approved artifacts that need public URLs for downstream model providers.
- Asset/version hashes are stored in metadata and render manifests.
- Render caches can be lifecycle-expired; canonical assets, scene revisions and published-shot manifests are retained according to project retention policy.
- Large 3DGS/world files should use dedicated lifecycle and preview derivatives to prevent expensive browser downloads.

## 18. 3D Editor UX

### 18.1 Recommended layout

```
┌───────────────────────────────────────────────────────────────┐
│ Toolbar: Select | Move | Rotate | Scale | Camera | Render      │
├──────────────┬────────────────────────────────┬───────────────┤
│ Scene Tree   │                                │ Inspector     │
│ Environment  │          3D VIEWPORT           │ Transform     │
│ Characters   │                                │ Camera        │
│ Props        │                                │ Semantic      │
│ Cameras      │                                │ Asset         │
├──────────────┴────────────────────────────────┴───────────────┤
│ Shot 05.01 | Shot 05.02 | Shot 05.03 | + Add Shot           │
├───────────────────────────────────────────────────────────────┤
│ Director Command: “Move puppy closer to drain...”             │
└───────────────────────────────────────────────────────────────┘
```

### 18.2 Browser rendering strategy

The browser viewport should load the preview GLB derivatives and scene transform metadata rather than full production source files. A Three.js-class WebGL viewer can display selection, bounding boxes, cameras and transform gizmos; authoritative edits are persisted through the Scene API. The browser does not need to understand every OpenUSD feature.

### 18.3 Undo/redo

Every edit command should be recorded as a scene operation. The UI can optimistically apply it locally, then persist it as a new scene revision or as a batched working revision. Undo/redo is implemented by operation history rather than by mutating raw mesh data in the browser.

## 19. Character, Rigging and Animation Strategy

### 19.1 MVP

- Use existing rigged character assets when available.
- For still reference frames, allow pre-posed meshes or static pose variants.
- For AI-generated unrigged character meshes, treat them as static/rigid until a rigging workflow is available.
- Downstream video model generates fine motion from the approved pose/reference frame.
- Character semantic anchors (eyes, head, hands/paws, feet) are required for reliable camera targets and interaction staging.

### 19.2 Phase 2

- Automatic rigging service and skeleton normalization.
- Pose library and retargeting.
- Keyframe pose interpolation for deterministic body blocking.
- Camera trajectory animation and multi-frame control renders.
- IK constraints for hand/paw-to-object interactions.
- Optional facial blendshape/viseme metadata for dialogue-oriented workflows.

### 19.3 Important limitation

> **Generated mesh ≠ controllable character.** Image-to-3D models can produce visually strong meshes, but a production character requires topology/rig/weights/anchors to pose reliably. StoryStudio should not make automatic character animation a prerequisite for the first 3D release.

## 20. Validation and Quality Control

### 20.1 Asset validation

| Check | Action on failure |
| --- | --- |
| Readable geometry | Reject asset version. |
| No external/missing texture references | Attempt package repair; otherwise reject/warn. |
| Finite bounds / no NaN transforms | Reject. |
| Reasonable scale for asset class | Auto-normalize when confidence is high; otherwise require review. |
| Ground origin | Auto-adjust for grounded classes. |
| Texture budget | Downscale preview derivative; preserve source if allowed. |
| Polygon/vertex budget | Generate LOD/decimated preview; production mesh retained separately. |
| Material compatibility | Bake/convert unsupported nodes to supported PBR maps where feasible. |

### 20.2 Scene validation

- All asset references resolve to immutable versions.
- No camera is inside invalid geometry unless explicitly allowed.
- Grounded nodes are not floating beyond tolerance.
- Hero characters and required props are inside camera frustum for the shot.
- Object intersections above configured severity are flagged, not always blocked.
- Scene bounds remain within configured maximum size.
- No missing required anchor used by a shot or agent command.
- Shot references the intended scene revision.

### 20.3 Render QC

- Output dimensions and aspect ratio match requested StoryStudio format (16:9, 9:16, 1:1, etc.).
- RGB image is non-empty and camera sees at least one expected subject.
- Depth and mask passes have valid range/labels.
- Camera metadata exactly matches render camera.
- No unresolved texture checker/error material in approved reference render.
- Content hash and manifest are written before job is marked complete.

## 21. Security, Licensing and Provenance

### 21.1 Security rules

- Treat uploaded .blend, FBX, GLB, USD and texture files as untrusted data.
- Do not execute user-provided Python, drivers, scripts or arbitrary node code.
- Prefer conversion/import into a fresh controlled Blender process/container.
- Run workers with restricted filesystem/network permissions and resource limits.
- Scan archive contents and reject path traversal, executable payloads and unsupported formats.
- Use signed short-lived URLs for private assets supplied to workers/providers.

### 21.2 Licensing policy

Each provider profile must include model/code license metadata, commercial-use status, dependency notes and the date/legal review version. TRELLIS.2 currently states that its model and code are released under the MIT License while noting that some dependencies have separate terms. [R3] Tencent Hunyuan repositories include model-specific license files; StoryStudio should perform license review before production commercial use rather than infer rights from “open-source” wording alone.

### 21.3 Provenance

```
provenance {
  sourceType: generated | uploaded | library
  provider / model / version
  sourceImageIds[]
  sourcePromptHash
  seed
  generatedAt
  normalizedByVersion
  canonicalHash
  licenseProfileId
}
```

## 22. Observability, Cost and Token Accounting

### 22.1 Metrics

| Metric group | Examples |
| --- | --- |
| Generation | provider latency, queue time, GPU seconds, output size, retry count. |
| Normalization | import time, polygon count before/after, texture memory, warnings. |
| Scene | node count, asset count, scene compile time, preview size. |
| Render | Blender startup/cache hit, render seconds, resolution, passes, GPU/CPU memory. |
| AI handoff | reference render ID, downstream model, video latency, regeneration count. |
| Quality | asset rejection rate, scene QC warnings, user replacement rate, shot approval rate. |

### 22.2 Token/cost event model

Do not hard-code customer token charges until provider benchmarks are stable. Instead emit normalized billable events. Pricing can then map events to StoryStudio tokens by plan or quality tier.

```
billableEvent {
  type: "3d_asset_generate" | "3d_world_generate" | "3d_reference_render",
  modelProfile: "prop_high_v1",
  gpuSeconds: 17.4,
  providerCostUsd: 0.00,
  storageBytes: 14322890,
  outputAssetId: "ast_...",
  projectId: "prj_..."
}
```

### 22.3 Cache economics

The primary savings mechanism is reuse. If 12 shots use the same street and five props, those assets are generated once. Camera changes and object transforms should trigger only scene/shot recompilation and rendering, not 3D regeneration. Content-addressed caching should therefore exist at asset generation, asset normalization, scene compilation and shot rendering layers.

## 23. Performance and Scaling

### 23.1 GPU planning

Current official TRELLIS.2 documentation states at least 24 GB NVIDIA GPU memory and reports H100 test timings of approximately 3 s at 512³, 17 s at 1024³ and 60 s at 1536³; these figures are vendor-reported reference measurements and must be re-benchmarked on StoryStudio infrastructure. [R3] Hunyuan3D 2.1 publishes approximately 10 GB VRAM for shape generation, 21 GB for texture generation and 29 GB combined. [R4] World-scale models should be benchmarked separately because their memory and storage profile is materially larger.

### 23.2 Worker pools

| Pool | Workload | Scaling signal |
| --- | --- | --- |
| asset-gpu | TRELLIS/Hunyuan object generation | Queue length + GPU utilization. |
| world-gpu | Hunyuan world generation/reconstruction | Explicit capacity / lower concurrency. |
| blender-render | Scene compile + reference render | Queue length + render time; CPU/GPU profile. |
| asset-cpu | Validation, conversion, thumbnails, LOD | CPU queue + memory. |

### 23.3 MVP performance targets

| Operation | Product target (not provider guarantee) |
| --- | --- |
| Scene metadata edit | Interactive; persist/acknowledge within normal web-app latency. |
| Viewport transform | Immediate local preview; server revision follows asynchronously. |
| Existing normalized asset load | Use cached preview derivative; avoid source-resolution download. |
| Reference render | Fast enough for shot iteration; optimize before beauty rendering. |
| Re-render after camera-only edit | Must not regenerate any 3D asset. |
| Re-render after object transform | Must not regenerate any 3D asset. |

## 24. Failure Handling and Fallbacks

| Failure | Required behavior |
| --- | --- |
| 3D provider timeout/OOM | Retry under provider policy; optionally switch configured fallback provider/profile. |
| Asset normalization fails | Preserve source job diagnostics; do not publish invalid canonical asset. |
| World output not semantically editable | Use as environment/background; important movable objects are separate mesh assets. |
| Missing rig/anchor | Use static pose or existing 2D reference generation; do not fabricate uncontrolled animation. |
| Blender render fails | Retry clean worker; preserve scene revision and deterministic inputs. |
| Viewport derivative too heavy | Regenerate lower LOD/texture derivative. |
| Reference image rejected by user | Change camera/objects and re-render; do not regenerate 3D assets unless asset appearance itself is wrong. |
| 3D path blocked | Set visualizationMode=2d and continue existing StoryStudio pipeline. |

### 24.1 Error code examples

```
3D_ASSET_PROVIDER_OOM
3D_ASSET_INVALID_GEOMETRY
3D_ASSET_TEXTURE_MISSING
3D_SCENE_REFERENCE_NOT_FOUND
3D_SCENE_REVISION_CONFLICT
3D_SHOT_CAMERA_INVALID
3D_RENDER_EMPTY_FRAME
3D_RENDER_WORKER_FAILED
3D_LICENSE_PROFILE_BLOCKED
```

## 25. Implementation Phases

### Phase 0 — Technical spike

- Create a Blender headless worker that imports canonical GLB, applies transform/camera JSON and renders RGB/depth/object masks.
- Benchmark TRELLIS.2 and Hunyuan3D 2.1 on representative StoryStudio assets.
- Validate USD → Blender and GLB → browser preview path.
- Build one golden test scene with 5–10 movable objects and 6 cameras.
- Decide first production 3D provider profile based on quality, latency, VRAM, license and operational stability.

### Phase 1 — MVP persistent scene layout

- 3D asset library/import and one AI asset-generation provider.
- Scene data model, versioning and object transforms.
- Web viewport with scene tree and transform gizmos.
- Camera/shot creation and reference rendering.
- RGB/depth/object-mask outputs.
- Existing image/video generation handoff.
- Step Functions workflows, retries, metrics and token events.
- MCP tools for scene creation, transforms, camera and rendering.

### Phase 2 — Characters and deterministic motion

- Rigging/pose service and normalized skeleton.
- Semantic body anchors and IK.
- Pose library and retargeting.
- Animated camera trajectories.
- Short control sequences (RGB/depth) for compatible video pipelines.

### Phase 3 — World generation and reconstruction

- HY-World/HunyuanWorld provider adapter.
- Generated/reconstructed environment worlds.
- 3DGS background support and mesh collision proxies.
- Semantic extraction/decomposition for movable objects.
- Scene expansion between connected locations.

### Phase 4 — AI virtual director

- Script-to-blocking conversion.
- Automatic camera coverage from cinematic intent.
- Continuity checks across shots.
- Agent-led scene adjustments from render QA.
- Reusable location/character libraries across episodes and channels.

## 26. MVP Acceptance Criteria

| ID | Acceptance criterion |
| --- | --- |
| AC-01 | A user can import or generate at least one GLB asset and see it in the StoryStudio 3D viewport. |
| AC-02 | A scene can contain an environment plus at least ten independently selectable/movable asset instances. |
| AC-03 | Move/rotate/scale operations persist and create reproducible scene revisions. |
| AC-04 | The same base scene can produce at least six distinct camera shots without duplicating or regenerating assets. |
| AC-05 | A reference-render job produces RGB, depth, object ID/mask and camera metadata. |
| AC-06 | Camera-only and transform-only changes do not invoke 3D asset generation. |
| AC-07 | An approved 3D reference can be handed into an existing StoryStudio image/video generation workflow. |
| AC-08 | A shot can fall back to the current 2D pipeline without corrupting project state. |
| AC-09 | All generated/imported assets have provenance, version, hash and validation state. |
| AC-10 | MCP can create a scene, add an object, transform it, create a camera and request a render using IDs/revisions. |
| AC-11 | Stale scene edits produce a revision conflict rather than silently overwriting newer data. |
| AC-12 | Render manifests capture enough information to reproduce an approved reference frame. |

## 27. End-to-End Example — Hero Puppy Storm-Drain Rescue

### 27.1 Base scene

```
Scene: scn_storm_drain_01

Environment
  ├── wet_street.glb
  ├── sidewalk.glb
  └── alley_background

Characters
  ├── hero_puppy_rigged.glb
  └── kitten_rigged.glb

Props
  ├── storm_drain.glb
  ├── drain_grate.glb
  └── trash_can.glb

FX
  ├── rain_profile
  └── water_surface

Cameras
  ├── cam_wide
  ├── cam_puppy_side
  ├── cam_paw_macro
  └── cam_inside_drain
```

### 27.2 Shot sequence from one scene

| Shot | Camera / override | 3D benefit |
| --- | --- | --- |
| 05.01 — Establishing | Wide, low street-level view | Defines exact street/drain geography. |
| 05.02 — Puppy at grate | Medium side camera; puppy moved into rescue pose | Continuity with drain location. |
| 05.03 — Reaching paws | 70 mm close-up; look-at midpoint between paw anchors | Reliable near-contact composition. |
| 05.04 — Kitten POV | Camera bound near kitten eye anchor looking up | Same scene, radically different viewpoint. |
| 05.05 — Under-grate low angle | Camera inside drain; grate foreground | No need to regenerate environment layout. |
| 05.06 — Rescue payoff | Camera returns to side; kitten transform/pose moved toward opening | Maintains screen direction and object positions. |

### 27.3 Director-agent operation

```
User: “The kitten should reach higher. Bring the puppy paw a little deeper through the grate and push the camera in.”

Resolved operations:
1. setPose(node_kitten, pose="reach_up_02")
2. setAnchorTarget(node_puppy.right_paw, node_kitten.left_paw, separation=0.08m)
3. moveCamera(cam_paw_macro, direction="forward", distance=0.22m)
4. keepLookAt(cam_paw_macro, midpoint(node_puppy.right_paw, node_kitten.left_paw))
5. render3DShot(shot_05_03, passes=[rgb,depth,object_id])
```

### 27.4 Final video prompt handoff

The 3D render becomes the structural reference. StoryStudio can still generate a cinematic motion prompt such as: “Slow side push-in; puppy’s paw strains deeper through the wet grate; kitten reaches upward; rain sprays across the puppy’s face; water rushes below; preserve exact character appearance and grate geometry.” The generative model supplies micro-motion and final visual richness while the 3D reference protects composition.

## 28. Architecture Decisions Pending

| Decision | Options | Recommendation / test |
| --- | --- | --- |
| First object provider | TRELLIS.2 vs Hunyuan3D 2.1 | Benchmark same 25 StoryStudio reference images; score geometry, texture, latency, VRAM, cleanup rate and license fit. |
| USD authoring library | OpenUSD Python directly vs generated USDA templates | Use OpenUSD Python for production; template only for early spike. |
| Browser viewer | Three.js direct vs React Three Fiber wrapper | Choose based on current StoryStudio frontend conventions; scene API remains independent. |
| Blender renderer profile | Fast raster reference vs path-traced beauty | MVP defaults to fast structural/reference render; beauty is optional. |
| World model timing | HunyuanWorld 1.0 vs HY-World 2.x | Do not block MVP; evaluate after object/scene path is proven. |
| Rigging provider | Open-source auto-rig vs authored character pipeline | Separate Phase 2 spike; use known rigged characters first. |
| Public 3D delivery | R2 GLB previews vs signed private delivery | Use public only for assets safe to expose; otherwise signed delivery. |

## 29. References

Technical assumptions in this specification were verified against primary project/specification documentation available on 19 August 2026.

| Ref | Source | Used for |
| --- | --- | --- |
| [R1] Tencent HunyuanWorld 1.0 | Official GitHub repository: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 | Semantically layered 3D mesh world generation, mesh export, disentangled object representations. |
| [R2] Tencent HY-World 2.0 / 2.1 update | Official GitHub repository: https://github.com/Tencent-Hunyuan/HY-World-2.0 | Persistent mesh/3DGS worlds, multi-modal generation/reconstruction; July 2026 repository update mentions HY World 2.1. |
| [R3] Microsoft TRELLIS.2 | Official GitHub repository: https://github.com/microsoft/TRELLIS.2 | Image-to-3D, O-Voxel, PBR surface attributes, published H100 timings/VRAM prerequisite, MIT license. |
| [R4] Tencent Hunyuan3D 2.1 | Official GitHub repository: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 | Image-to-shape and PBR texture generation with published VRAM guidance. |
| [R5] Khronos glTF 2.0 | Official specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html | Portable scene/asset representation including nodes, transformations, hierarchy, meshes, PBR materials, cameras, skins and animations. |
| [R6] OpenUSD | Official documentation: https://openusd.org/ | Scenegraph, composition, references, layers and sparse override workflows. |
| [R7] Blender | Official documentation: https://docs.blender.org/ | Headless/background command-line rendering, Python API, USD and glTF import/export support. |

## Appendix A — Recommended Initial Technology Profile

| Layer | Initial profile |
| --- | --- |
| Scene source of truth | StoryStudio metadata + OpenUSD stage/layers. |
| Object interchange | GLB/glTF 2.0. |
| 3D object generation | Benchmark TRELLIS.2 as primary candidate; Hunyuan3D 2.1 as alternative. |
| World generation | Deferred from MVP; evaluate HY-World 2.x / HunyuanWorld after core scene pipeline. |
| Render compiler | Pinned Blender LTS/headless container + controlled Python scripts. |
| Web viewport | Three.js-class GLB renderer with server-authoritative scene operations. |
| Orchestration | Existing StoryStudio Step Functions pattern for heavy jobs. |
| Metadata | Existing StoryStudio backend/Convex records for project state and revisions. |
| Private storage | S3 for source/canonical assets and render manifests. |
| Delivery storage | R2 for public/signed previews/reference artifacts where appropriate. |
| Final cinematic stage | Existing StoryStudio image/video models (WAN/Seedance/etc.) using 3D reference as composition input. |

## Appendix B — Definition of Done for the Technical Spike

- A headless Blender command accepts scene JSON + GLB assets + camera JSON and produces all required render passes.
- The same canonical scene can be rendered from six cameras with no asset regeneration.
- One AI-generated prop from each benchmarked provider can be normalized and moved correctly in the scene.
- The browser can load preview derivatives and manipulate transforms without needing production source meshes.
- A complete shot render can be handed into one existing StoryStudio I2V flow and produce a usable final clip.
- Rendering and provider jobs emit structured metrics/cost events and are safely retryable.
- All artifacts are traceable to immutable versions and hashes.