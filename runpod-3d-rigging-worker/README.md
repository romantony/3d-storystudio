# rigging-gpu — automatic character rigging (SkinTokens / TokenRig)

RunPod Serverless image wrapping [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens)
(MIT license) — spec §28's "Rigging provider" architecture decision, resolved
as a Phase 2 spike. Takes an unrigged GLB (e.g. straight out of the
TRELLIS.2/Hunyuan3D asset workers) and returns a rigged GLB with a real
skeleton + per-vertex skinning weights.

## Why SkinTokens, not UniRig

Both are from the same team (Tsinghua/Tripo); SkinTokens is UniRig's
explicitly-positioned successor. Chosen over UniRig after comparing both
against real source (not just marketing copy):

| | UniRig | SkinTokens |
|---|---|---|
| Pipeline | 3 stages (skeleton → skin → merge) | 1 unified call, GLB in/out |
| Recommended checkpoint availability | Best checkpoint (Rig-XL/VRoid) still "planned future" | Recommended checkpoint downloadable today |
| Claimed accuracy | baseline | 98–133% better skinning, 17–22% better bone prediction (same team's own benchmarks) |
| License | MIT | MIT |
| VRAM | 8GB min | 14GB min |

## Why a separate endpoint

Same reasoning as `runpod-3d-asset-worker/README.md`'s "Why two images
instead of one": torch 2.7.0/cu128 + its own flash-attn build + `bpy>=4.2`
is a different, incompatible pinned toolchain from TRELLIS.2 (torch
2.6.0/cu124), Hunyuan3D (torch 2.5.1/cu124), and the Blender render worker
(Blender 5.2 LTS as a binary, not the `bpy` pip package).

## bpy version (dedicated Python 3.13 env)

SkinTokens does all glTF import/export/skin-transfer through a background
`bpy_server.py` subprocess over localhost HTTP. The main env is Python 3.11
(pinned by torch 2.7.0 / flash-attn), where `bpy>=4.2` can only resolve to
5.0.1 -- bpy 5.1+ ships cp313 wheels only. So the Dockerfile builds a second
conda env (`bpy313`: Python 3.13, `bpy==5.2.3`, numpy/scipy/trimesh, CPU
torch for the `torch.save` wire format) and `handler.py` launches
`src/bpy_server_py313.py` with it when `BPY_SERVER_PYTHON` is set (it is, in
the image); unset falls back to upstream's `start_bpy_server()`.
`provenance.exporter` reports the exporter string from the output GLB, so a
live job shows which bpy build actually ran.

Upstream quirk, unrelated to the bpy version: `bpy_server` segfaults on input
meshes under 990 vertices (`vertex_groups[...].remove(range(990))`).

## GPU / volume

- **No network volume needed.** Model weights (~1.6GB: the TokenRig
  checkpoint + FSQ-CVAE tokenizer + Qwen3-0.6B config) are baked directly
  into the Docker image at build time — small enough that this sidesteps
  the whole pre-warm-a-network-volume pattern the other two asset workers
  need for their ~15GB checkpoints.
- **GPU**: 14GB VRAM minimum per upstream. Not yet benchmarked against
  StoryStudio's actual asset complexity. Recommend provisioning on the
  L40S class already live for TRELLIS.2 (48GB, comfortable headroom) over
  the render worker's smaller RTX A4000/A4500 class (20GB, thin margin
  over the stated minimum) until real VRAM usage is observed in a live job.

## Request / response contract

```json
POST /run
{"input": {
  "assetUrl": "https://.../unrigged_model.glb",
  "project_id": "prj_hero_puppy",
  "frame_id": "hero_puppy_rig"
}}
```

| Field | Required | Notes |
|---|---|---|
| `assetUrl` | yes | Publicly fetchable URL of the unrigged source GLB |
| `topK`, `topP`, `temperature`, `repetitionPenalty`, `numBeams` | no | TokenRig sampling params, upstream defaults (5 / 0.95 / 1.0 / 2.0 / 10) |
| `usePostprocess` | no (default `false`) | Voxel-based skin-weight postprocessing (upstream `--use_postprocess`) |
| `project_id`, `frame_id`, `output_key` | no | Same asset-key convention as the other two asset workers |

```json
{"modelUrl": "https://.../rig_generate.glb",
 "modelKey": "storystudio/models_3d_rigged/....glb",
 "skeleton": {"jointCount": 22, "jointNames": ["..."]},
 "anchors": {"head": "...", "left_hand": "...", "root": "..."},
 "provenance": {"provider": "skintokens",
                "modelVersion": "TokenRig-articulation_xl_quantization_256_token_4",
                "exporter": "Khronos glTF Blender I/O v5.2.x",
                "genTimeS": 14.2}}
```

**`anchors` is a heuristic, not a confirmed taxonomy.** The recommended
checkpoint is trained on the broad ArticulationXL2.0 dataset (not
constrained to a fixed Mixamo/VRoid bone-name scheme), so the exact
joint-naming convention it predicts hasn't been verified against real
inference output yet — `map_semantic_anchors()` in `src/handler.py` does
loose substring matching against common naming families and degrades
gracefully (fewer anchors filled in) rather than crashing on an
unexpected scheme. **Verification status: not yet run against a live
endpoint.** First real jobs should be (1) the upstream example
`examples/giraffe.glb` bundled in the SkinTokens repo — a good
non-humanoid sanity check — and (2) an actual TRELLIS.2/Hunyuan3D-
generated character, to see real joint names and correct the anchor
patterns if they don't match. `skeleton.jointNames` is always returned in
full specifically so this can be inspected/refined without re-running
inference.

### Error codes (spec §24.1, same extension convention as the other asset workers)

| Code | Meaning |
|---|---|
| `3D_ASSET_INVALID_INPUT` | Missing `assetUrl` |
| `3D_ASSET_INVALID_GEOMETRY` | Output GLB has no mesh, or — rigging-specific — has a mesh but zero skins/joints (rigging silently produced an unrigged result) |
| `3D_ASSET_PROVIDER_OOM` | CUDA out-of-memory |
| `3D_ASSET_GENERATION_FAILED` | Anything else |

### Webhook passthrough

Same as the other three workers — no code needed, pass a top-level
`"webhook"` URL alongside `"input"` in the `/run` request; see the render
worker's README for the exact shape.

## What this doesn't do yet

- **Skin-only mode** (reusing an already-authored skeleton — upstream's
  `--use_skeleton` flag) isn't wired up. MVP is full auto-rig
  (skeleton + skinning) only, matching spec §19.1's "AI-generated unrigged
  character meshes" scope.
- **Bone attribute prediction** (physics stiffness, etc.) — upstream lists
  this as "coming soon" on its own roadmap, not available in the current
  checkpoint.
- Pose library, retargeting, IK application, and animated camera
  trajectories are explicitly out of scope for this repo (Part B) — see
  spec §19.2 / the implementation plan's Part A/Part B split.

## Build

```sh
docker build -t storystudio/3d-rigging-skintokens:latest .
```

Compiles flash-attn from source (no version pin — see the Dockerfile
comment on why) and downloads ~1.6GB of model weights at build time.
