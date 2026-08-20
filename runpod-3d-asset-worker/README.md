# asset-gpu — 3D object generation workers

Two RunPod Serverless images, one per provider: [`trellis2/`](trellis2/) and
[`hunyuan3d/`](hunyuan3d/). Both implement the **same request/response
contract** below so the Step Functions asset-generation step (spec
`E2E-3D-Asset-Generate`) can call either behind a `providerProfile` without
caring which one answered — the spec's `ThreeDProvider` abstraction (§6.3)
lives at that orchestration layer, not inside a single Docker image.

## Why two images instead of one

TRELLIS.2 and Hunyuan3D-2.1 pin **different, incompatible toolchains**:

| | TRELLIS.2 | Hunyuan3D-2.1 |
|---|---|---|
| torch | 2.6.0 / cu124 | 2.5.1 / cu124 |
| Compiled CUDA extensions | flash-attn 2.7.3, nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel | custom_rasterizer, DifferentiableRenderer (compiled against the exact conda-pinned toolchain) |

Merging them into one image means every compiled extension has to survive
being built next to a torch version it wasn't tested against. That's a real
ABI risk for a marginal win — they'd still be **separate RunPod endpoints**
regardless, the same way `qwen-image-gen` and `Wan2-14b-fp8` are separate
endpoints today, because GPU pool sizing and scale-to-zero behavior should
be independent per model.

## Model decision

| | Decision | Why |
|---|---|---|
| Primary | **TRELLIS.2** | MIT license, cleanest legal fit, official H100 numbers (3s/17s/60s @ 512³/1024³/1536³) match the spec's cited figures |
| Secondary / fallback profile | **Hunyuan3D 2.1** | Strong published quality benchmarks; kept live from day one as the configured fallback, not bolted on after an outage |
| Deferred | HunyuanWorld / HY-World | Phase 3 — scene-scale, not object-scale; out of this worker's scope entirely |

**License flag — read before enabling Hunyuan3D in production:** it ships
under the **Tencent Hunyuan 3D 2.1 Community License**, not MIT, and that
license **explicitly excludes the EU, UK and South Korea** from its grant.
This is exactly the case spec §21.2 warns about — don't infer commercial
rights from "open-source" wording. `threeDProviderConfigs.licenseProfileId`
for the `hunyuan3d` profile must point to a completed legal review, scoped
to whichever territories StoryStudio actually serves, before this image
takes production traffic. TRELLIS.2 has no such restriction.

**Both handlers are written against each project's documented README-level
Python API**, not the full pipeline source — in particular
`Hunyuan3DPaintPipeline.__call__`'s exact return type isn't specified
upstream (`handler.py` defensively handles either a path string or a
trimesh-exportable object). Confirm both APIs against the pinned commit
during the Phase 0 bake-off before treating either handler as
production-hardened.

## GPU / volume

Reuses the exact GPU class and network-volume pattern already live for
`qwen-image-gen`/`qwen-image-edit` (see `~/runpod/API.md`) — no new GPU
vocabulary for the fleet to operate.

| | GPU | Why |
|---|---|---|
| trellis2 | **L40S, 48GB** (~$0.86/hr) | TRELLIS.2 needs ≥24GB; L40S clears that with 2x headroom for cheaper than A40/A6000 (~$1.22/hr) for the same 48GB, and it's the same GPU class already picked for `runpod-3d-render-worker` — one GPU class across the whole 3D pipeline instead of two. Already covered by the Dockerfile's `TORCH_CUDA_ARCH_LIST` (includes `8.9`/Ada). Step up to A100/H100 only if the Phase 0 bake-off shows `quality: "high"` needs it. |
| hunyuan3d | A40 / A6000 / A100, 48GB+ | Combined shape+texture needs ~29GB — same class already used elsewhere leaves headroom |

Suggested volumes (network volume, mounted at `/workspace` on both Pods and Serverless workers on this account):
- `trellis2-4b-a40` — the ~15GB TRELLIS.2-4B checkpoint, must land at `/workspace/models/Trellis2` specifically (that's the persistent network volume mount; anywhere else on a Pod is local container disk and gets thrown away). Pre-warm it once by attaching the volume to a RunPod Pod and running [`trellis2/scripts/download_model.py`](trellis2/scripts/download_model.py) — `handler.py` prefers this over a live Hugging Face pull automatically, and falls back to one only if the volume wasn't pre-warmed.
- `hunyuan3d-2-1-a40` — caches the Hunyuan3D-Shape-2.1 + Hunyuan3D-Paint-2.1 checkpoints (larger — texture model included). Pre-warm with [`hunyuan3d/scripts/download_model.py`](hunyuan3d/scripts/download_model.py). Note this worker's caching isn't uniform like TRELLIS.2's: the shape pipeline uses a custom loader that ignores `HF_HOME` entirely (reads `HY3DGEN_MODELS` instead — set in the Dockerfile to `/workspace/models/hy3dgen`), while the paint pipeline's multiview weights and `facebook/dinov2-giant` use standard HF caching and do respect `HF_HOME`. The pre-warm script handles both paths; confirmed against the actual upstream `smart_load_model` source, not assumed.

## Scope: generation only, not normalization

Both handlers return a **raw generated GLB** plus provenance — nothing
else. Thumbnail generation, LOD derivatives, unit/origin normalization and
manifest/hash writing are deliberately **not** done here; that's the
`asset-cpu` pool's job (spec §23.2, §9.1 steps 4–9), running on cheap CPU
workers so the expensive GPU pool isn't held open for I/O-bound work.

## Request / response contract

```json
POST /run
{"input": {
  "image_url": "https://.../reference.png",
  "quality": "standard",
  "project_id": "prj_hero_puppy",
  "frame_id": "storm_drain_01",
  "seed": 19203,
  "sourceImageId": "img_123"
}}
```

`quality`: `draft` | `standard` | `high` (trellis2) — controls
`decimation_target`/`texture_size`. `shape_only` | `standard` (hunyuan3d) —
`shape_only` skips the texture pass entirely (10GB VRAM vs 29GB combined).

```json
{"modelUrl": "https://pub-xxx.r2.dev/storystudio/models_3d/prj_hero_puppy_storm_drain_01_asset_generate.glb",
 "provenance": {"provider": "trellis2", "modelVersion": "TRELLIS.2-4B", "quality": "standard", "seed": 19203, "genTimeS": 21.4}}
```

Errors follow the shared shape: `{"error": "...", "traceback": "..."}`.

## Build

```sh
docker build -t storystudio/3d-asset-trellis2:latest   trellis2/
docker build -t storystudio/3d-asset-hunyuan3d:latest   hunyuan3d/
```

Both builds compile CUDA extensions from source and will take a while
(TRELLIS.2's own docs warn about this) — budget real CI time, matching the
build-time gotchas already logged for the Qwen workers.
