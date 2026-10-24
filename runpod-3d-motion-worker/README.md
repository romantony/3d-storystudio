# motion-gpu — text-to-motion (HY-Motion 1.0)

RunPod Serverless image wrapping [Tencent-Hunyuan/HY-Motion-1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0)
(1.0B DiT, `tencent-hunyuan-community` license — same territorial caveats as
Hunyuan3D-2.1, needs legal review before production). Text prompt in,
SMPL-H skeleton animation out (FBX + raw NPZ), uploaded to R2.

## Why a separate image

Own pins (transformers 4.53.3, bitsandbytes, `fbxsdkpy` from an INRIA GitLab
index) that shouldn't leak into the other workers' envs. No compiled
extensions, so it uses a plain CUDA runtime base + Python 3.10 (no conda).

## GPU

Model card: 26GB min VRAM (Standard), 24GB (Lite) — and that excludes
nothing about the Qwen3-8B text encoder loaded alongside it. Use a 48GB+
card. Not yet verified on a live endpoint.

## Weights

Not baked in (~20GB). Pre-warm the network volume from a Pod:
`python scripts/download_model.py [--variant HY-Motion-1.0-Lite]`.
Otherwise the DiT downloads on first job and Qwen3-8B / CLIP via HF cache
(`USE_HF_MODELS=1`, `HF_HOME=/workspace/hf-cache`).

## Job input

| field | default | notes |
|---|---|---|
| `prompt` | required | English, ideally < 60 words |
| `durationSec` | 5.0 | 0.5–`MAX_DURATION_S` (10); keep < 5s for lower VRAM |
| `seeds` / `seed`+`numSeeds` | `[42]` | one clip per seed, max `MAX_SEEDS` (4) |
| `cfgScale` | 5.0 | upstream default |
| `variant` | `HY-Motion-1.0` | or `HY-Motion-1.0-Lite` |
| `project_id`, `frame_id` | – | R2 key naming, as in the other workers |

Output: `{clips: [{seed, fbxUrl, fbxKey, npzUrl, npzKey}], durationSec, fps: 30, skeleton: "smplh", provenance}`.

## Not included

The upstream LLM prompt rewriter / duration estimator (Text2MotionPrompter
via vllm) — the caller supplies prompt and duration. FBX output targets
upstream's SMPL-H "wooden" rig; retargeting onto SkinTokens-rigged
characters is a separate step and not built yet.
