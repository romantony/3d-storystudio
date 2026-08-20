# Full-pipeline smoke test: Qwen -> TRELLIS.2 -> Blender

Chains all three RunPod endpoints for real: text prompt -> Qwen-generated
image -> TRELLIS.2 GLB -> three Blender/Cycles camera shots, for three
different generated objects (treasure chest, toy robot, ceramic mug).

This goes one step further than [`fixtures/3d/golden-scene/`](../golden-scene/),
which only exercises the render worker against known-good procedural
geometry — this exercises the *generation* half of the pipeline too, and
proves the actual handoff shape (`modelUrl` -> `nodes[0].assetUrl`)
described in the "Chaining them in StoryStudio" section of the Trellis &
Blender Pipeline doc, starting from a prompt instead of a pre-existing
reference image.

## Running it

```sh
pip install python-dotenv   # no boto3 needed here -- no R2 upload step
python3 run_full_pipeline_test.py
```

Reads `RUNPOD_API_KEY`, `QWEN_ENDPOINT_ID`, `TRELLIS2_ENDPOINT_ID`,
`BLENDER_ENDPOINT_ID` from `.env` at the repo root (auto-loaded). Takes a
while — 3 test cases x (1 Qwen job + 1 TRELLIS.2 job + 3 Blender shots) =
15 sequential jobs, with TRELLIS.2 cold-start alone running ~3 minutes.
Writes `last_run_results.json` (gitignored) with every URL on completion.

## Known unknowns

- **Qwen's output field name isn't confirmed against real source** — that
  worker isn't in this repo, so `extract_image_url()` guesses across a
  few common shapes (`imageUrl`, `image_url`, `url`, `images[0]`) and
  fails loudly with the raw payload if none match. If it fails here,
  paste the raw payload back and the extractor gets a real fix instead of
  another guess.
- **Camera framing assumes TRELLIS.2's fixed aabb** — every generated GLB
  is normalized to roughly a 1m cube centered on the origin
  (`handler.py`'s `o_voxel.postprocess.to_glb` call uses a fixed
  `[-0.5,-0.5,-0.5]..[0.5,0.5,0.5]` bound), so the three camera shots are
  framed against that, not measured per-asset. There's no asset
  normalization/re-centering step yet (spec §9.1) to correct for an
  unusually shaped generation.
- **Prompts are deliberately "isolated product shot" style** — single
  object, plain background — because TRELLIS.2's background-removal step
  (briaai/RMBG-2.0) and image-conditioning model both assume one clean
  foreground subject. A busy generated scene is likely to produce a
  broken or partial mesh.
