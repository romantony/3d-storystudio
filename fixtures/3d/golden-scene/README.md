# Golden test scene

Phase 0 exit criteria (spec Appendix B, implementation plan §P0/§06):

> The same canonical scene can be rendered from six cameras with no asset
> regeneration.

This fixture is the concrete test for that line, and for AC-04/AC-05/AC-06.

## What's here

- `scene.json` — one environment (ground plane) + 8 movable props, 9 nodes
  total. Every prop's mesh origin is baked to its ground-contact point
  (footprint center at local z=0) per spec §7.2, so grounded nodes carry
  `translation.z = 0`.
- `cameras/*.json` — 6 shots covering the spec §10.2 preset vocabulary that
  doesn't require a character anchor: extreme-wide/establishing, wide,
  medium, close-up (with DOF), low angle, high angle/top-down.
  Over-the-shoulder/POV are skipped here — they need a rigged character
  anchor, which is explicitly Phase 2 (spec §19).
- `assets/*.glb` — 9 procedurally generated primitives (cube/sphere/
  cylinder/cone/capsule/pyramid + ground plane), not sourced/downloaded CC0
  assets. This fixture exists to regression-test the render worker's scene
  compile + camera + pass pipeline, not asset quality — procedural geometry
  gets that with zero external downloads and zero licensing questions.
  Regenerate with `generate_assets.py` if you need to tweak the scene.
- `run_golden_scene.py` — uploads the 9 GLBs to R2 **once**, then fires all
  6 shots against that same upload and asserts every requested pass
  (rgb/depth/normal/object_id/alpha + masks/manifest) comes back non-empty
  for every shot. That upload-once/render-six-times structure is what
  actually proves "no asset regeneration," not just an assertion in a
  comment.

## Running it

```sh
pip install boto3
export RUNPOD_API_KEY=...          # your key -- never commit or paste this
export BLENDER_ENDPOINT_ID=...     # the runpod-3d-render-worker endpoint id
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_PUBLIC_URL=...           # e.g. https://pub-xxx.r2.dev
python3 run_golden_scene.py
```

Exits non-zero if any shot fails or any pass comes back empty. Prints a
per-shot pass/fail summary and the render worker's `renderTimeS` for each.

## What this does and doesn't prove

Proves: scene compile, camera framing (6 distinct presets), all 5 render
passes, DOF handling, non-destructive multi-shot reuse of one uploaded
asset set — i.e. the render worker half of Phase 0's Definition of Done.

Doesn't prove: USD round-tripping (this fixture's `scene.json` is fed to
the worker directly as flat JSON, same as the manual smoke tests so far —
there's still no OpenUSD stage anywhere in the pipeline), browser preview
loading, or anything about the asset-*generation* providers (TRELLIS.2/
Hunyuan3D) — those are exercised by the separate provider bake-off, not
this fixture.
