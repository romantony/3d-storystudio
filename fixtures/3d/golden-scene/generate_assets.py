#!/usr/bin/env python3
"""Regenerates assets/*.glb for the golden test scene.

Deliberately procedural (trimesh primitives), not sourced/downloaded CC0
assets — this fixture exists to regression-test the render worker's scene
compile + camera + pass pipeline (spec Appendix B), not asset quality, so
zero external downloads and zero licensing questions beats "real-looking"
geometry. Every mesh is translated so its local origin sits at its
ground-contact point (footprint center at z=0), per the canonical origin
policy in spec §7.2 — scene.json can then place grounded props with
translation.z == 0.

    python3 -m venv .venv && .venv/bin/pip install trimesh numpy scipy
    .venv/bin/python3 generate_assets.py
"""
import numpy as np
import trimesh

OUT = "assets"

# Meshes below are authored Z-up (ground at z=0, per spec §7.2), but glTF is
# Y-up by spec and Blender's importer converts (x, y, z) -> (x, -z, y). Without
# this rotation the GLBs import 90deg off (ground plane standing on its edge,
# props half-buried and centered on z=0). (x, y, z) -> (x, z, -y) is the exact
# inverse, so a Z-up scene round-trips through the importer unchanged.
ZUP_TO_YUP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


def ground_to_zero(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Shift so the mesh's lowest point sits at z=0 (ground-contact origin)."""
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def color(mesh: trimesh.Trimesh, rgb) -> trimesh.Trimesh:
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh, face_colors=[*rgb, 255] * np.ones((len(mesh.faces), 4), dtype=np.uint8)
    )
    return mesh


def pyramid(base=0.4, height=0.5) -> trimesh.Trimesh:
    h = base / 2
    verts = np.array([
        [-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0],  # base, z=0
        [0, 0, height],                                    # apex
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],           # base
        [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],  # sides
    ])
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


ASSETS = {
    "prop_cube_red":       lambda: color(ground_to_zero(trimesh.creation.box([0.5, 0.5, 0.5])), (196, 60, 48)),
    "prop_cube_blue":      lambda: color(ground_to_zero(trimesh.creation.box([0.35, 0.35, 0.35])), (52, 98, 176)),
    "prop_sphere_green":   lambda: color(ground_to_zero(trimesh.creation.icosphere(radius=0.30, subdivisions=3)), (66, 150, 84)),
    "prop_sphere_teal":    lambda: color(ground_to_zero(trimesh.creation.icosphere(radius=0.22, subdivisions=3)), (40, 148, 143)),
    "prop_cylinder_yellow": lambda: color(ground_to_zero(trimesh.creation.cylinder(radius=0.20, height=0.60, sections=32)), (214, 172, 44)),
    "prop_cone_purple":    lambda: color(ground_to_zero(trimesh.creation.cone(radius=0.25, height=0.50, sections=32)), (128, 78, 168)),
    "prop_capsule_orange": lambda: color(ground_to_zero(trimesh.creation.capsule(radius=0.15, height=0.35)), (222, 128, 44)),
    "prop_pyramid_pink":   lambda: color(pyramid(base=0.4, height=0.5), (208, 96, 150)),
    "prop_cube_teal_small": lambda: color(ground_to_zero(trimesh.creation.box([0.22, 0.22, 0.22])), (36, 140, 132)),
    "prop_sphere_gold":    lambda: color(ground_to_zero(trimesh.creation.icosphere(radius=0.18, subdivisions=3)), (196, 158, 40)),
    "env_ground_plane":    lambda: color(trimesh.creation.box([8, 8, 0.05]).apply_translation([0, 0, -0.025]), (120, 120, 112)),
}


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    for name, build in ASSETS.items():
        mesh = build()
        b = mesh.bounds  # Z-up bounds, reported before the glTF axis rotation
        path = f"{OUT}/{name}.glb"
        mesh.apply_transform(ZUP_TO_YUP).export(path, file_type="glb")
        print(f"{name:22s} bounds z=[{b[0][2]:.3f},{b[1][2]:.3f}]  -> {path}")


if __name__ == "__main__":
    main()
