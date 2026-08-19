"""Runs inside Blender's bundled Python via `blender --background --python
render_scene.py -- <args>`. Not a RunPod handler — the outer handler.py
(system python3, in ../handler.py) downloads assets, writes scene/camera
JSON, shells out to this script, then uploads whatever lands in --out-dir.

Contract (spec Appendix B, Definition of Done): accept scene JSON + GLB
assets + camera JSON, produce RGB/depth/normal/object_id/alpha passes plus
a masks.json anchor and a Blender version stamp for the render manifest.
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    args = {}
    it = iter(argv)
    for a in it:
        if a.startswith("--"):
            args[a[2:]] = next(it, None)
    return args


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.film_transparent = True


def import_nodes(nodes):
    """Each node becomes a parent Empty at the node's transform, with the
    imported glTF objects underneath — the transform is applied once at the
    anchor rather than distributed across however many objects the GLB
    contains, and pass_index is stamped on every child for the object_id
    pass to key off."""
    imported = []
    for idx, node in enumerate(nodes, start=1):
        local_path = node.get("_localPath")
        if not local_path:
            continue

        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=local_path)
        new_objs = [o for o in bpy.context.scene.objects if o not in before]
        if not new_objs:
            continue

        node_id = node.get("nodeId", f"node_{idx}")
        anchor = bpy.data.objects.new(node_id, None)
        bpy.context.scene.collection.objects.link(anchor)
        for o in new_objs:
            if o.parent is None:
                o.parent = anchor
            o.pass_index = idx

        transform = node.get("transform", {})
        t = transform.get("translation", [0, 0, 0])
        q = transform.get("rotationQuat", [0, 0, 0, 1])  # spec order [x,y,z,w]
        s = transform.get("scale", [1, 1, 1])
        anchor.location = Vector(t)
        anchor.rotation_mode = 'QUATERNION'
        anchor.rotation_quaternion = Quaternion((q[3], q[0], q[1], q[2]))  # mathutils order (w,x,y,z)
        anchor.scale = Vector(s)

        imported.append({"nodeId": node_id, "passIndex": idx})
    return imported


def add_default_lighting():
    """Structural/reference renders need *some* light even when the scene
    JSON doesn't define one yet (Phase 0 golden scene has no /Lights nodes)
    — a flat sun + world fill keeps the render non-black without editorializing
    on final look, which spec §11.2 leaves to the generative stage."""
    sun_data = bpy.data.lights.new("key_sun", type='SUN')
    sun_data.energy = 3.0
    sun = bpy.data.objects.new("key_sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(55), 0, math.radians(35))

    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[1].default_value = 0.35  # ambient fill strength


def setup_camera(camera_cfg, width, height):
    cam_data = bpy.data.cameras.new("shot_camera")
    cam_data.lens = float(camera_cfg.get("focalLengthMm", 50))
    cam = bpy.data.objects.new("shot_camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    t = camera_cfg.get("translation", [0, -5, 1.6])
    cam.location = Vector(t)
    cam.rotation_mode = 'QUATERNION'

    look_at = camera_cfg.get("lookAt") or {}
    world_target = look_at.get("_worldPosition")  # resolved by the caller from nodeId+anchor
    if world_target:
        direction = Vector(world_target) - cam.location
        cam.rotation_quaternion = direction.to_track_quat('-Z', 'Y')
    else:
        q = camera_cfg.get("rotationQuat", [0, 0, 0, 1])
        cam.rotation_quaternion = Quaternion((q[3], q[0], q[1], q[2]))

    dof = camera_cfg.get("dof", {})
    if dof.get("enabled"):
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = float(dof.get("focusDistanceM", 3.0))

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    return cam


def configure_engine(engine, samples):
    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == 'CYCLES':
        # Cycles' OptiX/CUDA compute path needs no display or EGL context —
        # the reliable default for headless server rendering. See README:
        # background EEVEE-in-Docker GPU rendering is known-fragile (EGL
        # context setup, driver-specific bugs).
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = 'GPU'
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
    else:
        # BLENDER_EEVEE_NEXT — opt-in only, verify NVIDIA_DRIVER_CAPABILITIES
        # and libegl1/libgl1 work on the target GPU/driver before relying on it.
        scene.eevee.taa_render_samples = samples


def setup_passes(passes, out_dir):
    scene = bpy.context.scene
    vl = scene.view_layers[0]
    vl.use_pass_z = "depth" in passes
    vl.use_pass_normal = "normal" in passes
    vl.use_pass_object_index = "object_id" in passes
    vl.use_pass_combined = True

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")

    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], composite.inputs["Image"])

    def file_output(socket_name, filename_prefix, file_format, color_depth=None):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.base_path = str(out_dir)
        node.file_slots[0].path = filename_prefix
        node.format.file_format = file_format
        if color_depth:
            node.format.color_depth = color_depth
        tree.links.new(rl.outputs[socket_name], node.inputs[0])

    file_output("Image", "rgb_", 'PNG', '8')
    if "alpha" in passes:
        file_output("Alpha", "alpha_", 'PNG', '8')
    if "depth" in passes:
        file_output("Depth", "depth_", 'OPEN_EXR')
    if "normal" in passes:
        file_output("Normal", "normal_", 'OPEN_EXR')
    if "object_id" in passes:
        file_output("IndexOB", "object_id_", 'PNG', '16')


def normalize_output_filenames(out_dir: Path):
    """CompositorNodeOutputFile appends Blender's frame number to each
    filename (rgb_0001.png) — strip it so the handler can upload fixed,
    predictable names."""
    renames = {"rgb_": "rgb", "alpha_": "alpha", "depth_": "depth",
               "normal_": "normal", "object_id_": "object_id"}
    for f in list(out_dir.iterdir()):
        for prefix, canonical in renames.items():
            if f.name.startswith(prefix):
                f.rename(out_dir / f"{canonical}{f.suffix}")
                break


def main():
    args = parse_args()
    scene_json = json.loads(Path(args["scene"]).read_text())
    camera_json = json.loads(Path(args["camera"]).read_text())
    out_dir = Path(args["out-dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = int(args["width"]), int(args["height"])
    passes = args["passes"].split(",")

    reset_scene()
    imported = import_nodes(scene_json.get("nodes", []))
    add_default_lighting()
    setup_camera(camera_json, width, height)
    configure_engine(args["engine"], int(args["samples"]))
    setup_passes(passes, out_dir)

    bpy.ops.render.render(write_still=True)

    normalize_output_filenames(out_dir)
    (out_dir / "masks.json").write_text(json.dumps({"nodes": imported}))
    (out_dir / "version.txt").write_text(bpy.app.version_string)


if __name__ == "__main__":
    main()
