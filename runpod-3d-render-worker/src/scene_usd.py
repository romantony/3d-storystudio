"""OpenUSD composition/parsing for the render worker (system python3, not
Blender's bundled python -- see handler.py). Spec line 59/184/320 puts
canonical scene composition (transforms, references, cameras) in USD, with
GLB staying the portable geometry format referenced *from* it. This module
is deliberately narrow: it authors/reads just enough of a `.usda` stage
(one Xform per scene node, translate/orient/scale ops, a custom
`storystudio:assetUrl` attribute carrying the GLB reference) to round-trip
the exact node shape handler.py already builds from flat scene+nodes JSON --
USD becomes an alternate input encoding for the same data, not a parallel
scene representation. Architecture decision (implementation plan §02):
"OpenUSD Python, in-worker" -- runs here rather than a second service.
"""
from typing import Any, Dict, List, Optional, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom

_STORYSTUDIO_NS = "storystudio"


def compose_usda(scene: Dict[str, Any], nodes: List[Dict[str, Any]]) -> str:
    """Builds a `.usda` text stage from the same scene+nodes shape
    handler.py already accepts as flat JSON (see fixtures/3d/golden-scene/
    scene.json + run_golden_scene.py's build_input())."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, scene.get("upAxis", "Z"))
    stage.SetMetadata("metersPerUnit", 1.0 if scene.get("units", "meter") == "meter" else 0.01)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey(f"{_STORYSTUDIO_NS}:sceneId", scene.get("sceneId", ""))
    world.GetPrim().SetCustomDataByKey(f"{_STORYSTUDIO_NS}:revision", scene.get("revision", 1))

    for node in nodes:
        xform = UsdGeom.Xform.Define(stage, f"/World/{node['nodeId']}")
        prim = xform.GetPrim()

        t = node.get("transform", {})
        translate = t.get("translation", [0, 0, 0])
        q = t.get("rotationQuat", [0, 0, 0, 1])  # spec order [x, y, z, w]
        scale = t.get("scale", [1, 1, 1])

        xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
        xform.AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))  # Gf.Quatf(real, i, j, k)
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))

        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:assetUrl", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("assetUrl", "")
        )
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:nodeType", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("type", "")
        )
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:name", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("name", "")
        )

    return stage.GetRootLayer().ExportToString()


def parse_usda(usda_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Inverse of compose_usda(): opens a `.usda` stage and reconstructs the
    scene+nodes shape handler.py's existing render path already consumes,
    so USD input is normalized once here and the rest of the pipeline
    (asset download, scene.json/camera.json, Blender subprocess) is
    reused completely unchanged."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(usda_text)
    stage = Usd.Stage.Open(layer)

    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise ValueError("USD stage has no /World prim")

    scene = {
        "sceneId": world.GetCustomDataByKey(f"{_STORYSTUDIO_NS}:sceneId") or "",
        "revision": world.GetCustomDataByKey(f"{_STORYSTUDIO_NS}:revision") or 1,
        "upAxis": UsdGeom.GetStageUpAxis(stage),
        "units": "meter" if stage.GetMetadata("metersPerUnit") == 1.0 else "centimeter",
    }

    nodes: List[Dict[str, Any]] = []
    for prim in world.GetChildren():
        xf = UsdGeom.Xformable(prim)
        local = xf.GetLocalTransformation(Usd.TimeCode.Default())
        translation = local.ExtractTranslation()
        rot_quat = local.ExtractRotationQuat()
        imag = rot_quat.GetImaginary()
        scale = [
            Gf.Vec3d(local[0][0], local[0][1], local[0][2]).GetLength(),
            Gf.Vec3d(local[1][0], local[1][1], local[1][2]).GetLength(),
            Gf.Vec3d(local[2][0], local[2][1], local[2][2]).GetLength(),
        ]
        asset_url = prim.GetAttribute(f"{_STORYSTUDIO_NS}:assetUrl").Get()
        if not asset_url:
            # A node with no asset reference can't be imported/rendered --
            # same "nothing to render" contract as an empty nodes list.
            continue
        nodes.append({
            "nodeId": prim.GetName(),
            "name": prim.GetAttribute(f"{_STORYSTUDIO_NS}:name").Get() or "",
            "type": prim.GetAttribute(f"{_STORYSTUDIO_NS}:nodeType").Get() or "",
            "assetUrl": asset_url,
            "transform": {
                "translation": [translation[0], translation[1], translation[2]],
                "rotationQuat": [imag[0], imag[1], imag[2], rot_quat.GetReal()],
                "scale": scale,
            },
        })

    return scene, nodes
