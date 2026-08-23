"""OpenUSD composition/parsing for the render worker (system python3, not
Blender's bundled python -- see handler.py). Spec line 59/184/320 puts
canonical scene composition (transforms, references, cameras) in USD, with
GLB staying the portable geometry format referenced *from* it. Architecture
decision (implementation plan §02): "OpenUSD Python, in-worker" -- runs
here rather than a second service.

Two composition shapes, both round-tripping into the same scene+nodes(+
camera+lights) dict shape handler.py already builds from flat JSON:

- A single flat `.usda` stage (compose_usda/parse_usda) -- one Xform per
  scene node, translate/orient/scale ops, a custom `storystudio:assetUrl`
  reference attribute, plus optional Camera/UsdLux prims and semantic
  attributes. Phase 0's original shape, extended.
- A base `scene.usda` plus a *sparse* `shot.usda` override layer
  (compose_shot_usda/parse_composed_stage) per spec §7.4/§17.1: the shot
  layer only carries override opinions -- camera selection, node transform
  overrides, visibility -- never a full scene copy. This is what spec §10
  means by shots storing diffs, not duplicates, mapped onto real USD
  sublayer composition rather than flattened by hand.
"""
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

_STORYSTUDIO_NS = "storystudio"
_LIGHT_TYPE_TO_USD = {"sun": "DistantLight", "point": "SphereLight", "area": "RectLight"}
_USD_TYPE_TO_LIGHT = {v: k for k, v in _LIGHT_TYPE_TO_USD.items()}


def _set_xform_ops(xform: UsdGeom.Xformable, transform: Dict[str, Any]) -> None:
    t = transform.get("translation", [0, 0, 0])
    q = transform.get("rotationQuat", [0, 0, 0, 1])  # spec order [x, y, z, w]
    s = transform.get("scale", [1, 1, 1])
    xform.AddTranslateOp().Set(Gf.Vec3d(*t))
    xform.AddOrientOp().Set(Gf.Quatf(q[3], q[0], q[1], q[2]))  # Gf.Quatf(real, i, j, k)
    xform.AddScaleOp().Set(Gf.Vec3f(*s))


def _read_xform(prim) -> Dict[str, Any]:
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
    return {
        "translation": [translation[0], translation[1], translation[2]],
        "rotationQuat": [imag[0], imag[1], imag[2], rot_quat.GetReal()],
        "scale": scale,
    }


def _author_camera(stage: Usd.Stage, camera: Dict[str, Any]) -> None:
    camera_id = camera.get("cameraId", "shot_camera")
    cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{camera_id}")
    prim = cam.GetPrim()
    cam.CreateFocalLengthAttr().Set(float(camera.get("focalLengthMm", 50)))
    _set_xform_ops(cam, {
        "translation": camera.get("translation", [0, -5, 1.6]),
        "rotationQuat": camera.get("rotationQuat", [0, 0, 0, 1]),
        "scale": [1, 1, 1],
    })
    # lookAt/dof are custom to our pipeline (not standard UsdGeom.Camera
    # fields) -- round-tripped as opaque custom attrs rather than modeled
    # as real USD concepts, same treatment as assetUrl on nodes.
    look_at = camera.get("lookAt")
    if look_at:
        node_target = look_at.get("nodeId", "")
        world_pos = look_at.get("_worldPosition")
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:lookAtNodeId", Sdf.ValueTypeNames.String, custom=True).Set(node_target)
        if world_pos:
            prim.CreateAttribute(f"{_STORYSTUDIO_NS}:lookAtWorldPosition", Sdf.ValueTypeNames.Float3, custom=True).Set(
                Gf.Vec3f(*world_pos)
            )
    dof = camera.get("dof") or {}
    if dof.get("enabled"):
        cam.CreateFocusDistanceAttr().Set(float(dof.get("focusDistanceM", 3.0)))
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:dofEnabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
    prim.CreateAttribute(f"{_STORYSTUDIO_NS}:cameraId", Sdf.ValueTypeNames.String, custom=True).Set(camera_id)


def _read_camera(prim) -> Dict[str, Any]:
    cam = UsdGeom.Camera(prim)
    xform_data = _read_xform(prim)
    camera: Dict[str, Any] = {
        "cameraId": prim.GetAttribute(f"{_STORYSTUDIO_NS}:cameraId").Get() or prim.GetName(),
        "focalLengthMm": cam.GetFocalLengthAttr().Get() or 50.0,
        "translation": xform_data["translation"],
        "rotationQuat": xform_data["rotationQuat"],
    }
    node_target = prim.GetAttribute(f"{_STORYSTUDIO_NS}:lookAtNodeId").Get()
    world_pos = prim.GetAttribute(f"{_STORYSTUDIO_NS}:lookAtWorldPosition").Get()
    if node_target or world_pos:
        camera["lookAt"] = {}
        if node_target:
            camera["lookAt"]["nodeId"] = node_target
        if world_pos:
            camera["lookAt"]["_worldPosition"] = [world_pos[0], world_pos[1], world_pos[2]]
    if prim.GetAttribute(f"{_STORYSTUDIO_NS}:dofEnabled").Get():
        camera["dof"] = {"enabled": True, "focusDistanceM": cam.GetFocusDistanceAttr().Get() or 3.0}
    return camera


def _author_light(stage: Usd.Stage, light: Dict[str, Any]) -> None:
    light_id = light["lightId"]
    usd_type = _LIGHT_TYPE_TO_USD.get(light.get("type", "sun"), "DistantLight")
    path = f"/World/Lights/{light_id}"
    if usd_type == "DistantLight":
        usd_light = UsdLux.DistantLight.Define(stage, path)
    elif usd_type == "SphereLight":
        usd_light = UsdLux.SphereLight.Define(stage, path)
    else:
        usd_light = UsdLux.RectLight.Define(stage, path)
    prim = usd_light.GetPrim()
    usd_light.CreateIntensityAttr().Set(float(light.get("energy", 5.0)))
    color = light.get("color", [1.0, 1.0, 1.0])
    usd_light.CreateColorAttr().Set(Gf.Vec3f(*color))
    _set_xform_ops(UsdGeom.Xformable(prim), {
        "translation": light.get("translation", [0, 0, 0]),
        "rotationQuat": light.get("rotationQuat", [0, 0, 0, 1]),
        "scale": [1, 1, 1],
    })
    prim.CreateAttribute(f"{_STORYSTUDIO_NS}:lightId", Sdf.ValueTypeNames.String, custom=True).Set(light_id)
    prim.CreateAttribute(f"{_STORYSTUDIO_NS}:lightType", Sdf.ValueTypeNames.String, custom=True).Set(
        light.get("type", "sun")
    )


def _read_light(prim) -> Dict[str, Any]:
    xform_data = _read_xform(prim)
    intensity_attr = prim.GetAttribute("inputs:intensity")
    color_attr = prim.GetAttribute("inputs:color")
    color = color_attr.Get() if color_attr else None
    return {
        "lightId": prim.GetAttribute(f"{_STORYSTUDIO_NS}:lightId").Get() or prim.GetName(),
        "type": prim.GetAttribute(f"{_STORYSTUDIO_NS}:lightType").Get() or "sun",
        "energy": intensity_attr.Get() if intensity_attr else 5.0,
        "color": [color[0], color[1], color[2]] if color else [1.0, 1.0, 1.0],
        "translation": xform_data["translation"],
        "rotationQuat": xform_data["rotationQuat"],
    }


def compose_usda(
    scene: Dict[str, Any],
    nodes: List[Dict[str, Any]],
    camera: Optional[Dict[str, Any]] = None,
    lights: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Builds a `.usda` text stage from the same scene+nodes(+camera+lights)
    shape handler.py already accepts as flat JSON (see fixtures/3d/
    golden-scene/scene.json + run_golden_scene.py's build_input()).
    `camera`/`lights` are optional -- omitting them preserves the original
    Phase 0 scene-only stage shape exactly."""
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
        _set_xform_ops(xform, node.get("transform", {}))

        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:assetUrl", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("assetUrl", "")
        )
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:nodeType", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("type", "")
        )
        prim.CreateAttribute(f"{_STORYSTUDIO_NS}:name", Sdf.ValueTypeNames.String, custom=True).Set(
            node.get("name", "")
        )
        semantic = node.get("semantic") or {}
        if semantic.get("role"):
            prim.CreateAttribute(f"{_STORYSTUDIO_NS}:semanticRole", Sdf.ValueTypeNames.String, custom=True).Set(
                semantic["role"]
            )
        if semantic.get("forward"):
            prim.CreateAttribute(f"{_STORYSTUDIO_NS}:semanticForward", Sdf.ValueTypeNames.Float3, custom=True).Set(
                Gf.Vec3f(*semantic["forward"])
            )
        visibility = node.get("visibility", True)
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if visibility else UsdGeom.Tokens.invisible
        )

    if camera:
        _author_camera(stage, camera)
    for light in lights or []:
        _author_light(stage, light)

    return stage.GetRootLayer().ExportToString()


def _extract_from_stage(stage: Usd.Stage) -> Dict[str, Any]:
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
    camera: Optional[Dict[str, Any]] = None
    lights: List[Dict[str, Any]] = []

    for prim in world.GetChildren():
        name = prim.GetName()
        if name == "Cameras":
            cam_children = list(prim.GetChildren())
            if cam_children:
                camera = _read_camera(cam_children[0])
            continue
        if name == "Lights":
            for light_prim in prim.GetChildren():
                lights.append(_read_light(light_prim))
            continue

        asset_url = prim.GetAttribute(f"{_STORYSTUDIO_NS}:assetUrl").Get()
        if not asset_url:
            # A node with no asset reference can't be imported/rendered --
            # same "nothing to render" contract as an empty nodes list.
            continue
        xform_data = _read_xform(prim)
        visibility_attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
        node = {
            "nodeId": prim.GetName(),
            "name": prim.GetAttribute(f"{_STORYSTUDIO_NS}:name").Get() or "",
            "type": prim.GetAttribute(f"{_STORYSTUDIO_NS}:nodeType").Get() or "",
            "assetUrl": asset_url,
            "transform": xform_data,
            "visibility": (visibility_attr.Get() != UsdGeom.Tokens.invisible) if visibility_attr else True,
        }
        role = prim.GetAttribute(f"{_STORYSTUDIO_NS}:semanticRole").Get()
        forward = prim.GetAttribute(f"{_STORYSTUDIO_NS}:semanticForward").Get()
        if role or forward:
            node["semantic"] = {}
            if role:
                node["semantic"]["role"] = role
            if forward:
                node["semantic"]["forward"] = [forward[0], forward[1], forward[2]]
        nodes.append(node)

    return {"scene": scene, "nodes": nodes, "camera": camera, "lights": lights}


def parse_usda(usda_text: str) -> Dict[str, Any]:
    """Inverse of compose_usda(): opens a `.usda` stage and reconstructs the
    scene+nodes(+camera+lights) shape handler.py's render path consumes.
    Returns a dict (not a 2-tuple) so new fields (camera/lights) can be
    added without breaking positional-unpack call sites -- see handler.py."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(usda_text)
    stage = Usd.Stage.Open(layer)
    return _extract_from_stage(stage)


def compose_shot_usda(
    shot: Dict[str, Any],
    base_scene: Optional[Dict[str, Any]] = None,
    base_layer_path: str = "./scene.usda",
) -> str:
    """Sparse shot override layer per spec §7.4/§17.1: never a full scene
    copy. `shot` is `{"camera": {...}?, "nodeOverrides": [{"nodeId",
    "transform"?, "visibility"?}]?, "lights": [...]?}` -- everything
    optional, since a shot may override only the camera, only a couple of
    transforms, or nothing at all. `base_layer_path` is the relative/
    absolute path this shot layer will sublayer at render time (see
    parse_composed_stage, which writes both texts to temp files with this
    exact relationship before opening).

    `base_scene` (the base scene.usda's own scene dict, e.g. what
    compose_usda's `scene` argument was) is used to re-stamp upAxis/
    metersPerUnit onto *this* layer -- confirmed empirically that unlike
    prim-level opinions (which list/dict-compose across the full sublayer
    stack), UsdGeom's stage-level metadata (upAxis, metersPerUnit) is only
    ever read from a stage's root layer, not resolved through sublayers.
    Without this, a composed shot stage silently falls back to USD's
    defaults (Y-up, centimeters) even though the base scene.usda correctly
    declares Z-up/meters."""
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().subLayerPaths.append(base_layer_path)
    if base_scene:
        UsdGeom.SetStageUpAxis(stage, base_scene.get("upAxis", "Z"))
        stage.SetMetadata("metersPerUnit", 1.0 if base_scene.get("units", "meter") == "meter" else 0.01)

    camera = shot.get("camera")
    if camera:
        _author_camera(stage, camera)

    for override in shot.get("nodeOverrides", []):
        node_id = override["nodeId"]
        # OverridePrim authors a pure "over" spec -- no new def opinion --
        # so this layer only ever carries diffs against whatever the base
        # scene.usda already defined at /World/{node_id}.
        prim = stage.OverridePrim(f"/World/{node_id}")
        if "transform" in override:
            xform = UsdGeom.Xformable(prim)
            # Clear any inherited xformOpOrder before authoring override
            # ops, so this layer's ops fully replace rather than compose
            # on top of the base layer's ops for the same node.
            xform.ClearXformOpOrder()
            _set_xform_ops(xform, override["transform"])
        if "visibility" in override:
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if override["visibility"] else UsdGeom.Tokens.invisible
            )

    for light in shot.get("lights", []):
        _author_light(stage, light)

    if shot.get("shotId"):
        stage.GetRootLayer().customLayerData = {f"{_STORYSTUDIO_NS}:shotId": shot["shotId"]}

    return stage.GetRootLayer().ExportToString()


def parse_composed_stage(base_usda_text: str, shot_usda_text: str) -> Dict[str, Any]:
    """Composes a base scene.usda + sparse shot.usda override layer
    on-disk (USD sublayer resolution needs real files, not in-memory
    text -- the two blobs arrive over HTTP as separate strings, so this
    materializes them into a temp dir with the exact relative-path
    relationship compose_shot_usda() assumed) and extracts the merged
    scene+nodes+camera+lights, with the shot layer's opinions winning
    (USD's normal strongest-layer-wins sublayer rule)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        base_path = tmp_dir / "scene.usda"
        shot_path = tmp_dir / "shot.usda"
        base_path.write_text(base_usda_text)
        shot_path.write_text(shot_usda_text)
        stage = Usd.Stage.Open(str(shot_path))
        return _extract_from_stage(stage)
