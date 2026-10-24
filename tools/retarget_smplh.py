"""Retarget HY-Motion SMPL-H NPZ clips onto a SkinTokens-rigged GLB (52-joint,
SMPL-H-shaped skeleton named bone_0..bone_51) and export a GLB with one glTF
animation per clip.

    blender -b --python retarget.py -- in.glb out.glb name1=clip1.npz [name2=clip2.npz ...]

Only the 22 body joints are driven; finger joints stay at rest (the generated
hand poses are just SMPL's mean hand, not real motion).

Method (rest-pose-corrected world-rotation transfer):
  W(j) = B * G_smpl(j) * B^-1 * C(j) * R_rest(j)
where G_smpl is the SMPL world rotation of joint j, B maps SMPL axes
(x=left, y=up, z=forward) onto the target's axes, R_rest is the target bone's
rest rotation and C(j) is the shortest-arc rotation taking the target's rest
bone direction onto SMPL's T-pose direction (so an A-pose target and a T-pose
SMPL agree). Local pose = (W(parent) * Rrel)^-1 * W(j).
"""
import sys
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
IN_GLB, OUT_GLB = argv[0], argv[1]
CLIPS = [a.split("=", 1) for a in argv[2:]]
FPS = 30

# target bone index -> SMPL-H joint index (topology match; verified against
# the SkinTokens output: bone_6/25 collars off spine3, 44/48 hips off pelvis).
# "L" here is the side of bone_44 chain; swapped below if it isn't SMPL-left.
SIDE_A = {0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15,
          6: 13, 7: 16, 8: 18, 9: 20,
          44: 1, 45: 4, 46: 7, 47: 10}
SIDE_B = {25: 14, 26: 17, 27: 19, 28: 21,
          48: 2, 49: 5, 50: 8, 51: 11}
HAND_FINGER_BASES_A = (13, 16, 19, 22)   # first joints of the 4 non-thumb fingers, bone_9's hand
HAND_FINGER_BASES_B = (32, 35, 38, 41)   # bone_28's hand
SMPL_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=IN_GLB)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
bpy.context.view_layer.objects.active = arm
data = arm.data
bones = {int(b.name.split("_")[1]): b for b in data.bones if b.name.startswith("bone_")}
assert len(bones) == 52, f"expected 52 bones, got {len(bones)}"
rest_rot = {i: b.matrix_local.to_3x3() for i, b in bones.items()}
head = {i: b.head_local.copy() for i, b in bones.items()}
# every bone's parent index, from the armature hierarchy
parent = {i: (int(b.parent.name.split("_")[1]) if b.parent else None) for i, b in bones.items()}

# ---- target frame: up / forward / left ----------------------------------
# glTF import is Z-up; derive axes from the skeleton instead of assuming.
def snap(v):
    # nearest signed world axis: a leaning spine / splayed toes must not tilt the whole clip
    i = max(range(3), key=lambda k: abs(v[k]))
    out = Vector((0, 0, 0)); out[i] = 1.0 if v[i] > 0 else -1.0
    return out
up = snap(head[5] - head[0])
toe = (head[47] + head[51]) / 2 - (head[46] + head[50]) / 2
fwd = snap(toe - up * toe.dot(up))
left = up.cross(fwd).normalized()
side_a_is_left = (head[44] - head[48]).dot(left) > 0
print(f"[frame] up={tuple(round(x,3) for x in up)} fwd={tuple(round(x,3) for x in fwd)} "
      f"left={tuple(round(x,3) for x in left)} bone_44 is {'LEFT' if side_a_is_left else 'RIGHT'}")
if not side_a_is_left:
    # swap which chain plays SMPL-left: SMPL idx pairs differ by one (L odd/even)
    swap = {13: 14, 14: 13, 16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20,
            1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10}
    SIDE_A = {k: swap.get(v, v) for k, v in SIDE_A.items()}
    SIDE_B = {k: swap.get(v, v) for k, v in SIDE_B.items()}
MAP = {**SIDE_A, **SIDE_B}            # target bone -> SMPL joint
INV = {v: k for k, v in MAP.items()}  # SMPL joint -> target bone
B = Matrix(((left.x, up.x, fwd.x), (left.y, up.y, fwd.y), (left.z, up.z, fwd.z)))  # columns = SMPL x,y,z

# ---- rest-pose correction C(j) -------------------------------------------
def unit(*v):
    return Vector(v).normalized()

# idealised SMPL T-pose bone directions in SMPL axes (x=left, y=up, z=fwd)
IDEAL = {0: unit(0, 1, 0), 3: unit(0, 1, 0), 6: unit(0, 1, 0), 9: unit(0, 1, 0), 12: unit(0, 1, 0),
         13: unit(1, .1, 0), 16: unit(1, 0, 0), 18: unit(1, 0, 0), 20: unit(1, 0, 0),
         14: unit(-1, .1, 0), 17: unit(-1, 0, 0), 19: unit(-1, 0, 0), 21: unit(-1, 0, 0),
         1: unit(0, -1, 0), 4: unit(0, -1, 0), 7: unit(0, -.45, .9),
         2: unit(0, -1, 0), 5: unit(0, -1, 0), 8: unit(0, -.45, .9)}
# target bone each SMPL joint's direction points toward
def target_dir(sj):
    tb = INV[sj]
    if sj in (20, 21):  # wrist -> mean of finger bases
        bases = HAND_FINGER_BASES_A if tb == 9 else HAND_FINGER_BASES_B
        tip = sum((head[b] for b in bases), Vector()) / len(bases)
        return (tip - head[tb]).normalized()
    child_smpl = {0: 3, 3: 6, 6: 9, 9: 12, 12: 15, 13: 16, 16: 18, 18: 20, 14: 17, 17: 19, 19: 21,
                  1: 4, 4: 7, 7: 10, 2: 5, 5: 8, 8: 11}[sj]
    return (head[INV[child_smpl]] - head[tb]).normalized()

C = {}
for sj, d_s in IDEAL.items():
    d_t = target_dir(sj)
    C[sj] = d_t.rotation_difference(B @ d_s).to_matrix()
# Joints whose own rest direction isn't a meaningful limb axis (the head
# joint sits forward of the neck, finger bases splay) inherit instead:
# neck/head follow spine3, hands follow the forearm, feet follow the ankle.
for j, src in ((12, 9), (15, 9), (20, 18), (21, 19), (10, 7), (11, 8)):
    C[j] = C[src]
for sj in sorted(C):
    ang = np.degrees(Quaternion(C[sj].to_quaternion()).angle)
    print(f"[C] smpl {sj:2d} <- bone_{INV[sj]:<2d} correction {ang:5.1f} deg")

# ---- scale / ground ----------------------------------------------------------
ground = min((head[i] @ up) for i in head)               # lowest joint along up
pelvis_h = (head[0] @ up) - ground

def aa_to_mat(v):
    ang = float(np.linalg.norm(v))
    return Matrix.Identity(3) if ang < 1e-8 else Matrix.Rotation(ang, 3, Vector(v / ang))

bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
scene = bpy.context.scene
scene.render.fps = FPS

order = []  # SMPL joints parent-first (indices already topologically sorted)
for sj in range(22):
    order.append(sj)

actions = []
for name, npz_path in CLIPS:
    npz = np.load(npz_path)
    poses = npz["poses"].reshape(-1, 52, 3)[:, :22]
    trans = npz["trans"]
    T = poses.shape[0]
    s = pelvis_h / float(trans[0, 1])      # SMPL pelvis height at frame 0 -> target
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    arm.animation_data_create().action = act
    prev_q = {}
    foot_low = []
    for t in range(T):
        G = {}
        W = {}
        for sj in order:
            R = aa_to_mat(poses[t, sj])
            G[sj] = R if SMPL_PARENT[sj] < 0 else G[SMPL_PARENT[sj]] @ R
            W[sj] = B @ G[sj] @ B.inverted() @ C[sj] @ rest_rot[INV[sj]]
        for sj in order:
            tb = INV[sj]
            pb = arm.pose.bones[f"bone_{tb}"]
            p = parent[tb]
            if SMPL_PARENT[sj] < 0:
                L = rest_rot[tb].inverted() @ W[sj]
            else:
                pt = INV[SMPL_PARENT[sj]]
                Rrel = rest_rot[pt].inverted() @ rest_rot[tb]
                L = (W[SMPL_PARENT[sj]] @ Rrel).inverted() @ W[sj]
            q = L.to_quaternion()
            if tb in prev_q and q.dot(prev_q[tb]) < 0:
                q.negate()                      # keep quaternion continuity
            prev_q[tb] = q
            pb.rotation_quaternion = q
            pb.keyframe_insert("rotation_quaternion", frame=t + 1)
        d = s * (B @ Vector(trans[t] - trans[0]))
        root = arm.pose.bones["bone_0"]
        root.location = rest_rot[0].inverted() @ d
        root.keyframe_insert("location", frame=t + 1)
    actions.append(act)
    print(f"[clip] {name}: {T} frames, scale {s:.3f}")

# park the armature at rest and let the exporter pick up every action
arm.animation_data.action = None
scene.frame_start, scene.frame_end = 1, int(max(a.frame_range[1] for a in actions))
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format="GLB",
                          export_animation_mode="ACTIONS", export_force_sampling=True,
                          export_anim_single_armature=True, export_frame_range=False,
                          export_apply=False)
print("[done]", OUT_GLB)
