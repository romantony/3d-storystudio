"""bpy_server.py launcher for the dedicated Python 3.13 / bpy 5.2.3 env.

Upstream's `bpy_server.py` imports `src.server.spec`, which at module top
pulls in torch *and* the whole TokenRig model (transformers, lightning,
...) just for `object_to_bytes`/`bytes_to_object` (torch.save/torch.load)
and two port constants -- none of which the bpy side needs. This env only
has bpy, numpy/scipy/trimesh (what `src.rig_package` needs) and CPU torch,
so pre-register a stand-in `src.server.spec` with the identical wire format
(byte-for-byte the upstream implementations) before importing upstream's
real server. The main process (torch 2.7.0 / py3.11) keeps using upstream's
own spec.py, so both ends speak the same torch.save pickles.

Run from /workspace/SkinTokens (like upstream's `python bpy_server.py`) so
`src` resolves.
"""
import io
import sys
import types

import torch

BPY_PORT = 59876


def object_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def bytes_to_object(b, map_location=None):
    return torch.load(io.BytesIO(b), weights_only=False, map_location=map_location)


_spec = types.ModuleType("src.server.spec")
_spec.BPY_PORT = BPY_PORT
_spec.BPY_SERVER = f"http://localhost:{BPY_PORT}"
_spec.object_to_bytes = object_to_bytes
_spec.bytes_to_object = bytes_to_object
sys.modules["src.server.spec"] = _spec

from src.server.bpy_server import run  # noqa: E402

if __name__ == "__main__":
    import bpy  # noqa: E402
    print(f"[bpy_server_py313] Blender {bpy.app.version_string}, python {sys.version.split()[0]}", flush=True)
    run()
