# rigging-gpu — automatic character rigging via VAST-AI-Research/SkinTokens
# (TokenRig), MIT license. Spec §28's "Rigging provider" architecture
# decision names this a separate Phase 2 spike -- SkinTokens (not its
# predecessor UniRig) was chosen after comparing both against real source:
# same team, claims 98-133% better skinning accuracy / 17-22% better bone
# prediction than UniRig, a single unified GLB-in/GLB-out call instead of
# UniRig's three-stage skeleton/skin/merge pipeline, and its recommended
# checkpoint (trained on ArticulationXL2.0) is actually downloadable today
# rather than "planned future" like UniRig's best checkpoint.
#
# Separate endpoint from the other three 3D workers, matching runpod-3d-
# asset-worker/README.md's "Why two images instead of one": torch 2.7.0/
# cu128 + its own flash-attn build + bpy>=4.2 is a different, incompatible
# pinned toolchain from TRELLIS.2 (torch 2.6.0/cu124), Hunyuan3D (torch
# 2.5.1/cu124), and the Blender render worker (real Blender 4.5, not the
# bpy pip package).
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MAX_JOBS=2 \
    HF_HOME=/tmp/hf-cache \
    TRANSFORMERS_CACHE=/tmp/hf-cache

WORKDIR /workspace

# The first real build's live test failed with bpy_server.py (the
# background subprocess doing all mesh import/export via `bpy`) never
# coming up -- confirmed via a live job's ConnectionRefusedError on its
# localhost port, not a timing issue (60s is generous for a bottle HTTP
# server). `bpy`'s compiled extension needs real X11/GL client libs even
# for headless mesh-only operations, not just when actually rendering --
# this repo's hunyuan3d/Dockerfile already hit and solved the identical
# class of problem for its own bpy usage; matching that proven library
# set here rather than the thinner one this Dockerfile started with.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget cmake ninja-build \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
        libxi6 libxkbcommon-x11-0 libxrender-dev libxfixes3 libxxf86vm1 \
        libgl1-mesa-dev libegl1-mesa-dev libglvnd0 libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# bpy>=4.2 has no PyPI wheel below cp311 (confirmed the same way for the
# hunyuan3d worker's Dockerfile in this repo -- bpy wheels start at 4.2.0,
# cp311+ only). Ubuntu 22.04's apt python3 is 3.10, so this reuses that
# Dockerfile's exact fix: a conda env pinned to 3.11 rather than a
# deadsnakes PPA or building CPython from source.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /workspace/miniconda3 \
    && rm /tmp/miniconda.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n skintokens python=3.11 \
    && conda config --set always_yes true
ENV PATH="/workspace/miniconda3/envs/skintokens/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pinned to SkinTokens' documented combo (torch 2.7.0 / cu128). Do not
# float this -- flash-attn below is compiled against this exact ABI.
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Pinned to the commit this Dockerfile was written and verified against
# (main HEAD as of 2026-08-23). Bump deliberately, not by letting "main"
# float, if upstream ever changes.
ARG SKINTOKENS_COMMIT=273b691d35989d71cd17ff2895fdc735097b92d1
RUN git clone https://github.com/VAST-AI-Research/SkinTokens.git /workspace/SkinTokens \
    && cd /workspace/SkinTokens && git checkout ${SKINTOKENS_COMMIT}

WORKDIR /workspace/SkinTokens

RUN pip install --no-cache-dir -r requirements.txt

# --no-build-isolation: flash-attn's setup.py does `import torch` at build
# time to detect the CUDA/ABI to compile against -- default pip build
# isolation can't see the torch already installed above (same fix already
# proven in this repo's trellis2/Dockerfile). No version pin here: unlike
# trellis2 (which pins to a specific upstream-documented combo), SkinTokens'
# own README doesn't name a tested flash-attn version for torch 2.7.0/
# cu128, so this follows upstream's own untested `pip install flash-attn`
# instruction rather than guessing a pin that might not have a prebuilt
# wheel for this exact torch/cuda/python combo. Revisit with a real pin
# once a build has actually gone green.
RUN pip install --no-cache-dir --no-build-isolation flash-attn

# Model weights (~1.6GB total: the TokenRig checkpoint + FSQ-CVAE tokenizer
# + Qwen3-0.6B's config/tokenizer, per download.py -- the LLM's actual
# weights are NOT pulled here, since get_model() below loads them from the
# fine-tuned grpo_1400.ckpt instead) are small enough to bake directly into
# the image at build time -- unlike TRELLIS.2/Hunyuan3D's ~15GB checkpoints,
# this doesn't need a network-volume pre-warm step (and sidesteps that
# whole Pod-vs-Serverless-mount-path bug class entirely).
RUN python download.py --model

# Worker glue -- not part of upstream SkinTokens.
RUN pip install --no-cache-dir runpod boto3

# src/rig_package/info/asset.py imports scipy, but upstream's requirements.txt
# doesn't list it -- it only arrived transitively, and an unpinned rebuild
# dropped it (workers crashed on import). Install it explicitly.
RUN pip install --no-cache-dir scipy

# Dedicated Python 3.13 env for bpy_server.py only (placed after the heavy
# torch/flash-attn/weights layers so editing it doesn't bust their build cache). bpy 5.1+ ships cp313
# wheels only, and this image's main env is pinned to Python 3.11 by
# torch 2.7.0 / flash-attn, so `bpy>=4.2` in SkinTokens' requirements.txt
# resolved to 5.0.1 (the newest cp311 wheel; confirmed from a live job's GLB
# exporter string, "Khronos glTF Blender I/O v5.0.21"). demo.start_bpy_server()
# already runs bpy as a separate subprocess over localhost HTTP, so just
# launch it with this interpreter instead (handler.py reads BPY_SERVER_PYTHON;
# src/bpy_server_py313.py explains why CPU torch is here and the TokenRig
# model code is not). Pinned, not floated: output GLBs record the exporter
# version, same reproducibility reasoning as the render worker's Blender pin.
# Verified locally on CPU: SkinTokens' own load/export/transfer round-trip
# through the real HTTP protocol with a py3.11/numpy-1.26 client against this
# env (bpy 5.2.3, py3.13, numpy 2.5).
ARG BPY_SERVER_BPY_VERSION=5.2.3
RUN conda create -y -n bpy313 python=3.13 \
    && /workspace/miniconda3/envs/bpy313/bin/pip install --no-cache-dir \
        bpy==${BPY_SERVER_BPY_VERSION} numpy scipy trimesh bottle tornado requests \
    && /workspace/miniconda3/envs/bpy313/bin/pip install --no-cache-dir \
        torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple
ENV BPY_SERVER_PYTHON=/workspace/miniconda3/envs/bpy313/bin/python

# handler.py must live inside the repo dir: demo.py's model checkpoint path
# (experiments/.../grpo_1400.ckpt) and bpy_server.py are referenced as
# relative paths from the process cwd, and `from demo import run_rig`
# needs demo.py on sys.path -- same constraint already solved for trellis2/
# hunyuan3d's handler.py placement in their own Dockerfiles.
COPY src/handler.py /workspace/SkinTokens/handler.py
COPY src/bpy_server_py313.py /workspace/SkinTokens/bpy_server_py313.py

CMD ["python", "-u", "handler.py"]
