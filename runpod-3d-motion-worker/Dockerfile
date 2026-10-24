# motion-gpu — text-to-motion via Tencent-Hunyuan/HY-Motion-1.0 (1.0B DiT
# flow-matching model, SMPL-H skeleton output).
#
# LICENSE FLAG: tencent-hunyuan-community license, same family as
# Hunyuan3D-2.1 (see ../runpod-3d-asset-worker/hunyuan3d/Dockerfile) — it
# carries territorial exclusions. Complete a legal review before this
# image serves production traffic.
#
# Separate image from the other 3D workers: torch 2.5.1/cu124 matches
# Hunyuan3D, but this one pulls transformers 4.53.3 + bitsandbytes +
# the proprietary-ish Autodesk FBX SDK wheel (fbxsdkpy) from an INRIA
# GitLab index, none of which should leak into the other pinned envs.
#
# Plain runtime CUDA image + Ubuntu 22.04's Python 3.10: every dependency
# here ships a prebuilt wheel (fbxsdkpy has cp310 manylinux), so no nvcc,
# conda or source compiles are needed. Code lives in /app, NOT /workspace,
# so a network volume mounted at /workspace can't shadow it; only weights
# (HF_HOME / HY_MOTION_CKPTS) go on the volume.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HF_HOME only: transformers' legacy TRANSFORMERS_CACHE would point it at
    # hf-cache/ while huggingface_hub (download_model.py) writes to hf-cache/hub/,
    # so the worker would miss the pre-warmed weights and re-download ~16GB.
    HF_HOME=/workspace/hf-cache \
    HY_MOTION_CKPTS=/workspace/models/hy-motion \
    USE_HF_MODELS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git git-lfs ca-certificates \
        libgl1 libglib2.0-0 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Pinned to the commit this Dockerfile was written against (master HEAD as
# of 2026-10-24). Bump deliberately, not by letting master float.
# `git lfs pull` is required: the FBX exporter's template asset is an LFS
# object, and without it fbx export silently falls back to "dict" format.
ARG HYMOTION_COMMIT=4e426f5a1021cbcf7f375458c37b840ee7225229
RUN git clone https://github.com/Tencent-Hunyuan/HY-Motion-1.0.git /app \
    && cd /app && git checkout ${HYMOTION_COMMIT} && git lfs pull

WORKDIR /app

# requirements.txt re-pins torch==2.5.1/torchvision==0.20.1 (already
# satisfied above, so the cu124 build is kept) and adds an extra index for
# fbxsdkpy; PyPI stays primary for everything else.
RUN pip install -r requirements.txt

# Worker glue — not part of upstream. setuptools<82 for the same
# pkg_resources reason as the hunyuan3d image.
RUN pip install runpod boto3 "setuptools<82"

COPY src/handler.py /app/handler.py

# Weights are fetched on first cold start into the network volume (see
# scripts/download_model.py to pre-warm). ~16GB Qwen3-8B + ~1.7GB CLIP +
# the DiT checkpoint; baking them in would make the image ~25GB+.
CMD ["python", "-u", "handler.py"]
