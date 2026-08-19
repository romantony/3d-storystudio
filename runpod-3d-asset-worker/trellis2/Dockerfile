# asset-gpu / trellis2 — image-to-3D via microsoft/TRELLIS.2 (MIT license).
# Needs the CUDA *devel* image (not runtime) because nvdiffrast/nvdiffrec/
# CuMesh/FlexGEMM/flash-attn are compiled from source against nvcc at build
# time — see https://github.com/microsoft/TRELLIS.2 setup.sh, which this
# Dockerfile mirrors without the conda env (the container is already the
# isolation boundary).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf-cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache \
    MODEL_BASE=/runpod-volume \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=flash-attn \
    MAX_JOBS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        git wget ninja-build cmake gcc g++ \
        libjpeg-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

# Pinned to TRELLIS.2's documented combo (torch 2.6.0 / cu124). Do not float
# this — flash-attn/nvdiffrast below are compiled against this exact ABI.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Pin TRELLIS2_COMMIT to the SHA benchmarked in the Phase 0 spike once
# chosen — floating "main" is only acceptable for the initial bake-off.
ARG TRELLIS2_COMMIT=main
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
    && cd /app/TRELLIS.2 \
    && git checkout ${TRELLIS2_COMMIT} \
    && git submodule update --init --recursive

WORKDIR /app/TRELLIS.2

# ---- setup.sh --basic, without conda ----
RUN pip install --no-cache-dir \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers tensorboard pandas lpips zstandard kornia timm \
    && pip install --no-cache-dir \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    && pip install --no-cache-dir pillow-simd

# ---- setup.sh --flash-attn ----
# MAX_JOBS caps ninja's parallel nvcc compiles — flash-attn's default
# parallelism wants ~2GB RAM per job and will OOM-kill the build on
# constrained CI/build hosts otherwise; trades build time for reliability.
RUN pip install --no-cache-dir flash-attn==2.7.3

# ---- setup.sh --nvdiffrast ----
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/ext/nvdiffrast \
    && pip install --no-cache-dir --no-build-isolation /tmp/ext/nvdiffrast \
    && rm -rf /tmp/ext/nvdiffrast

# ---- setup.sh --nvdiffrec ----
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/ext/nvdiffrec \
    && pip install --no-cache-dir --no-build-isolation /tmp/ext/nvdiffrec \
    && rm -rf /tmp/ext/nvdiffrec

# ---- setup.sh --cumesh / --flexgemm ----
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/ext/CuMesh \
    && pip install --no-cache-dir --no-build-isolation /tmp/ext/CuMesh \
    && rm -rf /tmp/ext/CuMesh
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/ext/FlexGEMM \
    && pip install --no-cache-dir --no-build-isolation /tmp/ext/FlexGEMM \
    && rm -rf /tmp/ext/FlexGEMM

# ---- setup.sh --o-voxel (vendored inside the repo itself, per upstream) ----
RUN pip install --no-cache-dir --no-build-isolation ./o-voxel

# Worker glue — not part of upstream TRELLIS.2.
RUN pip install --no-cache-dir runpod boto3 Pillow numpy

# handler.py must live inside the repo dir, not /app: `from trellis2.pipelines
# import ...` only resolves because the `trellis2` package sits next to
# whatever script Python was invoked on — sys.path[0] is the *script's own*
# directory, not the process cwd. Confirmed by actually running the built
# image; a sibling /app/handler.py fails with ModuleNotFoundError.
COPY src/handler.py /app/TRELLIS.2/handler.py

CMD ["python3", "-u", "handler.py"]
