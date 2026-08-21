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
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    MODEL_BASE=/workspace \
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

# nvdiffrast's pyproject.toml declares its package name via PEP 621
# ([project] table), which needs setuptools>=64 to read. --no-build-isolation
# below means pip builds every extension with whatever setuptools is already
# here rather than what each package's build-system.requires asks for --
# Ubuntu 22.04's stock setuptools is too old, silently can't see the name,
# and pip installs the wheel as a package literally called "UNKNOWN" instead
# of "nvdiffrast" (confirmed in a real build log: "Successfully installed
# UNKNOWN-0.0.0" -- the compile succeeded, the package identity didn't, so
# `import nvdiffrast` fails at runtime despite the build going green).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pinned to TRELLIS.2's documented combo (torch 2.6.0 / cu124). Do not float
# this — flash-attn/nvdiffrast below are compiled against this exact ABI.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Pinned to the exact commit this Dockerfile's build/runtime fixes were
# verified against (main HEAD as of the last successful build, 2026-08-20 --
# main had been static since 2026-06-05, so this is also current upstream).
# Bump deliberately, not by letting "main" float, if upstream ever changes.
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git /app/TRELLIS.2 \
    && cd /app/TRELLIS.2 \
    && git checkout ${TRELLIS2_COMMIT} \
    && git submodule update --init --recursive

WORKDIR /app/TRELLIS.2

# trellis2/modules/image_feature_extractor.py's DinoV3FeatureExtractor does
# `enumerate(self.model.layer)`, assuming the transformer blocks sit
# directly on the DINOv3ViTModel instance. In the transformers version
# this image installs (unpinned, floats to latest), DINOv3ViTModel wraps
# them one level deeper: DINOv3ViTModel.model is a DINOv3ViTEncoder, and
# .layer lives on *that* -- confirmed against transformers' current
# modeling_dinov3_vit.py and against a real run: `self.model.embeddings`/
# `self.model.rope_embeddings` (siblings of DINOv3ViTModel.model) resolve
# fine, only `self.model.layer` 404s with AttributeError. Patching the
# vendored source at build time since neither this repo's commit nor
# transformers' version is pinned yet (still in the Phase 0 bake-off).
RUN sed -i 's/self\.model\.layer)/self.model.model.layer)/' \
    trellis2/modules/image_feature_extractor.py

# ---- setup.sh --basic, without conda ----
# plyfile isn't in upstream's --basic list but IS a real runtime dependency
# of at least one SC-VAE model class (SparseUnetVaeDecoder /
# FlexiDualGridVaeDecoder) -- without it, base.py's from_pretrained()
# catches the resulting ModuleNotFoundError in a bare `except Exception`
# and silently falls back to a code path with its own bug (an unqualified
# model path gets mis-split into a bogus HF repo_id). Confirmed against a
# live crash log + the actual base.py/models/__init__.py source.
RUN pip install --no-cache-dir \
        imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
        trimesh transformers tensorboard pandas lpips zstandard kornia timm \
        plyfile \
    && pip install --no-cache-dir \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
    && pip install --no-cache-dir pillow-simd

# pillow-simd (just installed, from upstream's --basic list) is an
# abandoned fork (no release since ~2021) whose compiled _webp C extension
# predates Pillow's HAVE_WEBPANIM addition. handler.py's glb.export(...,
# extension_webp=True) triggers trimesh -> PIL.Image.save(format="WEBP"),
# which imports PIL.WebPImagePlugin, which does `if _webp.HAVE_WEBPANIM`
# at module level -- AttributeError on pillow-simd's older _webp, only
# surfacing at actual export time deep in the pipeline. Force-reinstalling
# stock Pillow after it overwrites the same PIL/ import path with a build
# that has the attribute. Confirmed against a real crash log.
RUN pip install --no-cache-dir --force-reinstall Pillow

# ---- setup.sh --flash-attn ----
# MAX_JOBS caps ninja's parallel nvcc compiles — flash-attn's default
# parallelism wants ~2GB RAM per job and will OOM-kill the build on
# constrained CI/build hosts otherwise; trades build time for reliability.
# --no-build-isolation: flash-attn's setup.py does `import torch` at build
# time to detect the CUDA/ABI to compile against. Default pip build
# isolation builds in a fresh env that can't see the torch already
# installed above, so the build fails with ModuleNotFoundError: no module
# named 'torch' even though torch is installed in the image.
RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.7.3

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
