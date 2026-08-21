# asset-gpu / hunyuan3d — image-to-3D via Tencent-Hunyuan/Hunyuan3D-2.1.
#
# LICENSE FLAG: this model ships under the Tencent Hunyuan 3D 2.1 Community
# License, not MIT, and that license explicitly excludes the EU, UK and
# South Korea from its grant. Populate threeDProviderConfigs.licenseProfileId
# with a completed legal review before this image serves production traffic
# — see spec §21.2. TRELLIS.2 (the sibling image in this repo) is MIT.
#
# This Dockerfile mirrors Hunyuan3D-2.1's own docker/Dockerfile as closely
# as possible (including the conda toolchain) rather than simplifying it —
# the custom_rasterizer / DifferentiableRenderer compile steps are known
# finicky against ABI drift, and the upstream recipe is the one that's
# actually been verified to build.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# HY3DGEN_MODELS matters as much as HF_HOME here: the shape pipeline
# (hy3dshape's smart_load_model) does NOT use standard HF caching -- it
# builds its own local dir from os.environ.get('HY3DGEN_MODELS',
# '~/.cache/hy3dgen') and passes that as snapshot_download's local_dir=,
# bypassing HF_HOME entirely. Confirmed against the actual upstream
# source (hy3dshape/hy3dshape/utils/utils.py). Without this set, the
# shape checkpoint re-downloads on every cold start regardless of the
# network volume, even though HF_HOME is already pointed at it. The
# paint pipeline's multiview weights and facebook/dinov2-giant DO use
# plain HF caching and are already covered by HF_HOME.
#
# HF_MODULES_CACHE is set explicitly (rather than left to its diffusers
# default of HF_HOME/modules) to keep it OFF the network volume. It's
# where diffusers' custom_pipeline loader (the paint pipeline's
# multiviewDiffusionNet) extracts Python source at runtime -- into a
# fixed, non-namespaced diffusers_modules/local/ path, the same for every
# repo it's ever asked to load code from. That's fine as an ephemeral
# per-process scratch dir; it's not something that should persist across
# cold starts or be shared with whatever else writes to this volume, so
# route it to the container's own /tmp instead of the volume HF_HOME sits on.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYOPENGL_PLATFORM=egl \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    MODEL_BASE=/workspace \
    HY3DGEN_MODELS=/workspace/models/hy3dgen \
    HF_MODULES_CACHE=/tmp/hf-modules-cache

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip git-lfs \
        pkg-config libglvnd0 libgl1 libglx0 libegl1 libgles2 libglvnd-dev \
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev cmake curl \
        mesa-utils-extra libxrender1 libeigen3-dev python3-dev python3-setuptools \
        libcgal-dev libxi6 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /workspace/miniconda3 \
    && rm /tmp/miniconda.sh
ENV PATH="/workspace/miniconda3/bin:${PATH}"
RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --channel https://repo.anaconda.com/pkgs/r \
    && conda create -n hunyuan3d21 python=3.11 \
    && conda config --set always_yes true
ENV PATH="/workspace/miniconda3/envs/hunyuan3d21/bin:${PATH}"

RUN conda install Ninja \
    && conda install cuda -c nvidia/label/cuda-12.4.1 \
    && conda install -c conda-forge libstdcxx-ng

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Pin HUNYUAN3D_COMMIT to the SHA benchmarked in the Phase 0 spike once
# chosen — floating "main" is only acceptable for the initial bake-off.
ARG HUNYUAN3D_COMMIT=main
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /workspace/Hunyuan3D-2.1 \
    && cd /workspace/Hunyuan3D-2.1 && git checkout ${HUNYUAN3D_COMMIT}

# Strip upstream's --extra-index-url lines (Tencent/Aliyun mirrors) before
# installing: they're reachable from outside China but extremely slow, and
# pip queries every extra index for every package. On GitHub-hosted runners
# this stalled the build for ~2h before pip finally errored out. PyPI proper
# has everything these pin.
#
# bpy==4.0 has never had a PyPI wheel for any Python version (bpy wheels
# start at 4.2.0, cp311+ only) -- that's also why the conda env above is
# 3.11, not upstream's 3.10. bpy is a hard runtime dependency here: the
# paint pipeline (textureGenPipeline.py -> DifferentiableRenderer/mesh_utils.py)
# does `import bpy` at module load, it's not an optional render-tool extra.
RUN sed -i \
        -e '/^--extra-index-url/d' \
        -e 's/^bpy==4\.0$/bpy==4.2.0/' \
        /workspace/Hunyuan3D-2.1/requirements.txt \
    && pip install --no-cache-dir -r /workspace/Hunyuan3D-2.1/requirements.txt

# basicsr 1.4.2 (a realesrgan dependency, used by the paint pipeline's
# super-resolution step) imports from the private torchvision.transforms.
# functional_tensor module, which was deprecated in torchvision 0.15 and
# removed entirely in 0.17+ -- we're on 0.20.1. rgb_to_grayscale has always
# also been in the public torchvision.transforms.functional module with the
# same signature; this is the standard fix for this well-known basicsr/
# Real-ESRGAN incompatibility. Confirmed only one occurrence in basicsr 1.4.2.
RUN sed -i \
        's/from torchvision\.transforms\.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
        /workspace/miniconda3/envs/hunyuan3d21/lib/python3.11/site-packages/basicsr/data/degradations.py

# custom_rasterizer's setup.py does `import torch` directly and ships no
# pyproject.toml, but pip still build-isolates it by default (PEP 517, via
# the implicit setuptools legacy backend) into a venv that only has
# setuptools/wheel -- not the torch already installed in this env. Without
# --no-build-isolation it fails with "ModuleNotFoundError: No module named
# 'torch'" before ever reaching the CUDA compile.
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer \
    && export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    && export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
    && pip install --no-cache-dir --no-build-isolation -e .

RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer \
    && bash compile_mesh_painter.sh

RUN mkdir -p /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt \
    && wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
        -P /workspace/Hunyuan3D-2.1/hy3dpaint/ckpt

# Same two upstream config-path fixes the official Dockerfile applies.
RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint \
    && sed -i 's/self\.multiview_cfg_path = "cfgs\/hunyuan-paint-pbr\.yaml"/self.multiview_cfg_path = "hy3dpaint\/cfgs\/hunyuan-paint-pbr.yaml"/' textureGenPipeline.py \
    && cd utils \
    && sed -i 's/custom_pipeline = config\.custom_pipeline/custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")/' multiview_utils.py

# convert_obj_to_glb wraps everything in a bare `except Exception: return
# False`, and its only caller (textureGenPipeline.py) ignores the return
# value -- so on failure the pipeline silently keeps handing back the .obj
# path and _export_mesh (handler.py) uploads that, mislabeled as .glb, with
# zero diagnostics anywhere. Confirmed against a real job: the uploaded
# "textured.glb" was actually Wavefront OBJ, but a local repro of this exact
# function (same bpy==4.2.0, same import order with torch loaded first)
# succeeded cleanly against both a trivial mesh and the real downloaded OBJ
# -- so whatever's failing in the container specifically needs the actual
# traceback, not another guess.
RUN sed -i '/except Exception:/a\        import traceback; traceback.print_exc()' \
    /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer/mesh_utils.py

ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"

# Worker glue — not part of upstream Hunyuan3D-2.1.
RUN pip install --no-cache-dir runpod boto3

# pytorch_lightning's lightning_fabric (pulled in by hunyuanpaintpbr/unet/
# model.py's `import pytorch_lightning`) does `__import__("pkg_resources")
# .declare_namespace(...)` at import time -- a legacy namespace-package
# pattern. pkg_resources was removed from setuptools starting at 82.0.0
# (still present in 81.0.0); nothing in requirements.txt pins setuptools,
# so the earlier pip installs above pull whatever's latest and silently
# lose it. Confirmed against a real job traceback (ModuleNotFoundError:
# No module named 'pkg_resources') and against the actual setuptools
# wheels on PyPI (81.0.0 bundles pkg_resources/, 82.0.0 doesn't). Kept as
# the last pip step so nothing after this re-upgrades it.
RUN pip install --no-cache-dir "setuptools<82"

WORKDIR /workspace/Hunyuan3D-2.1
COPY src/handler.py /workspace/Hunyuan3D-2.1/handler.py

CMD ["python", "-u", "handler.py"]
