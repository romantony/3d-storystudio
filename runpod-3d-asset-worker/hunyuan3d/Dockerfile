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
    HY3DGEN_MODELS=/workspace/models/hy3dgen

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
    && conda create -n hunyuan3d21 python=3.10 \
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
RUN sed -i '/^--extra-index-url/d' /workspace/Hunyuan3D-2.1/requirements.txt \
    && pip install --no-cache-dir -r /workspace/Hunyuan3D-2.1/requirements.txt

RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer \
    && export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    && export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
    && pip install --no-cache-dir -e .

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

ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/hunyuan3d21/lib:${LD_LIBRARY_PATH}"

# Worker glue — not part of upstream Hunyuan3D-2.1.
RUN pip install --no-cache-dir runpod boto3

WORKDIR /workspace/Hunyuan3D-2.1
COPY src/handler.py /workspace/Hunyuan3D-2.1/handler.py

CMD ["python", "-u", "handler.py"]
