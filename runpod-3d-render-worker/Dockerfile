# blender-render — headless Blender 4.5 LTS scene compile + deterministic
# reference render. No CUDA extensions to compile here (Blender ships its
# own CUDA/OptiX kernels internally), so this stays on the *runtime* CUDA
# image and installs fast compared to the asset-gpu images.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
        wget xz-utils \
        libx11-6 libxi6 libxrender1 libxxf86vm1 libxfixes3 libxkbcommon0 \
        libgl1 libegl1 libglu1-mesa libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Blender 4.5 is the current LTS (blender.org/download/lts) — pin the patch,
# don't float it; render output must stay reproducible across cold starts
# per spec NFR-01 (renders record exact renderer version).
ARG BLENDER_VERSION=4.5.12
RUN wget -q "https://download.blender.org/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz" -O /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz
ENV PATH="/opt/blender:${PATH}"

# usd-core is Pixar/AOUSD's official pip distribution of OpenUSD's Python
# bindings (`pxr`) -- pure Python-facing wheel, no CUDA/GPU dependency, has
# a cp311 manylinux wheel (confirmed against a real PyPI JSON API query
# against this pin, not guessed). Runs under this image's system python3.11
# (handler.py), not Blender's bundled python -- scene composition/parsing
# happens before the Blender subprocess is ever invoked. Architecture
# decision (implementation plan §02): "OpenUSD Python, in-worker."
RUN pip install --no-cache-dir runpod boto3 usd-core==26.8

COPY src/handler.py /app/handler.py
COPY src/scene_usd.py /app/scene_usd.py
COPY src/blender/render_scene.py /app/blender/render_scene.py

CMD ["python3", "-u", "/app/handler.py"]
