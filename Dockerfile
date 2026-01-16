# CUDA runtime + Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # PyVista / VTK offscreen hints
    PYVISTA_OFF_SCREEN=true \
    VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=1 \
    DISPLAY=:99

WORKDIR /app

# ---- System deps (VTK/PyVista offscreen + fonts) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git ca-certificates curl \
    # OpenGL / X11 runtime libs
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libglu1-mesa mesa-utils \
    # offscreen
    xvfb \
    # fonts (for DejaVuSans.ttf in Pillow)
    fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps ----
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /app/requirements.txt

# ---- App code ----
COPY . /app

EXPOSE 8010

# 启动 Xvfb 后再启动 uvicorn
CMD ["bash", "-lc", "Xvfb :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & uvicorn http_code:app --host 0.0.0.0 --port 8010"]
