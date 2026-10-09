# Reproducible RunPod GPU image for the 1080p30 -> 4K120 pipeline.
# GPU: RIFE 4x interpolation + Real-ESRGAN x2 upscaling + NVENC encode.
#
# Build (from repo root):  docker build -f runpod/Dockerfile -t <user>/craps-4k120 .
# Push to a registry RunPod can pull, then set --image <user>/craps-4k120:tag.
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# NVENC needs the 'video' driver capability to be exposed.
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    INTERP=rife \
    RIFE_REPO=/opt/RIFE \
    FFMPEG=/opt/ffmpeg/ffmpeg \
    ESRGAN_MODEL=/opt/weights/RealESRGAN_x2plus.pth

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Modern ffmpeg with NVENC (distro ffmpeg can be too old for new GPUs/drivers).
RUN wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
        -O /tmp/ff.tar.xz \
    && tar --no-same-owner -xf /tmp/ff.tar.xz -C /tmp \
    && mkdir -p /opt/ffmpeg \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /opt/ffmpeg/ffmpeg \
    && cp /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /opt/ffmpeg/ffprobe \
    && chmod +x /opt/ffmpeg/ffmpeg

# numpy<2 (torch 2.1) + pipeline deps.
RUN pip install --no-cache-dir "numpy==1.26.4" opencv-python-headless av realesrgan basicsr gdown

# RIFE: repo + model weights (train_log/flownet.pkl).
RUN git clone --depth 1 https://github.com/hzwer/Practical-RIFE /opt/RIFE \
    && gdown 1gViYvvQrtETBgU1w8axZSsr7YUuw31uy -O /tmp/rife.zip \
    && python -m zipfile -e /tmp/rife.zip /opt/RIFE

# Real-ESRGAN x2 weights.
RUN mkdir -p /opt/weights \
    && wget -q -O /opt/weights/RealESRGAN_x2plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

WORKDIR /app
COPY . /app

# Override the base image entrypoint so our command runs directly.
ENTRYPOINT []
CMD ["bash", "-lc", "cd /app && python runpod/run_gpu.py"]