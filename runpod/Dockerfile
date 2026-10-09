# RunPod GPU image: 1080p/30 -> FILM 4x -> Real-ESRGAN x2 (4K) -> HEVC NVENC
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# The distro ffmpeg must expose NVENC; warn loudly if it does not.
RUN (ffmpeg -hide_banner -encoders 2>/dev/null | grep -q nvenc && echo "NVENC: OK") \
    || echo "WARNING: ffmpeg lacks NVENC - use an NVENC-enabled ffmpeg build"

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# FILM inference additionally needs the repo's own deps (gin-config, etc.);
# clone the repo and follow its requirements.
RUN git clone https://github.com/google-research/frame-interpolation /opt/frame-interpolation || true

WORKDIR /app
COPY . /app

ENV FILM_REPO=/opt/frame-interpolation \
    FILM_MODEL=/opt/frame-interpolation/pretrained_models/film_net/Style/saved_model \
    ESRGAN_MODEL=/opt/weights/RealESRGAN_x2plus.pth

CMD ["bash", "runpod/run_pod.sh"]