#!/usr/bin/env bash
# Runs INSIDE a RunPod GPU pod: sets up RIFE + Real-ESRGAN + modern ffmpeg,
# then runs the GPU pipeline on /workspace/in/clip.mp4 (uploaded separately).
set -euo pipefail

CLONE_DIR="${CLONE_DIR:-/workspace/craps}"
RIFE_DIR="${RIFE_DIR:-/workspace/RIFE}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/workspace/weights}"
OUT_DIR="${OUT_DIR:-/workspace/out}"
FFMPEG_DIR="${FFMPEG_DIR:-/opt/ffmpeg}"
FFMPEG_BIN="$FFMPEG_DIR/ffmpeg"

mkdir -p /workspace "$WEIGHTS_DIR" "$OUT_DIR" "$FFMPEG_DIR"

echo "== system deps =="
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y ffmpeg
fi

echo "== modern ffmpeg (NVENC) =="
if [ ! -x "$FFMPEG_BIN" ]; then
  wget -q https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
    -O /root/ff.tar.xz
  tar --no-same-owner -xf /root/ff.tar.xz -C /root
  cp /root/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg "$FFMPEG_BIN"
  cp /root/ffmpeg-master-latest-linux64-gpl/bin/ffprobe "$FFMPEG_DIR/ffprobe" || true
  chmod +x "$FFMPEG_BIN"
fi
"$FFMPEG_BIN" -hide_banner -encoders 2>/dev/null | grep -i nvenc | head -3 || echo "no nvenc"

echo "== python deps =="
pip install --no-cache-dir -U pip >/dev/null
pip install --no-cache-dir "numpy==1.26.4" opencv-python-headless av realesrgan basicsr gdown

echo "== pipeline repo =="
[ -d "$CLONE_DIR/.git" ] || git clone --depth 1 "${REPO_URL:?set REPO_URL}" "$CLONE_DIR"

echo "== RIFE =="
[ -d "$RIFE_DIR/.git" ] || git clone --depth 1 https://github.com/hzwer/Practical-RIFE "$RIFE_DIR"
if [ ! -f "$RIFE_DIR/train_log/flownet.pkl" ]; then
  gdown 1gViYvvQrtETBgU1w8axZSsr7YUuw31uy -O /workspace/rife_model.zip
  python -m zipfile -e /workspace/rife_model.zip "$RIFE_DIR"
fi

echo "== Real-ESRGAN x2 weights =="
ESRGAN_MODEL="$WEIGHTS_DIR/RealESRGAN_x2plus.pth"
[ -f "$ESRGAN_MODEL" ] || wget -q -O "$ESRGAN_MODEL" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

echo "== run GPU pipeline (RIFE 4x -> Real-ESRGAN x2 -> encode) =="
[ -f /workspace/in/clip.mp4 ] || { echo "ERROR: /workspace/in/clip.mp4 missing"; exit 1; }
cd "$CLONE_DIR"
INTERP="${INTERP:-rife}" RIFE_REPO="$RIFE_DIR" FFMPEG="$FFMPEG_BIN" \
  PYTHONPATH="$CLONE_DIR" \
  python runpod/run_gpu.py /workspace/in/clip.mp4 "$OUT_DIR/out_4k120.mp4" "$ESRGAN_MODEL"
echo "== DONE =="
ls -lh "$OUT_DIR/out_4k120.mp4"