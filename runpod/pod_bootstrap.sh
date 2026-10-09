#!/usr/bin/env bash
# Runs INSIDE a RunPod GPU pod. Self-contained: clones the repo, installs FILM +
# Real-ESRGAN, fetches the source clip, slices it, and runs the 4x + 4K pipeline.
#
# Required env: REPO_URL, INPUT_URL, START (s), DURATION (s)
set -euo pipefail

REPO_URL="${REPO_URL:?set REPO_URL}"
INPUT_URL="${INPUT_URL:?set INPUT_URL}"
START="${START:?set START}"
DURATION="${DURATION:?set DURATION}"

CLONE_DIR="${CLONE_DIR:-/workspace/craps}"
FILM_DIR="${FILM_DIR:-/workspace/frame-interpolation}"
WEIGHTS_DIR="${WEIGHTS_DIR:-/workspace/weights}"
OUT_DIR="${OUT_DIR:-/workspace/out}"

mkdir -p /workspace "$WEIGHTS_DIR" "$OUT_DIR"

echo "== clone pipeline =="
[ -d "$CLONE_DIR/.git" ] || git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
cd "$CLONE_DIR"

echo "== system deps =="
if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update -y && apt-get install -y ffmpeg
fi
ffmpeg -hide_banner -encoders 2>/dev/null | grep -i nvenc || echo "WARNING: no NVENC in pod ffmpeg"

echo "== python deps =="
pip install --no-cache-dir -U pip
pip install --no-cache-dir numpy opencv-python-headless av realesrgan basicsr yt-dlp

echo "== FILM =="
[ -d "$FILM_DIR/.git" ] || git clone --depth 1 https://github.com/google-research/frame-interpolation "$FILM_DIR"
( cd "$FILM_DIR" && pip install --no-cache-dir -r requirements.txt || true )
( cd "$FILM_DIR" && bash download_models.sh || true )
FILM_MODEL="${FILM_MODEL:-$FILM_DIR/pretrained_models/film_net/Style/saved_model}"

echo "== Real-ESRGAN x2 weights =="
ESRGAN_MODEL="$WEIGHTS_DIR/RealESRGAN_x2plus.pth"
[ -f "$ESRGAN_MODEL" ] || wget -q -O "$ESRGAN_MODEL" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

echo "== fetch + slice source clip =="
mkdir -p /workspace/in
python -m scripts.download_video "$INPUT_URL" --max-height 1080 --out /workspace/in
SRC="$(ls /workspace/in/*.mp4 | head -n1)"
ffmpeg -y -ss "$START" -i "$SRC" -t "$DURATION" \
  -c:v libx264 -crf 18 -pix_fmt yuv420p -an /workspace/in/clip.mp4

echo "== run GPU pipeline (FILM 4x -> Real-ESRGAN x2 -> NVENC) =="
python -m scripts.convert_video \
  --input /workspace/in/clip.mp4 --output "$OUT_DIR/out_4k120.mp4" --backend film \
  --out-fps 120 --width 3840 --height 2160 --work-dir /workspace/work \
  --film-repo "$FILM_DIR" --film-model "$FILM_MODEL" --film-times 2 \
  --esrgan-model "$ESRGAN_MODEL" --esrgan-tile 256

echo "== DONE =="
ls -lh "$OUT_DIR/out_4k120.mp4"