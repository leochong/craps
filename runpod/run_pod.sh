#!/usr/bin/env bash
# RunPod Pod entrypoint (run inside tmux so it survives SSH disconnects).
#   INPUT=/data/clip_1080p30.mp4 OUTPUT=/data/clip_4k120.mp4 bash runpod/run_pod.sh
set -euo pipefail

: "${INPUT:?set INPUT=/path/to/input_1080p30.mp4}"
: "${OUTPUT:?set OUTPUT=/path/to/output_4k120.mp4}"

python -m scripts.convert_video \
  --input "$INPUT" --output "$OUTPUT" --backend film \
  --out-fps 120 --width 3840 --height 2160 \
  --work-dir "${WORK_DIR:-/workspace/work}" \
  --film-repo "${FILM_REPO:-/opt/frame-interpolation}" \
  --film-model "${FILM_MODEL:-/opt/frame-interpolation/pretrained_models/film_net/Style/saved_model}" \
  --film-times "${FILM_TIMES:-2}" \
  --esrgan-model "${ESRGAN_MODEL:-/opt/weights/RealESRGAN_x2plus.pth}" \
  --esrgan-tile "${ESRGAN_TILE:-256}"
