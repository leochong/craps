"""Pod-side driver: GPU RIFE 4x interpolation + GPU Real-ESRGAN x2 + encoder.

Env: INTERP=rife|cpu, RIFE_REPO, FFMPEG (path to a modern ffmpeg), ESRGAN model path.
"""

import os
import sys

from video.encode import make_encoder
from video.interpolate_cpu import CpuInterpolator
from video.pipeline import convert_video
from video.upscale_esrgan import EsrganUpscaler

clip = sys.argv[1] if len(sys.argv) > 1 else "/workspace/in/clip.mp4"
out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out/out_4k120.mp4"
esrgan = sys.argv[3] if len(sys.argv) > 3 else "/workspace/weights/RealESRGAN_x2plus.pth"

mode = os.environ.get("INTERP", "rife").lower()
rife_repo = os.environ.get("RIFE_REPO", "/workspace/RIFE")
ffmpeg = os.environ.get("FFMPEG") or ("/opt/ffmpeg/ffmpeg" if os.path.exists("/opt/ffmpeg/ffmpeg") else "ffmpeg")

if mode == "rife":
    from video.interpolate_rife import RifeInterpolator
    interp = RifeInterpolator(4, rife_repo)
else:
    interp = CpuInterpolator(4, "dis")

print(f"clip={clip} out={out} interp={mode} ffmpeg={ffmpeg}", flush=True)
upscaler = EsrganUpscaler(esrgan, scale=2, tile=256)
encoder = make_encoder(out, 120, (3840, 2160), prefer_nvenc=True, ffmpeg=ffmpeg)
print("encoder:", type(encoder).__name__, flush=True)
res = convert_video(
    clip, out, out_fps=120, out_size=(3840, 2160),
    interpolator=interp, upscaler=upscaler, encoder=encoder, progress=True,
)
print("RESULT", res, flush=True)