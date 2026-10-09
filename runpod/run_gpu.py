"""Pod-side driver: GPU Real-ESRGAN x2 + NVENC on a clip (CPU optical-flow 4x interp)."""

import sys

from video.encode import OpenCVEncoder
from video.interpolate_cpu import CpuInterpolator
from video.pipeline import convert_video
from video.upscale_esrgan import EsrganUpscaler

clip = sys.argv[1] if len(sys.argv) > 1 else "/workspace/in/clip.mp4"
out = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out/out_4k120.mp4"
esrgan = sys.argv[3] if len(sys.argv) > 3 else "/workspace/weights/RealESRGAN_x2plus.pth"

print(f"clip={clip} out={out} esrgan={esrgan}", flush=True)
upscaler = EsrganUpscaler(esrgan, scale=2, tile=256)
encoder = OpenCVEncoder(out, 120, (3840, 2160))
res = convert_video(
    clip, out, out_fps=120, out_size=(3840, 2160),
    interpolator=CpuInterpolator(4, "dis"), upscaler=upscaler, encoder=encoder,
    progress=True,
)
print("RESULT", res, flush=True)