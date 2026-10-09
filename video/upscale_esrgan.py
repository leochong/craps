"""GPU 4K upscaling via Real-ESRGAN (x2plus). RunPod-validated (needs torch + realesrgan)."""

import os

import cv2
import numpy as np


def build_upsampler(model_path, scale=2, tile=256, half=True, device=None):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    return RealESRGANer(scale=scale, model_path=model_path, model=model,
                        tile=tile, tile_pad=10, pre_pad=0, half=half, device=device)


class EsrganUpscaler:
    def __init__(self, model_path, scale=2, tile=256, half=True, device=None):
        self.scale = scale
        self.upsampler = build_upsampler(model_path, scale=scale, tile=tile, half=half, device=device)

    def __call__(self, bgr):
        out, _ = self.upsampler.enhance(bgr, outscale=self.scale)
        return out


def upscale_png_sequence(in_dir, out_dir, model_path, scale=2, tile=256, out_size=None):
    """Real-ESRGAN over a PNG directory. `out_size` overrides final (w,h) if given."""
    from video.decode import list_png_sequence

    up = EsrganUpscaler(model_path, scale=scale, tile=tile)
    os.makedirs(out_dir, exist_ok=True)
    paths = list_png_sequence(in_dir)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        out = up(img)
        if out_size is not None:
            out = cv2.resize(out, out_size, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(p)), out)
    return len(paths)