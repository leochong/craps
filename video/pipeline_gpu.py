"""RunPod GPU pipeline: 1080p/30 -> FILM (4x) -> Real-ESRGAN x2 (4K) -> HEVC NVENC.

Directory-based (FFmpeg PNG sequences), matching the RunPod plan. Validated on a
GPU pod; NOT runnable on the local CPU-only host (needs torch, realesrgan, FFmpeg
with NVENC, and the FILM model repo/weights).
"""

import os
import shutil

from .decode import write_png_sequence
from .encode import encode_png_sequence
from .interpolate_film import find_output_dir, run_film_cli
from .upscale_esrgan import upscale_png_sequence


def convert_video_gpu(in_path, out_path, *,
                      out_fps=120, out_size=(3840, 2160),
                      work_dir="work",
                      film_repo=None, film_model=None, film_times=2,
                      esrgan_model=None, esrgan_scale=2, esrgan_tile=256,
                      python="python", ffmpeg="ffmpeg", keep_intermediate=False):
    os.makedirs(work_dir, exist_ok=True)

    frames_in = os.path.join(work_dir, "frames_in")
    if os.path.isdir(frames_in):
        shutil.rmtree(frames_in)
    n_src = write_png_sequence(in_path, frames_in, prefix="frame_")

    run_film_cli(frames_in, film_repo, film_model, times_to_interpolate=film_times, python=python)
    interp_dir = find_output_dir(frames_in)

    frames_4k = os.path.join(work_dir, "frames_4k")
    if os.path.isdir(frames_4k):
        shutil.rmtree(frames_4k)
    n_out = upscale_png_sequence(interp_dir, frames_4k, esrgan_model,
                                 scale=esrgan_scale, tile=esrgan_tile, out_size=out_size)

    encode_png_sequence(frames_4k, out_path, out_fps, ffmpeg=ffmpeg)

    if not keep_intermediate:
        shutil.rmtree(frames_in, ignore_errors=True)
        shutil.rmtree(frames_4k, ignore_errors=True)

    return {
        "input": in_path,
        "output": out_path,
        "src_frames": n_src,
        "out_frames": n_out,
        "out_fps": out_fps,
        "out_size": list(out_size),
    }