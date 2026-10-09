from .decode import FrameReader
from .encode import make_encoder
from .interpolate_cpu import CpuInterpolator
from .upscale_cpu import CpuUpscaler


def convert_video(in_path, out_path, out_fps=120, out_size=(3840, 2160),
                  interpolator=None, upscaler=None, encoder=None,
                  max_seconds=None, prefer_nvenc=True, progress=False):
    """Stream 1080p/30 -> (interpolate) -> (upscale) -> 4K/120.

    pluggable interpolator/upscaler/encoder; defaults are the CPU fallbacks.
    """
    reader = FrameReader(in_path)
    factor = max(1, int(round(out_fps / reader.fps)))

    interpolator = interpolator or CpuInterpolator(factor)
    upscaler = upscaler or CpuUpscaler(out_size)
    encoder = encoder or make_encoder(out_path, out_fps, out_size, prefer_nvenc=prefer_nvenc)

    limit = None if max_seconds is None else int(round(max_seconds * reader.fps))

    def source():
        for i, frame in enumerate(reader.frames()):
            if limit is not None and i >= limit:
                break
            yield frame

    count = 0
    try:
        for frame in interpolator.stream(source()):
            encoder.write(upscaler(frame))
            count += 1
            if progress and count % 24 == 0:
                print(f"  wrote {count} frames", flush=True)
    finally:
        encoder.release()
        reader.release()

    return {
        "input": in_path,
        "output": out_path,
        "src_fps": reader.fps,
        "out_fps": out_fps,
        "factor": factor,
        "out_size": list(out_size),
        "frames": count,
    }