import argparse

from video.decode import FrameReader
from video.pipeline import convert_video


def main():
    ap = argparse.ArgumentParser(description="1080p/30 -> (interpolate) -> (upscale) -> 4K/120")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=["cpu", "film"], default="cpu")
    ap.add_argument("--out-fps", type=int, default=120)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--method", choices=["dis", "farneback"], default="dis")
    ap.add_argument("--no-nvenc", action="store_true")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--film-repo", default=None)
    ap.add_argument("--film-model", default=None)
    ap.add_argument("--film-times", type=int, default=2)
    ap.add_argument("--esrgan-model", default=None)
    ap.add_argument("--esrgan-tile", type=int, default=256)
    args = ap.parse_args()

    size = (args.width, args.height)

    if args.backend == "cpu":
        from video.interpolate_cpu import CpuInterpolator
        from video.upscale_cpu import CpuUpscaler

        with FrameReader(args.input) as reader:
            factor = max(1, int(round(args.out_fps / reader.fps)))
        res = convert_video(
            args.input, args.output, out_fps=args.out_fps, out_size=size,
            interpolator=CpuInterpolator(factor, args.method),
            upscaler=CpuUpscaler(size),
            max_seconds=args.max_seconds, prefer_nvenc=not args.no_nvenc, progress=True,
        )
    else:
        from video.pipeline_gpu import convert_video_gpu

        res = convert_video_gpu(
            args.input, args.output, out_fps=args.out_fps, out_size=size,
            work_dir=args.work_dir, film_repo=args.film_repo, film_model=args.film_model,
            film_times=args.film_times, esrgan_model=args.esrgan_model, esrgan_tile=args.esrgan_tile,
        )

    print(res)


if __name__ == "__main__":
    main()