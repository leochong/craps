import os
import shutil
import sys

import cv2

from video.decode import FrameReader
from video.pipeline import convert_video

SRC = "IMG_2111.MOV"
WORK = "out/video"


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def make_proxy(src, path, seconds=1.0, size=(1920, 1080), stride=4):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, size)
    max_n = int(seconds * fps / stride)
    n = 0
    i = 0
    while n < max_n:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            n += 1
        i += 1
    writer.release()
    cap.release()
    return n


def bench_pipeline():
    os.makedirs(WORK, exist_ok=True)
    proxy = os.path.join(WORK, "proxy_1080p30.mp4")
    n_src = make_proxy(SRC, proxy, seconds=1.0)
    src = FrameReader(proxy)
    expected = (n_src - 1) * 4 + 1 if n_src > 0 else 0
    check("proxy built (1080p30)", n_src > 0 and (src.width, src.height) == (1920, 1080),
          f"{n_src} frames @ {src.fps:.1f}fps {src.width}x{src.height}")
    src.release()

    out = os.path.join(WORK, "out_4k120.mp4")
    res = convert_video(proxy, out, out_fps=120, out_size=(3840, 2160),
                        max_seconds=1.0, prefer_nvenc=False)

    cap = cv2.VideoCapture(out)
    ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ofps = cap.get(cv2.CAP_PROP_FPS)
    oframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    ok_size = (ow, oh) == (3840, 2160)
    ok_fps = abs(ofps - 120.0) < 0.5
    ok_count = abs(res["frames"] - expected) <= 4
    ok_file = os.path.getsize(out) > 0
    return check("4K/120 conversion", ok_size and ok_fps and ok_count and ok_file,
                 f"size={ow}x{oh} fps={ofps:.1f} frames={res['frames']} "
                 f"(expected~{expected}) bytes={os.path.getsize(out) if ok_file else 0}")


def main():
    if not os.path.exists(SRC):
        print(f"SKIP: {SRC} not found")
        sys.exit(0)
    results = [bench_pipeline()]
    print(f"\n{sum(results)}/{len(results)} video benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()