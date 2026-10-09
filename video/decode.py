import glob
import os

import cv2


class FrameReader:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def write_png_sequence(video_path, out_dir, prefix="frame_", max_frames=None, resize=None):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    with FrameReader(video_path) as reader:
        for i, frame in enumerate(reader.frames()):
            if max_frames is not None and i >= max_frames:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, f"{prefix}{i:05d}.png"), frame)
            written += 1
    return written


def list_png_sequence(in_dir):
    return sorted(glob.glob(os.path.join(in_dir, "*.png")))