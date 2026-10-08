import json
import os

import cv2

from .frame_grab import FrameSource


def extract_clip(path, start_frame, end_frame, out_path, step=1, target_size=None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = None
    count = 0
    with FrameSource(path, target_size=target_size) as src:
        for idx, frame in src.frames(start=start_frame, end=end_frame, step=step):
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, src.fps / step, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            count += 1
    if writer is not None:
        writer.release()
    return count


def save_annotations(path, entries):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)


def load_annotations(path):
    with open(path) as f:
        return json.load(f)