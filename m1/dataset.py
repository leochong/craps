"""Dataset ingestion + manifest for CV training assets.

Videos are registered in a JSONL manifest (metadata + provenance + content hash);
training code samples frames from the referenced video via `frame_iter`. Synthetic
assets (interpolated/upscaled) are flagged and must not be used as label ground truth.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import cv2

MANIFEST_NAME = "manifest.jsonl"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = round(info["frames"] / info["fps"], 3) if info["fps"] else None
    return info


def frame_iter(path, stride=1, max_frames=None, size=None):
    cap = cv2.VideoCapture(path)
    i = 0
    taken = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            if size is not None:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            yield i, frame
            taken += 1
            if max_frames is not None and taken >= max_frames:
                break
        i += 1
    cap.release()


class VideoDataset:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, MANIFEST_NAME)

    def register(self, video_path, asset_id, provenance=None, extract_frames=0,
                 frame_size=None, frame_stride=1):
        info = probe_video(video_path)
        entry = {
            "id": asset_id,
            "source_path": os.path.abspath(video_path),
            "sha256": sha256_file(video_path),
            "fps": info["fps"],
            "width": info["width"],
            "height": info["height"],
            "frames": info["frames"],
            "duration_s": info["duration_s"],
            "source_bytes": os.path.getsize(video_path),
            "registered_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance or {},
            "synthetic": bool((provenance or {}).get("synthetic", False)),
        }
        if extract_frames:
            frames_dir = os.path.join(self.root, asset_id, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            n = 0
            for idx, frame in frame_iter(video_path, stride=frame_stride,
                                         max_frames=extract_frames, size=frame_size):
                cv2.imwrite(os.path.join(frames_dir, f"f{idx:05d}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
            entry["extracted_frames"] = n
            entry["frames_dir"] = os.path.relpath(frames_dir, self.root)
        self._append(entry)
        return entry

    def _append(self, entry):
        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def entries(self):
        if not os.path.exists(self.manifest_path):
            return []
        with open(self.manifest_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def find(self, asset_id):
        for e in self.entries():
            if e["id"] == asset_id:
                return e
        return None