"""Keypoint training dataset bridging the CV video manifest and label schema.

`KeypointDataset` is torch-free (yields numpy CHW images and numpy target dicts);
call `to_torch()` or use `collate_fn` to get tensors when torch is installed.
Frames are streamed from the manifest video via `dataset.frame_iter`. Synthetic
(interpolated/upscaled) assets are excluded from label supervision by default;
they may only be used as input augmentation with `allow_synthetic=True`.
"""

import json
import os

import numpy as np

from .dataset import frame_iter
from .labels import (
    NUM_KEYPOINTS,
    FrameLabel,
    encode_heatmaps,
    normalize_keypoints,
)

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_HEATMAP_SIZE = (64, 64)
TARGET_KEYS = ("keypoints", "visible", "face", "valid")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def has_torch():
    """True if torch is importable (checked lazily, never at module import)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def supervisable(entry):
    """False for synthetic/derived assets that must not provide label ground truth."""
    prov = entry.get("provenance") or {}
    if entry.get("synthetic") or prov.get("synthetic"):
        return False
    return True


class _FrameCursor:
    """Forward-only frame cursor over `frame_iter`, resetting on backward seeks."""

    def __init__(self, path, size=None):
        self.path = path
        self.size = size
        self._reset()

    def _reset(self):
        self._it = frame_iter(self.path, stride=1, size=self.size)
        self._next = 0

    def read(self, idx):
        if idx < self._next:
            self._reset()
        for i, frame in self._it:
            self._next = i + 1
            if i == idx:
                return frame
        return None


class KeypointDataset:
    """Frame keypoint samples resolved from a VideoDataset manifest + training JSONL."""

    def __init__(self, manifest_path, training_path, split=None, image_size=DEFAULT_IMAGE_SIZE,
                 max_dice=2, heatmap=False, heatmap_size=DEFAULT_HEATMAP_SIZE,
                 heatmap_sigma=2.0, allow_synthetic=False, strict=False):
        self.manifest_path = manifest_path
        self.training_path = training_path
        self.split = split
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_dice = int(max_dice)
        self.heatmap = bool(heatmap)
        self.heatmap_size = (int(heatmap_size[0]), int(heatmap_size[1]))
        self.heatmap_sigma = float(heatmap_sigma)
        self.strict = bool(strict)

        self.entries = {e["id"]: e for e in _read_jsonl(manifest_path)}
        self.skipped_missing = 0
        self.skipped_synthetic = 0
        self.samples = []
        self._cursor = None
        self._cursor_id = None

        for record in _read_jsonl(training_path):
            label = FrameLabel.from_dict(record) if isinstance(record, dict) else record
            if split is not None and label.split not in (None, split):
                continue
            entry = self.entries.get(label.asset_id)
            if entry is None:
                if strict:
                    raise KeyError(f"asset '{label.asset_id}' absent from {manifest_path}")
                self.skipped_missing += 1
                continue
            if not supervisable(entry) and not allow_synthetic:
                self.skipped_synthetic += 1
                continue
            self.samples.append((entry, label))

        self.samples.sort(key=lambda s: (s[0]["id"], s[1].frame))

    def __len__(self):
        return len(self.samples)

    def _read_frame(self, entry, idx):
        if self._cursor_id != entry["id"]:
            self._cursor = _FrameCursor(entry["source_path"], size=self.image_size)
            self._cursor_id = entry["id"]
        return self._cursor.read(int(idx))

    def _encode_target(self, label):
        k = NUM_KEYPOINTS
        keypoints = np.zeros((self.max_dice, k, 2), dtype=np.float32)
        visible = np.zeros((self.max_dice, k), dtype=np.float32)
        face = np.full((self.max_dice,), -1, dtype=np.int64)
        valid = np.zeros((self.max_dice,), dtype=np.float32)
        for j, die in enumerate(label.dice[: self.max_dice]):
            keypoints[j] = encode_target_keypoints(die, label.width, label.height)
            visible[j] = die.visible
            face[j] = -1 if die.face is None else int(die.face)
            valid[j] = 1.0
        target = {"keypoints": keypoints, "visible": visible, "face": face, "valid": valid}
        if self.heatmap:
            heat = np.zeros((self.max_dice, k, self.heatmap_size[0], self.heatmap_size[1]),
                            dtype=np.float32)
            for j in range(self.max_dice):
                if valid[j] > 0.5:
                    heat[j] = encode_heatmaps(keypoints[j], self.heatmap_size,
                                              sigma=self.heatmap_sigma, visible=visible[j])
            target["heatmaps"] = heat
        target["meta"] = {"asset_id": label.asset_id, "frame": int(label.frame),
                          "width": label.width, "height": label.height}
        return target

    def __getitem__(self, index):
        entry, label = self.samples[index]
        frame = self._read_frame(entry, label.frame)
        if frame is None:
            raise IndexError(f"could not read frame {label.frame} of '{entry['id']}'")
        image = np.ascontiguousarray(frame.transpose(2, 0, 1)).astype(np.float32) / 255.0
        return image, self._encode_target(label)

    def close(self):
        self._cursor = None
        self._cursor_id = None

    def __del__(self):
        self.close()

    def to_torch(self):
        """Wrap as a torch.utils.data.Dataset yielding tensor images and targets."""
        import torch
        from torch.utils.data import Dataset

        base = self

        class _TorchKeypointDataset(Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, index):
                image, target = base[index]
                out = {}
                for key, value in target.items():
                    if key == "meta":
                        out[key] = value
                    else:
                        out[key] = torch.from_numpy(np.asarray(value))
                return torch.from_numpy(image), out

        return _TorchKeypointDataset()


def encode_target_keypoints(die, width, height):
    """Direct-regression target for one die: normalized (K,2) pixel keypoints."""
    return normalize_keypoints(die.keypoints, width, height).astype(np.float32)


def collate_fn(batch):
    """Collate (image, target) samples into batched tensors (requires torch)."""
    import torch

    images = torch.stack([torch.as_tensor(sample[0]) for sample in batch])
    keys = set()
    for _, target in batch:
        keys.update(k for k in target if k != "meta")
    out = {}
    for key in TARGET_KEYS + ("heatmaps",):
        if key in keys:
            out[key] = torch.stack([torch.as_tensor(np.asarray(t[key])) for _, t in batch])
    out["meta"] = [t.get("meta") for _, t in batch]
    return images, out


def build_datasets(manifest_path, training_path, image_size=DEFAULT_IMAGE_SIZE,
                   val_split="val", train_split="train", **kwargs):
    """Convenience: construct disjoint train/val KeypointDataset pairs."""
    train = KeypointDataset(manifest_path, training_path, split=train_split,
                            image_size=image_size, **kwargs)
    val = KeypointDataset(manifest_path, training_path, split=val_split,
                          image_size=image_size, **kwargs)
    return train, val
