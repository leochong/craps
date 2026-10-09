"""Build a keypoint training manifest (frame -> 14 keypoints/die) from HITL rolls.

Labels are synthesized from annotated DieMarks at original video resolution and
written as a JSONL of FrameLabel records with a deterministic train/val split.
The source video is registered as a non-synthetic supervision asset.
"""

import argparse
import os

from m1.dataset import VideoDataset, probe_video
from m1.labels import (
    frame_labels_from_roll,
    split_for_key,
    validate_label,
    save_frame_labels,
)
from m1.roll_annotation import RollSet


def ensure_asset(root, video_path, asset_id):
    """Return an existing manifest entry or register the original video as supervision."""
    ds = VideoDataset(root)
    target = os.path.abspath(video_path)
    for entry in ds.entries():
        if entry["id"] == asset_id or os.path.abspath(entry.get("source_path", "")) == target:
            return entry
    provenance = {
        "origin": video_path,
        "derivation": "none",
        "synthetic": False,
        "ground_truth_labels": True,
        "intended_use": "keypoint_supervision",
        "note": "original-resolution frames; labels from HITL rolls.json",
    }
    return ds.register(video_path, asset_id, provenance=provenance)


def build(rolls_path="annotations/rolls.json", out_path="datasets/cv/train_keypoints.jsonl",
          video="IMG_2111.MOV", manifest_root="datasets/cv", asset_id=None,
          val_frac=0.2, seed=0, register=True, image_size=None, strict=False, verbose=True):
    """Convert rolls.json into a JSONL training manifest; returns a summary dict."""
    if not os.path.exists(rolls_path):
        if verbose:
            print(f"SKIP: no annotations at {rolls_path}")
        return {"written": 0, "reason": "no_rolls", "train": 0, "val": 0}

    rolls = RollSet.load(rolls_path)
    if image_size is None:
        if os.path.exists(video):
            info = probe_video(video)
            image_size = (info["width"], info["height"])
        else:
            image_size = (1920, 1080)
            if verbose:
                print(f"WARN: {video} not found; defaulting to {image_size}")
    image_size = (int(image_size[0]), int(image_size[1]))

    asset_id = asset_id or os.path.splitext(os.path.basename(video))[0]
    if register and os.path.exists(video):
        entry = ensure_asset(manifest_root, video, asset_id)
        asset_id = entry["id"]

    records = []
    split_of = {}
    n_invalid = 0
    for roll in rolls.rolls:
        split = split_for_key(roll.id, val_frac=val_frac, seed=seed)
        split_of[roll.id] = split
        for label in frame_labels_from_roll(roll, image_size, asset_id=asset_id):
            label.split = split
            errors = validate_label(label)
            if errors:
                n_invalid += 1
                if verbose:
                    print(f"WARN: dropping {roll.id}/{label.phase}: {errors}")
                continue
            records.append(label)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_frame_labels(records, out_path)

    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    summary = {
        "written": len(records),
        "train": n_train,
        "val": n_val,
        "invalid": n_invalid,
        "rolls": len(rolls.rolls),
        "split_of": split_of,
        "asset_id": asset_id,
        "image_size": list(image_size),
        "out": out_path,
    }
    if verbose:
        print(f"wrote {len(records)} frame labels -> {out_path}")
        print(f"  asset={asset_id} size={image_size[0]}x{image_size[1]} "
              f"train={n_train} val={n_val} invalid={n_invalid}")
    if strict and not records:
        raise RuntimeError("no valid frame labels produced")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Build keypoint training manifest from annotated rolls")
    ap.add_argument("--rolls", default="annotations/rolls.json")
    ap.add_argument("--out", default="datasets/cv/train_keypoints.jsonl")
    ap.add_argument("--video", default="IMG_2111.MOV")
    ap.add_argument("--root", default="datasets/cv")
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default=None, help="WxH override")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    image_size = None
    if args.size:
        w, h = args.size.lower().split("x")
        image_size = (int(w), int(h))
    build(rolls_path=args.rolls, out_path=args.out, video=args.video,
          manifest_root=args.root, asset_id=args.asset_id, val_frac=args.val_frac,
          seed=args.seed, register=not args.no_register, image_size=image_size,
          strict=args.strict)


if __name__ == "__main__":
    main()
