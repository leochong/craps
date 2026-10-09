"""Validation for the M1 keypoint training scaffold.

Checks label round-trip, manifest->dataset sample shapes/types, split disjointness,
target encode->decode accuracy, synthetic-asset exclusion, and (when torch is
installed) that a smoke train step reduces loss. Prints [PASS]/[FAIL]; exits
nonzero on any failure.
"""

import json
import os
import shutil
import sys

import numpy as np

from m1.dataset import VideoDataset, probe_video
from m1.labels import (
    NUM_KEYPOINTS,
    FrameLabel,
    decode_heatmaps,
    decode_keypoints,
    encode_heatmaps,
    encode_keypoints,
    frame_labels_from_roll,
    validate_label,
)
from m1.roll_annotation import DieMark, Roll, RollSet
from m1.torch_dataset import KeypointDataset, has_torch

WORK = "out/train_validate"
CLIP = os.path.join(WORK, "clip.mp4")
ROLLS = os.path.join(WORK, "rolls.json")
TRAIN = os.path.join(WORK, "train_keypoints.jsonl")
SYNTH_TRAIN = os.path.join(WORK, "synth_only.jsonl")
ROOT = os.path.join(WORK, "datasets")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def make_clip(path, n_frames=8, size=(96, 64), fps=10.0):
    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        return False
    rng = np.random.RandomState(0)
    for i in range(n_frames):
        frame = rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        import cv2 as _cv2
        _cv2.rectangle(frame, (10, 10), (30, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return os.path.exists(path) and probe_video(path)["frames"] >= n_frames


def make_rolls(path, n_rolls=12):
    rolls = RollSet(CLIP)
    for i in range(n_rolls):
        initial = [DieMark((10, 10, 20, 20), face=3), DieMark((60, 30, 20, 20), face=4)]
        final = [DieMark((12, 12, 20, 20), face=5), DieMark((58, 32, 20, 20), face=2)]
        roll = Roll(
            id=f"roll_{i + 1:04d}",
            initial_frame=0,
            final_frame=5,
            initial_dice=initial,
            final_dice=final,
        )
        roll.compute_sum()
        rolls.append(roll)
    rolls.save(path)
    return RollSet.load(path)


def bench_label_roundtrip():
    rolls = make_rolls(ROLLS)
    labels = frame_labels_from_roll(rolls.rolls[0], (96, 64), asset_id="orig_clip")
    label = labels[0]
    label.split = "train"
    restored = FrameLabel.from_dict(json.loads(json.dumps(label.to_dict())))
    kp_ok = np.allclose(restored.dice[0].keypoints, label.dice[0].keypoints)
    vis_ok = np.allclose(restored.dice[0].visible, label.dice[0].visible)
    face_ok = restored.dice[0].face == label.dice[0].face
    meta_ok = (restored.asset_id == label.asset_id and restored.frame == label.frame
               and restored.split == label.split and restored.phase == label.phase)
    errors = validate_label(label)
    ok = kp_ok and vis_ok and face_ok and meta_ok and not errors
    return check("label round-trip + validation", ok,
                 f"kp={kp_ok} visible={vis_ok} face={face_ok} meta={meta_ok} errors={errors}")


def bench_split_disjoint():
    from scripts.build_training_set import build

    summary = build(rolls_path=ROLLS, out_path=TRAIN, video=CLIP, manifest_root=ROOT,
                    asset_id="orig_clip", val_frac=0.5, seed=1, verbose=False)
    again = build(rolls_path=ROLLS, out_path=TRAIN, video=CLIP, manifest_root=ROOT,
                  asset_id="orig_clip", val_frac=0.5, seed=1, verbose=False)
    split_of = summary["split_of"]
    train = {k for k, v in split_of.items() if v == "train"}
    val = {k for k, v in split_of.items() if v == "val"}
    deterministic = summary["split_of"] == again["split_of"]
    ok = (not (train & val)) and (train | val == set(split_of)) and deterministic
    ok = ok and len(train) > 0 and len(val) > 0 and summary["written"] > 0
    return check("deterministic disjoint split", ok,
                 f"train={len(train)} val={len(val)} overlap={train & val} det={deterministic}")


def bench_dataset_samples():
    ds = KeypointDataset(os.path.join(ROOT, "manifest.jsonl"), TRAIN, split="train",
                         image_size=(96, 64), heatmap=False)
    ok_len = len(ds) > 0
    image, target = ds[0]
    img_ok = (image.shape == (3, 64, 96) and image.dtype == np.float32
              and 0.0 <= float(image.min()) and float(image.max()) <= 1.0)
    kp = target["keypoints"]
    tgt_ok = (kp.shape == (2, NUM_KEYPOINTS, 2) and kp.dtype == np.float32
              and target["visible"].shape == (2, NUM_KEYPOINTS)
              and target["face"].shape == (2,) and target["valid"].shape == (2,))
    meta_ok = target["meta"]["asset_id"] == "orig_clip"
    heat = ds.heatmap
    ds_heat = KeypointDataset(os.path.join(ROOT, "manifest.jsonl"), TRAIN, split="train",
                              image_size=(96, 64), heatmap=True, heatmap_size=(16, 24))
    _, htarget = ds_heat[0]
    heat_ok = htarget["heatmaps"].shape == (2, NUM_KEYPOINTS, 16, 24)
    ok = ok_len and img_ok and tgt_ok and meta_ok and heat_ok
    return check("manifest->dataset samples", ok,
                 f"len={len(ds)} img={image.shape} kp={kp.shape} heat={heat_ok}")


def bench_synthetic_excluded():
    ds_reg = VideoDataset(ROOT)
    entry = ds_reg.register(CLIP, "synth_clip",
                            provenance={"synthetic": True, "ground_truth_labels": False,
                                        "intended_use": "cv_augmentation"})
    labels = frame_labels_from_roll(make_rolls(ROLLS).rolls[0], (96, 64), asset_id="synth_clip")
    with open(SYNTH_TRAIN, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label.to_dict()) + "\n")
    ds = KeypointDataset(os.path.join(ROOT, "manifest.jsonl"), SYNTH_TRAIN, split=None)
    ok = entry["synthetic"] and len(ds) == 0 and ds.skipped_synthetic > 0
    return check("synthetic assets excluded", ok,
                 f"synthetic={entry['synthetic']} samples={len(ds)} skipped={ds.skipped_synthetic}")


def bench_encode_decode():
    width, height = 640, 480
    kp = np.array([[100.0, 200.0], [320.5, 240.25], [17.0, 55.0]], dtype=np.float64)
    enc = encode_keypoints(kp, width, height)
    dec = decode_keypoints(enc, width, height)
    direct_err = float(np.abs(dec - kp).max())
    heat = encode_heatmaps(encode_keypoints(kp, width, height), (64, 64), sigma=2.0)
    dec_norm, conf = decode_heatmaps(heat)
    heat_err = float(np.abs(dec_norm * np.array([width, height]) - kp).max())
    ok = direct_err < 1e-3 and heat_err < 2.0
    return check("target encode->decode", ok,
                 f"direct={direct_err:.2e}px heatmap={heat_err:.2f}px")


def bench_smoke():
    if not has_torch():
        return check("smoke train (torch)", True, "SKIP torch not installed")
    from m1.train import run_smoke

    result = run_smoke(epochs=2, steps_per_epoch=16, seed=0, verbose=False)
    ok = result is not None and result["reduced"]
    detail = "n/a" if result is None else f"{result['history'][0]:.4f}->{result['history'][-1]:.4f}"
    return check("smoke train reduces loss", ok, detail)


def bench_torch_collate():
    if not has_torch():
        return check("torch collate_fn", True, "SKIP torch not installed")
    import torch
    from torch.utils.data import DataLoader

    from m1.torch_dataset import collate_fn

    ds = KeypointDataset(os.path.join(ROOT, "manifest.jsonl"), TRAIN, split="train",
                         image_size=(96, 64))
    loader = DataLoader(ds.to_torch(), batch_size=2, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    ok = (tuple(images.shape) == (2, 3, 64, 96) and images.dtype == torch.float32
          and targets["keypoints"].shape == (2, 2, NUM_KEYPOINTS, 2)
          and isinstance(targets["meta"], list))
    return check("torch collate_fn", ok, f"images={tuple(images.shape)}")


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK, exist_ok=True)
    if not make_clip(CLIP):
        print(f"[FAIL] could not create test clip at {CLIP}")
        sys.exit(1)
    make_rolls(ROLLS)
    results = [
        bench_label_roundtrip(),
        bench_split_disjoint(),
        bench_dataset_samples(),
        bench_synthetic_excluded(),
        bench_encode_decode(),
        bench_smoke(),
        bench_torch_collate(),
    ]
    print(f"\n{sum(results)}/{len(results)} training benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
