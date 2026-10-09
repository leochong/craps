"""M1 keypoint detector training (torch imported lazily).

Predicts 14 keypoints per die (8 corners + 6 face centers) from normalized image
coordinates. Direct-regression heads (normalized (K,2) + face logits) are the
default because they are cheap, deterministic, and their pixel residual matches
the `pose.solve_die_pose` metric; Gaussian heatmaps are available in the dataset
for a spatial Soft-Argmax variant on GPU. Runs a CPU smoke path without a GPU.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

from .labels import (
    NUM_KEYPOINTS,
    decode_heatmaps,
    decode_keypoints,
    encode_heatmaps,
    encode_keypoints,
    split_for_key,
)


def torch_or_none():
    """Import torch lazily; return None if unavailable."""
    try:
        import torch
        return torch
    except Exception:
        return None


def parse_size(text):
    w, h = str(text).lower().split("x")
    return (int(w), int(h))


def build_model(num_dice=2, num_keypoints=NUM_KEYPOINTS, num_faces=6, width=32):
    """Small configurable fully-convolutional keypoint + face regressor."""
    import torch.nn as nn

    class KeypointNet(nn.Module):
        def __init__(self):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                )

            self.num_dice = num_dice
            self.num_keypoints = num_keypoints
            self.num_faces = num_faces
            self.features = nn.Sequential(
                block(3, width), block(width, width * 2), block(width * 2, width * 4),
                nn.AdaptiveAvgPool2d(1),
            )
            self.keypoint_head = nn.Sequential(
                nn.Flatten(), nn.Linear(width * 4, 256), nn.ReLU(inplace=True),
                nn.Linear(256, num_dice * num_keypoints * 2),
            )
            self.face_head = nn.Sequential(
                nn.Flatten(), nn.Linear(width * 4, num_dice * num_faces),
            )

        def forward(self, x):
            import torch

            feat = self.features(x)
            kp = torch.sigmoid(self.keypoint_head(feat))
            kp = kp.view(-1, self.num_dice, self.num_keypoints, 2)
            face_logits = self.face_head(feat).view(-1, self.num_dice, self.num_faces)
            return {"keypoints": kp, "face_logits": face_logits}

    return KeypointNet()


def keypoint_loss(outputs, targets, face_weight=0.1):
    """Masked Smooth-L1 keypoint loss plus optional face cross-entropy."""
    import torch
    import torch.nn.functional as F

    pred = outputs["keypoints"]
    gt = targets["keypoints"]
    visible = targets["visible"]
    valid = targets["valid"]
    mask = visible * valid.unsqueeze(-1)
    diff = F.smooth_l1_loss(pred, gt, reduction="none").sum(dim=-1)
    kp_loss = (diff * mask).sum() / mask.sum().clamp_min(1.0)

    logits = outputs["face_logits"]
    face = targets["face"].clamp_min(0).long()
    face_mask = (targets["face"] >= 0).float() * valid
    b, d, c = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, c), face.reshape(-1), reduction="none").view(b, d)
    face_loss = (ce * face_mask).sum() / face_mask.sum().clamp_min(1.0)
    return kp_loss + face_weight * face_loss


def evaluate(model, loader, device, image_size):
    """PCK@0.05 (normalized) and mean pixel residual consistent with solve_die_pose."""
    import torch

    model.eval()
    res_sum = 0.0
    n = 0
    hit = 0
    diag = math.hypot(float(image_size[0]), float(image_size[1]))
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            pred = model(images)["keypoints"].cpu().numpy()
            gt = np.asarray(targets["keypoints"])
            vis = np.asarray(targets["visible"]) * np.asarray(targets["valid"])[..., None]
            dx = (pred[..., 0] - gt[..., 0]) * float(image_size[0])
            dy = (pred[..., 1] - gt[..., 1]) * float(image_size[1])
            res = np.sqrt(dx * dx + dy * dy) * vis
            res_sum += float(res.sum())
            n += float(vis.sum())
            hit += int(((res <= 0.05 * diag) & (vis > 0.5)).sum())
    return {
        "residual_px": res_sum / max(n, 1.0),
        "pck@0.05": hit / max(n, 1.0),
        "n_keypoints": int(n),
    }


def _to_device(batch, device):
    images, targets = batch
    images = images.to(device)
    targets = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in targets.items()}
    return images, targets


def run_smoke(epochs=2, steps_per_epoch=16, batch=8, image_size=(64, 64),
              num_dice=2, seed=0, device="cpu", lr=1e-2, verbose=True):
    """Tiny CPU/GPU smoke train: verifies loss decreases end-to-end."""
    torch = torch_or_none()
    if torch is None:
        if verbose:
            print("[SKIP] torch not installed; smoke train unavailable")
        return None

    torch.manual_seed(seed)
    height, width = image_size
    model = build_model(num_dice=num_dice).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def random_batch():
        images = torch.rand(batch, 3, height, width, device=device)
        targets = {
            "keypoints": torch.rand(batch, num_dice, NUM_KEYPOINTS, 2, device=device) * 0.6 + 0.2,
            "visible": torch.ones(batch, num_dice, NUM_KEYPOINTS, device=device),
            "face": torch.randint(1, 7, (batch, num_dice), device=device),
            "valid": torch.ones(batch, num_dice, device=device),
        }
        return images, targets

    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            images, targets = random_batch()
            opt.zero_grad()
            loss = keypoint_loss(model(images), targets)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        history.append(epoch_loss / steps_per_epoch)
        if verbose:
            print(f"  smoke epoch {epoch + 1}/{epochs} loss={history[-1]:.4f}")
    reduced = history[-1] < history[0]
    if verbose:
        print(f"[{'PASS' if reduced else 'FAIL'}] smoke loss reduced {history[0]:.4f} -> {history[-1]:.4f}")
    return {"history": history, "reduced": reduced}


def train(video_manifest, train_manifest, epochs=2, image_size=(256, 256), batch=8,
          lr=1e-3, out="out/train", device=None, seed=0, smoke=False, verbose=True):
    """Train the keypoint net from the manifest-backed dataset; save a checkpoint."""
    torch = torch_or_none()
    if torch is None:
        print("[SKIP] torch not installed; install torch to train")
        return None
    from torch.utils.data import DataLoader

    from .torch_dataset import KeypointDataset, collate_fn

    if smoke:
        result = run_smoke(epochs=min(int(epochs), 2), image_size=(64, 64), verbose=verbose)
        return {"smoke": result}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = KeypointDataset(video_manifest, train_manifest, split="train", image_size=image_size)
    val_ds = KeypointDataset(video_manifest, train_manifest, split="val", image_size=image_size)
    if verbose:
        print(f"device={device} train={len(train_ds)} val={len(val_ds)} "
              f"(skipped synthetic={train_ds.skipped_synthetic} missing={train_ds.skipped_missing})")
    if len(train_ds) == 0:
        print("[FAIL] no training samples; run scripts/build_training_set.py first")
        return None

    loader = DataLoader(train_ds.to_torch(), batch_size=batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds.to_torch(), batch_size=batch, collate_fn=collate_fn) if len(val_ds) else None
    model = build_model(num_dice=train_ds.max_dice).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        start = time.time()
        epoch_loss = 0.0
        for images, targets in loader:
            images, targets = _to_device((images, targets), device)
            opt.zero_grad()
            loss = keypoint_loss(model(images), targets)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        avg = epoch_loss / max(len(loader), 1)
        msg = f"epoch {epoch + 1}/{epochs} loss={avg:.4f} {time.time() - start:.1f}s"
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device, image_size)
            msg += f" val_resid={metrics['residual_px']:.2f}px pck@0.05={metrics['pck@0.05']:.3f}"
        if verbose:
            print(msg)

    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "keypoint_net.pt")
    torch.save({"model": model.state_dict(), "num_dice": train_ds.max_dice,
                "image_size": list(image_size), "epochs": epochs}, ckpt)
    if verbose:
        print(f"saved checkpoint -> {ckpt}")
    return {"checkpoint": ckpt}


def numpy_stub_check(verbose=True):
    """Validate dataloader/target logic without torch (encode/decode + split)."""
    ok = True
    width, height = 640, 480
    kp = np.array([[100.0, 200.0], [320.5, 240.25]], dtype=np.float64)
    enc = encode_keypoints(kp, width, height)
    dec = decode_keypoints(enc, width, height)
    err = float(np.abs(dec - kp).max())
    ok &= err < 1e-3
    if verbose:
        print(f"[{'PASS' if err < 1e-3 else 'FAIL'}] numpy keypoint encode->decode err={err:.2e}px")

    heat = encode_heatmaps(enc, (64, 64), sigma=2.0)
    dec_norm, conf = decode_heatmaps(heat)
    heat_err = float(np.abs(dec_norm - enc).max() * max(width, height))
    ok &= heat_err < 2.0
    if verbose:
        print(f"[{'PASS' if heat_err < 2.0 else 'FAIL'}] numpy heatmap encode->decode err={heat_err:.2f}px")

    a = [split_for_key(f"roll_{i:04d}", 0.2, 7) for i in range(20)]
    b = [split_for_key(f"roll_{i:04d}", 0.2, 7) for i in range(20)]
    ok &= a == b
    if verbose:
        print(f"[{'PASS' if a == b else 'FAIL'}] deterministic split (val={a.count('val')}/20)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Train the M1 keypoint detector")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic train (no dataset needed)")
    ap.add_argument("--video-manifest", default="datasets/cv/manifest.jsonl")
    ap.add_argument("--train-manifest", default="datasets/cv/train_keypoints.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image-size", default="256x256")
    ap.add_argument("--out", default="out/train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if torch_or_none() is None:
        print("[SKIP] torch not installed: falling back to numpy dataloader/target check")
        sys.exit(numpy_stub_check())

    if args.smoke or not os.path.exists(args.train_manifest):
        if not args.smoke:
            print(f"note: {args.train_manifest} missing; running --smoke instead")
        result = run_smoke(epochs=min(args.epochs, 2), seed=args.seed)
        sys.exit(0 if (result and result["reduced"]) else 1)

    train(args.video_manifest, args.train_manifest, epochs=args.epochs,
          image_size=parse_size(args.image_size), batch=args.batch, lr=args.lr,
          out=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
