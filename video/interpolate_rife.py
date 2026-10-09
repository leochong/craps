"""GPU frame interpolation via RIFE (Practical-RIFE, PyTorch) with arbitrary t.

RIFE weights (train_log/{flownet.pkl,*py}) must be present under `rife_repo`.
"""

import os
import sys

import cv2
import numpy as np


class RifeInterpolator:
    def __init__(self, factor, rife_repo, model_dir=None, device="cuda"):
        self.factor = max(1, int(factor))
        self.rife_repo = rife_repo
        self.model_dir = model_dir or os.path.join(rife_repo, "train_log")
        self.device = device
        self.model = None
        self._torch = None

    def _load(self):
        import torch
        import torch.nn.functional as F  # noqa: F401

        if self.rife_repo not in sys.path:
            sys.path.insert(0, self.rife_repo)
        from train_log.RIFE_HDv3 import Model

        model = Model()
        model.load_model(self.model_dir, -1)
        model.eval()
        model.device()
        self.model = model
        self._torch = torch

    def _infer(self, a, b, t):
        torch = self._torch
        import torch.nn.functional as F

        i0 = torch.tensor(a.transpose(2, 0, 1)).to(self.device) / 255.0
        i1 = torch.tensor(b.transpose(2, 0, 1)).to(self.device) / 255.0
        i0 = i0.unsqueeze(0)
        i1 = i1.unsqueeze(0)
        _, _, h, w = i0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        pad = (0, pw - w, 0, ph - h)
        i0 = F.pad(i0, pad)
        i1 = F.pad(i1, pad)
        with torch.no_grad():
            out = self.model.inference(i0, i1, t)
        return (out[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    def stream(self, frames):
        it = iter(frames)
        try:
            prev = next(it)
        except StopIteration:
            return
        yield prev
        for cur in it:
            if self.model is None:
                self._load()
            if self.factor > 1:
                for k in range(1, self.factor):
                    yield self._infer(prev, cur, k / self.factor)
            yield cur
            prev = cur