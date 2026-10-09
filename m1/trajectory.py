"""Trajectory schema: per-die machine-frame position tracks from video.

A Trajectory is the measured bridge from M1 pose extraction into M2 inverse
calibration. Positions are in metres in the machine frame (origin at the plate
centre, +Z up). Synthetic/interpolated-derived tracks carry `synthetic=True` and
must not supervise calibration.
"""

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Trajectory:
    asset_id: str
    die_index: int
    times: np.ndarray
    pos: np.ndarray
    quality: np.ndarray = None
    face: list = field(default_factory=list)
    residual_px: list = field(default_factory=list)
    synthetic: bool = False
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        if self.pos.shape[0] != self.times.shape[0]:
            raise ValueError("times and pos length mismatch")
        if self.quality is None:
            self.quality = np.ones(self.times.shape[0], dtype=np.float64)
        else:
            self.quality = np.asarray(self.quality, dtype=np.float64).reshape(-1)

    def __len__(self):
        return int(self.times.shape[0])

    def to_dict(self):
        return {
            "asset_id": self.asset_id,
            "die_index": int(self.die_index),
            "times": self.times.tolist(),
            "pos": self.pos.tolist(),
            "quality": self.quality.tolist(),
            "face": list(self.face),
            "residual_px": [float(v) for v in self.residual_px],
            "synthetic": bool(self.synthetic),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            asset_id=d["asset_id"],
            die_index=d["die_index"],
            times=d["times"],
            pos=d["pos"],
            quality=d.get("quality"),
            face=d.get("face", []),
            residual_px=d.get("residual_px", []),
            synthetic=bool(d.get("synthetic", False)),
            provenance=d.get("provenance", {}),
        )


def validate_trajectory(traj):
    """Return a list of error strings for a Trajectory."""
    errors = []
    if len(traj) < 2:
        errors.append("trajectory needs at least 2 samples")
    if not np.isfinite(traj.times).all() or not np.isfinite(traj.pos).all():
        errors.append("non-finite times/pos")
    elif np.any(np.diff(traj.times) <= 0):
        errors.append("times must be strictly increasing")
    if traj.quality.shape[0] != len(traj):
        errors.append("quality length mismatch")
    if traj.synthetic and not traj.provenance:
        errors.append("synthetic trajectory must record provenance")
    return errors


def save_trajectories(trajectories, path):
    """Write Trajectory records to a JSONL file."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict()) + "\n")


def load_trajectories(path):
    """Load Trajectory records from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [Trajectory.from_dict(json.loads(line)) for line in f if line.strip()]
