"""Keypoint/label schema for M1 die pose training.

Each die is described by 14 keypoints: 8 cube corners followed by the 6 face
centers, ordered by face value 1..6. Coordinates are in die-side units with the
cube spanning [-0.5, 0.5]; the +Z face is the "1" face and the pip plane at
z=0.5 matches `pose.pip_layout_3d`, so labels stay congruent with PnP.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from .roll_annotation import DieMark

NUM_CORNERS = 8
NUM_FACE_CENTERS = 6
NUM_KEYPOINTS = NUM_CORNERS + NUM_FACE_CENTERS
FACE_Z = 0.5

CORNERS_LOCAL = np.array(
    [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ],
    dtype=np.float64,
)

FACE_NORMALS = np.array(
    [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    ],
    dtype=np.float64,
)

KEYPOINT_NAMES = [f"corner_{i}" for i in range(NUM_CORNERS)] + [
    f"face_{v}" for v in range(1, NUM_FACE_CENTERS + 1)
]
FACE_TO_KEYPOINT = {v: NUM_CORNERS + (v - 1) for v in range(1, NUM_FACE_CENTERS + 1)}
KEYPOINT_TO_FACE = {k: v for v, k in FACE_TO_KEYPOINT.items()}


def die_keypoints_3d(scale=1.0):
    """Return the 14 canonical keypoints (corners then face centers); scale=die side."""
    face_centers = FACE_Z * FACE_NORMALS
    return np.vstack([CORNERS_LOCAL, face_centers]) * float(scale)


def face_center_3d(face_value, scale=1.0):
    """Return the 3D center of a face value in die-side units."""
    return FACE_Z * FACE_NORMALS[face_value - 1] * float(scale)


def normalize_keypoints(keypoints, width, height):
    """Map pixel keypoints to [0,1] with x/width, y/height."""
    kp = np.asarray(keypoints, dtype=np.float64).copy()
    kp[..., 0] = kp[..., 0] / float(width)
    kp[..., 1] = kp[..., 1] / float(height)
    return kp


def denormalize_keypoints(keypoints_norm, width, height):
    """Map [0,1] keypoints back to pixels."""
    kp = np.asarray(keypoints_norm, dtype=np.float64).copy()
    kp[..., 0] = kp[..., 0] * float(width)
    kp[..., 1] = kp[..., 1] * float(height)
    return kp


def encode_keypoints(keypoints, width, height):
    """Direct-regression target: normalized pixel keypoints."""
    return normalize_keypoints(keypoints, width, height).astype(np.float32)


def decode_keypoints(keypoints_norm, width, height):
    """Invert `encode_keypoints` back to pixels."""
    return denormalize_keypoints(keypoints_norm, width, height)


def encode_heatmaps(keypoints_norm, out_hw, sigma=2.0, visible=None):
    """Encode normalized keypoints as Gaussian heatmaps of shape (K, oh, ow)."""
    kp = np.asarray(keypoints_norm, dtype=np.float64)
    oh, ow = int(out_hw[0]), int(out_hw[1])
    k = kp.shape[0]
    ys = np.arange(oh, dtype=np.float64)[:, None]
    xs = np.arange(ow, dtype=np.float64)[None, :]
    heat = np.zeros((k, oh, ow), dtype=np.float32)
    denom = 2.0 * float(sigma) ** 2
    for i in range(k):
        if visible is not None and not bool(visible[i]):
            continue
        cx = kp[i, 0] * (ow - 1)
        cy = kp[i, 1] * (oh - 1)
        g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / denom)
        heat[i] = g
    return heat


def decode_heatmaps(heatmaps, out_hw=None):
    """Argmax + quadratic sub-pixel decode into normalized keypoints and confidences."""
    heat = np.asarray(heatmaps, dtype=np.float64)
    k, oh, ow = heat.shape
    flat = heat.reshape(k, -1)
    idx = flat.argmax(axis=1)
    conf = flat[np.arange(k), idx]
    pix = np.stack([idx % ow, idx // ow], axis=1).astype(np.float64)
    for i in range(k):
        for axis, size in ((0, ow), (1, oh)):
            c = int(pix[i, axis])
            if 0 < c < size - 1:
                if axis == 0:
                    v0, v1, v2 = heat[i, int(pix[i, 1]), c - 1], heat[i, int(pix[i, 1]), c], heat[i, int(pix[i, 1]), c + 1]
                else:
                    v0, v1, v2 = heat[i, c - 1, int(pix[i, 0])], heat[i, c, int(pix[i, 0])], heat[i, c + 1, int(pix[i, 0])]
                denom = 2.0 * v1 - v0 - v2
                if abs(denom) > 1e-9:
                    pix[i, axis] += float(np.clip(0.5 * (v2 - v0) / denom, -0.5, 0.5))
    xs = pix[:, 0] / max(ow - 1, 1)
    ys = pix[:, 1] / max(oh - 1, 1)
    return np.stack([xs, ys], axis=1), conf


@dataclass
class DieLabel:
    keypoints: np.ndarray
    visible: np.ndarray = None
    face: int = None
    rect: list = None

    def __post_init__(self):
        self.keypoints = np.asarray(self.keypoints, dtype=np.float64).reshape(-1, 2)
        if self.visible is None:
            self.visible = np.ones(self.keypoints.shape[0], dtype=np.float64)
        else:
            self.visible = np.asarray(self.visible, dtype=np.float64).reshape(-1)

    def to_dict(self):
        return {
            "keypoints": self.keypoints.tolist(),
            "visible": self.visible.tolist(),
            "face": None if self.face is None else int(self.face),
            "rect": None if self.rect is None else [float(v) for v in self.rect],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            keypoints=d["keypoints"],
            visible=d.get("visible"),
            face=d.get("face"),
            rect=d.get("rect"),
        )


@dataclass
class FrameLabel:
    asset_id: str
    frame: int
    width: int
    height: int
    dice: list = field(default_factory=list)
    source: str = None
    roll_id: str = None
    phase: str = None
    split: str = None

    def to_dict(self):
        return {
            "asset_id": self.asset_id,
            "frame": int(self.frame),
            "width": int(self.width),
            "height": int(self.height),
            "dice": [d.to_dict() if isinstance(d, DieLabel) else d for d in self.dice],
            "source": self.source,
            "roll_id": self.roll_id,
            "phase": self.phase,
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            asset_id=d["asset_id"],
            frame=d["frame"],
            width=d["width"],
            height=d["height"],
            dice=[DieLabel.from_dict(x) for x in d.get("dice", [])],
            source=d.get("source"),
            roll_id=d.get("roll_id"),
            phase=d.get("phase"),
            split=d.get("split"),
        )


def split_for_key(key, val_frac=0.2, seed=0):
    """Deterministic train/val assignment by hashing a grouping key."""
    if val_frac <= 0.0:
        return "train"
    if val_frac >= 1.0:
        return "val"
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return "val" if bucket < int(round(val_frac * 10000)) else "train"


def _rotation_a_to_b(a, b):
    """Rotation matrix mapping unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0.0:
            return np.eye(3)
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        kmat = _skew(axis)
        return np.eye(3) + 2.0 * kmat @ kmat
    kmat = _skew(v)
    return np.eye(3) + kmat + kmat @ kmat * (1.0 / (1.0 + c))


def _skew(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def keypoints_from_die_mark(mark, image_size):
    """Synthesize 14 keypoints from a HITL DieMark under a frontal-cube assumption.

    The marked face is rotated to face the camera and the annotated rect maps to
    that face square, so the projected keypoints are consistent with `pip_layout_3d`.
    """
    width, height = int(image_size[0]), int(image_size[1])
    x, y, w, h = [float(v) for v in mark.rect]
    cx, cy = x + w / 2.0, y + h / 2.0
    side_px = max(1.0, (w + h) / 2.0)
    f = float(max(width, height))
    tz = max(f / side_px - FACE_Z, FACE_Z + 1e-3)
    ox = (cx - width / 2.0) * tz / f
    oy = (cy - height / 2.0) * tz / f

    local = die_keypoints_3d(1.0)
    if mark.face is not None:
        rot = _rotation_a_to_b(FACE_NORMALS[mark.face - 1], np.array([0.0, 0.0, 1.0]))
        local = local @ rot.T
        norms = FACE_NORMALS @ rot.T
    else:
        norms = FACE_NORMALS

    pts_cam = local + np.array([ox, oy, tz])
    proj = np.empty((NUM_KEYPOINTS, 2), dtype=np.float64)
    proj[:, 0] = f * pts_cam[:, 0] / pts_cam[:, 2] + width / 2.0
    proj[:, 1] = f * pts_cam[:, 1] / pts_cam[:, 2] + height / 2.0

    visible = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
    visible[:NUM_CORNERS] = (local[:NUM_CORNERS, 2] > 1e-6).astype(np.float64)
    visible[NUM_CORNERS:] = (norms[:, 2] > 1e-6).astype(np.float64)
    if mark.face is not None:
        visible[FACE_TO_KEYPOINT[mark.face]] = 1.0
    return DieLabel(keypoints=proj, visible=visible, face=mark.face,
                    rect=[x, y, w, h])


def die_mark_from_label(label):
    """Convert a DieLabel back to a DieMark (uses stored rect or visible-keypoint bbox)."""
    rect = label.rect
    if rect is None:
        mask = label.visible > 0.5
        pts = label.keypoints[mask] if np.any(mask) else label.keypoints
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        rect = [x0, y0, x1 - x0, y1 - y0]
    return DieMark(tuple(float(v) for v in rect), label.face)


def frame_labels_from_roll(roll, image_size, asset_id=None):
    """Convert one annotated Roll into initial/final FrameLabels via the schema."""
    width, height = int(image_size[0]), int(image_size[1])
    labels = []
    for phase, frame, dice in (
        ("initial", roll.initial_frame, roll.initial_dice),
        ("final", roll.final_frame, roll.final_dice),
    ):
        if frame is None:
            continue
        marks = [keypoints_from_die_mark(m, (width, height)) for m in dice]
        labels.append(
            FrameLabel(
                asset_id=asset_id or "",
                frame=int(frame),
                width=width,
                height=height,
                dice=marks,
                source="hitl_roll",
                roll_id=roll.id,
                phase=phase,
            )
        )
    return labels


def validate_die_label(label, width=None, height=None):
    errors = []
    kp = np.asarray(label.keypoints)
    if kp.shape != (NUM_KEYPOINTS, 2):
        errors.append(f"keypoints shape {kp.shape} != {(NUM_KEYPOINTS, 2)}")
    elif not np.isfinite(kp).all():
        errors.append("non-finite keypoints")
    if np.asarray(label.visible).reshape(-1).shape[0] != NUM_KEYPOINTS:
        errors.append("visible length != keypoints")
    elif float(np.sum(np.asarray(label.visible) > 0.5)) < 1.0:
        errors.append("no visible keypoints")
    if label.face is not None and not (1 <= int(label.face) <= NUM_FACE_CENTERS):
        errors.append(f"face out of range: {label.face}")
    if width and height and np.isfinite(kp).all():
        if (kp[:, 0] < -0.05 * width).any() or (kp[:, 0] > 1.05 * width).any():
            errors.append("keypoints outside image width")
        if (kp[:, 1] < -0.05 * height).any() or (kp[:, 1] > 1.05 * height).any():
            errors.append("keypoints outside image height")
    return errors


def validate_label(label):
    """Validate a DieLabel or FrameLabel; returns a list of error strings."""
    if isinstance(label, FrameLabel) or hasattr(label, "dice"):
        errors = []
        if label.width is None or label.height is None or label.width <= 0 or label.height <= 0:
            errors.append("invalid image size")
        if not label.dice:
            errors.append("frame has no dice")
        for i, die in enumerate(label.dice):
            for e in validate_die_label(die, label.width, label.height):
                errors.append(f"die[{i}]: {e}")
        return errors
    return validate_die_label(label)


def load_frame_labels(path):
    """Load a JSONL training manifest into FrameLabel records."""
    labels = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(FrameLabel.from_dict(json.loads(line)))
    return labels


def save_frame_labels(labels, path):
    """Write FrameLabel records to a JSONL training manifest."""
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label.to_dict()) + "\n")
