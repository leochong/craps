"""Per-frame die keypoint annotation (auto-detect + human-in-the-loop).

Auto-runs the classical detector + PnP and projects the 14-keypoint die model,
then lets a human accept/correct frames into a FrameLabel JSONL training manifest.
`--auto-only` bootstraps labels headlessly (e.g. on a GPU box). Labels must come
from original-resolution frames; synthetic assets are rejected for supervision.
"""

import argparse
import os

import cv2
import numpy as np

from m1 import pose as pose_mod
from m1.frame_grab import FrameSource
from m1.homography import calibrate_plate, load_calibration
from m1.labels import (
    FrameLabel,
    keypoints_from_die_mark,
    keypoints_from_pose,
    save_frame_labels,
    split_for_key,
    validate_label,
)
from m1.roll_annotation import DieMark

_FACE_KEYS = {ord(str(n)): n for n in range(1, 7)}
_COLORS = [(0, 255, 0), (0, 200, 255), (255, 128, 0), (255, 0, 255)]


def detect_frame_labels(frame, calib=None, max_residual_px=50.0):
    """Return DieLabels from classical detection + PnP keypoint projection."""
    height, width = frame.shape[:2]
    K = (np.asarray(calib["K"], dtype=np.float64) if calib and "K" in calib
         else pose_mod.default_intrinsics(width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    labels = []
    for d in pose_mod.detect_dice_and_pips(gray):
        solved = pose_mod.solve_die_pose(d["pips"], d.get("face_value"), K)
        if solved is None or solved["residual_px"] > max_residual_px:
            continue
        labels.append(keypoints_from_pose(solved["R"], solved["tvec"], K,
                                          face=d.get("face_value")))
    return labels


def auto_annotate(video, out_path, asset_id, calib=None, stride=1, frame_range=None,
                  max_residual_px=50.0, val_frac=0.2, seed=0, min_dice=1):
    """Headless pass: write a FrameLabel JSONL from auto-detected frames."""
    src = FrameSource(video)
    start, end = frame_range if frame_range else (0, src.frame_count)
    split = split_for_key(asset_id, val_frac, seed)
    labels = []
    for idx, frame in src.frames(start=start, end=end, step=stride):
        dice = detect_frame_labels(frame, calib, max_residual_px)
        if len(dice) < min_dice:
            continue
        label = FrameLabel(asset_id=asset_id, frame=idx, width=frame.shape[1],
                           height=frame.shape[0], dice=dice, source="auto_keypoints",
                           split=split)
        if not validate_label(label):
            labels.append(label)
    src.release()
    save_frame_labels(labels, out_path)
    return labels


class KeypointAnnotator:
    def __init__(self, video, out_path, asset_id, calib=None, scale=1.0, stride=1,
                 val_frac=0.2, seed=0, max_residual_px=50.0):
        self.src = FrameSource(video)
        self.out_path = out_path
        self.asset_id = asset_id
        self.calib = calib
        self.scale = scale if scale != 1.0 else None
        self.stride = stride
        self.split = split_for_key(asset_id, val_frac, seed)
        self.max_residual_px = max_residual_px
        self.frame_idx = 0
        self.frame = None
        self.dice = []
        self.accepted = []
        self.drag_start = None
        cv2.namedWindow("Keypoint Annotator")
        cv2.setMouseCallback("Keypoint Annotator", self._mouse)

    def _s(self, value):
        return int(value * self.scale) if self.scale else int(value)

    def _from_display(self, value):
        return value / self.scale if self.scale else value

    def _goto(self, idx):
        idx = max(0, min(idx, self.src.frame_count - 1))
        self.frame_idx = idx
        raw = self.src.read(idx)
        if raw is not None:
            self.frame = raw
            self.dice = detect_frame_labels(raw, self.calib, self.max_residual_px)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            self.drag_start = None
            x0, y0 = self._from_display(x0), self._from_display(y0)
            x1, y1 = self._from_display(x), self._from_display(y)
            x, y = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w >= 3 and h >= 3 and self.frame is not None:
                self.dice.append(keypoints_from_die_mark(
                    DieMark((x, y, w, h)), (self.frame.shape[1], self.frame.shape[0])))

    def _draw(self):
        out = self.frame.copy()
        for j, die in enumerate(self.dice):
            color = _COLORS[j % len(_COLORS)]
            for i, (px, py) in enumerate(die.keypoints):
                if die.visible[i] > 0.5:
                    cv2.circle(out, (self._s(px), self._s(py)), 4, color, -1)
                    cv2.putText(out, str(i), (self._s(px) + 5, self._s(py)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    cv2.circle(out, (self._s(px), self._s(py)), 2, color, 1)
            label = f"die{j} face={die.face}"
            cv2.putText(out, label, (self._s(die.keypoints[:, 0].min()), self._s(die.keypoints[:, 1].min()) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        banner = f"{self.asset_id} frame {self.frame_idx}/{self.src.frame_count - 1} dice={len(self.dice)} accepted={len(self.accepted)}"
        cv2.putText(out, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        help_line = "a:accept  n/p:+/-1  N/P:+/-30  r:redetect  d:delete  b:box-add  1-6:face  s:save  esc:quit"
        cv2.putText(out, help_line, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out

    def _accept(self):
        if not self.dice:
            print("no dice to accept")
            return
        label = FrameLabel(asset_id=self.asset_id, frame=self.frame_idx,
                           width=self.frame.shape[1], height=self.frame.shape[0],
                           dice=list(self.dice), source="hitl_keypoints", split=self.split)
        errors = validate_label(label)
        if errors:
            print(f"cannot accept: {errors}")
            return
        self.accepted.append(label)
        print(f"accepted frame {self.frame_idx} ({len(self.dice)} dice)")
        self._goto(self.frame_idx + self.stride)

    def _handle_key(self, key):
        if key == ord("a"):
            self._accept()
        elif key == ord("n"):
            self._goto(self.frame_idx + 1)
        elif key == ord("p"):
            self._goto(self.frame_idx - 1)
        elif key == ord("N"):
            self._goto(self.frame_idx + 30)
        elif key == ord("P"):
            self._goto(self.frame_idx - 30)
        elif key == ord("r"):
            self._goto(self.frame_idx)
        elif key == ord("d") and self.dice:
            self.dice.pop()
        elif key in _FACE_KEYS and self.dice:
            self.dice[-1].face = _FACE_KEYS[key]
        elif key == ord("s"):
            save_frame_labels(self.accepted, self.out_path)
            print(f"saved {len(self.accepted)} labels -> {self.out_path}")

    def run(self):
        self._goto(0)
        while True:
            cv2.imshow("Keypoint Annotator", self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break
            self._handle_key(key)
        cv2.destroyAllWindows()
        self.src.release()
        save_frame_labels(self.accepted, self.out_path)
        print(f"total labels: {len(self.accepted)}")


def parse_range(text):
    if not text:
        return None
    start, end = text.split(":")
    return (int(start), int(end))


def resolve_calibration(args, video):
    if args.calib and os.path.exists(args.calib):
        return load_calibration(args.calib)
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return calibrate_plate(frame, plate_radius_m=args.plate_radius_m, focal=args.focal)


def main():
    ap = argparse.ArgumentParser(description="Per-frame die keypoint annotator (auto + HITL)")
    ap.add_argument("--video", default="IMG_2111.MOV")
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--root", default="datasets/cv")
    ap.add_argument("--plate-radius-m", type=float, default=0.15)
    ap.add_argument("--focal", type=float, default=None)
    ap.add_argument("--frames", default=None, help="start:end")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--auto-only", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.synthetic and not args.force:
        print("FAIL: refusing to label synthetic/interpolated frames (use --force to override)")
        return
    if not os.path.exists(args.video):
        print(f"SKIP: {args.video} not found")
        return

    asset_id = args.asset_id or os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or os.path.join("datasets", "cv", f"keypoints_{asset_id}.jsonl")

    if not args.no_register and not args.synthetic:
        from scripts.build_training_set import ensure_asset

        ensure_asset(args.root, args.video, asset_id)

    calib = resolve_calibration(args, args.video)
    if calib is None:
        print("WARN: plate calibration failed; using default intrinsics")

    frame_range = parse_range(args.frames)
    if frame_range and args.max_frames:
        frame_range = (frame_range[0], min(frame_range[1], frame_range[0] + args.max_frames))

    if args.auto_only:
        labels = auto_annotate(args.video, out_path, asset_id, calib=calib,
                               stride=args.stride, frame_range=frame_range,
                               val_frac=args.val_frac, seed=args.seed)
        print(f"auto-annotated {len(labels)} frames -> {out_path}")
        return

    KeypointAnnotator(args.video, out_path, asset_id, calib=calib, scale=args.scale,
                      stride=args.stride, val_frac=args.val_frac, seed=args.seed).run()


if __name__ == "__main__":
    main()