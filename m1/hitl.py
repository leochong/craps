import json
import os

import cv2

from . import pose as pose_mod
from .frame_grab import FrameSource


def draw_overlays(frame, plate, dice, keypoints=None):
    out = frame.copy()
    if plate:
        cv2.circle(out, (plate["cx"], plate["cy"]), plate["radius"], (0, 255, 0), 2)
        cv2.drawMarker(out, (plate["cx"], plate["cy"]), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    for d in dice:
        x, y, w, h = d["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
        for (px, py) in d["pips"]:
            cv2.circle(out, (int(px), int(py)), 4, (0, 0, 255), -1)
        fv = d.get("face_value")
        if fv is not None:
            cv2.putText(out, f"face={fv}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        p = d.get("pose")
        if p:
            cv2.putText(out, f"resid={p['residual_px']:.1f}px", (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    if keypoints is not None:
        for i, (px, py) in enumerate(keypoints):
            cv2.circle(out, (int(px), int(py)), 5, (0, 255, 255), -1)
            cv2.putText(out, str(i), (int(px) + 6, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def _mouse_cb(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["keypoints"].append([float(x), float(y)])
    elif event == cv2.EVENT_RBUTTONDOWN and state["keypoints"]:
        state["keypoints"].pop()
    elif event == cv2.EVENT_MBUTTONDOWN:
        state["auto_accept"] = not state["auto_accept"]


def _detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return pose_mod.detect_dice_and_pips(gray)


def run_hitl(path, target_size=None, out_json="annotations/hitl.json"):
    cv2.namedWindow("M1 HITL")
    state = {"keypoints": [], "auto_accept": True}
    cv2.setMouseCallback("M1 HITL", _mouse_cb, state)

    annotations = []
    frame_idx = 0
    K = None
    with FrameSource(path, target_size=target_size) as src:
        while True:
            frame = src.read(frame_idx)
            if frame is None:
                break
            if K is None:
                K = pose_mod.default_intrinsics(frame.shape[1], frame.shape[0])
            dice = _detect(frame)
            for d in dice:
                d["pose"] = None
                if d.get("face_value") is not None and d["pips"].size:
                    d["pose"] = pose_mod.solve_die_pose(d["pips"], d["face_value"], K)

            overlay = draw_overlays(frame, None, dice, keypoints=state["keypoints"])
            cv2.putText(overlay, f"frame {frame_idx}  auto={state['auto_accept']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("M1 HITL", overlay)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break
            elif key == ord("a"):
                annotations.append({"frame": frame_idx, "keypoints": state["keypoints"], "dice": dice})
                state["keypoints"] = []
            elif key == ord("d"):
                state["keypoints"] = []
            elif key in (ord("n"), ord(" ")):
                frame_idx += 1
            elif key == ord("p"):
                frame_idx = max(0, frame_idx - 1)

    cv2.destroyAllWindows()
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(annotations, f, indent=2)
    return annotations