"""Track die poses across frames and emit machine-frame trajectories.

Runs the classical M1 detector + PnP per frame, associates detections into tracks
by nearest centroid, and converts each track's camera-frame translation to the
machine frame via the plate calibration from `homography`.
"""

import cv2
import numpy as np

from . import pose as pose_mod
from .frame_grab import FrameSource
from .homography import camera_to_machine, has_plane_pose, tvec_to_machine
from .trajectory import Trajectory


def _machine_pos(tvec, calib):
    if has_plane_pose(calib):
        return camera_to_machine(tvec, calib)
    return tvec_to_machine(tvec, calib)


def detect_frame_poses(frame, calib, max_residual_px=50.0, die_side=0.016):
    """Detect dice in a BGR frame and return solved-pose detections with machine positions."""
    K = np.asarray(calib["K"], dtype=np.float64)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = []
    for d in pose_mod.detect_dice_and_pips(gray):
        solved = pose_mod.solve_die_pose(d["pips"], d.get("face_value"), K)
        if solved is None or solved["residual_px"] > max_residual_px:
            continue
        x, y, w, h = d["bbox"]
        tvec = np.asarray(solved["tvec"], dtype=np.float64).reshape(3) * die_side
        detections.append({
            "centroid": (x + w / 2.0, y + h / 2.0),
            "tvec": tvec,
            "pos": _machine_pos(tvec, calib),
            "residual_px": float(solved["residual_px"]),
            "face": d.get("face_value"),
        })
    return detections


def _centroid_distance(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _to_trajectory(track, calib, fps, asset_id, synthetic, provenance):
    samples = track["samples"]
    times = np.array([idx / fps for idx, _ in samples], dtype=np.float64)
    pos = np.stack([det["pos"] for _, det in samples])
    quality = np.array([1.0 / (1.0 + det["residual_px"]) for _, det in samples])
    faces = [det["face"] for _, det in samples]
    residual = [det["residual_px"] for _, det in samples]
    return Trajectory(
        asset_id=asset_id,
        die_index=int(track["die_index"]),
        times=times,
        pos=pos,
        quality=quality,
        face=faces,
        residual_px=residual,
        synthetic=synthetic,
        provenance=dict(provenance),
    )


def track_poses(video, calib, frame_range=None, stride=1, max_gap=4, match_px=50.0,
                min_samples=3, asset_id=None, synthetic=False, provenance=None,
                die_side=0.016, max_residual_px=50.0, max_height_m=1.0):
    """Track all solvable die poses in a clip; return a list of Trajectories."""
    src = FrameSource(video)
    start, end = frame_range if frame_range else (0, src.frame_count)
    fps = src.fps or 30.0
    asset_id = asset_id or calib.get("asset_id") or "clip"
    provenance = provenance if provenance is not None else {"source": video}

    tracks = []
    active = []
    die_index = 0
    for idx, frame in src.frames(start=start, end=end, step=stride):
        detections = detect_frame_poses(frame, calib, max_residual_px=max_residual_px,
                                        die_side=die_side)
        detections = [d for d in detections if -0.05 <= d["pos"][2] <= max_height_m]
        used = set()
        for track in active:
            best, best_dist = -1, match_px
            for j, det in enumerate(detections):
                if j in used:
                    continue
                dist = _centroid_distance(track["last"]["centroid"], det["centroid"])
                if dist < best_dist:
                    best_dist, best = dist, j
            if best >= 0:
                used.add(best)
                track["samples"].append((idx, detections[best]))
                track["last"] = detections[best]
                track["gap"] = 0
            else:
                track["gap"] += 1
        for j, det in enumerate(detections):
            if j not in used:
                active.append({"samples": [(idx, det)], "last": det, "gap": 0, "die_index": None})
        remaining = []
        for track in active:
            if track["gap"] > max_gap:
                if len(track["samples"]) >= min_samples:
                    track["die_index"] = die_index
                    die_index += 1
                    tracks.append(track)
            else:
                remaining.append(track)
        active = remaining

    for track in active:
        if len(track["samples"]) >= min_samples:
            track["die_index"] = die_index
            die_index += 1
            tracks.append(track)
    src.release()

    trajectories = [
        _to_trajectory(t, calib, fps, asset_id, synthetic, provenance)
        for t in tracks
    ]
    trajectories.sort(key=lambda tr: (tr.times[0], tr.die_index))
    for i, traj in enumerate(trajectories):
        traj.die_index = i
    return trajectories
