"""Validation for geometric plate calibration (Phase C1).

Checks circle-from-ellipse plane pose recovery on synthetic cameras, plane
back-projection round-trips, the 4-point homography, and graceful fallback when
no plane pose is available. Prints [PASS]/[FAIL]; exits nonzero on failure.
"""

import sys

import cv2
import numpy as np

from m1 import homography as H

W, HH, F = 1920, 1080, 1400.0


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def intrinsics():
    return np.array([[F, 0.0, W / 2.0], [0.0, F, HH / 2.0], [0.0, 0.0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def synthetic_ellipse(R, t, K, radius_m, n=400):
    phi = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    world = np.stack([radius_m * np.cos(phi), radius_m * np.sin(phi), np.zeros(n)], axis=1)
    cam = world @ R.T + t
    uv = (cam[:, :2] / cam[:, 2:3]) @ K[:2, :2].T + K[:2, 2]
    (cx, cy), (a, b), angle = cv2.fitEllipse(uv.astype(np.float32))
    return {"cx": float(cx), "cy": float(cy), "major_px": float(a / 2.0),
            "minor_px": float(b / 2.0), "angle": float(angle),
            "radius": float((a + b) / 4.0)}


def bench_circle_pose():
    K = intrinsics()
    radius_m = 0.15
    center = np.array([0.05, 0.08, 0.6])
    worst_n, worst_t, worst_res = 0.0, 0.0, 0.0
    for tilt in (0.2, 0.6, 0.9, 1.1):
        R = rot_x(np.pi) @ rot_y(tilt) @ rot_x(0.2 * tilt)
        t = -R @ center
        ellipse = synthetic_ellipse(R, t, K, radius_m)
        pose = H.pose_from_circle(ellipse, K, radius_m)
        if pose is None:
            return check("circle-from-ellipse pose", False, f"tilt={tilt} returned None")
        n_err = float(np.degrees(np.arccos(np.clip(abs(np.dot(R[:, 2], pose["normal"])), -1, 1))))
        t_err = float(np.linalg.norm(t - pose["t"]))
        worst_n = max(worst_n, n_err)
        worst_t = max(worst_t, t_err)
        worst_res = max(worst_res, pose["residual"])
    ok = worst_n < 0.5 and worst_t < 0.002 and worst_res < 1e-2
    return check("circle-from-ellipse pose", ok,
                 f"max_normal_err={worst_n:.3f}deg max_t_err={worst_t*1000:.3f}mm")


def bench_plane_roundtrip():
    K = intrinsics()
    R = rot_x(np.pi) @ rot_y(0.7) @ rot_x(0.15)
    t = -R @ np.array([0.03, -0.02, 0.55])
    calib = {"K": K.tolist(), "R": R.tolist(), "t": t.tolist()}
    world = np.array([0.04, -0.03, 0.0])
    cam = R @ world + t
    back = H.camera_to_machine(cam, calib)
    cam_err = float(np.linalg.norm(back - world))
    uv = (K @ cam)[:2] / (K @ cam)[2]
    plane = H.machine_from_image([uv], calib)[0]
    plane_err = float(np.linalg.norm(plane - world))
    ok = cam_err < 1e-9 and plane_err < 1e-9
    return check("plane back-projection round-trip", ok,
                 f"cam_err={cam_err:.2e} plane_err={plane_err:.2e}")


def bench_homography():
    world = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.08], [0.0, 0.08]])
    H_true = np.array([[900.0, 40.0, 950.0], [20.0, 880.0, 520.0], [0.05, 0.02, 1.0]])
    homog = np.concatenate([world, np.ones((4, 1))], axis=1)
    img = (H_true @ homog.T).T
    img = img[:, :2] / img[:, 2:3]
    H_est = H.homography_from_points(img, world)
    img_h = np.concatenate([img, np.ones((4, 1))], axis=1)
    pred = (H_est @ img_h.T).T
    pred = pred[:, :2] / pred[:, 2:3]
    err = float(np.abs(pred - world).max())
    return check("4-point homography", err < 1e-6, f"max_err={err:.2e}m")


def bench_fallback():
    pts = np.array([[960.0, 540.0], [1060.0, 540.0]])
    calib = {"cx": 960.0, "cy": 540.0, "scale_m_per_px": 0.001, "width": W, "height": HH}
    xy = H.machine_from_image(pts, calib)
    flat_ok = xy.shape == (2, 3) and np.allclose(xy[:, 2], 0.0) and abs(xy[1, 0] - 0.1) < 1e-9
    return check("top-down fallback (no plane pose)", flat_ok,
                 f"xy={np.round(xy, 4).tolist()}")


def main():
    results = [
        bench_circle_pose(),
        bench_plane_roundtrip(),
        bench_homography(),
        bench_fallback(),
    ]
    print(f"\n{sum(results)}/{len(results)} calibration-geometry benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()