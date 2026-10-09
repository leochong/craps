"""Validation for the M1->M2 trajectory bridge (Phase A).

Checks trajectory round-trip, plate/machine-frame mapping, tvec->machine
invertibility, the adjoint target bridge (including synthetic rejection), tracking
smoke, and that a synthetic self-fit still reduces loss. Prints [PASS]/[FAIL].
"""

import os
import shutil
import sys

import numpy as np

from m1 import homography
from m1.track import track_poses
from m1.trajectory import Trajectory, load_trajectories, save_trajectories, validate_trajectory
from m2 import adjoint
from scripts.fit_synth import THETA0, TRUE_THETA, build_spec

WORK = "out/traj_validate"
CLIP = os.path.join(WORK, "clip.mp4")


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def make_clip(path, n_frames=6, size=(96, 64), fps=10.0):
    import cv2

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        return False
    rng = np.random.RandomState(0)
    for i in range(n_frames):
        frame = rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        cv2.rectangle(frame, (20, 15), (45, 40), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return os.path.exists(path)


def bench_roundtrip():
    traj = Trajectory(asset_id="sim", die_index=1,
                      times=[0.0, 0.1, 0.2, 0.3],
                      pos=[[0, 0, 0.01], [0.01, 0.02, 0.05], [0.03, 0.01, 0.02], [0.04, 0, 0.008]],
                      quality=[1.0, 0.8, 0.9, 1.0], provenance={"source": "unit"})
    path = os.path.join(WORK, "traj.jsonl")
    save_trajectories([traj], path)
    loaded = load_trajectories(path)[0]
    ok = (np.allclose(loaded.times, traj.times) and np.allclose(loaded.pos, traj.pos)
          and np.allclose(loaded.quality, traj.quality)
          and loaded.die_index == traj.die_index and not validate_trajectory(loaded))
    return check("trajectory round-trip", ok,
                 f"n={len(loaded)} errors={validate_trajectory(loaded)}")


def _calib(width=640, height=480, f=640.0, scale=0.0005, cx=320.0, cy=240.0, dist=0.5):
    from m1.pose import default_intrinsics

    return {"cx": cx, "cy": cy, "scale_m_per_px": scale, "focal": f,
            "camera_distance_m": dist, "width": width, "height": height,
            "K": default_intrinsics(width, height).tolist()}


def bench_plate_mapping():
    calib = _calib()
    pts = np.array([[calib["cx"], calib["cy"]], [calib["cx"] + 100, calib["cy"] - 50]])
    xy = homography.image_to_machine_xy(pts, calib)
    expected = np.array([[0.0, 0.0], [100 * calib["scale_m_per_px"], 50 * calib["scale_m_per_px"]]])
    ok = np.allclose(xy, expected)
    return check("plate image->machine XY", ok, f"xy={np.round(xy, 5).tolist()}")


def bench_tvec_inverse():
    calib = _calib()
    X, Y, Z = 0.05, -0.02, 0.03
    u = calib["cx"] + X / calib["scale_m_per_px"]
    v = calib["cy"] - Y / calib["scale_m_per_px"]
    zc = calib["camera_distance_m"] - Z
    tvec = np.array([(u - calib["width"] / 2) * zc / calib["focal"],
                     (v - calib["height"] / 2) * zc / calib["focal"], zc])
    rec = homography.tvec_to_machine(tvec, calib)
    err = float(np.abs(rec - np.array([X, Y, Z])).max())
    ok = err < 1e-6
    return check("tvec -> machine invertible", ok, f"err={err:.2e} m")


def bench_bridge():
    spec = build_spec(steps=400, elevation=20.0, omega=(0.0, 0.0, 0.0))
    ts, xs = adjoint.run_trajectory(TRUE_THETA, spec)
    traj = Trajectory(asset_id="sim", die_index=0, times=ts, pos=xs,
                      provenance={"source": "sim"})
    target = adjoint.target_from_trajectory(traj)
    loss0 = adjoint.loss(THETA0, target, spec)
    res = adjoint.fit_cmaes(THETA0, target, spec, active=[0, 2], maxiter=40, sigma0=0.2, seed=3)
    ok = np.isfinite(loss0) and res["loss"] < loss0
    return check("adjoint bridge fit reduces loss", ok,
                 f"loss {loss0:.3e} -> {res['loss']:.3e}")


def bench_synthetic_rejected():
    traj = Trajectory(asset_id="rife", die_index=0, times=[0.0, 0.1, 0.2],
                      pos=[[0, 0, 0.0], [0, 0, 0.1], [0, 0, 0.0]],
                      synthetic=True, provenance={"source": "rife-interp"})
    raised = False
    try:
        adjoint.target_from_trajectory(traj)
    except ValueError:
        raised = True
    allowed = adjoint.target_from_trajectory(traj, allow_synthetic=True)
    ok = raised and allowed["pos"].shape == (3, 3)
    return check("synthetic trajectory rejected", ok, f"raised={raised}")


def bench_track_smoke():
    if not make_clip(CLIP):
        return check("track_poses runs", False, "could not create clip")
    calib = _calib(width=96, height=64, f=96.0, scale=0.002, cx=48.0, cy=32.0, dist=0.2)
    tracks = track_poses(CLIP, calib, min_samples=1, asset_id="clip")
    ok = isinstance(tracks, list) and all(isinstance(t, Trajectory) for t in tracks)
    return check("track_poses runs", ok, f"tracks={len(tracks)}")


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK, exist_ok=True)
    results = [
        bench_roundtrip(),
        bench_plate_mapping(),
        bench_tvec_inverse(),
        bench_bridge(),
        bench_synthetic_rejected(),
        bench_track_smoke(),
    ]
    print(f"\n{sum(results)}/{len(results)} trajectory benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
