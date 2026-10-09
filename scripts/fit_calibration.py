"""Multi-clip inverse calibration: fit shared machine parameters across clips.

Aggregates measured trajectories (one clip per die track) into a single robust
inverse problem, optionally activating extended geometry parameters
(mu/e for felt/dome/bumper and angular damping). 30 fps-observable launch/felt
parameters should be preferred; contact restitution needs higher-fps capture.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from m1.trajectory import load_trajectories
from m2 import adjoint
from m2.montecarlo import DEFAULT_GEOM


def parse_overrides(text):
    out = {}
    if not text:
        return out
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=")
        out[key.strip()] = float(value)
    return out


def parse_active(text):
    if not text:
        return None
    names = [t.strip() for t in text.split(",") if t.strip()]
    for name in names:
        if name not in adjoint.ALL_PARAM_NAMES:
            raise ValueError(f"unknown parameter '{name}'")
    return [adjoint.ALL_PARAM_NAMES.index(name) for name in names]


def load_clip_files(paths, patterns):
    files = list(paths)
    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))
    return sorted(set(files))


def build_clip(traj, dt=0.001, steps=None, specs=None, geom=None):
    span = float(traj.times[-1] - traj.times[0])
    if not steps:
        steps = max(int(round(span / dt)), 1)
    launch = (specs or {}).get(traj.asset_id, {})
    pos = traj.pos - np.array([0.0, 0.0, traj.pos[0, 2]])
    spec = dict(
        geom=dict(geom or DEFAULT_GEOM), side=0.016, mass=0.006,
        pos=tuple(float(v) for v in pos[0]), quat=[0.0, 0.0, 0.0, 1.0],
        omega=(0.0, 0.0, 0.0),
        yaw=float(launch.get("yaw", 0.0)),
        elevation=float(launch.get("elevation", 1.2)),
        dt=dt, steps=steps, solver_iterations=12,
    )
    target = adjoint.target_from_trajectory(traj)
    target["times"] = target["times"] - target["times"][0]
    target["pos"] = pos
    return {"spec": spec, "target": target, "asset_id": traj.asset_id,
            "die_index": traj.die_index}


def collect_clips(trajectories, dt, steps, specs):
    clips = []
    skipped = 0
    for traj in trajectories:
        if traj.synthetic:
            skipped += 1
            continue
        clips.append(build_clip(traj, dt=dt, steps=steps, specs=specs))
    return clips, skipped


def main():
    ap = argparse.ArgumentParser(description="Fit shared machine params from many trajectory clips")
    ap.add_argument("--trajectory", action="append", default=[], help="trajectory JSONL (repeatable)")
    ap.add_argument("--pattern", action="append", default=[], help="glob of trajectory JSONLs")
    ap.add_argument("--specs", default=None, help="JSON mapping asset_id -> {yaw, elevation}")
    ap.add_argument("--init", default=None, help="init overrides, e.g. mu_felt=0.45,impulse=0.02")
    ap.add_argument("--active", default="mu_felt,e_felt,impulse")
    ap.add_argument("--method", choices=["lm", "cmaes"], default="lm")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="out/fit_calibration.json")
    args = ap.parse_args()

    files = load_clip_files(args.trajectory, args.pattern)
    if not files:
        files = sorted(glob.glob(os.path.join("datasets", "trajectories", "*.jsonl")))
    if not files:
        print("SKIP: no trajectory files found (run scripts/track_clip.py first)")
        return

    trajectories = []
    for path in files:
        trajectories.extend(load_trajectories(path))
    specs = {}
    if args.specs:
        with open(args.specs, encoding="utf-8") as f:
            specs = json.load(f)

    clips, skipped = collect_clips(trajectories, args.dt, args.steps or None, specs)
    if not clips:
        print(f"FAIL: no usable clips ({skipped} synthetic skipped)")
        sys.exit(1)

    theta0 = adjoint.default_theta(overrides=parse_overrides(args.init))
    active = parse_active(args.active)
    L0 = adjoint.multi_loss(theta0, clips)
    result = adjoint.fit_multi(theta0, clips, method=args.method, active=active,
                               iters=args.iters, maxiter=args.iters, seed=args.seed)
    ident = adjoint.multi_identifiability(result["theta"], clips, active=active)

    names = adjoint.ALL_PARAM_NAMES
    print(f"clips={len(clips)} (synthetic skipped={skipped}) files={len(files)} "
          f"method={result.get('method', args.method)} active={args.active}")
    print(f"{'param':>16} {'init':>10} {'fitted':>10}")
    for i, name in enumerate(names):
        tag = "" if (active is None or i in active) else " (frozen)"
        print(f"{name:>16} {theta0[i]:>10.5f} {result['theta'][i]:>10.5f}{tag}")
    print(f"\nloss: initial={L0:.4e} final={result['loss']:.4e} "
          f"reduction={100 * (1 - result['loss'] / max(L0, 1e-30)):.2f}%")
    print(f"identifiability: cond(J)={ident['condition_number']:.2f} sigma_min={ident['sigma_min']:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "clips": len(clips),
        "synthetic_skipped": skipped,
        "files": files,
        "init": theta0.tolist(),
        "fitted": result["theta"].tolist(),
        "param_names": list(names),
        "active": args.active,
        "loss_initial": L0,
        "loss_final": result["loss"],
        "method": result.get("method", args.method),
        "identifiability": ident,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()