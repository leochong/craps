"""Inverse-calibrate machine parameters from a measured die trajectory.

Phase A fits only the low-order, 30 fps-observable parameters (impulse + mu_felt)
by default; contact restitution is left at priors until higher-fps capture exists.
"""

import argparse
import json
import os

import numpy as np

from m1.trajectory import load_trajectories
from m2 import adjoint
from m2.montecarlo import DEFAULT_GEOM
from scripts.fit_synth import parse_active

THETA0 = [0.5, 0.4, 0.02]


def build_spec(traj, dt=0.001, elevation=20.0, yaw=0.0, steps=None):
    span = float(traj.times[-1] - traj.times[0])
    if not steps:
        steps = max(int(round(span / dt)), 1)
    return dict(
        geom=dict(DEFAULT_GEOM), side=0.016, mass=0.006,
        pos=tuple(float(v) for v in traj.pos[0]), quat=[0.0, 0.0, 0.0, 1.0],
        omega=(0.0, 0.0, 0.0), yaw=yaw, elevation=elevation,
        dt=dt, steps=steps, solver_iterations=12,
    )


def main():
    ap = argparse.ArgumentParser(description="Fit physics params from a measured trajectory")
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--die-index", type=int, default=0)
    ap.add_argument("--active", default="mu_felt,impulse")
    ap.add_argument("--method", choices=["cmaes", "lm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="out/fit_clip.json")
    args = ap.parse_args()

    asset_id = args.asset_id or "clip"
    path = args.trajectory or os.path.join("datasets", "trajectories", f"{asset_id}.jsonl")
    if not os.path.exists(path):
        print(f"SKIP: no trajectory at {path} (run scripts/track_clip.py first)")
        return

    trajectories = load_trajectories(path)
    match = [t for t in trajectories if t.die_index == args.die_index]
    if not match:
        print(f"FAIL: die_index={args.die_index} not found in {path}")
        return
    traj = match[0]
    if traj.synthetic:
        print("FAIL: refusing to calibrate on a synthetic trajectory")
        return

    target = adjoint.target_from_trajectory(traj)
    target["times"] = target["times"] - target["times"][0]
    target["pos"] = target["pos"] - np.array([0.0, 0.0, target["pos"][0, 2]])
    spec = build_spec(traj, dt=args.dt, elevation=args.elevation, yaw=args.yaw,
                      steps=args.steps or None)
    spec["pos"] = (float(traj.pos[0, 0]), float(traj.pos[0, 1]), 0.0)
    active = parse_active(args.active)

    loss0 = adjoint.loss(THETA0, target, spec)
    result = adjoint.fit(THETA0, target, spec, method=args.method, active=active,
                         iters=args.iters, maxiter=args.iters, seed=args.seed)
    ident = adjoint.identifiability(result["theta"], target, spec, active=active)

    print(f"asset={traj.asset_id} die={traj.die_index} samples={len(traj)} "
          f"method={result.get('method', args.method)} active={args.active}")
    print(f"{'param':>10} {'init':>10} {'fitted':>10}")
    for i, name in enumerate(adjoint.PARAM_NAMES):
        tag = "" if (active is None or i in active) else " (frozen)"
        print(f"{name:>10} {THETA0[i]:>10.5f} {result['theta'][i]:>10.5f}{tag}")
    print(f"\nloss: initial={loss0:.3e} final={result['loss']:.3e} "
          f"reduction={100 * (1 - result['loss'] / max(loss0, 1e-30)):.2f}%")
    print(f"identifiability: cond(J)={ident['condition_number']:.2f} sigma_min={ident['sigma_min']:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "asset_id": traj.asset_id,
        "die_index": traj.die_index,
        "samples": len(traj),
        "init": THETA0,
        "fitted": result["theta"].tolist(),
        "active": args.active,
        "loss_initial": loss0,
        "loss_final": result["loss"],
        "method": result.get("method", args.method),
        "identifiability": ident,
        "spec": {k: v for k, v in spec.items() if k != "geom"},
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
