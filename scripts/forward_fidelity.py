"""Forward-fidelity report: simulated vs measured trajectories for a calibration.

Replays each measured trajectory through the forward model with the fitted
parameters and reports cumulative spatial divergence over the first impacts. The
M3 gate (< 8% relative divergence) is only enforced with --gate; by default this
is a report, since fitted parameters may still be unvalidated.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

from m1.trajectory import load_trajectories
from m2 import adjoint, metrics
from scripts.fit_calibration import build_clip, load_clip_files


def main():
    ap = argparse.ArgumentParser(description="Simulated-vs-measured trajectory divergence report")
    ap.add_argument("--calibration", default="out/fit_calibration.json")
    ap.add_argument("--trajectory", action="append", default=[])
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--specs", default=None)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=metrics.DIVERGENCE_GATE)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default="out/forward_fidelity.json")
    args = ap.parse_args()

    if not os.path.exists(args.calibration):
        print(f"SKIP: no calibration at {args.calibration}")
        return
    with open(args.calibration, encoding="utf-8") as f:
        payload = json.load(f)
    names = payload.get("param_names", list(adjoint.ALL_PARAM_NAMES))
    theta = adjoint.default_theta()
    for i, name in enumerate(adjoint.ALL_PARAM_NAMES):
        if name in names:
            theta[i] = float(payload["fitted"][names.index(name)])

    files = load_clip_files(args.trajectory, args.pattern)
    if not files:
        files = sorted(glob.glob(os.path.join("datasets", "trajectories", "*.jsonl")))
    if not files:
        print("SKIP: no trajectory files found")
        return
    specs = {}
    if args.specs:
        with open(args.specs, encoding="utf-8") as f:
            specs = json.load(f)

    rows = []
    for path in files:
        for traj in load_trajectories(path):
            if traj.synthetic:
                continue
            clip = build_clip(traj, dt=args.dt, steps=args.steps or None, specs=specs)
            pred_t, pred_x = adjoint.run_trajectory(theta, clip["spec"])
            report = metrics.trajectory_divergence(
                pred_t, pred_x, clip["target"]["times"], clip["target"]["pos"])
            rows.append({"asset_id": traj.asset_id, "die_index": traj.die_index,
                         "n": len(traj), **report})

    if not rows:
        print("FAIL: no usable (non-synthetic) trajectories")
        sys.exit(1)

    print(f"{'asset':>22} {'die':>3} {'n':>4} {'impacts':>7} {'relative':>10} {'abs_mean_m':>11}")
    for r in rows:
        print(f"{r['asset_id']:>22} {r['die_index']:>3} {r['n']:>4} {r['n_impacts']:>7} "
              f"{r['relative']:>10.4f} {r['absolute_mean_m']:>11.4f}")
    worst = max(rows, key=lambda r: r["relative"])
    passed = metrics.divergence_gate(worst, args.threshold)
    print(f"\nworst relative divergence={worst['relative']:.4f} "
          f"gate={args.threshold} -> {'PASS' if passed else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"threshold": args.threshold, "passed": passed, "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")
    if args.gate and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()