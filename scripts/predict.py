"""End-to-end prediction: calibrated parameters -> Monte Carlo -> EV table.

Produces the terminal-sum distribution P(2..12) and the per-wager EV/Kelly table
from a fitted calibration. Output is explicitly UNVALIDATED until the forward
fidelity gate (Phase B) passes; treat the EV table as a pipeline demonstration.
"""

import argparse
import json
import os

from m2 import ev, montecarlo
from m2.calibration import load_calibration, quality_flags, theta_to_spec

VALIDATED = False


def run_prediction(spec, n=32, seed=0, workers=1, threshold=0.02, provenance=None):
    """Run Monte Carlo + EV from a spec; return (dist, ev_table, payload)."""
    dist, _ = montecarlo.monte_carlo(spec, n=n, seed=seed, workers=workers)
    summary = montecarlo.summarize(dist)
    table = ev.ev_table(dist["P"], threshold=threshold)
    payload = {
        "validated": VALIDATED,
        "provenance": provenance or {},
        "n": dist["n"],
        "distribution": {str(s): dist["P"][s] for s in range(2, 13)},
        "counts": {str(s): dist["counts"][s] for s in range(2, 13)},
        "kl_vs_fair": dist["kl_vs_fair"],
        "mean_sum": dist["mean_sum"],
        "settled_fraction": dist["settled_fraction"],
        "ev_table": table,
        "best_opportunity": ev.best_opportunity(dist["P"], threshold=threshold),
        "house_edge_reference": ev.house_edge_reference(),
        "geometry": spec["geom"],
        "spec": {k: v for k, v in spec.items() if k != "geom"},
        "summary": summary,
    }
    return dist, table, payload


def main():
    ap = argparse.ArgumentParser(description="Predict P(2..12) + EV from calibrated params (unvalidated)")
    ap.add_argument("--calibration", default="out/fit_calibration.json")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--yaw", type=float, default=None)
    ap.add_argument("--out", default="out/predict")
    args = ap.parse_args()

    if not os.path.exists(args.calibration):
        print(f"SKIP: no calibration at {args.calibration} (run scripts/fit_calibration.py)")
        return

    report = load_calibration(args.calibration)
    launch = {}
    if args.max_time is not None:
        launch["max_time"] = args.max_time
    if args.elevation is not None:
        launch["elevation"] = args.elevation
    if args.yaw is not None:
        launch["yaw"] = args.yaw
    spec = theta_to_spec(report["theta"], launch=launch)

    flags = quality_flags(report)
    print("=" * 64)
    print("UNVALIDATED PREDICTION - pipeline demonstration only")
    print("calibrated parameters are not yet validated by the forward-fidelity gate")
    if flags:
        print("WARNINGS: " + "; ".join(flags))
    print("=" * 64)

    provenance = {
        "calibration": os.path.abspath(args.calibration),
        "active": report.get("active"),
        "at_bounds": report["at_bounds"],
        "identifiability": report.get("identifiability", {}),
    }
    dist, table, payload = run_prediction(spec, n=args.n, seed=args.seed,
                                          workers=args.workers, threshold=args.threshold,
                                          provenance=provenance)

    print(f"rollouts: {dist['n']}   mean_sum: {dist['mean_sum']:.3f}   "
          f"KL(P||fair): {dist['kl_vs_fair']:.5f}   settled: {dist['settled_fraction']*100:.0f}%")
    print("sum   P       95% CI")
    for s in range(2, 13):
        lo, hi = dist["ci"][s]
        print(f"{s:>3}   {dist['P'][s]:.4f}   [{lo:.4f}, {hi:.4f}]")
    print("\ntarget  odds    p_win    EV       Kelly   signal")
    for r in table:
        print(f"{r['target']:>5}   {r['odds']:.3f}   {r['p_win']:.4f}   "
              f"{r['ev']*100:+.3f}%  {r['kelly']:.4f}  {'YES' if r['signal'] else ''}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "predict_summary.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path}  (validated={payload['validated']})")


if __name__ == "__main__":
    main()