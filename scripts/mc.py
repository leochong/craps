import argparse
import json
import os

from m2 import ev, montecarlo
from m2.montecarlo import DEFAULT_SPEC


def main():
    ap = argparse.ArgumentParser(description="Monte Carlo rollout distribution + EV table")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--impulse", type=float, default=DEFAULT_SPEC["impulse"])
    ap.add_argument("--impulse-std", type=float, default=DEFAULT_SPEC["impulse_std"])
    ap.add_argument("--yaw", type=float, default=DEFAULT_SPEC["yaw"])
    ap.add_argument("--yaw-std", type=float, default=DEFAULT_SPEC["yaw_std"])
    ap.add_argument("--spin-std", type=float, default=DEFAULT_SPEC["spin_std"])
    ap.add_argument("--pos-std", type=float, default=DEFAULT_SPEC["pos_std"])
    ap.add_argument("--angle-std", type=float, default=DEFAULT_SPEC["angle_std"])
    ap.add_argument("--max-time", type=float, default=DEFAULT_SPEC["max_time"])
    ap.add_argument("--threshold", type=float, default=0.02)
    ap.add_argument("--out", default="out/mc")
    args = ap.parse_args()

    spec = dict(DEFAULT_SPEC)
    spec.update(
        impulse=args.impulse,
        impulse_std=args.impulse_std,
        yaw=args.yaw,
        yaw_std=args.yaw_std,
        spin_std=args.spin_std,
        pos_std=args.pos_std,
        angle_std=args.angle_std,
        max_time=args.max_time,
    )

    dist, results = montecarlo.monte_carlo(spec, n=args.n, seed=args.seed, workers=args.workers)
    summary = montecarlo.summarize(dist)
    table = ev.ev_table(dist["P"], threshold=args.threshold)

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
    out = {
        "spec": {k: v for k, v in spec.items() if k != "geom"},
        "geometry": spec["geom"],
        "summary": summary,
        "distribution": {str(s): dist["P"][s] for s in range(2, 13)},
        "counts": {str(s): dist["counts"][s] for s in range(2, 13)},
        "kl_vs_fair": dist["kl_vs_fair"],
        "ev_table": table,
        "best_opportunity": ev.best_opportunity(dist["P"], threshold=args.threshold),
        "house_edge_reference": ev.house_edge_reference(),
    }
    with open(os.path.join(args.out, "mc_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}/mc_summary.json")


if __name__ == "__main__":
    main()