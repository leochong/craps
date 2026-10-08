import argparse
import json
import os

import numpy as np

from m2 import adjoint
from m2.montecarlo import DEFAULT_GEOM

FIT_GEOM = dict(DEFAULT_GEOM)
FIT_GEOM.update(radius=50.0, height=50.0, bumper_major=50.0, bumper_minor=0.001, bumper_z=50.0)

TRUE_THETA = [0.5, 0.4, 0.02]
THETA0 = [0.9, 0.7, 0.035]


def build_spec(steps=1000, elevation=20.0, omega=(0.0, 0.0, 0.0), dt=0.001, solver_iterations=12):
    return dict(
        geom=dict(FIT_GEOM),
        side=0.016,
        mass=0.006,
        pos=(0.0, 0.0, 0.008),
        quat=[0.0, 0.0, 0.0, 1.0],
        omega=omega,
        yaw=0.5,
        elevation=elevation,
        dt=dt,
        steps=steps,
        solver_iterations=solver_iterations,
    )


def parse_active(text):
    if not text:
        return None
    names = [t.strip() for t in text.split(",") if t.strip()]
    return [adjoint.PARAM_NAMES.index(n) for n in names]


def rel_error(true, est, idx):
    return {adjoint.PARAM_NAMES[i]: abs(est[i] - true[i]) / max(abs(true[i]), 1e-9) for i in idx}


def main():
    ap = argparse.ArgumentParser(description="Synthetic inverse-physics recovery (M3)")
    ap.add_argument("--method", choices=["cmaes", "lm"], default="cmaes")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--omega", default="0,0,0")
    ap.add_argument("--sample-n", type=int, default=60)
    ap.add_argument("--active", default="e_felt,impulse")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="out/fit_synth.json")
    args = ap.parse_args()

    omega = tuple(float(v) for v in args.omega.split(","))
    spec = build_spec(steps=args.steps, elevation=args.elevation, omega=omega)
    active = parse_active(args.active)

    target = adjoint.make_target(TRUE_THETA, spec, sample_n=args.sample_n, noise=args.noise, seed=args.seed)
    L0 = adjoint.loss(THETA0, target, spec)
    result = adjoint.fit(THETA0, target, spec, method=args.method, active=active,
                         iters=args.iters, maxiter=args.iters, seed=args.seed)
    ident = adjoint.identifiability(result["theta"], target, spec, active=active)

    print(f"method: {result.get('method', args.method)}   active: {args.active}   "
          f"horizon: {args.steps * spec['dt']:.2f}s   noise: {args.noise}")
    print(f"{'param':>10} {'true':>10} {'init':>10} {'recovered':>10} {'rel.err':>9}")
    for i, name in enumerate(adjoint.PARAM_NAMES):
        tag = "" if (active is None or i in active) else " (frozen)"
        print(f"{name:>10} {TRUE_THETA[i]:>10.5f} {THETA0[i]:>10.5f} "
              f"{result['theta'][i]:>10.5f} {rel_error(TRUE_THETA, result['theta'], [i])[name]*100:>8.2f}%{tag}")

    print(f"\nloss: initial={L0:.3e}  final={result['loss']:.3e}  "
          f"reduction={100*(1 - result['loss']/max(L0,1e-30)):.2f}%")
    print(f"identifiability: cond(J)={ident['condition_number']:.2f}  "
          f"sigma_min={ident['sigma_min']:.3e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        "true": TRUE_THETA,
        "init": THETA0,
        "recovered": result["theta"].tolist(),
        "active": args.active,
        "rel_error": rel_error(TRUE_THETA, result["theta"], active or list(range(3))),
        "loss_initial": L0,
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