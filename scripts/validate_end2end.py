"""Validation for the end-to-end prediction path (Phase A).

Checks calibrated-parameter -> spec mapping, calibration loading/bound flags,
Monte Carlo determinism and normalisation, KL behaviour, and that the prediction
payload is explicitly unvalidated. Prints [PASS]/[FAIL]; exits nonzero on failure.
"""

import json
import os
import shutil
import sys

import numpy as np

from m2 import adjoint, montecarlo
from m2.calibration import load_calibration, quality_flags, theta_to_spec
from m2.montecarlo import DEFAULT_SPEC
from scripts.predict import run_prediction

WORK = "out/end2end_validate"


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def cheap_spec():
    spec = dict(DEFAULT_SPEC)
    spec.update(max_time=0.4, dt=0.002, solver_iterations=6)
    return spec


def bench_theta_to_spec():
    theta = adjoint.default_theta(overrides={"mu_felt": 0.3, "e_dome": 0.7, "impulse": 0.05})
    spec = theta_to_spec(theta)
    ok = (abs(spec["geom"]["mu_felt"] - 0.3) < 1e-12
          and abs(spec["geom"]["e_dome"] - 0.7) < 1e-12
          and abs(spec["impulse"] - 0.05) < 1e-12
          and spec["geom"]["radius"] == DEFAULT_SPEC["geom"]["radius"]
          and spec["base_positions"] == DEFAULT_SPEC["base_positions"])
    return check("theta -> MC spec mapping", ok,
                 f"mu={spec['geom']['mu_felt']} e_dome={spec['geom']['e_dome']} imp={spec['impulse']}")


def bench_load_calibration():
    path = os.path.join(WORK, "calib.json")
    payload = {
        "param_names": list(adjoint.ALL_PARAM_NAMES),
        "fitted": [0.0, 0.4, 0.08, 0.2, 0.5, 0.4, 0.6, 0.5],
        "active": "mu_felt,e_felt,impulse",
        "identifiability": {"condition_number": 5.0},
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    report = load_calibration(path)
    flags = quality_flags(report)
    ok = (abs(report["theta"][0]) < 1e-12 and "mu_felt" in report["at_bounds"]
          and report["active"] == "mu_felt,e_felt,impulse"
          and any("bounds" in f for f in flags))
    return check("load_calibration + bound flags", ok,
                 f"at_bounds={report['at_bounds']} flags={flags}")


def bench_mc_normalisation():
    spec = cheap_spec()
    dist_a, _ = montecarlo.monte_carlo(spec, n=6, seed=0)
    dist_b, _ = montecarlo.monte_carlo(spec, n=6, seed=0)
    total = sum(dist_a["P"].values())
    counts = sum(dist_a["counts"].values())
    deterministic = all(abs(dist_a["P"][s] - dist_b["P"][s]) < 1e-12 for s in range(2, 13))
    ok = abs(total - 1.0) < 1e-12 and counts == 6 and dist_a["kl_vs_fair"] >= 0 and deterministic
    return check("MC normalisation + determinism", ok,
                 f"sum={total:.6f} counts={counts} kl={dist_a['kl_vs_fair']:.4f} det={deterministic}")


def bench_biased_kl():
    results = [{"sum": 7, "settled": True} for _ in range(10)]
    dist = montecarlo.aggregate(results, n=10)
    fair = montecarlo.aggregate([{"sum": s, "settled": True} for s in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)], n=10)
    ok = abs(dist["P"][7] - 1.0) < 1e-12 and dist["kl_vs_fair"] > 0 and fair["kl_vs_fair"] >= 0
    return check("biased distribution has KL > 0", ok,
                 f"kl_biased={dist['kl_vs_fair']:.4f} kl_mixed={fair['kl_vs_fair']:.4f}")


def bench_payload_unvalidated():
    spec = cheap_spec()
    _, table, payload = run_prediction(spec, n=6, seed=1, threshold=0.02, provenance={"x": 1})
    ev_ok = len(table) == 10 and all("ev" in row and "signal" in row for row in table)
    ok = (payload["validated"] is False and payload["provenance"] == {"x": 1}
          and ev_ok and abs(sum(payload["distribution"].values()) - 1.0) < 1e-12)
    return check("prediction payload flagged unvalidated", ok,
                 f"validated={payload['validated']} ev_rows={len(table)}")


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK, exist_ok=True)
    results = [
        bench_theta_to_spec(),
        bench_load_calibration(),
        bench_mc_normalisation(),
        bench_biased_kl(),
        bench_payload_unvalidated(),
    ]
    print(f"\n{sum(results)}/{len(results)} end-to-end benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()