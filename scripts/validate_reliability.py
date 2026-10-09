"""Validation for probability-calibration metrics (Phase B).

Checks Brier scoring, reliability curves, and expected calibration error on
synthetic calibrated/miscalibrated forecasts. Prints [PASS]/[FAIL]; exits
nonzero on failure.
"""

import sys

import numpy as np

from m2 import metrics


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def bench_brier_perfect():
    outcomes = [2, 5, 7, 11, 12]
    probs = [{s: (1.0 if s == o else 0.0) for s in metrics.CATEGORIES} for o in outcomes]
    score = metrics.brier_multiclass(probs, outcomes)
    return check("Brier perfect = 0", abs(score) < 1e-12, f"brier={score:.2e}")


def bench_brier_uniform():
    n = 500
    probs = [{s: 1.0 / len(metrics.CATEGORIES) for s in metrics.CATEGORIES} for _ in range(n)]
    outcomes = list(np.random.default_rng(0).choice(metrics.CATEGORIES, n))
    expected = (len(metrics.CATEGORIES) - 1) / len(metrics.CATEGORIES)
    score = metrics.brier_multiclass(probs, outcomes)
    return check("Brier uniform baseline", abs(score - expected) < 1e-9,
                 f"brier={score:.5f} expected={expected:.5f}")


def bench_reliability_calibrated():
    rng = np.random.default_rng(1)
    n = 20000
    probs = rng.uniform(0.05, 0.95, n)
    outcomes = (rng.uniform(0.0, 1.0, n) < probs).astype(float)
    bins = metrics.reliability_curve(probs, outcomes, n_bins=10)
    ece = metrics.expected_calibration_error(bins)
    return check("calibrated forecasts -> low ECE", metrics.calibration_gate(ece),
                 f"ece={ece:.4f} gate={metrics.ECE_GATE}")


def bench_reliability_miscalibrated():
    rng = np.random.default_rng(2)
    n = 5000
    probs = np.full(n, 0.5)
    outcomes = (rng.uniform(0.0, 1.0, n) < 0.9).astype(float)
    bins = metrics.reliability_curve(probs, outcomes, n_bins=10)
    ece = metrics.expected_calibration_error(bins)
    return check("miscalibrated forecasts -> high ECE", not metrics.calibration_gate(ece),
                 f"ece={ece:.4f}")


def bench_multiclass_reliability():
    rng = np.random.default_rng(3)
    true_p = {s: (0.2 if s == 7 else 0.8 / 10.0) for s in metrics.CATEGORIES}
    probs = [dict(true_p) for _ in range(4000)]
    outcomes = list(rng.choice(metrics.CATEGORIES, 4000, p=[true_p[s] for s in metrics.CATEGORIES]))
    score = metrics.brier_multiclass(probs, outcomes)
    indicators = [1.0 if o == 7 else 0.0 for o in outcomes]
    p7 = [true_p[7]] * len(outcomes)
    bins = metrics.reliability_curve(p7, indicators, n_bins=1)
    ece = metrics.expected_calibration_error(bins)
    expected_brier = 1.0 - sum(v * v for v in true_p.values())
    ok = abs(score - expected_brier) < 0.02 and ece < 0.03
    return check("multiclass distribution reliability", ok,
                 f"brier={score:.4f}(exp {expected_brier:.4f}) ece(sum7)={ece:.4f}")


def main():
    results = [
        bench_brier_perfect(),
        bench_brier_uniform(),
        bench_reliability_calibrated(),
        bench_reliability_miscalibrated(),
        bench_multiclass_reliability(),
    ]
    print(f"\n{sum(results)}/{len(results)} reliability benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()