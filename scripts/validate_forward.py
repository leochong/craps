"""Validation for forward-fidelity metrics and the M3 divergence gate (Phase B).

Checks impact detection, zero divergence for identical trajectories, and that a
parameter fit on one clip predicts a held-out clip better than the prior and
below the acceptance gate. Prints [PASS]/[FAIL]; exits nonzero on failure.
"""

import sys

import numpy as np

from m2 import adjoint, metrics

FREE_GEOM = dict(radius=50.0, height=50.0, bumper_major=50.0, bumper_minor=0.001, bumper_z=50.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def free_spec(elevation, yaw, steps=120):
    return dict(geom=dict(FREE_GEOM), side=0.016, mass=0.006, pos=(0.0, 0.0, 0.008),
                quat=[0.0, 0.0, 0.0, 1.0], omega=(0.0, 0.0, 0.0), yaw=yaw,
                elevation=elevation, dt=0.001, steps=steps, solver_iterations=6)


def bounce_trajectory(n=240, dt=0.005):
    times = np.arange(n) * dt
    z = 0.12 * np.abs(np.sin(np.pi * 3.0 * times)) * np.exp(-0.6 * times)
    pos = np.stack([times, np.zeros(n), z], axis=1)
    return times, pos


def bench_impact_detection():
    times, pos = bounce_trajectory()
    impacts = metrics.detect_impacts(times, pos, floor_z=0.0, band=0.02)
    return check("impact detection", 3 <= len(impacts) <= 6, f"impacts={len(impacts)}")


def bench_identical_divergence():
    times, pos = bounce_trajectory()
    report = metrics.trajectory_divergence(times, pos, times, pos, n_impacts=3, floor_z=0.0)
    ok = report["relative"] == 0.0 and report["n_impacts"] >= 3
    return check("identical trajectory -> zero divergence", ok,
                 f"relative={report['relative']:.2e} impacts={report['n_impacts']}")


def bench_divergence_gate():
    true = adjoint.default_theta(overrides={"impulse": 0.03, "mu_felt": 0.4, "e_felt": 0.4})
    theta0 = adjoint.default_theta(overrides={"impulse": 0.02, "mu_felt": 0.4, "e_felt": 0.4})
    spec_a = free_spec(8.0, 0.2)
    spec_b = free_spec(14.0, 1.0)
    train = {"spec": spec_a, "target": adjoint.make_target(true, spec_a, sample_n=30)}
    result = adjoint.fit_multi(theta0, [train], method="lm", active=[2], iters=25)

    meas_t, meas_x = adjoint.run_trajectory(true, spec_b)
    fit_t, fit_x = adjoint.run_trajectory(result["theta"], spec_b)
    prior_t, prior_x = adjoint.run_trajectory(theta0, spec_b)
    rep_fit = metrics.trajectory_divergence(fit_t, fit_x, meas_t, meas_x)
    rep_prior = metrics.trajectory_divergence(prior_t, prior_x, meas_t, meas_x)

    ok = (metrics.divergence_gate(rep_fit)
          and rep_fit["relative"] < rep_prior["relative"]
          and np.isfinite(rep_fit["relative"]))
    return check("held-out divergence gate", ok,
                 f"fitted={rep_fit['relative']:.4f} prior={rep_prior['relative']:.4f} "
                 f"gate={metrics.DIVERGENCE_GATE}")


def main():
    results = [
        bench_impact_detection(),
        bench_identical_divergence(),
        bench_divergence_gate(),
    ]
    print(f"\n{sum(results)}/{len(results)} forward-fidelity benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()