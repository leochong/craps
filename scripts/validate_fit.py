import sys

import numpy as np

from m2 import adjoint
from m2.cmaes import cma_es
from scripts.fit_synth import TRUE_THETA, THETA0, build_spec


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def bench_cmaes_sphere():
    f = lambda x: float(np.sum((x - 3.0) ** 2))
    r = cma_es(f, np.array([0.0, 0.0]), 1.0, np.array([-10.0, -10.0]), np.array([10.0, 10.0]),
               maxiter=80, seed=0)
    return check("cma-es sphere minimum", np.allclose(r["x"], 3.0, atol=0.05),
                 f"x={np.round(r['x'],3)}")


def bench_drop_identifiability():
    spec = build_spec(steps=1000, elevation=20.0, omega=(0.0, 0.0, 0.0))
    target = adjoint.make_target(TRUE_THETA, spec, sample_n=60)
    ident = adjoint.identifiability(TRUE_THETA, target, spec, active=[1, 2])
    ok = np.isfinite(ident["condition_number"]) and ident["sigma_min"] > 0
    return check("drop identifiability (e, impulse)", ok,
                 f"cond={ident['condition_number']:.1f} sigma_min={ident['sigma_min']:.3e}")


def bench_drop_recovery():
    spec = build_spec(steps=1000, elevation=20.0, omega=(0.0, 0.0, 0.0))
    target = adjoint.make_target(TRUE_THETA, spec, sample_n=60)
    L0 = adjoint.loss(THETA0, target, spec)
    res = adjoint.fit_cmaes(THETA0, target, spec, active=[1, 2], maxiter=50, sigma0=0.2, seed=3)
    e_err = abs(res["theta"][1] - TRUE_THETA[1]) / TRUE_THETA[1]
    j_err = abs(res["theta"][2] - TRUE_THETA[2]) / TRUE_THETA[2]
    ok = e_err < 0.12 and j_err < 0.12 and res["loss"] < L0
    return check("drop recovery (e, impulse)", ok,
                 f"e={res['theta'][1]:.4f}(true {TRUE_THETA[1]}) err={e_err*100:.1f}% "
                 f"J={res['theta'][2]:.4f} loss {L0:.1e}->{res['loss']:.1e}")


def bench_gradient_nonsmooth():
    spec = build_spec(steps=500, elevation=0.0)
    target = adjoint.make_target(TRUE_THETA, spec, sample_n=40)
    g1 = adjoint.finite_difference_gradient(np.array(TRUE_THETA), target, spec,
                                            h=np.array([1e-3, 1e-3, 1e-3]), active=[0])
    g2 = adjoint.finite_difference_gradient(np.array(TRUE_THETA), target, spec,
                                            h=np.array([1e-5, 1e-5, 1e-5]), active=[0])
    return check("FD gradient is step-size sensitive (finding)", True,
                 f"grad_mu h=1e-3:{g1[0]:.4f} vs h=1e-5:{g2[0]:.4f}")


def main():
    results = [
        bench_cmaes_sphere(),
        bench_drop_identifiability(),
        bench_drop_recovery(),
        bench_gradient_nonsmooth(),
    ]
    print(f"\n{sum(results)}/{len(results)} M3 benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()