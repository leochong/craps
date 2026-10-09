"""Validation for multi-clip inverse calibration (Phase B).

Checks the extended parameter space, weighted multi-clip residual/jacobian
identities, shared-parameter recovery, frozen-parameter isolation, multi-clip
identifiability gain, and the measured-trajectory clip bridge. Prints
[PASS]/[FAIL] and exits nonzero on failure.
"""

import sys

import numpy as np

from m1.trajectory import Trajectory
from m2 import adjoint
from m2.montecarlo import DEFAULT_GEOM

FREE_GEOM = dict(radius=50.0, height=50.0, bumper_major=50.0, bumper_minor=0.001, bumper_z=50.0)


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def free_spec(elevation, yaw, steps=120, omega=(0.0, 0.0, 0.0)):
    return dict(geom=dict(FREE_GEOM), side=0.016, mass=0.006, pos=(0.0, 0.0, 0.008),
                quat=[0.0, 0.0, 0.0, 1.0], omega=omega, yaw=yaw, elevation=elevation,
                dt=0.001, steps=steps, solver_iterations=6)


def machine_spec(elevation, yaw, steps=200, omega=(0.0, 0.0, 10.0)):
    return dict(geom=dict(DEFAULT_GEOM), side=0.016, mass=0.006, pos=(0.0, 0.0, 0.008),
                quat=[0.0, 0.0, 0.0, 1.0], omega=omega, yaw=yaw, elevation=elevation,
                dt=0.001, steps=steps, solver_iterations=6)


def bench_parameter_space():
    core = adjoint.names_for(len(adjoint.PARAM_NAMES))
    full = adjoint.names_for(len(adjoint.ALL_PARAM_NAMES))
    theta = adjoint.default_theta()
    clipped = adjoint.clip(np.full(theta.size, 99.0))
    core_vec = adjoint.clip(np.array([9.0, 0.4, 0.08]))
    ok = (core == adjoint.PARAM_NAMES and full == adjoint.ALL_PARAM_NAMES
          and theta.size == len(adjoint.ALL_PARAM_NAMES)
          and clipped[0] == adjoint.BOUNDS_ALL["mu_felt"][1]
          and core_vec.size == 3 and core_vec[0] == adjoint.BOUNDS["mu_felt"][1])
    return check("extended parameter space", ok,
                 f"core={len(core)} full={len(full)} mu_clip={clipped[0]}")


def bench_multi_identities():
    theta = adjoint.default_theta()
    specs = [free_spec(8.0, 0.2), free_spec(14.0, 1.0)]
    weights = [1.0, 2.0]
    clips = [{"spec": s, "target": adjoint.make_target(theta, s, sample_n=5), "weight": w}
             for s, w in zip(specs, weights)]
    r = adjoint.multi_residual(theta, clips)
    parts = [adjoint.residual(theta, c["target"], c["spec"]) for c in clips]
    expected_len = sum(p.size for p in parts)
    length_ok = r.size == expected_len
    manual = sum(w * float(np.dot(p, p)) for w, p in zip(weights, parts)) / expected_len
    loss_ok = abs(adjoint.multi_loss(theta, clips) - manual) < 1e-12
    active = [2]
    _, J = adjoint.multi_jacobian(theta, clips, active=active)
    stacked = np.vstack([adjoint.jacobian(theta, c["target"], c["spec"], active=active)[1]
                         * np.sqrt(c["weight"]) for c in clips])
    jac_ok = J.shape == (expected_len, len(active)) and np.allclose(J, stacked)
    ok = length_ok and loss_ok and jac_ok
    return check("multi-clip residual/jacobian identities", ok,
                 f"len={r.size} loss_ok={loss_ok} jac={J.shape}")


def bench_shared_recovery():
    true = adjoint.default_theta(overrides={"impulse": 0.03, "mu_felt": 0.4, "e_felt": 0.4})
    theta0 = adjoint.default_theta(overrides={"impulse": 0.02, "mu_felt": 0.4, "e_felt": 0.4})
    specs = [free_spec(8.0, 0.2), free_spec(14.0, 1.0)]
    clips = [{"spec": s, "target": adjoint.make_target(true, s, sample_n=25)} for s in specs]
    result = adjoint.fit_multi(theta0, clips, method="lm", active=[2], iters=25)
    recovered = float(result["theta"][2])
    err = abs(recovered - 0.03) / 0.03
    ok = err < 0.02 and result["loss"] < 1e-10
    return check("shared impulse recovery (2 clips)", ok,
                 f"impulse={recovered:.5f} err={err*100:.2f}% loss={result['loss']:.1e}")


def bench_frozen_params():
    true = adjoint.default_theta(overrides={"impulse": 0.03, "mu_felt": 0.4, "e_felt": 0.4})
    theta0 = adjoint.default_theta(overrides={"impulse": 0.02, "mu_felt": 0.4, "e_felt": 0.4})
    specs = [free_spec(8.0, 0.2), free_spec(14.0, 1.0)]
    clips = [{"spec": s, "target": adjoint.make_target(true, s, sample_n=25)} for s in specs]
    result = adjoint.fit_multi(theta0, clips, method="lm", active=[2], iters=20)
    frozen_ok = all(abs(result["theta"][i] - theta0[i]) < 1e-12
                    for i in range(theta0.size) if i != 2)
    moved = abs(result["theta"][2] - theta0[2]) > 1e-3
    return check("frozen params unchanged", frozen_ok and moved,
                 f"moved_impulse={result['theta'][2]:.4f}")


def bench_clip_identifiability():
    theta0 = adjoint.default_theta(overrides={"impulse": 0.02, "mu_felt": 0.4, "e_felt": 0.4})
    specs = [machine_spec(0.4, 0.2), machine_spec(0.6, 1.0), machine_spec(0.5, 2.0)]
    clips = [{"spec": s, "target": adjoint.make_target(theta0, s, sample_n=25)} for s in specs]
    single = adjoint.identifiability(theta0, clips[0]["target"], clips[0]["spec"], active=[1, 7])
    multi = adjoint.multi_identifiability(theta0, clips, active=[1, 7])
    improved = np.isfinite(multi["condition_number"]) and (
        not np.isfinite(single["condition_number"])
        or multi["condition_number"] <= single["condition_number"])
    return check("multi-clip identifiability gain", improved,
                 f"cond single={single['condition_number']:.3g} multi={multi['condition_number']:.3g}")


def bench_trajectory_clips():
    theta = adjoint.default_theta(overrides={"impulse": 0.02})
    spec = free_spec(8.0, 0.2, steps=60)
    ts, xs = adjoint.run_trajectory(theta, spec)
    traj = Trajectory(asset_id="sim", die_index=0, times=ts, pos=xs, provenance={"source": "sim"})
    target = adjoint.target_from_trajectory(traj)
    clip = {"spec": dict(spec, steps=60), "target": target}
    loss = adjoint.multi_loss(theta, [clip])
    tuples_ok = len(adjoint.normalize_clips([(spec, target, 0.5)])) == 1
    synthetic = Trajectory(asset_id="x", die_index=0, times=[0.0, 0.1], pos=[[0, 0, 1], [0, 0, 0]],
                           synthetic=True, provenance={"source": "rife"})
    rejected = False
    try:
        adjoint.target_from_trajectory(synthetic)
    except ValueError:
        rejected = True
    ok = np.isfinite(loss) and tuples_ok and rejected
    return check("trajectory clip bridge", ok,
                 f"loss={loss:.2e} tuples={tuples_ok} synthetic_rejected={rejected}")


def main():
    results = [
        bench_parameter_space(),
        bench_multi_identities(),
        bench_shared_recovery(),
        bench_frozen_params(),
        bench_clip_identifiability(),
        bench_trajectory_clips(),
    ]
    print(f"\n{sum(results)}/{len(results)} calibration benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()