import numpy as np

from .body import Body
from .world import Geometry, World

PARAM_NAMES = ("mu_felt", "e_felt", "impulse")
BOUNDS = {
    "mu_felt": (0.0, 1.5),
    "e_felt": (0.0, 0.95),
    "impulse": (0.004, 0.2),
}
_STEPS = {"mu_felt": 1e-3, "e_felt": 1e-3, "impulse": 2e-3}


def _direction(spec):
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([np.cos(spec["yaw"]), np.sin(spec["yaw"]), 0.0])
    d = spec["elevation"] * up + horizontal
    return d / np.linalg.norm(d)


def clip(theta):
    theta = np.asarray(theta, dtype=float).copy()
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS[name]
        theta[i] = min(max(theta[i], lo), hi)
    return theta


def run_trajectory(theta, spec):
    theta = clip(theta)
    gk = dict(spec["geom"])
    gk["mu_felt"] = float(theta[0])
    gk["e_felt"] = float(theta[1])
    world = World(geometry=Geometry(**gk), dt=spec["dt"], solver_iterations=spec["solver_iterations"])
    body = Body(side=spec["side"], mass=spec["mass"], pos=spec["pos"],
                quat=spec.get("quat"), omega=spec.get("omega", (0.0, 0.0, 0.0)))
    world.bodies = [body]
    body.vel = body.vel + _direction(spec) * (float(theta[2]) / body.mass)

    steps = spec["steps"]
    ts = np.empty(steps + 1)
    xs = np.empty((steps + 1, 3))
    ts[0] = 0.0
    xs[0] = body.pos.copy()
    for i in range(steps):
        world.step(record=False)
        ts[i + 1] = world.t
        xs[i + 1] = body.pos.copy()
    return ts, xs


def _interp(ts, xs, times):
    return np.stack([np.interp(times, ts, xs[:, d]) for d in range(3)], axis=1)


def make_target(theta_true, spec, sample_n=50, noise=0.0, seed=0):
    ts, xs = run_trajectory(theta_true, spec)
    times = np.linspace(ts[0], ts[-1], sample_n)
    pos = _interp(ts, xs, times)
    if noise > 0:
        rng = np.random.default_rng(seed)
        pos = pos + rng.normal(0.0, noise, pos.shape)
    return {"times": times, "pos": pos, "theta_true": np.asarray(theta_true, dtype=float)}


def target_from_trajectory(trajectory, allow_synthetic=False):
    """Build a calibration target from a measured trajectory (m1 Trajectory or dict).

    Synthetic/interpolated tracks are rejected unless `allow_synthetic=True`.
    """
    if isinstance(trajectory, dict):
        synthetic = bool(trajectory.get("synthetic", False))
        times = trajectory["times"]
        pos = trajectory["pos"]
        asset_id = trajectory.get("asset_id")
    else:
        synthetic = bool(getattr(trajectory, "synthetic", False))
        times = getattr(trajectory, "times")
        pos = getattr(trajectory, "pos")
        asset_id = getattr(trajectory, "asset_id", None)
    if synthetic and not allow_synthetic:
        raise ValueError("synthetic trajectory cannot supervise calibration")
    times = np.asarray(times, dtype=float).reshape(-1)
    pos = np.asarray(pos, dtype=float).reshape(-1, 3)
    return {"times": times, "pos": pos, "source": "trajectory", "asset_id": asset_id}


def residual(theta, target, spec):
    ts, xs = run_trajectory(theta, spec)
    pred = _interp(ts, xs, target["times"])
    return (pred - target["pos"]).ravel()


def loss(theta, target, spec):
    r = residual(theta, target, spec)
    return float(np.dot(r, r) / r.size)


def finite_difference_gradient(theta, target, spec, h=None, active=None):
    theta = np.asarray(theta, dtype=float)
    if h is None:
        h = np.array([_STEPS[n] for n in PARAM_NAMES])
    idx = _active_indices(theta, active)
    grad = np.zeros(len(theta))
    for i in idx:
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += h[i]
        tm[i] -= h[i]
        grad[i] = (loss(tp, target, spec) - loss(tm, target, spec)) / (2 * h[i])
    return grad


def jacobian(theta, target, spec, h=None, active=None):
    theta = np.asarray(theta, dtype=float)
    if h is None:
        h = np.array([_STEPS[n] for n in PARAM_NAMES])
    idx = _active_indices(theta, active)
    r0 = residual(theta, target, spec)
    J = np.empty((r0.size, len(idx)))
    for col, i in enumerate(idx):
        tp = theta.copy()
        tp[i] += h[i]
        J[:, col] = (residual(tp, target, spec) - r0) / h[i]
    return r0, J


def _active_indices(theta, active):
    if active is None:
        return list(range(len(theta)))
    return list(active)


def fit_lm(theta0, target, spec, iters=30, lam0=1e-2, active=None):
    theta = clip(theta0)
    idx = _active_indices(theta, active)
    L = loss(theta, target, spec)
    lam = lam0
    history = [{"iter": 0, "loss": L, "theta": theta.copy()}]
    for it in range(iters):
        r, J = jacobian(theta, target, spec, active=active)
        A = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(12):
            try:
                delta_a = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            delta = np.zeros(len(theta))
            for k, i in enumerate(idx):
                delta[i] = delta_a[k]
            cand = clip(theta + delta)
            Lc = loss(cand, target, spec)
            if Lc < L:
                theta = cand
                L = Lc
                lam = max(lam * 0.5, 1e-9)
                improved = True
                break
            lam *= 2.0
        history.append({"iter": it + 1, "loss": L, "theta": theta.copy()})
        if not improved:
            break
    return {"theta": theta, "loss": L, "history": history}


def fit_cmaes(theta0, target, spec, active=None, **kwargs):
    from .cmaes import cma_es

    theta0 = np.asarray(theta0, dtype=float)
    idx = _active_indices(theta0, active)
    lower = np.array([BOUNDS[PARAM_NAMES[i]][0] for i in idx])
    upper = np.array([BOUNDS[PARAM_NAMES[i]][1] for i in idx])

    def f(x):
        theta = theta0.copy()
        for k, i in enumerate(idx):
            theta[i] = x[k]
        return loss(theta, target, spec)

    result = cma_es(f, clip(theta0)[idx], sigma0=kwargs.get("sigma0", 0.1),
                    lower=lower, upper=upper, maxiter=kwargs.get("maxiter", 80),
                    seed=kwargs.get("seed", 0))
    theta = theta0.copy()
    for k, i in enumerate(idx):
        theta[i] = result["x"][k]
    return {"theta": clip(theta), "loss": result["f"], "history": result["history"]}


def fit(theta0, target, spec, method="lm", fallback=False, active=None, **kwargs):
    if method == "cmaes":
        return fit_cmaes(theta0, target, spec, active=active, **kwargs)
    result = fit_lm(theta0, target, spec, iters=kwargs.get("iters", 30), active=active)
    if fallback:
        initial = loss(clip(theta0), target, spec)
        if result["loss"] > 0.5 * initial:
            cma = fit_cmaes(theta0, target, spec, active=active, **kwargs)
            if cma["loss"] < result["loss"]:
                cma["method"] = "cmaes-fallback"
                return cma
    result["method"] = "lm"
    return result


def identifiability(theta, target, spec, h=None, active=None):
    _, J = jacobian(theta, target, spec, h, active=active)
    sv = np.linalg.svd(J, compute_uv=False)
    smin = float(sv[-1])
    return {
        "singular_values": sv.tolist(),
        "condition_number": float(sv[0] / smin) if smin > 1e-14 else float("inf"),
        "sigma_max": float(sv[0]),
        "sigma_min": smin,
    }