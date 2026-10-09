import numpy as np

from .body import Body
from .world import Geometry, World

PARAM_NAMES = ("mu_felt", "e_felt", "impulse")
GEOM_PARAM_NAMES = ("mu_dome", "e_dome", "mu_bumper", "e_bumper", "angular_damping")
ALL_PARAM_NAMES = PARAM_NAMES + GEOM_PARAM_NAMES

BOUNDS = {
    "mu_felt": (0.0, 1.5),
    "e_felt": (0.0, 0.95),
    "impulse": (0.004, 0.2),
}
BOUNDS_ALL = {
    **BOUNDS,
    "mu_dome": (0.0, 1.5),
    "e_dome": (0.0, 0.98),
    "mu_bumper": (0.0, 2.0),
    "e_bumper": (0.0, 0.98),
    "angular_damping": (0.0, 20.0),
}
_STEPS = {"mu_felt": 1e-3, "e_felt": 1e-3, "impulse": 2e-3}
_STEPS_ALL = {
    **_STEPS,
    "mu_dome": 1e-3,
    "e_dome": 1e-3,
    "mu_bumper": 1e-3,
    "e_bumper": 1e-3,
    "angular_damping": 1e-2,
}
DEFAULT_PARAMS = {
    "mu_felt": 0.5, "e_felt": 0.4, "impulse": 0.08,
    "mu_dome": 0.2, "e_dome": 0.5, "mu_bumper": 0.4, "e_bumper": 0.6,
    "angular_damping": 0.5,
}


def names_for(n):
    """Parameter name tuple for a vector of length n (core or full space)."""
    if n == len(PARAM_NAMES):
        return PARAM_NAMES
    if n == len(ALL_PARAM_NAMES):
        return ALL_PARAM_NAMES
    raise ValueError(f"theta length {n} is neither core {len(PARAM_NAMES)} nor full {len(ALL_PARAM_NAMES)}")


def _table_for(n, core_table, all_table):
    return core_table if n == len(PARAM_NAMES) else all_table


def default_theta(names=ALL_PARAM_NAMES, overrides=None):
    """Full parameter vector over `names`, seeded from machine defaults."""
    values = {**DEFAULT_PARAMS, **(overrides or {})}
    return np.array([float(values[name]) for name in names], dtype=float)


def theta_to_overrides(theta):
    """Map a theta vector or dict to a {param_name: value} dict."""
    if isinstance(theta, dict):
        return dict(theta)
    theta = np.asarray(theta, dtype=float)
    names = names_for(theta.size)
    return {name: float(theta[i]) for i, name in enumerate(names)}


def clip(theta):
    if isinstance(theta, dict):
        out = {}
        for key, value in theta.items():
            lo, hi = BOUNDS_ALL.get(key, (value, value))
            out[key] = min(max(float(value), lo), hi)
        return out
    theta = np.asarray(theta, dtype=float).copy()
    table = _table_for(theta.size, BOUNDS, BOUNDS_ALL)
    for i, name in enumerate(names_for(theta.size)):
        lo, hi = table[name]
        theta[i] = min(max(theta[i], lo), hi)
    return theta


def _direction(elevation, yaw):
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    d = elevation * up + horizontal
    return d / np.linalg.norm(d)


def run_trajectory(theta, spec):
    overrides = theta if isinstance(theta, dict) else theta_to_overrides(clip(theta))
    gk = dict(spec["geom"])
    for name in ("mu_felt", "e_felt") + GEOM_PARAM_NAMES:
        if name in overrides:
            gk[name] = float(overrides[name])
    impulse = float(overrides.get("impulse", spec.get("impulse", DEFAULT_PARAMS["impulse"])))
    elevation = float(overrides.get("elevation", spec["elevation"]))
    yaw = float(overrides.get("yaw", spec["yaw"]))
    world = World(geometry=Geometry(**gk), dt=spec["dt"], solver_iterations=spec["solver_iterations"])
    body = Body(side=spec["side"], mass=spec["mass"], pos=spec["pos"],
                quat=spec.get("quat"), omega=spec.get("omega", (0.0, 0.0, 0.0)))
    world.bodies = [body]
    body.vel = body.vel + _direction(elevation, yaw) * (impulse / body.mass)

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
        table = _table_for(theta.size, _STEPS, _STEPS_ALL)
        h = np.array([table[name] for name in names_for(theta.size)])

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
        table = _table_for(theta.size, _STEPS, _STEPS_ALL)
        h = np.array([table[name] for name in names_for(theta.size)])
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
    names = names_for(theta0.size)
    table = _table_for(theta0.size, BOUNDS, BOUNDS_ALL)
    lower = np.array([table[names[i]][0] for i in idx])
    upper = np.array([table[names[i]][1] for i in idx])

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


def normalize_clips(clips):
    """Coerce a list of clip dicts/tuples/objects into {spec, target, weight}."""
    out = []
    for c in clips:
        if isinstance(c, dict):
            spec, target, weight = c["spec"], c["target"], c.get("weight", 1.0)
        elif isinstance(c, (tuple, list)):
            spec, target, weight = c[0], c[1], (c[2] if len(c) > 2 else 1.0)
        else:
            spec, target = c.spec, c.target
            weight = getattr(c, "weight", 1.0)
        out.append({"spec": spec, "target": target, "weight": float(weight)})
    return out


def multi_residual(theta, clips):
    """Concatenate sqrt-weighted residuals across all clips."""
    clips = normalize_clips(clips)
    parts = []
    for c in clips:
        parts.append(residual(theta, c["target"], c["spec"]) * np.sqrt(c["weight"]))
    return np.concatenate(parts)


def multi_loss(theta, clips):
    r = multi_residual(theta, clips)
    return float(np.dot(r, r) / r.size)


def multi_jacobian(theta, clips, h=None, active=None):
    theta = np.asarray(theta, dtype=float)
    clips = normalize_clips(clips)
    if h is None:
        table = _table_for(theta.size, _STEPS, _STEPS_ALL)
        h = np.array([table[name] for name in names_for(theta.size)])
    rows_r, rows_J = [], []
    for c in clips:
        r, J = jacobian(theta, c["target"], c["spec"], h=h, active=active)
        sw = np.sqrt(c["weight"])
        rows_r.append(r * sw)
        rows_J.append(J * sw)
    return np.concatenate(rows_r), np.vstack(rows_J)


def multi_identifiability(theta, clips, h=None, active=None):
    _, J = multi_jacobian(theta, clips, h, active=active)
    sv = np.linalg.svd(J, compute_uv=False)
    smin = float(sv[-1])
    return {
        "singular_values": sv.tolist(),
        "condition_number": float(sv[0] / smin) if smin > 1e-14 else float("inf"),
        "sigma_max": float(sv[0]),
        "sigma_min": smin,
    }


def fit_multi_lm(theta0, clips, iters=30, lam0=1e-2, active=None):
    theta = clip(theta0)
    idx = _active_indices(theta, active)
    L = multi_loss(theta, clips)
    lam = lam0
    history = [{"iter": 0, "loss": L, "theta": theta.copy()}]
    for it in range(iters):
        r, J = multi_jacobian(theta, clips, active=active)
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
            Lc = multi_loss(cand, clips)
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
    return {"theta": theta, "loss": L, "history": history, "method": "lm"}


def fit_multi_cmaes(theta0, clips, active=None, **kwargs):
    from .cmaes import cma_es

    theta0 = np.asarray(theta0, dtype=float)
    idx = _active_indices(theta0, active)
    names = names_for(theta0.size)
    table = _table_for(theta0.size, BOUNDS, BOUNDS_ALL)
    lower = np.array([table[names[i]][0] for i in idx])
    upper = np.array([table[names[i]][1] for i in idx])

    def f(x):
        theta = theta0.copy()
        for k, i in enumerate(idx):
            theta[i] = x[k]
        return multi_loss(theta, clips)

    result = cma_es(f, clip(theta0)[idx], sigma0=kwargs.get("sigma0", 0.1),
                    lower=lower, upper=upper, maxiter=kwargs.get("maxiter", 80),
                    seed=kwargs.get("seed", 0))
    theta = theta0.copy()
    for k, i in enumerate(idx):
        theta[i] = result["x"][k]
    return {"theta": clip(theta), "loss": result["f"], "history": result["history"],
            "method": "cmaes"}


def fit_multi(theta0, clips, method="cmaes", fallback=False, active=None, **kwargs):
    """Fit shared parameters across multiple measured clips."""
    clips = normalize_clips(clips)
    if method == "lm":
        result = fit_multi_lm(theta0, clips, iters=kwargs.get("iters", 30), active=active)
        if fallback:
            initial = multi_loss(clip(theta0), clips)
            if result["loss"] > 0.5 * initial:
                cma = fit_multi_cmaes(theta0, clips, active=active, **kwargs)
                if cma["loss"] < result["loss"]:
                    cma["method"] = "cmaes-fallback"
                    return cma
        return result
    return fit_multi_cmaes(theta0, clips, active=active, **kwargs)