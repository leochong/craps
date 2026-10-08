import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import faces
from .body import Body
from .world import Geometry, World

DEFAULT_GEOM = dict(
    radius=0.15,
    height=0.25,
    bumper_major=0.13,
    bumper_minor=0.03,
    bumper_z=0.06,
    mu_felt=0.5,
    e_felt=0.4,
    mu_dome=0.2,
    e_dome=0.5,
    mu_bumper=0.4,
    e_bumper=0.6,
    mu_dice=0.3,
    e_dice=0.6,
)

DEFAULT_SPEC = dict(
    geom=DEFAULT_GEOM,
    side=0.016,
    mass=0.006,
    base_positions=[(-0.02, 0.0), (0.02, 0.0)],
    impulse=0.08,
    impulse_std=0.01,
    elevation=1.2,
    elevation_std=0.05,
    yaw=0.3,
    yaw_std=0.2,
    spin=(0.0, 0.0, 0.0),
    spin_std=30.0,
    pos_std=0.002,
    angle_std=0.05,
    max_time=3.0,
    dt=0.001,
    solver_iterations=12,
)

FAIR = {
    2: 1 / 36, 3: 2 / 36, 4: 3 / 36, 5: 4 / 36, 6: 5 / 36, 7: 6 / 36,
    8: 5 / 36, 9: 4 / 36, 10: 3 / 36, 11: 2 / 36, 12: 1 / 36,
}


def _small_quat(angles):
    q = np.array([angles[0] / 2.0, angles[1] / 2.0, angles[2] / 2.0, 1.0])
    return q / np.linalg.norm(q)


def simulate_once(seed, spec):
    rng = np.random.default_rng(seed)
    geom = Geometry(**spec["geom"])
    world = World(geometry=geom, dt=spec["dt"], solver_iterations=spec["solver_iterations"])

    half = spec["side"] / 2.0
    dice = []
    for (bx, by) in spec["base_positions"]:
        pos = np.array([
            bx + rng.normal(0.0, spec["pos_std"]),
            by + rng.normal(0.0, spec["pos_std"]),
            half,
        ])
        quat = _small_quat(rng.normal(0.0, spec["angle_std"], 3))
        dice.append(Body(side=spec["side"], mass=spec["mass"], pos=pos, quat=quat))
    world.bodies = dice

    j_mag = spec["impulse"] + rng.normal(0.0, spec["impulse_std"])
    elevation = spec["elevation"] + rng.normal(0.0, spec["elevation_std"])
    yaw = spec["yaw"] + rng.normal(0.0, spec["yaw_std"])
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    direction = elevation * up + horizontal
    direction /= np.linalg.norm(direction)

    spin_base = np.asarray(spec["spin"], dtype=float)
    for b in dice:
        b.vel += direction * (j_mag / b.mass)
        b.omega += spin_base + rng.normal(0.0, spec["spin_std"], 3)

    world.simulate(max_time=spec["max_time"])
    quats = [b.quat for b in dice]
    return {
        "seed": seed,
        "sum": faces.outcome(quats),
        "rest_time": float(world.t),
        "settled": bool(world.is_settled()),
        "terminal": [{"pos": b.pos.tolist(), "quat": b.quat.tolist()} for b in dice],
    }


def _worker(args):
    seed, spec = args
    return simulate_once(seed, spec)


def monte_carlo(spec=None, n=32, seed=0, workers=1):
    spec = spec or DEFAULT_SPEC
    seeds = [seed + i for i in range(n)]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, [(s, spec) for s in seeds]))
    else:
        results = [simulate_once(s, spec) for s in seeds]
    return aggregate(results, n), results


def aggregate(results, n=None):
    n = n or len(results)
    counts = {s: 0 for s in range(2, 13)}
    for r in results:
        counts[r["sum"]] = counts.get(r["sum"], 0) + 1
    P = {s: counts[s] / n for s in range(2, 13)}
    return {
        "n": n,
        "counts": counts,
        "P": P,
        "kl_vs_fair": kl_divergence(P, FAIR),
        "ci": {s: wald_interval(counts[s], n) for s in range(2, 13)},
        "mean_sum": sum(s * P[s] for s in range(2, 13)),
        "settled_fraction": (sum(1 for r in results if r["settled"]) / n) if n else 0.0,
    }


def kl_divergence(P, Q):
    total = 0.0
    for s in range(2, 13):
        if P[s] > 0 and Q[s] > 0:
            total += P[s] * math.log(P[s] / Q[s])
    return total


def wald_interval(count, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def summarize(dist, top=3):
    P = dist["P"]
    ranked = sorted(range(2, 13), key=lambda s: P[s], reverse=True)[:top]
    return {
        "n": dist["n"],
        "mean_sum": dist["mean_sum"],
        "kl_vs_fair": dist["kl_vs_fair"],
        "most_likely": [{"sum": s, "p": P[s], "ci": dist["ci"][s]} for s in ranked],
        "settled_fraction": dist["settled_fraction"],
    }