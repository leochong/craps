import sys

import numpy as np

from m2 import mathx as mx
from m2.body import Body
from m2.contacts import generate_contacts
from m2.world import Geometry, World

G = 9.81
SIDE = 0.016
MASS = 0.006
HALF = SIDE / 2.0
DT = 0.001


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    return bool(cond)


def bench_free_fall():
    world = World(collisions=False, gravity=(0.0, 0.0, -G), dt=DT)
    b = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, 1.0))
    world.bodies = [b]
    n = 200
    for _ in range(n):
        world.step(record=False)
    t = n * DT
    z_true = 1.0 - 0.5 * G * t * t
    tol = 0.5 * G * DT * t + 1e-6
    return check("free fall", abs(b.pos[2] - z_true) < tol,
                 f"z={b.pos[2]:.5f} vs {z_true:.5f} tol={tol:.5f}")


def bench_bounce():
    g = Geometry(e_felt=0.5)
    world = World(geometry=g, dt=DT)
    b = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, 0.2))
    world.bodies = [b]
    world.simulate(max_time=2.0, record=True)
    z = world.result()["pos"][:, 0, 2]
    contacts = np.where(z < (HALF + 0.01))[0]
    if len(contacts) == 0:
        return check("bounce", False, "never hit floor")
    first_hit = contacts[0]
    apex = np.max(z[first_hit:])
    h = 0.2 - HALF
    ratio = apex / h
    return check("bounce apex ratio ~= e^2", abs(ratio - 0.25) < 0.06,
                 f"ratio={ratio:.3f} (expect ~0.25)")


def bench_frictionless_slide():
    g = Geometry(mu_felt=0.0, e_felt=0.0, radius=100.0, height=100.0,
                 bumper_major=100.0, bumper_minor=0.001, bumper_z=100.0)
    world = World(geometry=g, dt=DT)
    b = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, HALF), vel=(1.0, 0.0, 0.0))
    world.bodies = [b]
    for _ in range(300):
        world.step(record=False)
    return check("frictionless slide", abs(b.vel[0] - 1.0) < 1e-3 and abs(b.pos[2] - HALF) < 1e-3,
                 f"vx={b.vel[0]:.5f} z={b.pos[2]:.4f} (expect 1.0 / {HALF:.4f})")


def bench_friction_stops():
    mu = 0.5
    v0 = 0.8
    d_max = v0 * v0 / (2 * mu * G)
    g = Geometry(mu_felt=mu, e_felt=0.0)
    world = World(geometry=g, dt=DT)
    b = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, HALF), vel=(v0, 0.0, 0.0))
    world.bodies = [b]
    world.simulate(max_time=3.0)
    dx = b.pos[0]
    speed = np.linalg.norm(b.vel)
    return check("friction stops sliding", speed < 0.05 and dx < d_max * 1.1 + 0.005,
                 f"dx={dx:.4f} bound={d_max:.4f} v={speed:.4f}")


def bench_elastic_dice_collision():
    world = World(geometry=Geometry(e_dice=1.0, mu_dice=0.0, mu_felt=0.0, e_felt=0.0,
                                    radius=100.0, height=100.0, bumper_major=100.0,
                                    bumper_minor=0.001, bumper_z=100.0),
                  gravity=(0.0, 0.0, 0.0), dt=DT)
    A = Body(side=SIDE, mass=MASS, pos=(-0.05, 0.0, HALF), vel=(1.0, 0.0, 0.0))
    B = Body(side=SIDE, mass=MASS, pos=(0.05, 0.0, HALF), vel=(-1.0, 0.0, 0.0))
    world.bodies = [A, B]
    ke0 = world.total_kinetic_energy()
    p0 = MASS * (A.vel + B.vel)
    for _ in range(300):
        world.step(record=False)
    ke1 = world.total_kinetic_energy()
    p1 = MASS * (A.vel + B.vel)
    ok = abs(A.vel[0] + 1.0) < 0.15 and abs(B.vel[0] - 1.0) < 0.15
    konserve = abs(ke1 - ke0) / ke0 < 0.05 and np.linalg.norm(p1 - p0) < 1e-6
    return check("elastic die-die exchange", ok and konserve,
                 f"vA={A.vel[0]:.3f} vB={B.vel[0]:.3f} ke_ratio={ke1/ke0:.3f}")


def bench_resting_stability():
    g = Geometry(mu_felt=0.5, e_felt=0.2)
    world = World(geometry=g, dt=DT, beta=0.4)
    b = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, HALF))
    world.bodies = [b]
    world.simulate(max_time=1.5, hold_steps=20)
    drift = np.linalg.norm(b.pos - np.array([0.0, 0.0, HALF]))
    quat_drift = np.linalg.norm(b.quat - mx.identity_quat())
    ok = drift < 1e-3 and quat_drift < 1e-3 and b.vel[2] == b.vel[2]
    return check("resting stability", ok, f"drift={drift:.6f} quat={quat_drift:.6f}")


def bench_inertia():
    invI = mx.cube_inv_inertia(MASS, SIDE)
    expected = 6.0 / (MASS * SIDE * SIDE)
    b = Body(side=SIDE, mass=MASS)
    return check("cube inertia", abs(invI - expected) < 1e-12 and abs(b.inv_inertia - expected) < 1e-12)


def bench_containment():
    g = Geometry()
    world = World(geometry=g, dt=DT)
    spin = np.array([0.0, 40.0, 25.0])
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([np.cos(0.3), np.sin(0.3), 0.0])
    direction = 1.2 * up + horizontal
    direction /= np.linalg.norm(direction)
    A = Body(side=SIDE, mass=MASS, pos=(0.0, 0.0, HALF))
    B = Body(side=SIDE, mass=MASS, pos=(0.03, 0.0, HALF))
    for b in (A, B):
        b.vel += direction * (0.08 / MASS)
        b.omega += spin
    world.bodies = [A, B]
    world.simulate(max_time=4.0)
    res = world.result()
    max_radial = np.max(np.hypot(res["pos"][:, :, 0], res["pos"][:, :, 1]))
    centroid = np.array([b.pos for b in world.bodies])
    has_nan = np.isnan(centroid).any()
    ok = (not has_nan) and max_radial <= g.radius + SIDE + 0.01
    return check("containment + no NaN", ok, f"max_radial={max_radial:.3f} R={g.radius + SIDE:.3f}")


def main():
    results = [
        bench_free_fall(),
        bench_bounce(),
        bench_frictionless_slide(),
        bench_friction_stops(),
        bench_elastic_dice_collision(),
        bench_resting_stability(),
        bench_inertia(),
        bench_containment(),
    ]
    n_pass = sum(results)
    print(f"\n{sum(results)}/{len(results)} benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()