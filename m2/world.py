from dataclasses import dataclass

import numpy as np

from .contacts import generate_contacts
from .solver import Solver


@dataclass
class Geometry:
    radius: float = 0.15
    height: float = 0.25
    bumper_major: float = 0.13
    bumper_minor: float = 0.03
    bumper_z: float = 0.06
    mu_felt: float = 0.5
    e_felt: float = 0.4
    mu_dome: float = 0.2
    e_dome: float = 0.5
    mu_bumper: float = 0.4
    e_bumper: float = 0.6
    mu_dice: float = 0.3
    e_dice: float = 0.6


class World:
    def __init__(self, geometry=None, bodies=None, gravity=(0.0, 0.0, -9.81),
                 dt=0.001, solver_iterations=12, beta=0.2, slop=0.001,
                 collisions=True):
        self.geometry = geometry or Geometry()
        self.bodies = bodies or []
        self.collisions = collisions
        self.gravity = np.asarray(gravity, dtype=float)
        self.dt = dt
        self.solver = Solver(self.bodies, iterations=solver_iterations, beta=beta, slop=slop)

        self.t = 0.0
        self.steps = 0
        self._record = {"t": [], "pos": [], "quat": [], "vel": [], "omega": []}
        self.max_penetration = 0.0
        self.contact_frames = 0
        self._rest_counter = 0

    def _snapshot(self, record=True):
        if not record:
            return
        self._record["t"].append(self.t)
        self._record["pos"].append([np.array(b.pos) for b in self.bodies])
        self._record["quat"].append([np.array(b.quat) for b in self.bodies])
        self._record["vel"].append([np.array(b.vel) for b in self.bodies])
        self._record["omega"].append([np.array(b.omega) for b in self.bodies])

    def step(self, record=True):
        self.solver.bodies = self.bodies
        for b in self.bodies:
            b.apply_gravity(self.gravity, self.dt)

        contacts = generate_contacts(self) if self.collisions else []
        if contacts:
            self.contact_frames += 1
            for c in contacts:
                self.max_penetration = max(self.max_penetration, c.depth)
            self.solver.solve_velocity(contacts, self.dt)

        for b in self.bodies:
            b.integrate(self.dt)

        self.solver.solve_position(contacts, self.dt)

        self.t += self.dt
        self.steps += 1
        self._snapshot(record)

        rest = self._at_rest()
        if rest:
            self._rest_counter += 1
        else:
            self._rest_counter = 0

    def _at_rest(self):
        for b in self.bodies:
            if np.linalg.norm(b.vel) > 0.01 or np.linalg.norm(b.omega) > 0.1:
                return False
        return True

    def is_settled(self, hold_steps=20):
        return self._rest_counter >= hold_steps

    def simulate(self, max_time=4.0, record=True, hold_steps=20):
        max_steps = int(round(max_time / self.dt))
        while self.steps < max_steps:
            self.step(record)
            if self.is_settled(hold_steps):
                break
        return self.result()

    def result(self):
        rec = self._record
        n = len(rec["t"])
        nb = len(self.bodies)
        out = {"time": np.asarray(rec["t"])}
        if nb:
            out["pos"] = np.asarray(rec["pos"]) if n else np.zeros((0, nb, 3))
            out["quat"] = np.asarray(rec["quat"]) if n else np.zeros((0, nb, 4))
            out["vel"] = np.asarray(rec["vel"]) if n else np.zeros((0, nb, 3))
            out["omega"] = np.asarray(rec["omega"]) if n else np.zeros((0, nb, 3))
        out["terminal"] = [
            {"pos": b.pos.tolist(), "quat": b.quat.tolist(),
             "vel": b.vel.tolist(), "omega": b.omega.tolist()}
            for b in self.bodies
        ]
        out["rest_time"] = float(self.t)
        out["steps"] = int(self.steps)
        out["max_penetration"] = float(self.max_penetration)
        out["contact_frames"] = int(self.contact_frames)
        out["settled"] = bool(self.is_settled())
        return out

    def total_kinetic_energy(self):
        return sum(b.kinetic_energy() for b in self.bodies)