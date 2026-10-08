import numpy as np

from . import mathx as mx


class Body:
    def __init__(self, side=0.016, mass=0.006, pos=(0.0, 0.0, 0.1),
                 quat=None, vel=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
        self.side = side
        self.mass = mass
        self.inv_mass = 1.0 / mass if mass > 0 else 0.0
        self.inv_inertia = mx.cube_inv_inertia(mass, side)

        self.pos = np.asarray(pos, dtype=float)
        self.quat = quat if quat is not None else mx.identity_quat()
        self.vel = np.asarray(vel, dtype=float)
        self.omega = np.asarray(omega, dtype=float)

    @property
    def half(self):
        return self.side / 2.0

    def R(self):
        return mx.R_from_quat(self.quat)

    def local_to_world(self, p_local):
        return self.pos + mx.quat_rotate(self.quat, p_local)

    def world_to_local(self, p_world):
        return mx.unrotate(self.quat, p_world - self.pos)

    def vertices(self):
        h = self.half
        corners = np.array([
            [-h, -h, -h], [h, -h, -h], [-h, h, -h], [h, h, -h],
            [-h, -h, h], [h, -h, h], [-h, h, h], [h, h, h],
        ])
        R = self.R()
        return self.pos + corners @ R.T

    def surface_samples(self):
        h = self.half
        local = []
        for sx in (-h, h):
            for sy in (-h, h):
                for sz in (-h, h):
                    local.append([sx, sy, sz])
        for axis in range(3):
            for s in (-h, h):
                p = [0.0, 0.0, 0.0]
                p[axis] = s
                local.append(p)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3
            for sj in (-h, h):
                for sk in (-h, h):
                    p = [0.0, 0.0, 0.0]
                    p[j] = sj
                    p[k] = sk
                    local.append(p)
        local = np.asarray(local)
        R = self.R()
        return self.pos + local @ R.T

    def velocity_at(self, point):
        return self.vel + mx.cross3(self.omega, point - self.pos)

    def apply_gravity(self, g, dt):
        self.vel += g * dt

    def integrate(self, dt):
        self.pos += self.vel * dt
        self.quat = mx.quat_integrate(self.quat, self.omega, dt)

    def kinetic_energy(self):
        linear = 0.5 * self.mass * np.dot(self.vel, self.vel)
        I = 1.0 / self.inv_inertia if self.inv_inertia > 0 else 0.0
        angular = 0.5 * I * np.dot(self.omega, self.omega)
        return linear + angular