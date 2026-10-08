import numpy as np

from . import mathx as mx


def _body_inv(bodies, idx):
    if idx < 0:
        return 0.0, 0.0, np.zeros(3), np.zeros(3)
    b = bodies[idx]
    return b.inv_mass, b.inv_inertia, b.pos, b.vel


def _effective_mass(inv_m_a, invI_a, r_a, inv_m_b, invI_b, r_b, n):
    denom = inv_m_a + inv_m_b
    ra_x_n = mx.cross3(r_a, n)
    rb_x_n = mx.cross3(r_b, n)
    denom += invI_a * np.dot(ra_x_n, ra_x_n)
    denom += invI_b * np.dot(rb_x_n, rb_x_n)
    return denom


class Solver:
    def __init__(self, bodies, iterations=12, beta=0.2, slop=0.001,
                 rest_threshold=0.1):
        self.bodies = bodies
        self.iterations = iterations
        self.beta = beta
        self.slop = slop
        self.rest_threshold = rest_threshold

    def _apply_impulse(self, idx, P, r):
        if idx < 0:
            return
        b = self.bodies[idx]
        b.vel += P * b.inv_mass
        b.omega += b.inv_inertia * mx.cross3(r, P)

    def _velocity(self, idx, r):
        if idx < 0:
            return np.zeros(3)
        return self.bodies[idx].vel + mx.cross3(self.bodies[idx].omega, r)

    def _solve_normal(self, c):
        inv_m_a, invI_a, pos_a, _ = _body_inv(self.bodies, c.a)
        inv_m_b, invI_b, pos_b, _ = _body_inv(self.bodies, c.b)
        r_a = c.point - pos_a
        r_b = c.point - pos_b

        rel = self._velocity(c.b, r_b) - self._velocity(c.a, r_a)
        rel_n = np.dot(rel, c.normal)

        if c.vbias is None:
            c.vbias = c.e * rel_n if rel_n < -self.rest_threshold else 0.0

        denom = _effective_mass(inv_m_a, invI_a, r_a, inv_m_b, invI_b, r_b, c.normal)
        if denom <= 0:
            return

        lam = -(rel_n + c.vbias) / denom

        old = c.jn
        new = max(0.0, old + lam)
        dj = new - old
        c.jn = new

        P = dj * c.normal
        self._apply_impulse(c.a, -P, r_a)
        self._apply_impulse(c.b, P, r_b)

    def _solve_friction(self, c):
        inv_m_a, invI_a, pos_a, _ = _body_inv(self.bodies, c.a)
        inv_m_b, invI_b, pos_b, _ = _body_inv(self.bodies, c.b)
        r_a = c.point - pos_a
        r_b = c.point - pos_b

        rel = self._velocity(c.b, r_b) - self._velocity(c.a, r_a)
        rel_n = np.dot(rel, c.normal)
        vt = rel - rel_n * c.normal
        norm_vt = np.linalg.norm(vt)
        if norm_vt < 1e-9:
            return
        t = vt / norm_vt

        denom_t = _effective_mass(inv_m_a, invI_a, r_a, inv_m_b, invI_b, r_b, t)
        if denom_t <= 0:
            return

        lam_t = -np.dot(rel, t) / denom_t
        lam_t = max(-c.mu * c.jn, min(c.mu * c.jn, lam_t))
        if lam_t == 0.0:
            return

        P = lam_t * t
        self._apply_impulse(c.a, -P, r_a)
        self._apply_impulse(c.b, P, r_b)

    def _relative_normal_velocity(self, c):
        inv_m_a, _, pos_a, _ = _body_inv(self.bodies, c.a)
        inv_m_b, _, pos_b, _ = _body_inv(self.bodies, c.b)
        return np.dot(self._velocity(c.b, c.point - pos_b) - self._velocity(c.a, c.point - pos_a), c.normal)

    def _set_restitution_bias(self, contacts):
        groups = {}
        for c in contacts:
            key = (c.a, c.b, round(c.normal[0], 2), round(c.normal[1], 2), round(c.normal[2], 2))
            groups.setdefault(key, []).append(c)
        for clist in groups.values():
            ref = float("inf")
            for c in clist:
                rn = self._relative_normal_velocity(c)
                if rn < ref:
                    ref = rn
            vb = clist[0].e * ref if ref < -self.rest_threshold else 0.0
            for c in clist:
                c.vbias = vb

    def solve_velocity(self, contacts, dt):
        self._set_restitution_bias(contacts)
        for _ in range(self.iterations):
            for c in contacts:
                self._solve_normal(c)
            for c in contacts:
                self._solve_friction(c)

    def solve_position(self, contacts, dt):
        for c in contacts:
            inv_m_a, _, _, _ = _body_inv(self.bodies, c.a)
            inv_m_b, _, _, _ = _body_inv(self.bodies, c.b)
            total_inv = inv_m_a + inv_m_b
            if total_inv <= 0:
                continue
            correction = max(0.0, c.depth - self.slop) * self.beta
            shift = (correction / total_inv) * c.normal
            if c.a >= 0:
                self.bodies[c.a].pos -= shift * inv_m_a
            if c.b >= 0:
                self.bodies[c.b].pos += shift * inv_m_b