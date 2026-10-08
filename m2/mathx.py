import numpy as np


def cross3(a, b):
    return np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def identity_quat():
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 0 else q


def quat_mult(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v):
    qv = q[:3]
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def unrotate(q, v):
    return quat_rotate(quat_conjugate(q), v)


def R_from_quat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_integrate(q, omega, dt):
    omega_q = np.array([omega[0], omega[1], omega[2], 0.0])
    dq = 0.5 * quat_mult(omega_q, q) * dt
    return quat_normalize(q + dq)


def cube_inv_inertia(mass, side):
    if mass <= 0:
        return 0.0
    return 6.0 / (mass * side * side)