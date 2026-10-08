import numpy as np


def box_local_sdf(p_local, h):
    q = np.abs(p_local) - h
    return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0)


def box_local_closest(p_local, h):
    return np.clip(p_local, -h, h)


def plane_sdf_axis(p, normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0)):
    n = np.asarray(normal, dtype=float)
    o = np.asarray(origin, dtype=float)
    return np.dot(p - o, n)


def cylinder_radial(p, radius, axis=(0.0, 0.0, 1.0)):
    ax = np.asarray(axis, dtype=float)
    radial_vec = p - np.dot(p, ax) * ax
    rad = np.linalg.norm(radial_vec)
    return rad, radial_vec / rad if rad > 0 else np.zeros(3)


def cylinder_wall_sdf(p, radius):
    rad, _ = cylinder_radial(p, radius)
    return rad - radius


def torus_sdf(p, major, minor, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    dx, dy, dz = p - c
    rad = np.hypot(dx, dy)
    q = np.hypot(rad - major, dz)
    return q - minor


def torus_normal(p, major, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    dx, dy, dz = p - c
    rad = np.hypot(dx, dy)
    if rad < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    q = np.hypot(rad - major, dz)
    if q < 1e-12:
        q = 1e-12
    radial_dir = np.array([dx / rad, dy / rad, 0.0])
    gr = (rad - major) / q * radial_dir
    gz = dz / q
    return np.array([gr[0], gr[1], gz])


def sphere_sdf(p, radius, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    return np.linalg.norm(p - c) - radius


def torus_sdf_array(points, major, minor, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    d = points - c
    rad = np.hypot(d[:, 0], d[:, 1])
    q = np.hypot(rad - major, d[:, 2])
    return q - minor


def torus_normal_array(points, major, center=(0.0, 0.0, 0.0)):
    c = np.asarray(center, dtype=float)
    d = points - c
    rad = np.hypot(d[:, 0], d[:, 1])
    q = np.hypot(rad - major, d[:, 2])
    safe_rad = np.where(rad < 1e-12, 1.0, rad)
    safe_q = np.where(q < 1e-12, 1e-12, q)
    n = np.empty_like(d)
    n[:, 0] = (rad - major) / safe_q * d[:, 0] / safe_rad
    n[:, 1] = (rad - major) / safe_q * d[:, 1] / safe_rad
    n[:, 2] = d[:, 2] / safe_q
    return n