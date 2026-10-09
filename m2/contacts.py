import numpy as np

from . import shapes as sh

_EPS = 1e-9


class Contact:
    __slots__ = ("a", "b", "normal", "point", "depth", "mu", "e", "jn", "jt", "vbias", "jt_vec")

    def __init__(self, a, b, normal, point, depth, mu, e):
        self.a = a
        self.b = b
        self.normal = np.asarray(normal, dtype=float)
        self.point = np.asarray(point, dtype=float)
        self.depth = float(depth)
        self.mu = float(mu)
        self.e = float(e)
        self.jn = 0.0
        self.jt = 0.0
        self.vbias = None
        self.jt_vec = None


def _static_contacts(world, ai):
    body = world.bodies[ai]
    g = world.geometry
    contacts = []

    verts = body.vertices()
    dz = -verts[:, 2]
    mask = dz > 0
    for v, d in zip(verts[mask], dz[mask]):
        contacts.append(Contact(ai, -1, np.array([0.0, 0.0, -1.0]), v, d, g.mu_felt, g.e_felt))

    ceiling = verts[:, 2] - g.height
    mask = ceiling > 0
    for v, d in zip(verts[mask], ceiling[mask]):
        contacts.append(Contact(ai, -1, np.array([0.0, 0.0, 1.0]), v, d, g.mu_dome, g.e_dome))

    rad = np.hypot(verts[:, 0], verts[:, 1])
    mask = rad > g.radius
    for v, r in zip(verts[mask], rad[mask]):
        outward = np.array([v[0] / r, v[1] / r, 0.0])
        contacts.append(Contact(ai, -1, outward, v, r - g.radius, g.mu_dome, g.e_dome))

    samples = body.surface_samples()
    sdf = sh.torus_sdf_array(samples, g.bumper_major, g.bumper_minor, (0.0, 0.0, g.bumper_z))
    mask = sdf < 0
    normals = sh.torus_normal_array(samples[mask], g.bumper_major, (0.0, 0.0, g.bumper_z))
    for p, n, s in zip(samples[mask], normals, -sdf[mask]):
        contacts.append(Contact(ai, -1, n, p, float(s), g.mu_bumper, g.e_bumper))

    return contacts


def _face_vertices(body, axis_i, sign):
    h = body.half
    j = (axis_i + 1) % 3
    k = (axis_i + 2) % 3
    verts = []
    for sj in (-h, h):
        for sk in (-h, h):
            p = np.zeros(3)
            p[axis_i] = sign * h
            p[j] = sj
            p[k] = sk
            verts.append(body.local_to_world(p))
    return np.array(verts)


def _clip_halfspace(points, axis, dist):
    pts = list(points)
    n = len(pts)
    if n == 0:
        return []
    out = []
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        da = np.dot(a, axis) - dist
        db = np.dot(b, axis) - dist
        if da <= 0:
            out.append(a)
        if (da <= 0) != (db <= 0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def _best_face(body, target):
    R = body.R()
    bi, bs = 0, 1
    best = None
    for i in range(3):
        d = np.dot(R[:, i], target)
        if best is None or abs(d) > abs(best):
            best = d
            bi = i
            bs = 1 if d >= 0 else -1
    return bi, bs, best


def _box_box_manifold(A, B, normal):
    da = _best_face(A, normal)
    db = _best_face(B, normal)

    if abs(da[2]) >= abs(db[2]):
        ref_body, inc_body = A, B
        ref_axis, ref_sign = da[0], da[1]
    else:
        ref_body, inc_body = B, A
        ref_axis, ref_sign = db[0], db[1]

    R_ref = ref_body.R()
    refn = R_ref[:, ref_axis] * ref_sign
    h_ref = ref_body.half
    refc = ref_body.pos

    inc_axis, inc_sign, _ = _best_face(inc_body, -refn)
    incident = _face_vertices(inc_body, inc_axis, inc_sign)

    poly = _clip_halfspace(incident, refn, np.dot(refc, refn) + h_ref)
    u = R_ref[:, (ref_axis + 1) % 3]
    v = R_ref[:, (ref_axis + 2) % 3]
    poly = _clip_halfspace(poly, u, np.dot(refc, u) + h_ref)
    poly = _clip_halfspace(poly, -u, -np.dot(refc, u) + h_ref)
    poly = _clip_halfspace(poly, v, np.dot(refc, v) + h_ref)
    poly = _clip_halfspace(poly, -v, -np.dot(refc, v) + h_ref)

    points = []
    for p in poly:
        depth = np.dot(refc, refn) + h_ref - np.dot(p, refn)
        if depth > 0:
            points.append((p, depth))
    return points


def _box_box_contacts(world, ai, bi):
    A = world.bodies[ai]
    B = world.bodies[bi]
    g = world.geometry

    broad = (A.half + B.half) * np.sqrt(3.0)
    if np.linalg.norm(B.pos - A.pos) > broad:
        return []

    RA = A.R()
    RB = B.R()
    axes_a = [RA[:, 0], RA[:, 1], RA[:, 2]]
    axes_b = [RB[:, 0], RB[:, 1], RB[:, 2]]

    t = B.pos - A.pos
    axes = list(axes_a) + list(axes_b)
    for a in axes_a:
        for b in axes_b:
            c = np.cross(a, b)
            if np.linalg.norm(c) > 1e-8:
                axes.append(c / np.linalg.norm(c))

    best_axis = None
    best_overlap = np.inf
    for axis in axes:
        axis = axis / (np.linalg.norm(axis) + _EPS)
        ra = A.half * (abs(np.dot(axis, axes_a[0])) + abs(np.dot(axis, axes_a[1])) + abs(np.dot(axis, axes_a[2])))
        rb = B.half * (abs(np.dot(axis, axes_b[0])) + abs(np.dot(axis, axes_b[1])) + abs(np.dot(axis, axes_b[2])))
        dist = abs(np.dot(axis, t))
        overlap = ra + rb - dist
        if overlap < 0:
            return []
        if overlap < best_overlap:
            best_overlap = overlap
            best_axis = axis

    if best_axis is None:
        return []

    normal = best_axis
    if np.dot(normal, t) < 0:
        normal = -normal

    manifold = _box_box_manifold(A, B, normal)
    contacts = []
    for p, depth in manifold[:8]:
        contacts.append(Contact(ai, bi, normal, p, depth, g.mu_dice, g.e_dice))
    return contacts


def generate_contacts(world):
    contacts = []
    n = len(world.bodies)
    for i in range(n):
        contacts.extend(_static_contacts(world, i))
    for i in range(n):
        for j in range(i + 1, n):
            contacts.extend(_box_box_contacts(world, i, j))
    return contacts