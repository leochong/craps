import numpy as np

from . import mathx as mx

FACE_TABLE = [
    (np.array([0.0, 0.0, 1.0]), 1),
    (np.array([0.0, 0.0, -1.0]), 6),
    (np.array([1.0, 0.0, 0.0]), 2),
    (np.array([-1.0, 0.0, 0.0]), 5),
    (np.array([0.0, 1.0, 0.0]), 3),
    (np.array([0.0, -1.0, 0.0]), 4),
]


def up_face(quat):
    best_val = None
    best_z = -2.0
    for n_local, value in FACE_TABLE:
        z = mx.quat_rotate(quat, n_local)[2]
        if z > best_z:
            best_z = z
            best_val = value
    return best_val


def outcome(quats):
    return int(sum(up_face(q) for q in quats))


def opposite_sum_ok():
    for n_local, value in FACE_TABLE:
        for m_local, other in FACE_TABLE:
            if np.allclose(n_local, -m_local) and value + other != 7:
                return False
    return True