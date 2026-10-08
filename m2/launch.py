import numpy as np

from . import mathx as mx


def apply_air_blast(bodies, impulse=0.08, elevation=1.2, yaw=0.0, spin=(0.0, 40.0, 0.0)):
    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    direction = elevation * up + horizontal
    direction = direction / np.linalg.norm(direction)

    for b in bodies:
        dv = direction * (impulse / b.mass)
        b.vel += dv
        b.omega += np.asarray(spin, dtype=float)


def resting_poses(n_dice=2, side=0.016, radius=0.03):
    poses = []
    for i in range(n_dice):
        pos = np.array([0.0, 0.0, side / 2.0])
        if i == 1:
            pos = np.array([radius, 0.0, side / 2.0])
        quat = mx.identity_quat()
        poses.append((pos, quat))
    return poses