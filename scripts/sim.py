import argparse
import json
import os

import numpy as np

from m2.body import Body
from m2.world import Geometry, World


def main():
    ap = argparse.ArgumentParser(description="Run one bubble-craps dice rollout")
    ap.add_argument("--out", default="out")
    ap.add_argument("--impulse", type=float, default=0.08)
    ap.add_argument("--elevation", type=float, default=1.2)
    ap.add_argument("--yaw", type=float, default=0.3)
    ap.add_argument("--spin", nargs=3, type=float, default=[0.0, 40.0, 25.0])
    ap.add_argument("--max-time", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    side = 0.016
    mass = 0.006
    dice = [
        Body(side=side, mass=mass, pos=(0.0, 0.0, side / 2), omega=np.asarray(args.spin)),
        Body(side=side, mass=mass, pos=(0.03, 0.0, side / 2), omega=np.asarray(args.spin)),
    ]
    world = World(geometry=Geometry(), bodies=dice)

    up = np.array([0.0, 0.0, 1.0])
    horizontal = np.array([np.cos(args.yaw), np.sin(args.yaw), 0.0])
    direction = args.elevation * up + horizontal
    direction = direction / np.linalg.norm(direction)
    for b in dice:
        b.vel += direction * (args.impulse / b.mass)

    world.simulate(max_time=args.max_time)
    res = world.result()

    os.makedirs(args.out, exist_ok=True)
    np.savez(
        os.path.join(args.out, "rollout.npz"),
        time=res["time"], pos=res["pos"], quat=res["quat"],
        vel=res["vel"], omega=res["omega"],
    )
    summary = {
        "settled": res["settled"],
        "rest_time_s": res["rest_time"],
        "steps": res["steps"],
        "max_penetration_m": res["max_penetration"],
        "contact_frames": res["contact_frames"],
        "terminal": res["terminal"],
    }
    with open(os.path.join(args.out, "rollout.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/rollout.npz + rollout.json")


if __name__ == "__main__":
    main()