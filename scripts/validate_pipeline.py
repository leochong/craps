import math
import sys

import numpy as np

from m2 import ev, faces, montecarlo
from m2.montecarlo import DEFAULT_SPEC, FAIR


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


def bench_face_readout():
    ident = np.array([0.0, 0.0, 0.0, 1.0])
    half_x = math.sqrt(0.5)
    q180x = np.array([1.0, 0.0, 0.0, 0.0])
    q90x = np.array([half_x, 0.0, 0.0, half_x])
    q180z = np.array([0.0, 0.0, 1.0, 0.0])
    ok = (
        faces.up_face(ident) == 1
        and faces.up_face(q180x) == 6
        and faces.up_face(q90x) == 3
        and faces.up_face(q180z) == 1
        and faces.opposite_sum_ok()
    )
    return check("face read-out", ok,
                 f"ident={faces.up_face(ident)} 180x={faces.up_face(q180x)} "
                 f"90x={faces.up_face(q90x)} opposite_ok={faces.opposite_sum_ok()}")


def bench_fair_distribution():
    total = sum(FAIR.values())
    mean = sum(s * FAIR[s] for s in FAIR)
    ok = abs(total - 1.0) < 1e-12 and abs(mean - 7.0) < 1e-12 and FAIR[7] == 6 / 36
    return check("fair distribution", ok, f"sum={total:.6f} mean={mean:.4f}")


def bench_kl():
    ok = abs(montecarlo.kl_divergence(FAIR, FAIR)) < 1e-12
    shifted = dict(FAIR)
    shifted[7] += 0.1
    shifted[2] -= 0.1
    return check("KL divergence", ok and montecarlo.kl_divergence(shifted, FAIR) > 0,
                 f"fair|fair={montecarlo.kl_divergence(FAIR, FAIR):.2e}")


def bench_house_edge():
    ref = {r["target"]: r["ev"] for r in ev.house_edge_reference()}
    expected = {
        6: -1 / 66, 8: -1 / 66,
        5: -0.04, 9: -0.04,
        4: -1 / 15, 10: -1 / 15,
        2: -1 / 14, 12: -1 / 14,
        3: -0.0625, 11: -0.0625,
    }
    ok = all(abs(ref[t] - expected[t]) < 1e-9 for t in expected)
    worst = min(expected, key=lambda t: ref[t])
    return check("craps house edges (place bets)", ok,
                 f"6={ref[6]*100:.3f}% 5={ref[5]*100:.2f}% 4={ref[4]*100:.3f}% 2={ref[2]*100:.3f}%")


def bench_aggregation():
    results = [{"sum": 7, "settled": True}] * 3 + [{"sum": 6, "settled": False}]
    dist = montecarlo.aggregate(results, n=4)
    total = sum(dist["P"].values())
    ok = abs(total - 1.0) < 1e-12 and abs(dist["P"][7] - 0.75) < 1e-12
    return check("MC aggregation", ok, f"P7={dist['P'][7]:.3f} total={total:.6f}")


def bench_single_rollout():
    spec = dict(DEFAULT_SPEC)
    spec["max_time"] = 1.0
    r = montecarlo.simulate_once(0, spec)
    ok = 2 <= r["sum"] <= 12 and len(r["terminal"]) == 2
    return check("single rollout", ok, f"sum={r['sum']} rest={r['rest_time']:.3f}s")


def main():
    results = [
        bench_face_readout(),
        bench_fair_distribution(),
        bench_kl(),
        bench_house_edge(),
        bench_aggregation(),
        bench_single_rollout(),
    ]
    print(f"\n{sum(results)}/{len(results)} pipeline benchmarks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()