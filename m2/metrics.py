"""Forward-fidelity and probability-calibration metrics.

`trajectory_divergence` compares a simulated trajectory against a measured one up
to the first few impacts (the M3 acceptance gate). The reliability helpers score a
predicted P(2..12) against held-out outcomes (Brier, reliability curve, ECE).
"""

import numpy as np

DIVERGENCE_GATE = 0.08
ECE_GATE = 0.10
CATEGORIES = tuple(range(2, 13))


def detect_impacts(times, pos, floor_z=0.008, band=0.02, min_gap=1):
    """Indices of floor impacts: local minima of Z within `band` of the floor."""
    z = np.asarray(pos, dtype=np.float64)[:, 2]
    candidates = [
        i for i in range(1, len(z) - 1)
        if z[i] <= z[i - 1] and z[i] <= z[i + 1] and z[i] <= floor_z + band
    ]
    impacts = []
    for i in candidates:
        if impacts and (i - impacts[-1]) < min_gap:
            continue
        impacts.append(i)
    return impacts


def _resample(times_src, pos_src, times_dst):
    return np.stack([np.interp(times_dst, times_src, pos_src[:, d]) for d in range(3)], axis=1)


def trajectory_divergence(pred_times, pred_pos, meas_times, meas_pos, n_impacts=3,
                          floor_z=0.008, band=0.02):
    """Cumulative spatial divergence of a predicted vs measured trajectory.

    `relative` is the summed position error divided by the measured path length up
    to the cut (the first `n_impacts` impacts, or the whole window if fewer).
    """
    pred_times = np.asarray(pred_times, dtype=np.float64)
    meas_times = np.asarray(meas_times, dtype=np.float64)
    pred_pos = np.asarray(pred_pos, dtype=np.float64).reshape(-1, 3)
    meas_pos = np.asarray(meas_pos, dtype=np.float64).reshape(-1, 3)

    pred = _resample(pred_times, pred_pos, meas_times)
    err = np.linalg.norm(pred - meas_pos, axis=1)
    impacts = detect_impacts(meas_times, meas_pos, floor_z=floor_z, band=band)
    cut = impacts[n_impacts - 1] if len(impacts) >= n_impacts else len(meas_times) - 1

    steps = np.linalg.norm(np.diff(meas_pos[: cut + 1], axis=0), axis=1)
    path = float(np.sum(steps))
    cumulative = float(np.sum(err[: cut + 1]))
    relative = cumulative / path if path > 1e-12 else (0.0 if cumulative < 1e-12 else float("inf"))
    return {
        "relative": relative,
        "absolute_mean_m": float(np.mean(err[: cut + 1])),
        "cumulative_m": cumulative,
        "measured_path_m": path,
        "n_impacts": len(impacts),
        "cut_index": int(cut),
        "impacts": impacts,
    }


def divergence_gate(report, threshold=DIVERGENCE_GATE):
    """True if the relative divergence is below the acceptance threshold."""
    return bool(report["relative"] < threshold)


def _prob_matrix(probabilities, categories=CATEGORIES):
    if isinstance(probabilities, np.ndarray):
        return np.asarray(probabilities, dtype=np.float64)
    return np.array([[float(p.get(s, 0.0)) for s in categories] for p in probabilities])


def brier_multiclass(probabilities, outcomes, categories=CATEGORIES):
    """Mean multi-class Brier score (0 perfect; ranges to 2)."""
    P = _prob_matrix(probabilities, categories)
    index = {s: i for i, s in enumerate(categories)}
    Y = np.zeros_like(P)
    for row, outcome in enumerate(outcomes):
        if outcome in index:
            Y[row, index[outcome]] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def reliability_curve(probabilities, indicators, n_bins=5):
    """Bin predicted probabilities and compare with observed frequencies."""
    probs = np.asarray(probabilities, dtype=np.float64)
    ind = np.asarray(indicators, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs >= lo) & (probs < hi) if b < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        bins.append({
            "lo": float(lo),
            "hi": float(hi),
            "p_mean": float(probs[mask].mean()),
            "frequency": float(ind[mask].mean()),
            "count": int(mask.sum()),
        })
    return bins


def expected_calibration_error(bins):
    """Count-weighted mean gap between predicted and observed frequency."""
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    return float(sum(b["count"] * abs(b["frequency"] - b["p_mean"]) for b in bins) / total)


def calibration_gate(ece, threshold=ECE_GATE):
    """True if expected calibration error is below the acceptance threshold."""
    return bool(ece < threshold)
