"""Map calibrated inverse-physics parameters into a Monte Carlo spec.

`theta_to_spec` writes fitted machine parameters (friction/restitution/damping,
launch impulse) into a `montecarlo.DEFAULT_SPEC`-shaped dict; `load_calibration`
reads a `fit_calibration.json` produced by `scripts/fit_calibration.py`. Output
used for prediction is explicitly unvalidated until the forward-fidelity gate
(Phase B) passes.
"""

import json

import numpy as np

from . import adjoint
from .montecarlo import DEFAULT_SPEC


def theta_to_spec(theta, base_spec=None, launch=None):
    """Return a Monte Carlo spec with fitted geometry/impulse from a theta vector."""
    base = dict(base_spec if base_spec is not None else DEFAULT_SPEC)
    overrides = adjoint.theta_to_overrides(theta)
    geom = dict(base["geom"])
    for name in ("mu_felt", "e_felt") + adjoint.GEOM_PARAM_NAMES:
        if name in overrides:
            geom[name] = float(overrides[name])
    base["geom"] = geom
    if "impulse" in overrides:
        base["impulse"] = float(overrides["impulse"])
    if launch:
        for key, value in launch.items():
            if key in base:
                base[key] = value
    return base


def load_calibration(path):
    """Load fitted parameters from a calibration JSON; report bound-pinned params."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    names = payload.get("param_names") or list(adjoint.ALL_PARAM_NAMES)
    fitted = payload.get("fitted")
    if fitted is None:
        raise ValueError(f"{path} has no 'fitted' parameters")

    theta = adjoint.default_theta()
    for i, name in enumerate(adjoint.ALL_PARAM_NAMES):
        if name in names:
            theta[i] = float(fitted[names.index(name)])

    at_bounds = []
    for i, name in enumerate(adjoint.ALL_PARAM_NAMES):
        lo, hi = adjoint.BOUNDS_ALL[name]
        if abs(theta[i] - lo) < 1e-6 or abs(theta[i] - hi) < 1e-6:
            at_bounds.append(name)
    return {
        "theta": theta,
        "active": payload.get("active"),
        "identifiability": payload.get("identifiability", {}),
        "at_bounds": at_bounds,
        "raw": payload,
    }


def quality_flags(report):
    """Human-readable warnings about a calibration report."""
    flags = []
    if report.get("at_bounds"):
        flags.append(f"params at bounds: {', '.join(report['at_bounds'])}")
    cond = report.get("identifiability", {}).get("condition_number")
    if cond is not None and (not np.isfinite(cond) or cond > 1e3):
        flags.append(f"poor conditioning (cond(J)={cond:.3g})")
    return flags
