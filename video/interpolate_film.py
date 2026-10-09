"""GPU frame interpolation via FILM (google-research/frame-interpolation). RunPod-validated.

This wraps the official FILM CLI, which operates on PNG directories and upsamples
the temporal rate by 2**times_to_interpolate (times=2 -> 4x -> 30->120 fps when
applied to a 30 fps sequence).
"""

import glob
import os
import subprocess


def run_film_cli(input_dir, repo_dir, model_path, times_to_interpolate=2,
                 python="python", extra_args=None):
    """Run `python -m eval.interpolator_cli` from the FILM repo.

    The FILM version determines where interpolated frames are written (commonly a
    subdirectory of `input_dir`). Discover it with `find_output_dir`.
    """
    cmd = [
        python, "-m", "eval.interpolator_cli",
        "--pattern", os.path.abspath(input_dir),
        "--model_path", model_path,
        "--times_to_interpolate", str(times_to_interpolate),
    ]
    if extra_args:
        cmd += list(extra_args)
    subprocess.run(cmd, cwd=repo_dir, check=True)
    return cmd


def find_output_dir(input_dir):
    """Return the directory (under input_dir) containing the most PNGs after FILM."""
    candidates = []
    for p in glob.glob(os.path.join(input_dir, "**", "*.png"), recursive=True):
        d = os.path.dirname(p)
        if d not in candidates:
            candidates.append(d)
    if not candidates:
        return None
    def count(d):
        return len(glob.glob(os.path.join(d, "*.png")))
    return max(candidates, key=count)