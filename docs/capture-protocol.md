# Capture Protocol — Bubble-Craps Calibration Footage

Scope: collecting footage that feeds M1 pose extraction and M2/M3 inverse
calibration. The current capture ceiling is **1080p30**, which determines what
physics parameters are recoverable (see "What 30 fps can identify").

## Research-integrity rules

- **Original-resolution, native-fps frames are the only label/measurement ground
  truth.** Interpolated (RIFE/optical-flow) or upscaled frames are *augmentation
  inputs only* and must never supervise calibration or keypoint labels.
- Register every derived asset with its provenance flag (`scripts/ingest_clip.py`),
  keeping `synthetic=true` / `ground_truth_labels=false`.
- Never commit footage, `.env`, or secrets (all media and `datasets/` are ignored).

## Hardware and framing

- Camera: fixed, tripod-mounted, locked to a single frame rate and resolution for
  a session. Do not change zoom mid-session (it invalidates calibration).
- Framing: near-overhead view of the plate, as close as the machine allows so the
  plate fills most of the frame (more pixels/die → better PnP residual).
- Lighting: bright, diffuse, steady; avoid strobing/mixed sources to reduce motion
  blur and rolling-shutter artifacts.
- Focus: manual focus locked on the plate plane.
- Include the **plate rim/circle** fully in view — it is the calibration reference.

## Session checklist

1. Record a short **calibration still/clip** with the dice at rest and the plate
   fully visible; use it for `scripts/track_clip.py` plate calibration.
2. Record **many rolls per session** (aim 10+ usable rolls per clip/session).
3. Vary launch conditions deliberately:
   - launch energy (soft vs hard),
   - direction/yaw across the plate,
   - spin,
   - one die vs two dice,
   - dice contacting felt vs dome vs bumper.
4. Keep the camera untouched between rolls in a session.
5. Note session metadata: machine id, plate radius (m), camera height estimate,
   settings, lighting.

## Per-clip metadata to record

- `asset_id`, machine id, date, resolution, fps, lens/zoom setting.
- Plate radius in metres (physical measurement, not estimated).
- Whether footage is original or interpolated/upscaled (`synthetic` flag).
- Any occlusions, glare, or dome reflections.

## What 30 fps can identify

At 1080p30, impacts last tens of milliseconds and are effectively sub-frame, so:

- **Fit from real footage**: launch energy/impulse, launch direction
  (elevation/yaw), effective `mu_felt`, and initial die poses.
- **Hold at priors (need 120/240 fps or simulation priors)**: restitution
  `e_felt`/`e_dome`/`e_bumper` and dome/bumper contact detail.
- Use `adjoint.identifiability` (condition number) as the gate: if it diverges for
  a parameter, do not trust it at this frame rate.

## Processing pipeline

1. `python -m scripts.track_clip.py --video <clip> --asset-id <id> [--frames 0:120]`
   → felt-ellipse plane calibration (circle-from-ellipse pose) + machine-frame
   `datasets/trajectories/<id>.jsonl`. Measure the felt radius; it fixes metric scale.
2. `python -m scripts.fit_clip.py --asset-id <id> --active mu_felt,impulse`
   → single-clip fit and identifiability report.
3. `python -m scripts.fit_calibration.py --active mu_felt,e_felt,impulse`
   → shared parameters fitted across all trajectory clips (multi-clip; adds
   `mu_dome/e_dome/mu_bumper/e_bumper/angular_damping` when active).
4. `python -m scripts.predict --calibration out/fit_calibration.json`
   → **unvalidated** P(2..12) + EV/Kelly table from the fitted twin. Do not read
   the edge as real until the forward-fidelity gate passes.
5. `python -m scripts.forward_fidelity --calibration out/fit_calibration.json [--gate]`
   → simulated-vs-measured divergence over the first impacts; `--gate` exits
   nonzero if the worst relative divergence exceeds 8%.
6. `python -m scripts.validate_trajectory.py`, `validate_calib_geom.py`,
   `validate_calibration.py`, `validate_end2end.py`, `validate_forward.py`,
   `validate_reliability.py` → plumbing/geometry/fidelity/calibration checks.

Multi-clip calibration is the reason to collect several rolls per session: a
single clip may leave a parameter unidentifiable (condition number `inf`), while
stacking diverse launches constrains it. Expand the active set only while
`cond(J)` stays finite and the held-out divergence gate holds.

## When to invest in more (or higher-fps) footage

More clips help only if they add **diversity of launch conditions** and keep
native fps/resolution. Before scaling capture, confirm the held-out divergence
gate (< 8% cumulative spatial divergence over the first three impacts). If contact
restitution matters, prioritise a 120/240 fps capture rig over more 30 fps volume.
