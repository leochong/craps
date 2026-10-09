"""Extract machine-frame die trajectories from a clip (M1 -> M3 bridge).

Calibrates the plate plane, runs per-frame PnP tracking, and writes Trajectory
records for downstream inverse calibration. Interpolated/derived footage must be
registered with --synthetic so it stays out of label supervision.
"""

import argparse
import os

import cv2

from m1.homography import calibrate_plate, load_calibration, plate_calibration, save_calibration
from m1.track import track_poses
from m1.trajectory import save_trajectories, validate_trajectory


def parse_range(text):
    if not text:
        return None
    start, end = text.split(":")
    return (int(start), int(end))


def resolve_calibration(args, video, asset_id):
    if args.calib and os.path.exists(args.calib):
        calib = load_calibration(args.calib)
        calib["asset_id"] = asset_id
        return calib
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if args.cx is not None and args.radius_px is not None:
        circle = {"cx": args.cx, "cy": args.cy, "radius": args.radius_px}
        calib = plate_calibration(circle, frame.shape[1], frame.shape[0],
                                  plate_radius_m=args.plate_radius_m, focal=args.focal)
    else:
        calib = calibrate_plate(frame, plate_radius_m=args.plate_radius_m, focal=args.focal)
    if calib is None:
        return None
    calib["asset_id"] = asset_id
    save_calibration(calib, args.calib_out)
    return calib


def main():
    ap = argparse.ArgumentParser(description="Extract die trajectories from a clip")
    ap.add_argument("--video", default="IMG_2111.MOV")
    ap.add_argument("--asset-id", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--calib", default=None, help="existing calibration JSON")
    ap.add_argument("--calib-out", default=None)
    ap.add_argument("--plate-radius-m", type=float, default=0.15,
                    help="physical felt radius in metres (measure per machine)")
    ap.add_argument("--focal", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--radius-px", type=float, default=None)
    ap.add_argument("--frames", default=None, help="start:end")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--max-residual", type=float, default=50.0)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"SKIP: {args.video} not found")
        return

    asset_id = args.asset_id or os.path.splitext(os.path.basename(args.video))[0]
    out = args.out or os.path.join("datasets", "trajectories", f"{asset_id}.jsonl")
    args.calib_out = args.calib_out or os.path.join("datasets", "trajectories", f"{asset_id}.calib.json")

    calib = resolve_calibration(args, args.video, asset_id)
    if calib is None:
        print("FAIL: could not calibrate plate (pass --cx/--cy/--radius-px or --calib)")
        return

    frame_range = parse_range(args.frames)
    if frame_range and args.max_frames:
        frame_range = (frame_range[0], min(frame_range[1], frame_range[0] + args.max_frames))

    provenance = {"source": args.video, "synthetic": bool(args.synthetic)}
    tracks = track_poses(args.video, calib, frame_range=frame_range, stride=args.stride,
                         min_samples=args.min_samples, asset_id=asset_id,
                         synthetic=args.synthetic, provenance=provenance,
                         max_residual_px=args.max_residual)
    valid = [t for t in tracks if not validate_trajectory(t)]
    save_trajectories(valid, out)
    print(f"calib: cx={calib['cx']:.1f} cy={calib['cy']:.1f} r_px={calib['radius_px']:.1f} "
          f"scale={calib['scale_m_per_px']*1000:.3f} mm/px")
    print(f"wrote {len(valid)}/{len(tracks)} trajectories -> {out}  synthetic={args.synthetic}")


if __name__ == "__main__":
    main()
