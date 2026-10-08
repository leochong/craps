import argparse
import json
import os

import cv2

from m1 import homography as hom
from m1 import pose as pose_mod
from m1.frame_grab import FrameSource
from m1.hitl import draw_overlays


def main():
    ap = argparse.ArgumentParser(description="M1 smoke test on the reference video")
    ap.add_argument("--video", default="IMG_2111.MOV")
    ap.add_argument("--frames", default="0,300,600,1200,2400")
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    indices = [int(t) for t in args.frames.split(",") if t.strip() != ""]
    os.makedirs(args.out, exist_ok=True)

    with FrameSource(args.video) as probe:
        src_w, src_h = probe.width, probe.height
    target = (int(src_w * args.scale), int(src_h * args.scale))

    summary = {"video": args.video, "fps": None, "frames": []}
    with FrameSource(args.video, target_size=target) as src:
        summary["fps"] = src.fps
        K = pose_mod.default_intrinsics(target[0], target[1])
        for idx in indices:
            frame = src.read(idx)
            if frame is None:
                print(f"frame {idx}: MISSING")
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate = hom.plate_origin(frame)
            dice = pose_mod.detect_dice_and_pips(gray)
            for d in dice:
                if d.get("face_value") is not None and d["pips"].size:
                    d["pose"] = pose_mod.solve_die_pose(d["pips"], d["face_value"], K)

            recap = {
                "frame": idx,
                "plate": {k: int(plate[k]) for k in ("cx", "cy", "radius")} if plate else None,
                "dice": [
                    {
                        "bbox": [int(v) for v in d["bbox"]],
                        "face_value": d["face_value"],
                        "num_pips": int(d["pips"].shape[0]),
                        "residual_px": round(d["pose"]["residual_px"], 3) if d.get("pose") else None,
                    }
                    for d in dice
                ],
            }
            summary["frames"].append(recap)

            overlay = draw_overlays(frame, plate, dice)
            out_png = os.path.join(args.out, f"probe_{idx}.png")
            cv2.imwrite(out_png, overlay)
            print(f"frame {idx}: plate={recap['plate']} dice={len(dice)}")

    out_json = os.path.join(args.out, "probe_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()