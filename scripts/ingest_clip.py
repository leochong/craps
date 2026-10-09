import argparse

from m1.dataset import VideoDataset


def parse_size(text):
    if not text:
        return None
    w, h = text.lower().split("x")
    return (int(w), int(h))


def main():
    ap = argparse.ArgumentParser(description="Register a video as a CV training asset")
    ap.add_argument("--video", required=True)
    ap.add_argument("--root", default="datasets/cv")
    ap.add_argument("--id", required=True)
    ap.add_argument("--origin", default=None, help="source video this was derived from")
    ap.add_argument("--derivation", default="none")
    ap.add_argument("--synthetic", action="store_true",
                    help="interpolated/upscaled content (not ground truth)")
    ap.add_argument("--intended-use", default="cv_augmentation")
    ap.add_argument("--note", default="")
    ap.add_argument("--extract-frames", type=int, default=0)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--frame-size", default=None, help="WxH")
    args = ap.parse_args()

    provenance = {
        "origin": args.origin,
        "derivation": args.derivation,
        "synthetic": bool(args.synthetic),
        "ground_truth_labels": False,
        "intended_use": args.intended_use,
        "note": args.note,
    }
    ds = VideoDataset(args.root)
    entry = ds.register(
        args.video, args.id, provenance=provenance,
        extract_frames=args.extract_frames, frame_size=parse_size(args.frame_size),
        frame_stride=args.frame_stride,
    )
    print(f"registered '{entry['id']}' -> {args.root}/manifest.jsonl")
    print(f"  {entry['width']}x{entry['height']} @ {entry['fps']:.1f}fps, "
          f"{entry['frames']} frames, {entry['duration_s']}s, synthetic={entry['synthetic']}")
    if entry.get("extracted_frames"):
        print(f"  extracted {entry['extracted_frames']} frames -> {entry['frames_dir']}")


if __name__ == "__main__":
    main()