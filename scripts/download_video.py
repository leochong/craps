import argparse
import os

import yt_dlp


def main():
    ap = argparse.ArgumentParser(description="Download a YouTube video (video-only) for pipeline testing")
    ap.add_argument("url")
    ap.add_argument("--out", default="data/downloads")
    ap.add_argument("--max-height", type=int, default=1080)
    ap.add_argument("--format", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fmt = args.format or (
        f"bestvideo[height<={args.max_height}][ext=mp4][vcodec^=avc1]/"
        f"bestvideo[height<={args.max_height}][ext=mp4]/bestvideo[height<={args.max_height}]/best"
    )
    opts = {
        "outtmpl": os.path.join(args.out, "%(id)s_%(height)sp.%(ext)s"),
        "format": fmt,
        "no_warnings": True,
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(args.url, download=True)
        print("downloaded:", ydl.prepare_filename(info))
        print("duration:", info.get("duration"), "s  height:", info.get("height"))


if __name__ == "__main__":
    main()