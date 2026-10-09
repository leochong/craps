import os
import subprocess

import cv2


class OpenCVEncoder:
    def __init__(self, path, fps, size, fourcc="mp4v"):
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for {path} ({fourcc})")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class FfmpegNvencEncoder:
    def __init__(self, path, fps, size, codec="hevc_nvenc", preset="p7", cq=18,
                 pix_fmt="yuv420p", level="5.2", ffmpeg="ffmpeg"):
        w, h = size
        cmd = [
            ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(int(fps)), "-i", "-",
            "-an", "-c:v", codec, "-preset", preset, "-cq", str(cq),
            "-pix_fmt", pix_fmt, "-profile:v", "main", "-level", str(level),
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def ffmpeg_has_nvenc(ffmpeg="ffmpeg"):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return "hevc_nvenc" in out.stdout or "h264_nvenc" in out.stdout


def nvenc_works(ffmpeg="ffmpeg"):
    if not ffmpeg_has_nvenc(ffmpeg):
        return False
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=256x144:rate=5",
             "-c:v", "hevc_nvenc", "-frames:v", "5", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30)
        return out.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def make_encoder(path, fps, size, prefer_nvenc=True, ffmpeg="ffmpeg"):
    if prefer_nvenc and nvenc_works(ffmpeg):
        return FfmpegNvencEncoder(path, fps, size, ffmpeg=ffmpeg)
    return OpenCVEncoder(path, fps, size)


def encode_png_sequence(in_dir, out_path, fps, pattern="frame_%05d.png", codec="hevc_nvenc",
                        preset="p7", cq=18, pix_fmt="yuv420p", level="5.2", ffmpeg="ffmpeg"):
    cmd = [
        ffmpeg, "-y", "-framerate", str(int(fps)),
        "-i", os.path.join(in_dir, pattern),
        "-an", "-c:v", codec, "-preset", preset, "-cq", str(cq),
        "-pix_fmt", pix_fmt, "-profile:v", "main", "-level", str(level), out_path,
    ]
    subprocess.run(cmd, check=True)
    return cmd