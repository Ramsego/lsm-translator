"""
Download one or more full mañanera conferences from YouTube.

Usage:
    # full video
    python phase2/download_mananera.py VIDEO_ID [VIDEO_ID ...]

    # section only (spike / testing)
    python phase2/download_mananera.py VIDEO_ID --start 00:15:00 --end 00:25:00
"""

import argparse
import subprocess
from pathlib import Path

OUT_BASE = Path("/Volumes/Crucial X8/LSM_Translator/videos/mananera")


def download(video_id: str, height: int, start: str = None, end: str = None):
    out_dir = OUT_BASE / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.mp4"

    if out_path.exists():
        print(f"Already downloaded: {video_id}, skipping.")
        return

    url = f"https://www.youtube.com/watch?v={video_id}"
    # Prefer h264 (avc1) so OpenCV can decode without extra codecs.
    # Omit --force-keyframes-at-cuts (conda ffmpeg 4.3 fails on it).
    fmt = (f"bv*[height<={height}][vcodec^=avc1]+ba/b[height<={height}][vcodec^=avc1]"
           f"/bv*[height<={height}]+ba/b[height<={height}]")
    cmd = [
        "yt-dlp", "-f", fmt,
        "--merge-output-format", "mp4",
        "--write-info-json",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
    ]
    if start and end:
        cmd += ["--download-sections", f"*{start}-{end}"]
    cmd.append(url)

    print(f"Downloading {video_id} {'(full)' if not start else f'{start}-{end}'}...")
    subprocess.run(cmd, check=True)
    print(f"  → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+", help="YouTube video ID(s).")
    ap.add_argument("--start", default=None, help="Section start HH:MM:SS (omit for full video).")
    ap.add_argument("--end", default=None, help="Section end HH:MM:SS.")
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    for vid in args.video_ids:
        download(vid, args.height, args.start, args.end)


if __name__ == "__main__":
    main()
