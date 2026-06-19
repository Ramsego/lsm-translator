"""
Phase 2 feasibility spike: download a section of one mañanera.

Full conferences are ~2h; for the spike we grab a ~10-minute section (needs ffmpeg
for the cut). The interpreter recuadro is burned into the official broadcast under
the channel's /streams tab.

Usage:
    python phase2/download_mananera.py VIDEO_ID --start 00:15:00 --end 00:25:00
"""

import argparse
import subprocess
from pathlib import Path

OUT_BASE = Path("/Volumes/Crucial X8/LSM_Translator/videos/mananera")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id", help="YouTube id of the conference.")
    ap.add_argument("--start", default="00:15:00", help="Section start (HH:MM:SS).")
    ap.add_argument("--end", default="00:25:00", help="Section end (HH:MM:SS).")
    ap.add_argument("--height", type=int, default=720, help="Max video height.")
    args = ap.parse_args()

    out_dir = OUT_BASE / args.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={args.video_id}"

    # Prefer h264 (avc1) so OpenCV can decode it; av1 often fails in cv2.
    # Omit --force-keyframes-at-cuts (the bundled ffmpeg 4.3 fails on it).
    fmt = (f"bv*[height<={args.height}][vcodec^=avc1]+ba/b[height<={args.height}][vcodec^=avc1]"
           f"/bv*[height<={args.height}]+ba/b[height<={args.height}]")
    cmd = [
        "yt-dlp",
        "-f", fmt,
        "--download-sections", f"*{args.start}-{args.end}",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        url,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
