"""
Dump sample frames from a mañanera section so we can locate the interpreter recuadro,
then (given a CROP box) emit a cropped sample to confirm framing.

Usage:
    python phase2/locate_recuadro.py <video.mp4>                  # dump full frames
    python phase2/locate_recuadro.py <video.mp4> --crop X Y W H   # also dump cropped
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

OUT_DIR = Path("phase2/recuadro_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--n", type=int, default=8, help="Frames to sample.")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="Crop box to preview.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    idxs = np.linspace(0, max(0, total - 1), args.n, dtype=int)
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fi}: read failed")
            continue
        cv2.imwrite(str(OUT_DIR / f"frame_{i:02d}.png"), frame)
        if args.crop:
            x, y, cw, ch = args.crop
            crop = frame[y:y+ch, x:x+cw]
            cv2.imwrite(str(OUT_DIR / f"crop_{i:02d}.png"), crop)
    cap.release()
    msg = f"Saved {len(idxs)} frames to {OUT_DIR}"
    if args.crop:
        msg += f" (+ crops {tuple(args.crop)})"
    print(msg)


if __name__ == "__main__":
    main()
