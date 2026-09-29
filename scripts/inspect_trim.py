"""
Visual before/after inspection for trimmed arrays.

Reads the 5 worst low_hand videos from quality_report.csv (or --ids),
loads the original from data/arrays_backup/ and the trimmed from data/arrays/,
and renders a side-by-side PNG grid to data/viz/trim_check/.

Usage:
    python scripts/inspect_trim.py                      # 5 worst low_hand
    python scripts/inspect_trim.py --ids ID1 ID2 ID3   # specific IDs
    python scripts/inspect_trim.py --n 10              # worst N
"""

import argparse
import csv
import sys
import numpy as np
import cv2
from pathlib import Path

DATA_DIR = Path("data")
ARRAYS_DIR = DATA_DIR / "arrays"
BACKUP_DIR = DATA_DIR / "arrays_backup"
QUALITY_CSV = DATA_DIR / "quality_report.csv"
OUT_DIR = DATA_DIR / "viz" / "trim_check"

HAND_ROWS = slice(0, 42)
POSE_ROWS = slice(42, 75)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
]


def draw_frame_from_array(arr: np.ndarray, frame_idx: int,
                           width=320, height=240) -> np.ndarray:
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 240

    def pt(x, y):
        return (int(x * width), int(y * height))

    # Left hand (rows 0–20) — blue
    for a, b in HAND_CONNECTIONS:
        xa, ya, _ = arr[frame_idx, a]
        xb, yb, _ = arr[frame_idx, b]
        if not (np.isnan(xa) or np.isnan(xb)):
            cv2.line(canvas, pt(xa, ya), pt(xb, yb), (255, 100, 100), 1)
    for i in range(21):
        x, y, _ = arr[frame_idx, i]
        if not np.isnan(x):
            cv2.circle(canvas, pt(x, y), 3, (255, 100, 100), -1)

    # Right hand (rows 21–41) — red
    for a, b in HAND_CONNECTIONS:
        xa, ya, _ = arr[frame_idx, 21 + a]
        xb, yb, _ = arr[frame_idx, 21 + b]
        if not (np.isnan(xa) or np.isnan(xb)):
            cv2.line(canvas, pt(xa, ya), pt(xb, yb), (100, 100, 255), 1)
    for i in range(21):
        x, y, _ = arr[frame_idx, 21 + i]
        if not np.isnan(x):
            cv2.circle(canvas, pt(x, y), 3, (100, 100, 255), -1)

    # Pose (rows 42–74) — green
    for a, b in POSE_CONNECTIONS:
        xa, ya, _ = arr[frame_idx, 42 + a]
        xb, yb, _ = arr[frame_idx, 42 + b]
        if not (np.isnan(xa) or np.isnan(xb)):
            cv2.line(canvas, pt(xa, ya), pt(xb, yb), (80, 180, 80), 1)

    cv2.putText(canvas, f"f{frame_idx}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    return canvas


def make_strip(arr: np.ndarray, n_frames: int = 6) -> np.ndarray:
    """Sample n_frames evenly from arr and concatenate horizontally."""
    indices = np.linspace(0, arr.shape[0] - 1, n_frames, dtype=int)
    frames = [draw_frame_from_array(arr, int(i)) for i in indices]
    return np.hstack(frames)


def inspect(youtube_id: str, label: str):
    orig_path = BACKUP_DIR / f"{youtube_id}.npy"
    trim_path = ARRAYS_DIR / f"{youtube_id}.npy"

    if not orig_path.exists():
        print(f"  {youtube_id}: no backup found — was trim_arrays.py run? skipping.")
        return
    if not trim_path.exists():
        print(f"  {youtube_id}: trimmed array missing, skipping.")
        return

    orig = np.load(orig_path)
    trimmed = np.load(trim_path)

    strip_orig = make_strip(orig)
    strip_trim = make_strip(trimmed)

    # Label bars
    def label_bar(text, width):
        bar = np.ones((24, width, 3), dtype=np.uint8) * 200
        cv2.putText(bar, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return bar

    w = strip_orig.shape[1]
    top = np.vstack([
        label_bar(f"BEFORE  {orig.shape[0]} frames", w),
        strip_orig,
    ])
    bot = np.vstack([
        label_bar(f"AFTER   {trimmed.shape[0]} frames  (removed {orig.shape[0]-trimmed.shape[0]})", w),
        strip_trim,
    ])

    header = np.ones((30, w, 3), dtype=np.uint8) * 255
    cv2.putText(header, f"{youtube_id}  |  {label}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    final = np.vstack([header, top, bot])
    out_path = OUT_DIR / f"{youtube_id}.png"
    cv2.imwrite(str(out_path), final)
    print(f"  Saved {out_path.name}  ({orig.shape[0]} → {trimmed.shape[0]} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", help="Specific youtube_ids to inspect.")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of worst low_hand videos to inspect (default: 5).")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.ids:
        # Build label lookup
        with open(QUALITY_CSV, encoding="utf-8") as f:
            lookup = {r["youtube_id"]: r["label"] for r in csv.DictReader(f)}
        targets = [(yid, lookup.get(yid, "")) for yid in args.ids]
    else:
        with open(QUALITY_CSV, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if "low_hand" in r["flags"]]
        rows.sort(key=lambda r: float(r["hand_coverage"]))
        targets = [(r["youtube_id"], r["label"]) for r in rows[:args.n]]

    if not targets:
        print("No low_hand videos found in quality_report.csv.")
        sys.exit(0)

    print(f"Inspecting {len(targets)} videos → {OUT_DIR}")
    for yid, label in targets:
        inspect(yid, label)

    print(f"\nDone. Open {OUT_DIR}/ to review.")


if __name__ == "__main__":
    main()
