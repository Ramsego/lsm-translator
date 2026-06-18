import csv
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lsm.db"
ARRAYS_DIR = DATA_DIR / "arrays"
OUT_CSV = DATA_DIR / "quality_report.csv"

HAND_ROWS = slice(0, 42)   # left + right hand (rows 0–41)
POSE_ROWS = slice(42, 75)  # pose (rows 42–74)
FACE_ROWS = slice(75, 116) # curated face (rows 75–115)

FPS_EST = 20.0

# Flag thresholds
SHORT_FRAMES   = 40    # < 2 s
LOW_HAND_COV   = 0.30
LOW_FACE_COV   = 0.10  # informational — do NOT auto-exclude
LOW_POSE_COV   = 0.50


def coverage(arr, row_slice):
    """Fraction of frames where at least one landmark in the slice is non-NaN."""
    chunk = arr[:, row_slice, :]          # [frames, landmarks, 3]
    has_data = ~np.isnan(chunk).all(axis=(1, 2))  # [frames]
    return has_data.mean()


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT youtube_id, label, source, array_path FROM videos ORDER BY label"
    ).fetchall()
    conn.close()

    results = []
    flag_counts = defaultdict(int)

    for youtube_id, label, source, array_path in rows:
        path = DATA_DIR / array_path
        if not path.exists():
            print(f"  MISSING array: {youtube_id}")
            continue

        arr = np.load(path)            # [frames, 116, 3]
        num_frames = arr.shape[0]
        duration_s = round(num_frames / FPS_EST, 1)

        hand_cov  = round(float(coverage(arr, HAND_ROWS)), 3)
        face_cov  = round(float(coverage(arr, FACE_ROWS)), 3)
        pose_cov  = round(float(coverage(arr, POSE_ROWS)), 3)
        nan_frac  = round(float(np.isnan(arr).mean()), 3)

        flags = []
        if num_frames < SHORT_FRAMES:
            flags.append("short")
        if hand_cov < LOW_HAND_COV:
            flags.append("low_hand")
        if face_cov < LOW_FACE_COV:
            flags.append("low_face")
        if pose_cov < LOW_POSE_COV:
            flags.append("low_pose")

        for f in flags:
            flag_counts[f] += 1

        results.append({
            "youtube_id":   youtube_id,
            "label":        label,
            "source":       source,
            "num_frames":   num_frames,
            "duration_s":   duration_s,
            "hand_coverage": hand_cov,
            "face_coverage": face_cov,
            "pose_coverage": pose_cov,
            "nan_fraction":  nan_frac,
            "flags":         ",".join(flags),
        })

    COLS = ["youtube_id", "label", "source", "num_frames", "duration_s",
            "hand_coverage", "face_coverage", "pose_coverage", "nan_fraction", "flags"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    any_flag = sum(1 for r in results if r["flags"])
    clean = total - any_flag

    print(f"\n=== Quality Report ===")
    print(f"  Total videos:  {total}")
    print(f"  Clean:         {clean}")
    print(f"  Flagged:       {any_flag}")
    print()
    for flag, count in sorted(flag_counts.items()):
        note = "  [informational, do not auto-exclude]" if flag == "low_face" else ""
        print(f"    {flag:15s} {count:4d}{note}")

    # Breakdown by source
    print()
    by_source = defaultdict(lambda: {"total": 0, "flagged": 0})
    for r in results:
        s = r["source"]
        by_source[s]["total"] += 1
        if r["flags"]:
            by_source[s]["flagged"] += 1
    for src, d in sorted(by_source.items()):
        print(f"  {src}: {d['total']} total, {d['flagged']} flagged")

    print(f"\nReport written to {OUT_CSV}")


if __name__ == "__main__":
    main()
