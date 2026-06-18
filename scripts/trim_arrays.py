"""
Strip leading/trailing frames where no hand is detected from every .npy array.

The signer often stands still for several seconds before and after signing,
leaving long runs of all-NaN hand rows at the array boundaries. Trimming to
the active window + a small buffer makes DTW matching much more reliable.

Usage:
    python scripts/trim_arrays.py [--buffer N] [--dry-run]
"""

import argparse
import sqlite3
import shutil
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lsm.db"
ARRAYS_DIR = DATA_DIR / "arrays"
BACKUP_DIR = DATA_DIR / "arrays_backup"

HAND_ROWS = slice(0, 42)  # rows 0–41: left + right hand


def trim_window(arr: np.ndarray, buffer: int = 5) -> np.ndarray | None:
    """Return trimmed array, or None if no hand frames detected.

    Finds the first and last frame where any hand landmark is non-NaN,
    then pads by `buffer` frames on each side (clamped to array bounds).
    """
    hand_chunk = arr[:, HAND_ROWS, :]
    has_hand = ~np.isnan(hand_chunk).all(axis=(1, 2))
    hand_indices = np.where(has_hand)[0]

    if len(hand_indices) == 0:
        return None

    start = max(0, int(hand_indices[0]) - buffer)
    end = min(arr.shape[0], int(hand_indices[-1]) + buffer + 1)
    return arr[start:end]


def active_segment(arr: np.ndarray, buffer: int = 5,
                   gap_threshold: int = 20) -> np.ndarray | None:
    """Return the single best contiguous signing segment, or None if no hands.

    Splits the hand-active frames wherever there is a run of >= gap_threshold
    consecutive no-hand frames (an outro card, a long rest, or an editing
    mistake), then keeps only the segment containing the most hand frames.
    A `buffer` of frames is kept on each side, clamped to array bounds.

    This both strips leading/trailing padding AND drops interior dead gaps
    (e.g. an embedded logo card between two takes).
    """
    hand_chunk = arr[:, HAND_ROWS, :]
    has_hand = ~np.isnan(hand_chunk).all(axis=(1, 2))
    hand_indices = np.where(has_hand)[0]

    if len(hand_indices) == 0:
        return None

    # Split hand_indices into segments wherever consecutive detections are
    # more than gap_threshold frames apart.
    splits = np.where(np.diff(hand_indices) > gap_threshold)[0]
    segments = np.split(hand_indices, splits + 1)

    # Keep the segment with the most hand frames (the real demo).
    best = max(segments, key=len)

    start = max(0, int(best[0]) - buffer)
    end = min(arr.shape[0], int(best[-1]) + buffer + 1)
    return arr[start:end]


def main():
    parser = argparse.ArgumentParser(description="Trim silent padding from landmark arrays.")
    parser.add_argument("--buffer", type=int, default=5,
                        help="Frames to keep before/after first/last hand detection (default: 5).")
    parser.add_argument("--gap-threshold", type=int, default=20,
                        help="Split on runs of >= this many no-hand frames; keep the "
                             "largest segment. Drops outro cards / editing tails (default: 20).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be trimmed without writing anything.")
    args = parser.parse_args()

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT youtube_id, num_frames, array_path FROM videos").fetchall()

    trimmed = 0
    skipped_no_hands = 0
    skipped_already_tight = 0
    total_frames_removed = 0

    updates = []

    for youtube_id, db_frames, array_path in rows:
        path = DATA_DIR / array_path
        if not path.exists():
            continue

        # Work from the pristine original when a backup exists, so re-runs are
        # idempotent and the window is computed on the untrimmed array.
        backup_path = BACKUP_DIR / path.name
        source_path = backup_path if backup_path.exists() else path
        arr = np.load(source_path)

        trimmed_arr = active_segment(arr, buffer=args.buffer,
                                     gap_threshold=args.gap_threshold)

        if trimmed_arr is None:
            skipped_no_hands += 1
            continue

        frames_removed = arr.shape[0] - trimmed_arr.shape[0]

        if frames_removed == 0:
            skipped_already_tight += 1
            continue

        total_frames_removed += frames_removed

        if not args.dry_run:
            # Back up original before overwriting (first run only)
            if not backup_path.exists():
                shutil.copy2(path, backup_path)

            np.save(path, trimmed_arr)

            # Recount hand frames in trimmed array
            hand_chunk = trimmed_arr[:, HAND_ROWS, :]
            new_hand_frames = int((~np.isnan(hand_chunk).all(axis=(1, 2))).sum())
            updates.append((trimmed_arr.shape[0], new_hand_frames, youtube_id))

        trimmed += 1

    if not args.dry_run and updates:
        conn.executemany(
            "UPDATE videos SET num_frames=?, num_hand_frames=? WHERE youtube_id=?",
            updates,
        )
        conn.commit()

    conn.close()

    avg_removed = total_frames_removed / trimmed if trimmed else 0
    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}=== Trim summary ===")
    print(f"  Trimmed:            {trimmed}")
    print(f"  Avg frames removed: {avg_removed:.1f}")
    print(f"  Total frames saved: {total_frames_removed}")
    print(f"  Already tight:      {skipped_already_tight}")
    print(f"  Skipped (no hands): {skipped_no_hands}  ← needs manual review")
    if not args.dry_run:
        print(f"  Originals backed up to {BACKUP_DIR}")


if __name__ == "__main__":
    main()
