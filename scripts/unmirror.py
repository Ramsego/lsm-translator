"""
Un-mirror landmarks.json files back to raw, as-detected orientation.

The old pipeline mirrored (x -> 1-x) any clip its count-based heuristic judged
left-dominant — including ~112 ChNt clips from a single right-handed signer, which
is wrong. We now store everything raw and handle handedness at DTW query time
(see handedness.mirror_array). This pass undoes the old mirror and rewrites the
dominant_hand field as informational, motion-based metadata.

No MediaPipe re-run: flipping x again exactly recovers the raw coordinates.

Usage:
    python scripts/unmirror.py [--dry-run]
"""

import argparse
import json
import shutil
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from handedness import flip_x_landmarks, motion_dominant

VIDEO_DIRS = [
    Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated",
    Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "wikisigns",
]


def iter_landmark_files():
    for base in VIDEO_DIRS:
        if not base.exists():
            print(f"  (skip, not mounted: {base})")
            continue
        for folder in sorted(base.iterdir()):
            if not folder.is_dir() or folder.name.startswith("._"):
                continue
            lf = folder / "landmarks.json"
            if lf.exists():
                yield lf


def main():
    parser = argparse.ArgumentParser(description="Un-mirror landmarks to raw orientation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without modifying any files.")
    args = parser.parse_args()

    unmirrored = 0
    already_raw = 0
    label_changes = Counter()
    total = 0

    for lf in iter_landmark_files():
        total += 1
        with open(lf) as f:
            data = json.load(f)

        landmarks = data.get("landmarks", [])
        old_dom = data.get("dominant_hand", "right")

        # The old mirror flipped x only when it judged the clip 'left'.
        if old_dom == "left":
            if not args.dry_run:
                flip_x_landmarks(landmarks)  # undo -> raw
            unmirrored += 1
        else:
            already_raw += 1

        # Recompute dominant_hand as informational metadata on the (now raw) data.
        # In dry-run we can't see the un-flipped coords, so only report the action.
        new_dom = motion_dominant(landmarks) if not args.dry_run else old_dom
        label_changes[(old_dom, new_dom)] += 1

        if not args.dry_run:
            bak = lf.with_suffix(".json.bak")
            if not bak.exists():
                shutil.copy2(lf, bak)
            data["landmarks"] = landmarks
            data["dominant_hand"] = new_dom            # now informational only
            data["orientation"] = "raw"                # marker: coords untransformed
            with open(lf, "w") as f:
                json.dump(data, f)

    mode = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{mode}=== Un-mirror summary ===")
    print(f"  Files seen:       {total}")
    print(f"  Un-mirrored:      {unmirrored}  (were 'left', flipped back to raw)")
    print(f"  Already raw:      {already_raw}")
    if not args.dry_run:
        print(f"  Backups written as <id>/landmarks.json.bak")
        print(f"  New motion-based dominant_hand label (informational):")
        agg = Counter()
        for (_old, new), n in label_changes.items():
            agg[new] += n
        for hand, n in sorted(agg.items()):
            print(f"    {hand}: {n}")


if __name__ == "__main__":
    main()
