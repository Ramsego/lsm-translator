"""
Forced-choice SignCLIP test -- locate + copy the 144 candidate (true + distractor)
source videos named in candidate_manifest.csv onto local disk, ready for
videos_to_poses. Videos live on the mounted drive at
videos/{isolated,wikisigns}/<youtube_id>/<youtube_id>.mp4 regardless of Phase-1 source
tag (chnt/wikisigns) -- both pools use the same per-id subdirectory layout.

Usage: python3 phase3/probe/fc_test/copy_candidate_videos.py
"""
import csv
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DRIVE = Path("/Volumes/Crucial X8/LSM_Translator/videos")
MANIFEST = Path(__file__).parent / "candidate_manifest.csv"
OUT_DIR = Path(__file__).parent / "candidate_clips"
OUT_DIR.mkdir(exist_ok=True)


def find_video(yid):
    for pool in ("isolated", "wikisigns"):
        p = DRIVE / pool / yid / f"{yid}.mp4"
        if p.exists():
            return p
    return None


def main():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["youtube_id"], []).append(r)

    found, missing = [], []
    for yid, refs in by_id.items():
        src = find_video(yid)
        if src is None:
            missing.append((yid, [r["label"] for r in refs]))
            continue
        dst = OUT_DIR / f"{yid}.mp4"
        if not dst.exists():
            shutil.copy(src, dst)
        found.append(yid)

    print(f"copied {len(found)}/{len(by_id)} distinct videos -> {OUT_DIR}")
    if missing:
        print(f"\n MISSING {len(missing)}:")
        for yid, labels in missing:
            print(f"  {yid}  (needed for: {labels})")


if __name__ == "__main__":
    main()
