"""
Forced-choice SignCLIP test — query side.

Cuts a tight clip for each of the 26 verified instances in review_2.csv (video
57TvyH9902U, verdict=='y', sign_start present), using the human-marked absolute
sign_start/sign_end (+/- BUFFER seconds, same padding cut_verified_interp_clips.py
uses -- 0.3s alone proved too tight for some marked spans there).

Usage (run from repo root, python3, no special env needed -- just ffmpeg on PATH):
    python3 phase3/probe/fc_test/cut_query_clips.py
"""
import csv
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC_VIDEO = Path("/Volumes/Crucial X8/LSM_Translator/videos/mananera/57TvyH9902U/57TvyH9902U.mp4")
REVIEW_CSV = REPO / "phase3/probe/review_2.csv"
OUT_DIR = REPO / "phase3/probe/fc_test/query_clips"
MANIFEST = REPO / "phase3/probe/fc_test/query_manifest.csv"
BUFFER = 0.6


def main():
    rows = [
        r for r in csv.DictReader(open(REVIEW_CSV))
        if r.get("verdict") == "y" and r.get("sign_start") and "57TvyH9902U" in r.get("file", "")
    ]
    assert len(rows) == 26, f"expected 26 usable instances, got {len(rows)}"

    out_rows = []
    for i, r in enumerate(rows):
        word = r["word"]
        st = float(r["sign_start"]) - BUFFER
        en = float(r["sign_end"]) + BUFFER
        clip_id = f"{i:02d}_{word}"
        out_path = OUT_DIR / f"{clip_id}.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(max(0, st)), "-to", str(en), "-i", str(SRC_VIDEO),
            "-c:v", "libx264", "-an", str(out_path),
        ], check=True)
        out_rows.append(dict(clip_id=clip_id, word=word, sign_start=r["sign_start"],
                              sign_end=r["sign_end"], cut_start=max(0, st), cut_end=en))
        print(f"  [{i+1:2d}/26] {clip_id:20s} [{st:.1f},{en:.1f}] -> {out_path.name}")

    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "word", "sign_start", "sign_end",
                                           "cut_start", "cut_end"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n26 clips cut -> {OUT_DIR}\nmanifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
