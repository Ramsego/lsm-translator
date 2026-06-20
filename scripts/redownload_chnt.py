"""
Re-download the ChNt isolated videos whose source .mp4 was deleted by the old
02_extract.py. Reads youtube_ids from lsm.db and fetches each back into its
existing folder (folder name == youtube_id), so the *.info.json + landmarks.json
already there are preserved. Resumable: skips folders that already have an .mp4.
"""
import sqlite3
import subprocess
import os
from pathlib import Path

VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated"
DB_PATH = Path("data/lsm.db")

def main():
    conn = sqlite3.connect(DB_PATH)
    ids = [row[0] for row in conn.execute("SELECT youtube_id FROM videos")]
    conn.close()
    print(f"ChNt videos in DB: {len(ids)}")

    ok = fail = skip = 0
    failures = []
    for i, yid in enumerate(sorted(ids), 1):
        folder = VIDEOS_DIR / yid
        folder.mkdir(parents=True, exist_ok=True)
        if list(folder.glob("*.mp4")):
            skip += 1
            continue

        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={yid}",
            "--output", str(folder / "%(id)s.%(ext)s"),
            "--format", "mp4",
            "--no-playlist",
            "--ignore-errors",
            "--sleep-requests", "2",
        ]
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0 and list(folder.glob("*.mp4")):
            ok += 1
        else:
            fail += 1
            failures.append(yid)
            print(f"  FAIL {yid}")

        if i % 25 == 0:
            print(f"  [{i}/{len(ids)}] ok={ok} fail={fail} skip={skip}")

    print(f"\nDone. downloaded={ok} failed={fail} already_had={skip}")
    if failures:
        print("Failed IDs (likely private/removed):")
        for yid in failures:
            print(f"  {yid}")

if __name__ == "__main__":
    main()
