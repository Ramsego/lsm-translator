import csv
import json
import re
import subprocess
import os
from pathlib import Path
from collections import defaultdict

MAP = Path("data/wikisigns_map.csv")
VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "wikisigns"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

def is_junk(word: str) -> bool:
    w = word.strip()
    if "Internacional" in w:          # International Sign, not LSM
        return True
    if len(w) < 2:
        return True
    if not re.search(r"[A-Za-zà-ÿ]", w):  # pure symbols / numbers
        return True
    return False

def load_map() -> dict:
    """Return {youtube_id: {'words': [...], 'entry_url': url}} after filtering."""
    by_id = defaultdict(lambda: {"words": [], "entry_url": ""})
    with open(MAP, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if is_junk(row["word"]):
                continue
            yid = row["youtube_id"]
            by_id[yid]["words"].append(row["word"])
            by_id[yid]["entry_url"] = row["entry_url"]
    return by_id

def download_one(yid: str, info: dict) -> bool:
    folder = VIDEOS_DIR / yid
    meta_path = folder / "info.json"
    if meta_path.exists() and list(folder.glob("*.mp4")):
        return True  # already done

    folder.mkdir(parents=True, exist_ok=True)
    words = info["words"]
    primary = words[0]

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
    if result.returncode != 0 or not list(folder.glob("*.mp4")):
        return False

    meta = {
        "youtube_id": yid,
        "label": primary.strip().lower(),
        "title": primary,
        "aliases": [w.strip().lower() for w in words],
        "source": "wikisigns",
        "source_url": info["entry_url"],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True

def main():
    by_id = load_map()
    print(f"Unique LSM videos to fetch: {len(by_id)}")

    ok = fail = skip = 0
    for i, (yid, info) in enumerate(sorted(by_id.items()), 1):
        folder = VIDEOS_DIR / yid
        if (folder / "info.json").exists() and list(folder.glob("*.mp4")):
            skip += 1
            continue
        success = download_one(yid, info)
        if success:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {yid} ({info['words'][0]})")
        if i % 25 == 0:
            print(f"  [{i}/{len(by_id)}] ok={ok} fail={fail} skip={skip}")

    print(f"\nDone. downloaded={ok} failed={fail} already_had={skip}")
    print(f"Re-run to retry failures (resumable). Videos in {VIDEOS_DIR}")

if __name__ == "__main__":
    main()
