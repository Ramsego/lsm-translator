import json
import sqlite3
import csv
from pathlib import Path
from datetime import date
import numpy as np

VIDEOS_DIR = Path("/Volumes/Crucial X8/LSM_Translator/videos/isolated")
DATA_DIR = Path("data")
ARRAYS_DIR = DATA_DIR / "arrays"
DB_PATH = DATA_DIR / "lsm.db"
CSV_PATH = DATA_DIR / "metadata.csv"

ARRAYS_DIR.mkdir(parents=True, exist_ok=True)

# Videos to leave out: 11 compilations + 2 bad tracking + 1 channel-metadata folder
EXCLUDE = {
    "JFYEdvq3kNU", "hR4BZNxbi1Y", "EX9SPT1ytx8", "Dm-neOnO0-E", "K56zs2zaIQE",
    "gH1vhh-V6oE", "FgThSMNNPpg", "Ra-0o-PKcJk", "19qD4s4ZsZU", "YiWJTcitkRc",
    "BP3R8czviC8",                 # compilations
    "-MIGKC3M07A", "Jk9qi2QsAZE",  # bad tracking (Papaya, Preocupado)
    "UC6N5RSv511MWWkq4BdzRl0A",    # channel metadata, not a video
}

# Where each source's landmarks live in the 75-row array
SOURCE_OFFSET = {"left_hand": 0, "right_hand": 21, "pose": 42}
NUM_LANDMARKS = 75  # 21 + 21 + 33

def build_array(landmarks: list):
    """Turn the list of detected points into a dense [frames, 75, 3] array (NaN = missing)."""
    if not landmarks:
        return None, 0, 0

    num_frames = max(lm["frame"] for lm in landmarks) + 1
    arr = np.full((num_frames, NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    hand_frames = set()

    for lm in landmarks:
        base = SOURCE_OFFSET[lm["source"]]
        row = base + lm["landmark_index"]
        arr[lm["frame"], row] = [lm["x"], lm["y"], lm["z"]]
        if lm["source"] in ("left_hand", "right_hand"):
            hand_frames.add(lm["frame"])

    return arr, num_frames, len(hand_frames)

def read_info(folder: Path) -> dict:
    info_files = list(folder.glob("*.info.json"))
    if not info_files:
        return {}
    with open(info_files[0], encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def setup_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY,
            youtube_id TEXT UNIQUE,
            title TEXT,
            label TEXT,
            channel TEXT,
            upload_date TEXT,
            duration INTEGER,
            dominant_hand TEXT,
            num_frames INTEGER,
            num_hand_frames INTEGER,
            array_path TEXT,
            video_type TEXT
        )
    """)
    conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    setup_db(conn)

    existing = {row[0] for row in conn.execute("SELECT youtube_id FROM videos")}
    today = date.today().isoformat()
    ingested = 0

    for folder in sorted(VIDEOS_DIR.iterdir()):
        if not folder.is_dir() or folder.name in EXCLUDE:
            continue
        if folder.name in existing:
            continue

        landmark_file = folder / "landmarks.json"
        if not landmark_file.exists():
            continue

        with open(landmark_file) as f:
            data = json.load(f)

        arr, num_frames, num_hand_frames = build_array(data.get("landmarks", []))
        if arr is None:
            print(f"  skip (no landmarks): {folder.name}")
            continue

        np.save(ARRAYS_DIR / f"{folder.name}.npy", arr)

        info = read_info(folder)
        title = info.get("title", "")
        conn.execute(
            "INSERT OR IGNORE INTO videos "
            "(youtube_id, title, label, channel, upload_date, duration, "
            " dominant_hand, num_frames, num_hand_frames, array_path, video_type) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                folder.name,
                title,
                title.strip().lower(),
                info.get("channel", ""),
                info.get("upload_date", ""),
                info.get("duration", 0),
                data.get("dominant_hand", "unknown"),
                num_frames,
                num_hand_frames,
                f"arrays/{folder.name}.npy",
                "isolated",
            ),
        )
        ingested += 1
        if ingested % 50 == 0:
            print(f"  {ingested} ingested...")

    conn.commit()

    # Export metadata to CSV for one-line pandas use
    cols = [d[1] for d in conn.execute("PRAGMA table_info(videos)")]
    rows = conn.execute(f"SELECT {','.join(cols)} FROM videos ORDER BY label")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    total = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"\nDone. {ingested} newly ingested, {total} total videos in DB.")
    print(f"Arrays in {ARRAYS_DIR}, metadata in {CSV_PATH}")
    conn.close()

if __name__ == "__main__":
    main()