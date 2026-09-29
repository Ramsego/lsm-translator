import sqlite3
import csv
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lsm.db"
CSV_PATH = DATA_DIR / "metadata.csv"
ARRAYS_DIR = DATA_DIR / "arrays"

EXCLUDE = {
    "JFYEdvq3kNU", "hR4BZNxbi1Y", "EX9SPT1ytx8", "Dm-neOnO0-E", "K56zs2zaIQE",
    "gH1vhh-V6oE", "FgThSMNNPpg", "Ra-0o-PKcJk", "19qD4s4ZsZU", "YiWJTcitkRc",
    "BP3R8czviC8", "-MIGKC3M07A", "Jk9qi2QsAZE", "UC6N5RSv511MWWkq4BdzRl0A",
}

def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition

def main():
    conn = sqlite3.connect(DB_PATH)
    all_passed = True

    # 1. Expected video count per source (chnt word bank + wikisigns)
    count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    by_source = dict(conn.execute("SELECT source, COUNT(*) FROM videos GROUP BY source"))
    all_passed &= check(f"chnt count == 490 (got {by_source.get('chnt')})", by_source.get("chnt") == 490)
    all_passed &= check(f"total videos > 490 (got {count})", count > 490)
    print(f"        by source: {by_source}")

    # 2. No excluded IDs leaked in
    ids = {row[0] for row in conn.execute("SELECT youtube_id FROM videos")}
    leaked = ids & EXCLUDE
    all_passed &= check(f"no excluded IDs present (leaked: {leaked or 'none'})", not leaked)

    # 3. Accents preserved in DB — find non-ASCII labels and prove round-trip
    accented = conn.execute(
        "SELECT label FROM videos WHERE label != CAST(label AS TEXT) OR label GLOB '*[^ -~]*'"
    ).fetchall()
    accented = [r[0] for r in accented]
    print(f"        {len(accented)} accented labels found, e.g. {accented[:5]}")
    # The 'á' character must appear somewhere and not be mangled to '?'
    has_clean_accent = any("?" not in lbl and any(ord(c) > 127 for c in lbl) for lbl in accented)
    all_passed &= check("accented labels intact (no '?' substitution)", has_clean_accent)
  

    # 4. CSV accents survive too
    with open(CSV_PATH, encoding="utf-8") as f:
        csv_labels = [row["label"] for row in csv.DictReader(f)]
    csv_accent_ok = any(any(ord(c) > 127 for c in lbl) for lbl in csv_labels)
    all_passed &= check("CSV has intact accented labels", csv_accent_ok)

    # 5. Every DB row has its .npy on disk, with the right shape
    rows = conn.execute("SELECT youtube_id, num_frames, array_path FROM videos").fetchall()
    missing = []
    bad_shape = []
    for yid, num_frames, path in rows:
        p = DATA_DIR / path
        if not p.exists():
            missing.append(yid)
            continue
        arr = np.load(p)
        if arr.shape != (num_frames, 116, 3):
            bad_shape.append((yid, arr.shape))
    all_passed &= check(f"all .npy files exist (missing: {len(missing)})", not missing)
    all_passed &= check(f"all arrays shaped [frames,116,3] (bad: {len(bad_shape)})", not bad_shape)

    # 6. Sanity on one array: pose should be mostly present, NaN fraction not total
    sample = conn.execute("SELECT array_path FROM videos LIMIT 1").fetchone()[0]
    arr = np.load(DATA_DIR / sample)
    nan_frac = np.isnan(arr).mean()
    print(f"        sample NaN fraction: {nan_frac:.2f}")
    all_passed &= check("sample array not fully NaN", nan_frac < 0.99)

    print("\nALL CHECKS PASSED" if all_passed else "\nSOME CHECKS FAILED")
    conn.close()

if __name__ == "__main__":
    main()