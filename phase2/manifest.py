"""
Local video manifest: one row per extracted conference so you never re-extract a video.

Auto-updated at the end of a successful extraction (called from extract_continuous.py),
or rebuilt from disk on demand. Kept LOCAL (git-ignored) until you choose to track it.

Columns: id, date, region, fps, signing_min, n_segs, transcript_source, title, url

Usage:
    python phase2/manifest.py --rebuild         # regenerate from everything on disk
    # (per-video upsert happens automatically after each extraction)
"""

import csv
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "videos_manifest.csv"
TRANS = REPO / "phase2" / "transcripts"
FIELDS = ["id", "date", "region", "fps", "signing_min", "n_segs",
          "transcript_source", "title", "url"]


def _region_from_title(t: str) -> str:
    t = (t or "").lower()
    if "sonora" in t:
        return "sonora"
    if any(k in t for k in ("ciudad de méxico", "cdmx", "ciudad de mexico", "brugada")):
        return "cdmx"
    if any(k in t for k in ("covid", "gatell", "pandemia")):
        return "covid"
    if any(k in t for k in ("sheinbaum", "presidenta")):
        return "federal-sheinbaum"
    if any(k in t for k in ("lópez obrador", "lopez obrador", "amlo", "matutina")):
        return "federal-amlo"
    return ""


def compute_row(video_path: Path, out_dir: Path) -> dict:
    """Build a manifest row from a video's info.json + its extraction segments.json."""
    video_path, out_dir = Path(video_path), Path(out_dir)
    vid = video_path.stem

    m = json.load(open(out_dir / "segments.json"))
    fps, ev = m.get("fps", 30), m.get("every", 1)
    segs = m.get("segments", [])
    mins = sum(s["n_frames"] for s in segs) / (fps / ev) / 60 if segs else 0

    title = date = url = ""
    info = video_path.parent / f"{vid}.info.json"
    if info.exists():
        j = json.load(open(info))
        title = j.get("title", "")
        up = j.get("upload_date", "")           # YYYYMMDD
        date = f"{up[:4]}-{up[4:6]}-{up[6:]}" if len(up) == 8 else ""
        url = j.get("webpage_url", f"https://www.youtube.com/watch?v={vid}")
    else:
        url = f"https://www.youtube.com/watch?v={vid}"

    src = "none"
    if date and list(TRANS.glob(f"{date}*.txt")):
        src = "gob.mx"
    elif list(TRANS.glob(f"*{vid}*.txt")):
        src = "gob.mx"
    elif list(video_path.parent.glob("*.es*.vtt")):
        src = "youtube-captions"
    elif (video_path.parent / f"{vid}.asr.json").exists():
        src = "whisper"

    return {"id": vid, "date": date, "region": _region_from_title(title),
            "fps": str(int(fps / ev)), "signing_min": f"{mins:.0f}",
            "n_segs": str(len(segs)), "transcript_source": src,
            "title": title, "url": url}


def upsert(row: dict, manifest_csv: Path = MANIFEST):
    """Insert or replace a video's row (keyed by id), preserving any others. Sorted by id.
    A manually-edited region cell is kept if the recomputed one is blank."""
    rows = {}
    if manifest_csv.exists():
        for r in csv.DictReader(open(manifest_csv)):
            rows[r["id"]] = r
    prev = rows.get(row["id"])
    if prev and not row.get("region"):
        row["region"] = prev.get("region", "")   # don't clobber a manual fill
    rows[row["id"]] = row
    with open(manifest_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in FIELDS})


def update_for_extraction(video_path, out_dir, manifest_csv: Path = MANIFEST):
    """Best-effort: called at the end of a successful extraction. Never raises."""
    try:
        upsert(compute_row(video_path, out_dir), manifest_csv)
        print(f"Manifest updated → {manifest_csv}")
    except Exception as e:
        print(f"(manifest update skipped: {e})")


def rebuild(manifest_csv: Path = MANIFEST):
    """Regenerate the whole manifest by scanning arrays/ + videos/ under LSM_DATA_ROOT."""
    base = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator"))
    arr, vid = base / "arrays" / "mananera", base / "videos" / "mananera"
    n = 0
    for d in sorted(arr.glob("*/")):
        if not (d / "segments.json").exists():
            continue
        vp = vid / d.name / f"{d.name}.mp4"
        try:
            upsert(compute_row(vp, d), manifest_csv)
            n += 1
        except Exception as e:
            print(f"  skip {d.name}: {e}")
    print(f"Rebuilt {manifest_csv} — {n} videos")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Regenerate from everything on disk.")
    args = ap.parse_args()
    if args.rebuild:
        rebuild()
    else:
        ap.error("nothing to do; pass --rebuild (per-video upsert is automatic after extraction)")
