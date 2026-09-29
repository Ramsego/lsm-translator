"""
QC gate 1/5: INTEGRITY — does every file that should exist, exist and load?

Builds a corpus-wide artifact manifest (path, sha256, bytes, mtime, kind, shape) and
cross-checks it two ways:

  1. segments.json <-> disk, in both directions:
       - a .npz sitting on disk that no segment references (orphan array)
       - a segment whose `file` field points at nothing on disk (dangling reference)
     This is only checkable for videos whose arrays are mirrored locally (an
     `arrays_<vid>/` dir under --root). For videos known only via
     `segments_all/<vid>.segments.json` (arrays live on the external drive, which is
     usually not mounted), file-existence is reported SKIPPED, not failed.

  2. videos_manifest.csv (repo root) <-> segments_all/*.segments.json: does the
     manifest's claimed n_segs match what's actually on disk? Known bad row:
     57TvyH9902U claims n_segs=0, segments_all shows 40.

Also flags videos whose segments.json references .npy while the corpus convention
is .npz (format split — 2XU5NTX4Xfs is the known instance).

Hard-fail: a referenced file is missing AND locally verifiable, or an .npz fails to
load / has the wrong landmarks shape. Everything else (format split, stale manifest
row, garbage manifest row) is a FLAG.

Run standalone:
    python qc/gate_integrity.py --root phase3/local_drive_mirror
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CANDIDATES = [REPO_ROOT / "videos_manifest.csv", REPO_ROOT / "phase2" / "videos_manifest.csv"]
EXPECTED_SCHEMA_ROWS = 124
HAND_ROWS = slice(0, 42)  # for shape sanity only
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


# --------------------------------------------------------------------------- #
# core, testable functions
# --------------------------------------------------------------------------- #

def guess_kind(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    if suf in ("npz", "npy", "json", "csv"):
        return suf
    return "other"


def sha256_of(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def npz_shape_and_schema(path: Path):
    """Returns (shape, schema_rows_guess, error). error is None on success."""
    try:
        with np.load(path) as d:
            if "landmarks" not in d:
                return None, None, "missing 'landmarks' key"
            arr = d["landmarks"]
            shape = tuple(arr.shape)
            schema = shape[1] if len(shape) >= 2 else None
            return shape, schema, None
    except Exception as e:  # noqa: BLE001 — genuinely want to catch any load failure
        return None, None, str(e)


def find_orphans(segments_meta: dict, array_dir: Path):
    """Two-way check between segments.json entries and files actually on disk.

    Returns (npz_not_referenced: list[str], referenced_missing: list[str]).
    """
    referenced = {s["file"] for s in segments_meta.get("segments", [])}
    on_disk = {p.name for p in array_dir.glob("*.npz")} | {p.name for p in array_dir.glob("*.npy")}
    not_referenced = sorted(on_disk - referenced)
    missing = sorted(f for f in referenced if f not in on_disk)
    return not_referenced, missing


def detect_format_split(segments_meta: dict) -> set:
    """Set of file extensions used inside one segments.json's `file` fields."""
    return {Path(s["file"]).suffix.lower() for s in segments_meta.get("segments", [])}


def cross_check_manifest(manifest_rows: list, segments_counts: dict):
    """manifest_rows: list of dict rows from videos_manifest.csv.
    segments_counts: {video_id: n_segments_on_disk} from segments_all/.

    Returns (mismatches, garbage_rows). mismatches: list of
    {id, claimed_n_segs, actual_n_segs}. garbage_rows: list of {id, reason}.
    """
    mismatches, garbage = [], []
    for r in manifest_rows:
        vid = r.get("id", "")
        claimed = r.get("n_segs", "")
        if not vid or not claimed.lstrip("-").isdigit():
            garbage.append({"id": vid or "<blank>", "reason": "unparseable id/n_segs"})
            continue
        if not YOUTUBE_ID_RE.match(vid):
            garbage.append({"id": vid, "reason": f"id is not an 11-char YouTube-ID shape ({vid!r})"})
            continue
        if vid not in segments_counts:
            continue  # no segments.json on disk to compare against — not this gate's business
        actual = segments_counts[vid]
        if int(claimed) != actual:
            mismatches.append({"id": vid, "claimed_n_segs": claimed, "actual_n_segs": actual})
    return mismatches, garbage


# --------------------------------------------------------------------------- #
# discovery helpers
# --------------------------------------------------------------------------- #

def discover_segments_all(root: Path):
    """{video_id: Path} for root/segments_all/<vid>.segments.json."""
    d = root / "segments_all"
    if not d.is_dir():
        return {}
    return {p.name[: -len(".segments.json")]: p for p in sorted(d.glob("*.segments.json"))}


def discover_local_arrays(root: Path):
    """{video_id: array_dir} for root/arrays_<vid>/segments.json (local .npz mirror)."""
    out = {}
    for d in sorted(root.glob("arrays_*")):
        if not d.is_dir():
            continue
        seg_json = d / "segments.json"
        if seg_json.exists():
            out[d.name[len("arrays_"):]] = d
    return out


def load_manifest_rows():
    for p in MANIFEST_CANDIDATES:
        if p.exists():
            return list(csv.DictReader(open(p))), p
    return None, None


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run(root: Path, out_dir: Path) -> dict:
    params = {"root": str(root), "expected_schema_rows": EXPECTED_SCHEMA_ROWS}
    if not root.exists():
        return {"gate": "integrity", "status": "SKIPPED", "reason": f"root not found: {root}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    manifest_path = out_dir / "artifact_manifest.csv"
    issues_path = out_dir / "integrity_issues.csv"

    # 1. manifest of every file under root
    manifest_rows = []
    all_files = [p for p in root.rglob("*") if p.is_file()]
    for p in all_files:
        kind = guess_kind(p)
        rel = p.relative_to(root)
        row = {"path": str(rel), "sha256": "", "bytes": p.stat().st_size,
               "mtime_iso": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(),
               "kind": kind, "shape": "", "schema_rows_guess": ""}
        try:
            row["sha256"] = sha256_of(p)
        except Exception as e:  # noqa: BLE001
            row["sha256"] = f"ERROR:{e}"
        if kind in ("npz", "npy"):
            shape, schema, err = npz_shape_and_schema(p) if kind == "npz" else (None, None, "npy-not-checked")
            if shape is not None:
                row["shape"] = "x".join(str(x) for x in shape)
                row["schema_rows_guess"] = schema
        manifest_rows.append(row)

    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes", "mtime_iso", "kind",
                                          "shape", "schema_rows_guess"])
        w.writeheader()
        w.writerows(manifest_rows)

    issues = []  # each: {severity: HARD|FLAG, video, kind, detail}
    n_checked = 0

    seg_all = discover_segments_all(root)
    local_arrays = discover_local_arrays(root)

    segments_counts = {}
    for vid, seg_path in seg_all.items():
        try:
            meta = json.load(open(seg_path))
        except Exception as e:  # noqa: BLE001
            issues.append({"severity": "HARD", "video": vid, "kind": "unreadable_segments_json",
                            "detail": str(e)})
            continue
        n_checked += 1
        segments_counts[vid] = len(meta.get("segments", []))

        exts = detect_format_split(meta)
        if exts - {".npz"}:
            issues.append({"severity": "FLAG", "video": vid, "kind": "format_split",
                            "detail": f"segments.json references {sorted(exts)} (corpus convention is .npz)"})

        if vid in local_arrays:
            not_ref, missing = find_orphans(meta, local_arrays[vid])
            for f in not_ref:
                issues.append({"severity": "FLAG", "video": vid, "kind": "orphan_npz",
                                "detail": f"{f} on disk but not referenced by segments.json"})
            for f in missing:
                issues.append({"severity": "HARD", "video": vid, "kind": "missing_referenced_file",
                                "detail": f"segments.json references {f}, not found on disk"})
            # corrupt/unreadable check, capped to what's actually present locally
            for s in meta.get("segments", []):
                p = local_arrays[vid] / s["file"]
                if p.suffix != ".npz" or not p.exists():
                    continue
                shape, schema, err = npz_shape_and_schema(p)
                n_checked += 1
                if err is not None:
                    issues.append({"severity": "HARD", "video": vid, "kind": "corrupt_npz",
                                    "detail": f"{s['file']}: {err}"})
                elif schema != EXPECTED_SCHEMA_ROWS:
                    issues.append({"severity": "HARD", "video": vid, "kind": "wrong_schema",
                                    "detail": f"{s['file']}: schema_rows={schema}, expected {EXPECTED_SCHEMA_ROWS}"})
        else:
            issues.append({"severity": "INFO", "video": vid, "kind": "arrays_not_mounted",
                            "detail": "no local arrays_<vid>/ mirror — file-existence not verifiable here"})

    # arrays_* dirs that have their own segments.json but aren't mirrored in segments_all
    # (e.g. arrays_57TvyH9902U/segments.json duplicates segments_all/57TvyH9902U.segments.json —
    # that's expected and not flagged; only check videos with local arrays not covered above)
    for vid, array_dir in local_arrays.items():
        if vid in seg_all:
            continue
        seg_path = array_dir / "segments.json"
        try:
            meta = json.load(open(seg_path))
        except Exception as e:  # noqa: BLE001
            issues.append({"severity": "HARD", "video": vid, "kind": "unreadable_segments_json", "detail": str(e)})
            continue
        n_checked += 1
        not_ref, missing = find_orphans(meta, array_dir)
        for f in not_ref:
            issues.append({"severity": "FLAG", "video": vid, "kind": "orphan_npz",
                            "detail": f"{f} on disk but not referenced by segments.json"})
        for f in missing:
            issues.append({"severity": "HARD", "video": vid, "kind": "missing_referenced_file",
                            "detail": f"segments.json references {f}, not found on disk"})

    # manifest cross-check
    manifest_rows_csv, manifest_used = load_manifest_rows()
    if manifest_rows_csv is None:
        issues.append({"severity": "INFO", "video": "", "kind": "no_manifest",
                        "detail": f"videos_manifest.csv not found at any of {[str(p) for p in MANIFEST_CANDIDATES]}"})
    else:
        mismatches, garbage = cross_check_manifest(manifest_rows_csv, segments_counts)
        n_checked += len(manifest_rows_csv)
        for m in mismatches:
            issues.append({"severity": "FLAG", "video": m["id"], "kind": "stale_manifest_row",
                            "detail": f"videos_manifest.csv n_segs={m['claimed_n_segs']} but "
                                      f"segments_all shows {m['actual_n_segs']}"})
        for g in garbage:
            issues.append({"severity": "FLAG", "video": g["id"], "kind": "garbage_manifest_row",
                            "detail": g["reason"]})
        params["manifest_used"] = str(manifest_used)

    with open(issues_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["severity", "video", "kind", "detail"])
        w.writeheader()
        w.writerows(issues)

    n_hard = sum(1 for i in issues if i["severity"] == "HARD")
    n_flag = sum(1 for i in issues if i["severity"] == "FLAG")
    status = "FAIL" if n_hard else ("FLAG" if n_flag else "PASS")

    return {"gate": "integrity", "status": status, "n_checked": n_checked, "n_failed": n_hard,
            "n_flagged": n_flag, "details_path": str(issues_path), "params": params,
            "manifest_path": str(manifest_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.root, out_dir)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
