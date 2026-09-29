"""
QC gate 3/5: PRESENCE — is the interpreter actually on screen for the segment?

Coverage = fraction of frames where at least one landmark in a row-slice is non-NaN,
ported from scripts/quality_report.py's 116-row version to the 124-row schema:
  hands 0:42, pose 42:75, face 75:124 (phase2/extract_continuous.py's layout).

Corpus-eligibility rule: duration >= 10s AND hand_coverage >= 0.6.

  - 10s floor: phase3/sea_test/filter_segments.py measured the segment-duration
    distribution as starkly bimodal (p50=7.2s, p75=81s, p90=437s); B-roll false
    positives (football players' arms, a trouser leg, a violin/child's hand
    caught in the interpreter's screen box) are 55% of segment COUNT but only
    1.17% of total DURATION. A duration-weighted audit of 16 windows >=10s came
    back 16/16 genuine. A 10s floor buys most of the cleanup for ~1% of the data.
  - 0.6 hand-coverage floor: real signing segments in the corpus sit >=0.55 with
    long/clean ones >=0.9 (see 57TvyH9902U/segments.json: 0.55, 0.986, ...); the
    known-bad seeds in R1QhGx6JdDk.segments.json sit at 0.286-0.545. 0.6 sits in
    the gap and is what this gate is calibrated against.

This gate NEVER hard-fails the run — see DATA_QUALITY.md sec. 2. It reports an
exclusion list (status FLAG when non-empty) rather than treating low coverage as a
corrupt-data condition; a segment excluded today may be re-included if a better
presence detector (e.g. the revalidate_segments.py pose-based re-gate) supersedes it.

Run standalone:
    python qc/gate_presence.py --root phase3/local_drive_mirror
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_ROWS = slice(0, 42)
POSE_ROWS = slice(42, 75)
FACE_ROWS = slice(75, 124)

MIN_DURATION_S = 10.0
MIN_HAND_COVERAGE = 0.6


# --------------------------------------------------------------------------- #
# core, testable functions
# --------------------------------------------------------------------------- #

def coverage(arr: np.ndarray, row_slice: slice) -> float:
    """Fraction of frames where at least one landmark in the slice is non-NaN."""
    if arr.shape[0] == 0:
        return 0.0
    chunk = arr[:, row_slice, :]
    has_data = ~np.isnan(chunk).all(axis=(1, 2))
    return float(has_data.mean())


def array_coverage(arr: np.ndarray):
    """Returns (hand_cov, pose_cov, face_cov, nan_frac)."""
    return (coverage(arr, HAND_ROWS), coverage(arr, POSE_ROWS), coverage(arr, FACE_ROWS),
            float(np.isnan(arr).mean()) if arr.size else 1.0)


def is_eligible(duration_s: float, hand_cov, min_duration=MIN_DURATION_S, min_hand_cov=MIN_HAND_COVERAGE):
    if hand_cov is None:
        return False
    return duration_s >= min_duration and hand_cov >= min_hand_cov


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def _discover_local_arrays(root: Path):
    out = {}
    for d in sorted(root.glob("arrays_*")):
        seg_json = d / "segments.json"
        if seg_json.exists():
            out[d.name[len("arrays_"):]] = d
    return out


def run(root: Path, out_dir: Path) -> dict:
    params = {"root": str(root), "min_duration_s": MIN_DURATION_S, "min_hand_coverage": MIN_HAND_COVERAGE}
    if not root.exists():
        return {"gate": "presence", "status": "SKIPPED", "reason": f"root not found: {root}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    rows = []
    local_arrays = _discover_local_arrays(root)

    # videos with local .npz — compute coverage directly from the array
    for vid, array_dir in local_arrays.items():
        meta = json.load(open(array_dir / "segments.json"))
        fps, every = meta.get("fps", 30.0), meta.get("every", 1)
        proc_fps = fps / every if fps else None
        for s in meta.get("segments", []):
            p = array_dir / s["file"]
            dur = s["end_sec"] - s["start_sec"]
            if not p.exists() or p.suffix != ".npz":
                rows.append({"video": vid, "seg": s["file"], "n_frames": s.get("n_frames", ""),
                              "dur_s": round(dur, 3), "hand_cov": "", "pose_cov": "", "face_cov": "",
                              "nan_frac": "", "flags": "unreadable,excluded"})
                continue
            with np.load(p) as d:
                arr = d["landmarks"]
            hand_cov, pose_cov, face_cov, nan_frac = array_coverage(arr)
            eligible = is_eligible(dur, hand_cov)
            flags = [] if eligible else ["excluded"]
            rows.append({"video": vid, "seg": s["file"], "n_frames": s.get("n_frames", ""),
                          "dur_s": round(dur, 3), "hand_cov": round(hand_cov, 4),
                          "pose_cov": round(pose_cov, 4), "face_cov": round(face_cov, 4),
                          "nan_frac": round(nan_frac, 4), "flags": ",".join(flags)})

    # videos with only segments_all/*.segments.json (arrays on the unmounted drive) —
    # fall back to the stored hand_coverage field
    seg_all_dir = root / "segments_all"
    if seg_all_dir.is_dir():
        for p in sorted(seg_all_dir.glob("*.segments.json")):
            vid = p.name[: -len(".segments.json")]
            if vid in local_arrays:
                continue  # already covered above with real array data
            meta = json.load(open(p))
            for s in meta.get("segments", []):
                dur = s["end_sec"] - s["start_sec"]
                hand_cov = s.get("hand_coverage")
                eligible = is_eligible(dur, hand_cov)
                flags = [] if eligible else ["excluded"]
                if hand_cov is None:
                    flags.append("no_coverage_field")
                rows.append({"video": vid, "seg": s["file"], "n_frames": s.get("n_frames", ""),
                              "dur_s": round(dur, 3), "hand_cov": hand_cov if hand_cov is not None else "",
                              "pose_cov": "n/a (no local array)", "face_cov": "n/a (no local array)",
                              "nan_frac": "n/a (no local array)", "flags": ",".join(flags)})

    details_path = out_dir / "presence_audit.csv"
    with open(details_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "seg", "n_frames", "dur_s", "hand_cov",
                                          "pose_cov", "face_cov", "nan_frac", "flags"])
        w.writeheader()
        w.writerows(rows)

    hours_before = sum(r["dur_s"] for r in rows) / 3600.0
    excluded = [r for r in rows if "excluded" in r["flags"]]
    hours_after = sum(r["dur_s"] for r in rows if "excluded" not in r["flags"]) / 3600.0

    status = "FLAG" if excluded else ("PASS" if rows else "SKIPPED")
    reason = None if rows else "no segments.json found under root"

    return {"gate": "presence", "status": status, "n_checked": len(rows), "n_failed": 0,
            "n_flagged": len(excluded), "details_path": str(details_path), "params": params,
            "hours_before": round(hours_before, 3), "hours_after": round(hours_after, 3),
            "reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.root, out_dir)
    print(json.dumps(result, indent=2))
    print(f"\nhours_before={result['hours_before']}  hours_after={result['hours_after']}")
    sys.exit(0)  # presence never fails the run


if __name__ == "__main__":
    main()
