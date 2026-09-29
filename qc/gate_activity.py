"""
QC gate 4/5: ACTIVITY (FLAG-only, new signal) — is the signer actually MOVING, not
just present?

Presence (gate_presence) only asks "are hand landmarks detected". A segment can pass
that and still be mostly the interpreter standing still between bursts (or, in a
worse case, a static false-positive that presence didn't catch). This gate reuses
phase2/estimate_lag.py's motion-energy primitive — per-frame hand motion
(HAND_ROWS=slice(0,42)), resampled to DT=0.25s bins — and asks what fraction of a
segment's bins clear a "something moved" threshold, and how long the longest still
stretch is.

CALIBRATION (run with --calibrate; numbers below are from the 2026-08-17 run against
57TvyH9902U, the only video with both local arrays and completed review verdicts):

    37/38 verdict=='y' review rows fell inside a locally-available segment; the DT-bin
    containing each row's video_start gives 35 usable "known-signing" energy samples
    (2 landed in a bin with no motion data, e.g. at a segment edge, and were skipped).

    global bin energy   (n=21789): p5=0.0031  p10=0.0052  p25=0.0100  p50=0.0166
    known-signing energy (n=35):   p5=0.0050  p10=0.0058  p25=0.0120  p50=0.0212

    Known-signing sits at or above the global 5th-10th percentile almost everywhere,
    which is what you'd expect (known-signing bins are a subset biased toward "yes,
    something moved") but the two distributions overlap heavily below the median —
    hand motion during isolated one-word signs is often small. ACTIVE_ENERGY_THRESHOLD
    is set to 0.005: at/just below the known-signing 5th percentile, so a "still" bin
    call very rarely throws out a real signing bin, while still cutting the bottom
    of the global distribution (frozen frames, B-roll wrongly gated as present).

Per-segment metrics: fraction_active (bins >= threshold) and longest_still_s (longest
run of below-threshold bins * DT). Default flag rule (used whether or not calibration
was rerun this session): fraction_active < 0.5 or longest_still_s > 8.

NEVER hard-fails. Only videos with a local .npz mirror can be scored (arrays_57TvyH9902U
today); everything else is reported as not-scoreable, not as a failure.

Run standalone:
    python qc/gate_activity.py --root phase3/local_drive_mirror
    python qc/gate_activity.py --root phase3/local_drive_mirror --calibrate
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_ROWS = slice(0, 42)
DT = 0.25

# See calibration note in the module docstring: known-signing p5 ~= 0.0050 on the one
# video where we can check (57TvyH9902U, n=35 matched bins, run 2026-08-17).
ACTIVE_ENERGY_THRESHOLD = 0.005
FRACTION_ACTIVE_MIN = 0.5
LONGEST_STILL_MAX_S = 8.0


# --------------------------------------------------------------------------- #
# core, testable functions
# --------------------------------------------------------------------------- #

def frame_energy(landmarks: np.ndarray) -> np.ndarray:
    """Per-frame hand motion energy, NaN-safe. landmarks: [T, 124, 3] -> [T-1]."""
    hands = landmarks[:, HAND_ROWS, :2]
    if hands.shape[0] < 2:
        return np.zeros(0)
    d = np.linalg.norm(np.diff(hands, axis=0), axis=2)
    with warnings.catch_warnings():
        # nanmean over a frame-pair where BOTH frames have all-NaN hands (a genuine
        # absence, not a bug) legitimately produces an all-NaN row -> RuntimeWarning.
        # nan_to_num converts it to 0 energy immediately after, which is correct.
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nan_to_num(np.nanmean(d, axis=1), nan=0.0)


def bin_energy(energy: np.ndarray, proc_fps: float, dt: float = DT) -> np.ndarray:
    """Resample per-frame energy into DT-second bins (mean within each bin)."""
    if len(energy) == 0 or not proc_fps:
        return np.zeros(0)
    bin_frames = max(1, int(round(dt * proc_fps)))
    n_bins = int(np.ceil(len(energy) / bin_frames))
    bins = np.zeros(n_bins)
    for b in range(n_bins):
        chunk = energy[b * bin_frames:(b + 1) * bin_frames]
        bins[b] = chunk.mean() if len(chunk) else 0.0
    return bins


def activity_metrics(bins: np.ndarray, threshold: float = ACTIVE_ENERGY_THRESHOLD, dt: float = DT):
    """Returns (fraction_active, longest_still_s)."""
    if len(bins) == 0:
        return 0.0, 0.0
    active = bins >= threshold
    fraction_active = float(active.mean())
    longest_run = 0
    cur = 0
    for a in active:
        if not a:
            cur += 1
            longest_run = max(longest_run, cur)
        else:
            cur = 0
    return fraction_active, longest_run * dt


def is_flagged(fraction_active, longest_still_s,
                frac_min=FRACTION_ACTIVE_MIN, still_max=LONGEST_STILL_MAX_S):
    return fraction_active < frac_min or longest_still_s > still_max


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #

def calibrate(root: Path, video_id="57TvyH9902U"):
    array_dir = root / f"arrays_{video_id}"
    review_csv = root / video_id / "review.csv"
    if not array_dir.exists() or not review_csv.exists():
        return None
    meta = json.load(open(array_dir / "segments.json"))
    fps, every = meta["fps"], meta["every"]
    proc_fps = fps / every

    seg_bins, all_bins = {}, []
    for s in meta["segments"]:
        p = array_dir / s["file"]
        if not p.exists():
            continue
        with np.load(p) as d:
            arr = d["landmarks"]
        bins = bin_energy(frame_energy(arr), proc_fps)
        seg_bins[s["file"]] = (s, bins)
        all_bins.append(bins)
    global_bins = np.concatenate(all_bins) if all_bins else np.zeros(0)

    rows = list(csv.DictReader(open(review_csv)))
    known = []
    for r in rows:
        if r.get("verdict") != "y":
            continue
        vs = float(r["video_start"])
        for s in meta["segments"]:
            if s["start_sec"] <= vs <= s["end_sec"]:
                bin_idx = int((vs - s["start_sec"]) / DT)
                _, bins = seg_bins.get(s["file"], (None, np.zeros(0)))
                if 0 <= bin_idx < len(bins):
                    known.append(bins[bin_idx])
                break
    known = np.array(known)

    pct = [5, 10, 25, 50, 75, 90, 95]
    return {
        "video": video_id, "n_global_bins": len(global_bins), "n_known_signing_bins": len(known),
        "global_percentiles": dict(zip(pct, np.percentile(global_bins, pct).round(4).tolist())) if len(global_bins) else {},
        "known_signing_percentiles": dict(zip(pct, np.percentile(known, pct).round(4).tolist())) if len(known) else {},
        "suggested_threshold": round(float(np.percentile(known, 5)), 4) if len(known) else None,
    }


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


def run(root: Path, out_dir: Path, threshold: float = ACTIVE_ENERGY_THRESHOLD) -> dict:
    params = {"root": str(root), "dt": DT, "active_energy_threshold": threshold,
              "fraction_active_min": FRACTION_ACTIVE_MIN, "longest_still_max_s": LONGEST_STILL_MAX_S,
              "calibration": "57TvyH9902U, n=35 known-signing bins, run 2026-08-17 (see module docstring)"}
    if not root.exists():
        return {"gate": "activity", "status": "SKIPPED", "reason": f"root not found: {root}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    local_arrays = _discover_local_arrays(root)
    if not local_arrays:
        return {"gate": "activity", "status": "SKIPPED",
                "reason": "no arrays_<vid>/ local mirror found — activity needs real landmark arrays",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    rows = []
    for vid, array_dir in local_arrays.items():
        meta = json.load(open(array_dir / "segments.json"))
        fps, every = meta.get("fps", 30.0), meta.get("every", 1)
        proc_fps = fps / every if fps else None
        for s in meta.get("segments", []):
            p = array_dir / s["file"]
            if not p.exists() or p.suffix != ".npz":
                continue
            with np.load(p) as d:
                arr = d["landmarks"]
            bins = bin_energy(frame_energy(arr), proc_fps)
            frac_active, longest_still = activity_metrics(bins, threshold)
            flagged = is_flagged(frac_active, longest_still)
            rows.append({"video": vid, "seg": s["file"], "fraction_active": round(frac_active, 4),
                          "longest_still_s": round(longest_still, 2), "flag": "still" if flagged else ""})

    details_path = out_dir / "activity_audit.csv"
    with open(details_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "seg", "fraction_active", "longest_still_s", "flag"])
        w.writeheader()
        w.writerows(rows)

    n_flagged = sum(1 for r in rows if r["flag"])
    status = "FLAG" if n_flagged else ("PASS" if rows else "SKIPPED")

    return {"gate": "activity", "status": status, "n_checked": len(rows), "n_failed": 0,
            "n_flagged": n_flagged, "details_path": str(details_path), "params": params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--threshold", type=float, default=ACTIVE_ENERGY_THRESHOLD)
    args = ap.parse_args()
    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.calibrate:
        cal = calibrate(args.root)
        if cal is None:
            print("calibration data not available (need arrays_<vid>/ + <vid>/review.csv)")
            sys.exit(0)
        print(json.dumps(cal, indent=2))
        sys.exit(0)

    result = run(args.root, out_dir, threshold=args.threshold)
    print(json.dumps(result, indent=2))
    sys.exit(0)  # activity never fails the run


if __name__ == "__main__":
    main()
