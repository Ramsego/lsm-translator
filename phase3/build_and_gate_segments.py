"""
Rebuild presence-validated segment boundaries from ALREADY-EXTRACTED landmark
arrays, using hand AND pose (not hand alone, not sparse pose-only revalidation)
-- for free, no new MediaPipe inference, since Stage 0 extraction already
computed and stored pose landmarks for every frame where hands fired.

Scope: only the videos that already have extracted .npz arrays (13 of 16 as of
2026-08-14; three videos -- 2XU5NTX4Xfs, pMpjBLT7J_g, Z-pG_c9HoT8 -- have none
yet and are skipped, not silently included).

For each video: walk its existing per-segment .npz files in order, mark each
frame present if BOTH hand rows (0:42) and pose rows (42:75) are non-NaN, then
re-run the same contiguous-run logic extract_continuous.py uses (gap tolerance,
min duration) to rebuild segment boundaries -- this time validated against both
signals, not just hand. Absolute time is recovered from each source segment's
own start_sec + frame_index/fps (segments.json's own fps field).

Output: <arrays-root>/<vid>/segments_and_gate.json, same schema as
segments.json/segments_revalidated.json, alongside them -- never overwrites
either.

Usage:
    python3 phase3/build_and_gate_segments.py \
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera"
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Expressed purely in SECONDS so they are correct regardless of proc_fps: they
# describe the same gap/min-duration extract_continuous.py used at its default
# --gap=20 / --min-frames=10 processed-frame thresholds, at whatever proc_fps
# those processed frames were sampled at (extract_continuous.py's own gap/
# min-frames are also counted in PROCESSED frames, i.e. units of 1/proc_fps).
GAP_SEC = 20 / 30      # matches extract_continuous.py's default --gap=20 processed frames
MIN_DURATION = 10 / 30 * 10  # matches --min-frames=10, in seconds

FACE_ROWS = slice(75, 124)


def and_gate_runs(arr, proc_fps, seg_start_sec):
    """Find contiguous (hand AND pose)-present runs in a segment's landmark array.

    All frame<->second conversions here use proc_fps = fps / every (the rate at
    which frames were actually SAMPLED and stored), never raw container fps --
    using raw fps for an every>1 video compresses recovered times by a factor
    of `every`.

    Returns a list of (start_t, end_t, present_frac) tuples, where present_frac
    is the fraction of the run's own processed frames with BOTH hand and pose
    present (== 1.0 by construction here, since the run is defined as a
    presence run split only on gaps -- included for schema symmetry with the
    per-segment output, which reports it over the recomputed sub-span).
    """
    hand_present = ~np.isnan(arr[:, 0:42, :]).all(axis=(1, 2))
    pose_present = ~np.isnan(arr[:, 42:75, :]).all(axis=(1, 2))
    present = hand_present & pose_present
    idx = np.where(present)[0]
    if len(idx) == 0:
        return []
    gap_frames = max(1, int(round(GAP_SEC * proc_fps)))
    splits = np.where(np.diff(idx) > gap_frames)[0]
    groups = np.split(idx, splits + 1)
    out = []
    for g in groups:
        start_t = seg_start_sec + g[0] / proc_fps
        end_t = seg_start_sec + (g[-1] + 1) / proc_fps
        if end_t - start_t >= MIN_DURATION:
            # presence fraction over the gated sub-span itself (processed frames
            # from g[0] to g[-1] inclusive), not just the run's own hit count --
            # a run can't contain internal gaps larger than gap_frames, but may
            # still contain isolated single-frame absences smaller than that.
            span = present[g[0]:g[-1] + 1]
            frac = float(span.mean()) if len(span) else 0.0
            out.append((round(start_t, 2), round(end_t, 2), round(frac, 3)))
    return out


def process_video(vdir: Path, write_output: bool = True):
    """Rebuild AND-gated segments for one already-extracted video directory.

    Returns a dict with per-video stats (before_dur, after_dur, n_orig, n_new,
    skipped_arrays, out) or None if the video was skipped entirely (no
    segments.json / no .npz files). Writes vdir/segments_and_gate.json unless
    write_output=False (used by tests that want the computed dict without
    touching disk).
    """
    seg_meta_path = vdir / "segments.json"
    npz_files = sorted(vdir.glob("*.npz"))
    if not seg_meta_path.exists() or not npz_files:
        print(f"{vdir.name:16s} SKIP (no segments.json or no .npz files)")
        return None

    meta = json.load(open(seg_meta_path))
    every = meta.get("every", 1)
    proc_fps = meta["fps"] / every       # frames were SAMPLED at fps/every, not raw fps
    orig_segs = {s["file"]: s for s in meta["segments"]}

    new_segments = []
    before_dur = 0.0
    skipped_arrays = 0
    for f in npz_files:
        orig = orig_segs.get(f.name)
        if orig is None:
            # .npz with no matching segments.json entry -- skip, don't guess
            skipped_arrays += 1
            continue
        arr = np.load(f)["landmarks"]
        before_dur += orig["end_sec"] - orig["start_sec"]
        for start_t, end_t, frac in and_gate_runs(arr, proc_fps, orig["start_sec"]):
            new_segments.append({
                "start_sec": start_t, "end_sec": end_t,
                "file": orig["file"], "presence_frac": frac,
            })

    if skipped_arrays:
        print(f"  WARNING: {skipped_arrays} array(s) skipped (no segments.json entry) "
              f"in {vdir.name}")

    after_dur = sum(s["end_sec"] - s["start_sec"] for s in new_segments)

    out = dict(meta)
    out["segments"] = new_segments
    out["and_gate"] = dict(criterion="hand AND pose, from already-extracted arrays",
                           hours_before=round(before_dur / 3600, 3),
                           hours_after=round(after_dur / 3600, 3),
                           proc_fps=proc_fps, every=every,
                           skipped_arrays=skipped_arrays)
    out_path = vdir / "segments_and_gate.json"
    if write_output:
        json.dump(out, open(out_path, "w"), indent=1)
    print(f"{vdir.name:16s} {len(orig_segs):>4d} -> {len(new_segments):>4d} segs   "
          f"{before_dur/3600:5.2f}h -> {after_dur/3600:5.2f}h")

    return dict(before_dur=before_dur, after_dur=after_dur, n_orig=len(orig_segs),
               n_new=len(new_segments), skipped_arrays=skipped_arrays, out=out,
               out_path=out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", type=Path, required=True)
    args = ap.parse_args()

    grand_before = grand_after = 0.0
    total_skipped_arrays = 0
    for vdir in sorted(args.arrays_root.iterdir()):
        if not vdir.is_dir() or vdir.name.startswith("."):
            continue
        result = process_video(vdir)
        if result is None:
            continue
        grand_before += result["before_dur"]
        grand_after += result["after_dur"]
        total_skipped_arrays += result["skipped_arrays"]

    print(f"\nCORPUS (already-extracted videos only): "
          f"{grand_before/3600:.2f}h -> {grand_after/3600:.2f}h")
    if total_skipped_arrays:
        print(f"WARNING: {total_skipped_arrays} array(s) total skipped across the corpus "
              f"(no segments.json entry)")


if __name__ == "__main__":
    main()
