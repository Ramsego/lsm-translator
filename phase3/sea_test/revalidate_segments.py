"""
Re-validate segments.json against actual frame content, and rebuild segment boundaries.

SUPERSEDED FOR NEW EXTRACTIONS (2026-08-14): phase2/extract_continuous.py now gates
on Pose AND Hand per-frame (not Hand alone), which was measured to match this
script's own accuracy without a separate revalidation pass -- see that file's
docstring. This script is still correct and still needed to reproduce/understand
the existing segments_revalidated.json files, which were built from the old
hand-only-then-pose-revalidated two-pass pipeline.

WHY (measured 2026-07-31): phase2/extract_continuous.py gates signer presence on
"hands detected inside the interpreter crop". That fails in both directions:

  false POSITIVE - when the broadcast cuts to B-roll, other people's hands land in
      that screen region and open a spurious window. Confirmed by cropping to the
      stored bbox: football players' arms (57TvyH9902U @3294s), a trouser leg
      (cjpPmeg-Doo @1206s), a violin and a child's hand (lrcmIZFvFgY @4021s), and --
      critically -- a hand holding paper beside a flower vase inside a 32.6s window
      (u3Gu2gSxUVY 1302.9-1335.6s). So this is NOT confined to short blips and a
      duration floor alone cannot fix it.
  false NEGATIVE - the interpreter rests between signs; hands vanish while the
      interpreter is plainly still there (pMpjBLT7J_g sampled at 50% hand presence).

Calibration on 28 held-out clips (26 known-good, 2 known-bad) separated perfectly on
POSE, not hands: pose detected in 100% of good samples and 0% of bad ones. An
interpreter always presents a torso; a crop of a flower vase does not. So this script
re-gates on pose and rebuilds boundaries from contiguous runs of pose-present samples,
which also splits a mixed segment instead of accepting or rejecting it wholesale.

Usage:
    python3 phase3/sea_test/revalidate_segments.py \
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera" \
        --videos-root "/Volumes/Crucial X8/LSM_Translator/videos/mananera" \
        --out-suffix _revalidated

Writes <arrays-root>/<vid>/segments<suffix>.json alongside the original (never
overwrites it) plus a per-sample CSV for auditing.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp


def crop_at(crop_changes, t):
    prior = [c for c in crop_changes if c["at_sec"] <= t]
    return (prior[-1] if prior else crop_changes[0])["crop"]


def runs_of_true(samples, step, seg_start, seg_end, min_duration):
    """samples: [(t, present)] ascending -> contiguous present runs as (start, end)."""
    out, run = [], None
    for t, present in samples:
        if present and run is None:
            run = [t, t]
        elif present:
            run[1] = t
        elif run is not None:
            out.append(tuple(run))
            run = None
    if run is not None:
        out.append(tuple(run))
    # widen each run by half a sampling step (the true boundary lies between samples)
    segs = []
    for a, b in out:
        a2 = max(seg_start, a - step / 2)
        b2 = min(seg_end, b + step / 2)
        if b2 - a2 >= min_duration:
            segs.append((a2, b2))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, required=True)
    ap.add_argument("--out-suffix", default="_revalidated")
    ap.add_argument("--sample-every", type=float, default=20.0,
                    help="seconds between content probes inside a segment")
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--min-duration", type=float, default=10.0,
                    help="drop rebuilt segments shorter than this")
    ap.add_argument("--only", nargs="*", help="restrict to these video ids")
    args = ap.parse_args()

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)

    vids = sorted(d for d in args.arrays_root.iterdir()
                  if (d / "segments.json").exists()
                  and (not args.only or d.name in args.only))

    grand_before = grand_after = 0.0
    t0 = time.time()
    for d in vids:
        vid = d.name
        m = json.load(open(d / "segments.json"))
        crops = sorted(m["crop_changes"], key=lambda c: c["at_sec"])
        src = args.videos_root / vid / f"{vid}.mp4"
        if not src.exists():
            print(f"{vid}: SKIP (no video at {src})", flush=True)
            continue

        cap = cv2.VideoCapture(str(src))
        audit, new_segments = [], []
        before = sum(s["end_sec"] - s["start_sec"] for s in m["segments"])

        for s in sorted(m["segments"], key=lambda s: s["start_sec"]):
            a, b = s["start_sec"], s["end_sec"]
            dur = b - a
            n = max(args.min_samples, int(dur / args.sample_every) + 1)
            step = dur / n
            samples = []
            for i in range(n):
                t = a + step * (i + 0.5)
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    samples.append((t, False))
                    continue
                x, y, w, h = crop_at(crops, t)
                H, W = frame.shape[:2]
                x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
                sub = frame[y:min(y + h, H), x:min(x + w, W)]
                if sub.size == 0:
                    samples.append((t, False))
                    continue
                r = holistic.process(cv2.cvtColor(sub, cv2.COLOR_BGR2RGB))
                present = r.pose_landmarks is not None
                samples.append((t, present))
                audit.append(dict(video=vid, t=round(t, 2), seg_start=round(a, 2),
                                  seg_end=round(b, 2), pose=int(present),
                                  hand=int(bool(r.left_hand_landmarks or r.right_hand_landmarks)),
                                  face=int(r.face_landmarks is not None)))
            for ra, rb in runs_of_true(samples, step, a, b, args.min_duration):
                new_segments.append({"start_sec": ra, "end_sec": rb})
        cap.release()

        after = sum(s["end_sec"] - s["start_sec"] for s in new_segments)
        grand_before += before
        grand_after += after
        out = dict(m)
        out["segments"] = new_segments
        out["revalidated"] = dict(criterion="mediapipe holistic pose_landmarks present",
                                  sample_every=args.sample_every,
                                  min_duration=args.min_duration,
                                  hours_before=round(before / 3600, 3),
                                  hours_after=round(after / 3600, 3))
        json.dump(out, open(d / f"segments{args.out_suffix}.json", "w"), indent=1)
        with open(d / f"segments{args.out_suffix}_audit.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video", "t", "seg_start", "seg_end",
                                              "pose", "hand", "face"])
            w.writeheader()
            w.writerows(audit)
        kept = after / before if before else 0
        print(f"{vid:16s} {len(m['segments']):>4d} -> {len(new_segments):>4d} segs   "
              f"{before/3600:5.2f}h -> {after/3600:5.2f}h  ({kept:5.1%} kept)  "
              f"[{len(audit)} probes, {time.time()-t0:.0f}s elapsed]", flush=True)

    print(f"\nCORPUS: {grand_before/3600:.2f}h -> {grand_after/3600:.2f}h "
          f"({grand_after/grand_before:.1%} kept)")


if __name__ == "__main__":
    main()
