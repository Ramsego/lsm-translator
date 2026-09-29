"""Workstream A1: drop SIGN-tier annotations that fall outside hand-presence-validated
segments (segments.json). SENTENCE/SUBTITLE/SUBTITLE_CORRECTED tiers untouched.

Generalized 2026-07-31 from the single-slice version (which hardcoded 57TvyH9902U,
SLICE_START=1200, SLICE_LEN=1800) so it can run across the whole corpus before SEA is
scaled up. Behaviour is unchanged on that slice: run with --min-segment-duration 0 and
the output is byte-identical to the previously validated segmentation_filtered/ file.

Why this matters more for training data than it did for the alignment metric: A1's
measured effect on retrieval was ~null (one instance of 26 flipped). But SEA spans that
land in signer-absent regions become (text, empty-video) pairs, which are actively
harmful as fine-tuning supervision rather than merely uninformative.

Single file:
    python3 phase3/sea_test/filter_segments.py \
        --eaf  phase3/sea_test/out/segmentation/E4s-1_30_50/57TvyH9902U_slice.eaf \
        --segments phase3/local_drive_mirror/arrays_57TvyH9902U/segments.json \
        --out-dir phase3/sea_test/out/segmentation_filtered/E4s-1_30_50 \
        --slice-start 1200 --slice-len 1800

Whole corpus (segments.json discovered per video id under --arrays-root):
    python3 phase3/sea_test/filter_segments.py \
        --eaf-dir phase3/sea_test/out/segmentation/E4s-1_30_50 \
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera" \
        --out-dir phase3/sea_test/out/segmentation_filtered/E4s-1_30_50
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_windows(segments_json, slice_start, slice_len, min_duration=0.0):
    """Validated signer-present windows, converted absolute -> slice-local and clipped.

    min_duration drops short windows BEFORE clipping. Rationale (measured 2026-07-31):
    extract_continuous.py's presence gate is "hands detected inside the interpreter
    crop", which false-positives on B-roll -- when the broadcast cuts to packaged
    footage, other people's hands land in that screen region and open a spurious
    "signer-present" window. Verified by cropping to the stored bbox at three such
    windows: football players' arms (57TvyH9902U @3294s, 2.4s window), a trouser leg
    (cjpPmeg-Doo @1206s, 1.1s), a violin and a child's hand (lrcmIZFvFgY @4021s, 2.9s).

    The duration distribution is starkly bimodal -- p50=7.2s but p75=81s, p90=437s --
    so these artefacts are 55% of segment COUNT and only 1.17% of total duration
    (14.8 min of 21.12h). A duration-weighted audit of 16 windows >=10s came back 16/16
    genuine interpreter footage (n=16 bounds the long-window false-positive rate at
    <19%, rule of three -- reassuring, not conclusive). So a 10s floor buys most of the
    cleanup for ~1% of the data.
    """
    segs = json.load(open(segments_json))["segments"]
    windows = []
    for s in segs:
        if s["end_sec"] - s["start_sec"] < min_duration:
            continue
        st = s["start_sec"] - slice_start
        en = s["end_sec"] - slice_start
        if en < 0 or st > slice_len:
            continue
        windows.append((max(0.0, st), min(slice_len, en)))
    windows.sort()
    return windows


def gaps_between(windows, slice_len, min_gap=0.5):
    gaps, prev_end = [], 0.0
    for a, b in windows:
        if a - prev_end > min_gap:
            gaps.append((prev_end, a))
        prev_end = max(prev_end, b)
    if slice_len - prev_end > min_gap:
        gaps.append((prev_end, slice_len))
    return gaps


def filter_one(eaf_path, segments_json, out_dir, slice_start, slice_len,
               min_duration=0.0, quiet=False):
    """Returns (before, after, matched, n_dropped) -- matched==n_dropped means every
    dropped annotation landed inside a real absence gap (the time-base sanity check)."""
    windows = load_windows(segments_json, slice_start, slice_len, min_duration)
    covered = sum(b - a for a, b in windows)
    if not quiet:
        print(f"  validated windows overlapping slice: {len(windows)}")
        print(f"  validated coverage: {covered:.1f}s / {slice_len:.0f}s ({covered/slice_len:.0%})")

    def in_window(mid):
        return any(a <= mid <= b for a, b in windows)

    tree = ET.parse(eaf_path)
    root = tree.getroot()
    time_slots = {}
    for ts in root.find("TIME_ORDER").findall("TIME_SLOT"):
        v = ts.get("TIME_VALUE")
        if v is not None:
            time_slots[ts.get("TIME_SLOT_ID")] = float(v) / 1000.0

    sign_tier = next((t for t in root.findall("TIER") if t.get("TIER_ID") == "SIGN"), None)
    if sign_tier is None:
        raise SystemExit(f"no SIGN tier in {eaf_path}")

    before = len(sign_tier.findall("ANNOTATION"))
    to_remove, dropped_ranges = [], []
    for ann in sign_tier.findall("ANNOTATION"):
        al = next(iter(ann), None)
        if al is None:
            continue
        st = time_slots.get(al.attrib.get("TIME_SLOT_REF1"))
        en = time_slots.get(al.attrib.get("TIME_SLOT_REF2"))
        if st is None or en is None:
            continue
        if not in_window((st + en) / 2):
            to_remove.append(ann)
            dropped_ranges.append((st, en))

    for ann in to_remove:
        sign_tier.remove(ann)
    after = len(sign_tier.findall("ANNOTATION"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_eaf = out_dir / Path(eaf_path).name
    tree.write(out_eaf, encoding="UTF-8", xml_declaration=True)

    # Sanity check: EVERY drop must fall inside a real gap between validated windows.
    # If any drop lands inside a validated window, the time-base conversion is wrong and
    # the output must not be used. (The original single-slice run passed 87/87.)
    gaps = gaps_between(windows, slice_len)
    matched = sum(1 for a, b in dropped_ranges
                  if any(g0 <= (a + b) / 2 <= g1 for g0, g1 in gaps))

    if not quiet:
        pct = (before - after) / before if before else 0
        print(f"  SIGN annotations: {before} -> {after}  (dropped {before-after}, {pct:.1%})")
        print(f"  sanity: {matched}/{len(dropped_ranges)} drops inside a real absence gap", end="")
        print("  OK" if matched == len(dropped_ranges) else "  *** MISMATCH -- DO NOT USE ***")
        print(f"  wrote {out_eaf}")
    return before, after, matched, len(dropped_ranges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eaf", type=Path, help="single .eaf to filter")
    ap.add_argument("--eaf-dir", type=Path, help="directory of .eaf files (corpus mode)")
    ap.add_argument("--segments", type=Path, help="segments.json (single-file mode)")
    ap.add_argument("--arrays-root", type=Path,
                    help="root holding <video_id>/segments.json (corpus mode)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--slice-start", type=float, default=0.0,
                    help="absolute time of the .eaf's t=0 (1200 for the original 30-min slice)")
    ap.add_argument("--slice-len", type=float, default=None,
                    help="length covered by the .eaf; default = last validated window end")
    ap.add_argument("--min-segment-duration", type=float, default=10.0,
                    help="drop validated windows shorter than this many seconds before "
                         "filtering; guards against the B-roll false-positive mode "
                         "(see load_windows docstring). 0 disables. Default 10.")
    args = ap.parse_args()

    if args.eaf:
        if not args.segments:
            ap.error("--eaf requires --segments")
        slice_len = args.slice_len
        if slice_len is None:
            segs = json.load(open(args.segments))["segments"]
            slice_len = max(s["end_sec"] for s in segs) - args.slice_start
        print(f"{args.eaf.name}:")
        _, _, matched, n = filter_one(args.eaf, args.segments, args.out_dir,
                                       args.slice_start, slice_len,
                                       args.min_segment_duration)
        sys.exit(0 if matched == n else 1)

    if not (args.eaf_dir and args.arrays_root):
        ap.error("corpus mode needs --eaf-dir and --arrays-root")

    eafs = sorted(args.eaf_dir.glob("*.eaf"))
    if not eafs:
        ap.error(f"no .eaf files under {args.eaf_dir}")

    total_before = total_after = 0
    failures = []
    for eaf in eafs:
        # video id = filename up to the first "_slice"/extension boundary
        vid = re.sub(r"_slice$", "", eaf.stem)
        segments = args.arrays_root / vid / "segments.json"
        if not segments.exists():
            print(f"{eaf.name}: SKIP -- no segments.json at {segments}")
            failures.append((eaf.name, "missing segments.json"))
            continue
        segs = json.load(open(segments))["segments"]
        slice_len = args.slice_len or (max(s["end_sec"] for s in segs) - args.slice_start)
        print(f"{eaf.name}:")
        b, a, matched, n = filter_one(eaf, segments, args.out_dir,
                                       args.slice_start, slice_len,
                                       args.min_segment_duration)
        total_before += b
        total_after += a
        if matched != n:
            failures.append((eaf.name, f"sanity {matched}/{n}"))

    print(f"\n=== corpus total: {total_before} -> {total_after} SIGN annotations "
          f"(dropped {total_before-total_after}) ===")
    if failures:
        print("FAILURES:")
        for name, why in failures:
            print(f"  {name}: {why}")
        sys.exit(1)
    print("all files passed the absence-gap sanity check")


if __name__ == "__main__":
    main()
