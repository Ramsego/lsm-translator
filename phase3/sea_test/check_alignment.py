"""
Judge whether SEA's alignment on our LSM data is meaningful — without needing to read LSM.

SEA takes subtitles timed to the SPEAKER's audio and re-times them to the INTERPRETER's
signing. We have an independent, empirical measurement of that offset: the interpreter lag,
measured per-video by phase2/estimate_lag.py (57TvyH9902U: 6.33s, spread 3.48s).

So the test is: how far did SEA move each subtitle, and does that shift look like the lag?

  - Shift clustered near the measured lag  -> SEA is finding the signing. WORKING.
  - Shift ~0 everywhere                    -> SEA just copied audio timings. NO-OP.
  - Shift random / huge spread             -> alignment is noise. BROKEN.

Second, independent check: for clips we hand-verified (word W really is signed at time T),
does the re-timed subtitle containing W now cover T? Uses phase3/probe/review.csv, so it
needs no new labeling and no LSM knowledge.

Usage:
    python phase3/sea_test/check_alignment.py \
        --original phase3/sea_test/subtitles/57TvyH9902U_slice.vtt \
        --aligned  phase3/sea_test/out/aligned_subtitles/57TvyH9902U_slice.vtt \
        --lag 6.33 --slice-start 1200 --video-id 57TvyH9902U
"""
import argparse
import csv
import re
import statistics as stats
from pathlib import Path


def parse_vtt(path):
    """-> [(start_sec, end_sec, text)] ; tolerant of WEBVTT headers/numbering."""
    cues, cur = [], None
    tpat = re.compile(
        r"(\d\d):(\d\d):(\d\d(?:\.\d+)?)\s*-->\s*(\d\d):(\d\d):(\d\d(?:\.\d+)?)"
    )
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = tpat.search(line)
        if m:
            h1, m1, s1, h2, m2, s2 = m.groups()
            st = int(h1) * 3600 + int(m1) * 60 + float(s1)
            en = int(h2) * 3600 + int(m2) * 60 + float(s2)
            cur = [st, en, []]
            cues.append(cur)
        elif cur is not None and line.strip() and line.strip() != "WEBVTT":
            cur[2].append(line.strip())
    return [(c[0], c[1], " ".join(c[2])) for c in cues if c[2]]


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--lag", type=float, required=True, help="measured interpreter lag (s)")
    ap.add_argument("--slice-start", type=float, default=0.0)
    ap.add_argument("--video-id", default=None)
    ap.add_argument("--review-csv", default="phase3/probe/review.csv")
    ap.add_argument("--interp-manifest", default="phase3/probe/clips_interpreter/manifest.csv")
    args = ap.parse_args()

    orig = parse_vtt(args.original)
    algn = parse_vtt(args.aligned)
    print(f"original cues: {len(orig)}   aligned cues: {len(algn)}")

    by_text = {}
    for st, en, tx in algn:
        by_text.setdefault(norm(tx), []).append((st, en))

    shifts, matched = [], 0
    for st, en, tx in orig:
        cands = by_text.get(norm(tx))
        if not cands:
            continue
        ast, _ = min(cands, key=lambda p: abs(p[0] - st))
        shifts.append(ast - st)
        matched += 1

    print(f"matched by text: {matched}/{len(orig)}")
    if not shifts:
        print("\n!! Could not match cues by text — inspect the aligned file format manually.")
        return

    med = stats.median(shifts)
    mean = stats.fmean(shifts)
    sd = stats.pstdev(shifts) if len(shifts) > 1 else 0.0
    print("\n--- Shift applied by SEA (aligned - original), seconds ---")
    print(f"  median {med:+.2f}   mean {mean:+.2f}   sd {sd:.2f}")
    print(f"  measured interpreter lag for this video: +{args.lag:.2f}")
    near_zero = sum(1 for s in shifts if abs(s) < 0.5) / len(shifts)
    print(f"  fraction of cues essentially unmoved (<0.5s): {near_zero:.0%}")

    print("\n  VERDICT:", end=" ")
    if near_zero > 0.8:
        print("NO-OP — SEA barely moved anything; it is not finding the signing.")
    elif abs(med - args.lag) < 2.5 and sd < 6:
        print("WORKING — shift clusters near the independently measured lag.")
    elif sd > 10:
        print("NOISY — shifts are scattered; alignment looks unreliable.")
    else:
        print(f"UNCLEAR — median {med:+.2f} vs expected +{args.lag:.2f}; inspect further.")

    # ---- second check: hand-verified sign timings ----
    # review.csv's "file" column (clips_interpreter/<clip_id>.mp4) carries no timestamp —
    # the clip's absolute offset in the source video lives in clips_interpreter/manifest.csv's
    # src_path (e.g. .../agua/agua_59m26s.mp4). Join on clip_id to recover it.
    rc = Path(args.review_csv)
    if not (rc.exists() and args.video_id):
        return
    rows = [
        r for r in csv.DictReader(open(rc))
        if r.get("verdict") == "y" and r.get("sign_start")
        and args.video_id in r.get("file", "")
    ]
    if not rows:
        return

    mf = Path(args.interp_manifest)
    clip_abs_by_id = {}
    if mf.exists():
        for r in csv.DictReader(open(mf)):
            m = re.search(r"_(\d+)m(\d+)s", r["src_path"])
            if m:
                clip_abs_by_id[r["clip_id"]] = int(m.group(1)) * 60 + int(m.group(2))

    slice_duration = max(en for _, en, _ in orig) if orig else None

    print(f"\n--- Verified-clip check ({len(rows)} clips for {args.video_id}) ---")
    print("Does the re-timed subtitle containing the word cover the moment you saw it signed?")
    hits, checked = 0, 0
    for r in rows:
        word = r["word"]
        video_start = float(r.get("video_start") or 0)
        if video_start > 0:
            # gold-pipeline format: video_start is already the clip's absolute offset in the
            # source video, and build_review_form.py's markStart/markEnd computed sign_start
            # as video_start + currentTime -> sign_start is ALREADY absolute. Don't re-add.
            sign_abs = float(r["sign_start"])
        else:
            # probe-pipeline format: video_start=="0" placeholder, sign_start is clip-relative;
            # recover the clip's real absolute offset via clips_interpreter/manifest.csv.
            clip_id = Path(r["file"]).stem
            clip_abs = clip_abs_by_id.get(clip_id)
            if clip_abs is None:
                print(f"  {word:10s} -> SKIP (no timestamp found for {clip_id} in {mf})")
                continue
            sign_abs = clip_abs + float(r["sign_start"])
        sign_local = sign_abs - args.slice_start
        if slice_duration is not None and not (0 <= sign_local <= slice_duration):
            print(f"  {word:10s} signed at {sign_local:7.1f}s (slice-local) -> SKIP (outside tested slice)")
            continue
        checked += 1
        covering = [
            (st, en, tx) for st, en, tx in algn
            if st - 1.0 <= sign_local <= en + 1.0 and word.lower() in tx.lower()
        ]
        ok = bool(covering)
        hits += ok
        print(f"  {word:10s} signed at {sign_local:7.1f}s (slice-local) -> {'HIT' if ok else 'miss'}")
    print(f"  {hits}/{checked} verified signs (in-slice) fall inside the re-timed subtitle containing that word")


if __name__ == "__main__":
    main()
