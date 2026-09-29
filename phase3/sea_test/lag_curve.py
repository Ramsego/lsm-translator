"""
Within-video interpreter-lag curve — "plot your statistics" (Zifan Jiang, 2026-08).

THE QUESTION
------------
`estimate_lag_v2.py` returns one median per video.  The spread around that median is
~3.5s and survives the detector fix, so it is real rather than an artifact.  This script
asks what SHAPE that spread has, because the answer decides the alignment model:

  bounded oscillation, no trend  -> a scalar bias is defensible; SEA's DP absorbs the rest
  trend within passages, resets  -> piecewise-LINEAR warping (Curiel et al. 2026 §4.1)
  structure tracks onset density -> we are measuring the detector, not the interpreter

DESIGN NOTES
------------
* Windows are indexed by EVENT COUNT, not by time.  A fixed 10-minute block lands in
  event-poor stretches and reports a confident number built on three matches; an
  N-event window carries constant statistical weight and makes event-poverty visible
  as gaps on the x-axis instead.

* `event_lag` takes the nearest FOLLOWING onset, so a MISSED onset silently reports the
  next one and inflates the delay.  Missed onsets cluster in dense passages — exactly
  where genuine drift would also appear.  The two are indistinguishable without a
  control, so local onset density is recorded per point and correlated against delay.
  If delay tracks sparsity, the curve is measuring the detector.

* The verified signs are plotted as an independent check with NO detector in the loop.
  They measure a DIFFERENT quantity (spoken word -> that word's sign) than the estimator
  (clause end -> next signing burst), so expect a level offset; it is the SHAPE that is
  being compared, not the absolute value.

Usage:
    python phase3/sea_test/lag_curve.py \
        --arrays "/Volumes/Crucial X8/LSM_Translator/arrays/mananera/57TvyH9902U" \
        --asr    phase3/local_drive_mirror/asr/57TvyH9902U.asr.json \
        --verified phase3/probe/review_2.csv \
        --out    phase3/sea_test/out/lag_curve/57TvyH9902U.png
"""

import argparse
import csv
import json
import unicodedata
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimate_lag_v2 import (DT, MAX_MATCH_LAG, motion_series, motion_onsets,
                             pause_events, _covered_intervals, _uncovered_between,
                             GAP_TOLERANCE)

DENSITY_WIN = 60.0   # seconds either side, for the local onset-density control


def _strip(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def matched_pairs(speech_ev, motion_ev, covered):
    """(speech_event_time, delay, local onset density /min) for every match."""
    motion_ev = np.sort(motion_ev)
    out = []
    for t in np.sort(speech_ev):
        lo = np.searchsorted(motion_ev, t, side="left")
        hi = np.searchsorted(motion_ev, t + MAX_MATCH_LAG, side="right")
        if hi <= lo:
            continue
        o = motion_ev[lo]
        if covered is not None and _uncovered_between(t, o, covered) > GAP_TOLERANCE:
            continue
        near = np.sum((motion_ev >= t - DENSITY_WIN) & (motion_ev <= t + DENSITY_WIN))
        out.append((float(t), float(o - t), near / (2 * DENSITY_WIN / 60.0)))
    return out


def verified_points(csv_path, asr_words):
    """(spoken word time, sign_start - word_end) for human-verified signs. No detector.

    Matching uses a 4-character prefix, which is cruder than the FORMS map in
    check_alignment_v2.py.  Adequate for a shape check; do NOT quote these numbers.
    """
    pts = []
    for r in csv.DictReader(open(csv_path)):
        if r["verdict"] != "y" or not r["sign_start"]:
            continue
        vs, ss = float(r["video_start"]), float(r["sign_start"])
        w = _strip(r["word"])[:4]
        cand = [a for a in asr_words
                if _strip(a["word"]).startswith(w) and vs - 20 <= a["end"] <= vs + 20]
        if not cand:
            continue
        m = min(cand, key=lambda a: abs(a["end"] - (vs - 2.33)))
        pts.append((m["end"], ss - m["end"]))
    return pts


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--asr", type=Path, required=True)
    ap.add_argument("--verified", type=Path, default=None)
    ap.add_argument("--window", type=int, default=20, help="events per sliding window")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    words = json.load(open(args.asr))["words"]
    motion = motion_series(args.arrays, DT)
    onsets = motion_onsets(motion, DT)
    speech = pause_events(words)
    seg = args.arrays / "segments.json"
    covered = _covered_intervals(seg) if seg.exists() else None

    pairs = matched_pairs(speech, onsets, covered)
    t = np.array([p[0] for p in pairs])
    lag = np.array([p[1] for p in pairs])
    dens = np.array([p[2] for p in pairs])

    # event-indexed sliding window
    W = args.window
    mid, med, iqr, dmed = [], [], [], []
    for i in range(0, len(pairs) - W + 1):
        sl = slice(i, i + W)
        mid.append(np.median(t[sl])); med.append(np.median(lag[sl]))
        q1, q3 = np.percentile(lag[sl], [25, 75]); iqr.append(q3 - q1)
        dmed.append(np.median(dens[sl]))
    mid, med, dmed = np.array(mid), np.array(med), np.array(dmed)

    rho_time = spearman(t, lag)
    rho_dens = spearman(dens, lag)

    print(f"\nmatches {len(pairs)} | onsets {len(onsets)} | global median lag {np.median(lag):.2f}s "
          f"(IQR/2 {(np.percentile(lag,75)-np.percentile(lag,25))/2:.2f})")
    print(f"{'decile':<9}{'t_mid_min':>11}{'n':>5}{'median_lag':>12}{'onsets/min':>12}")
    print("-" * 49)
    for k in range(10):
        s = slice(k * len(pairs) // 10, (k + 1) * len(pairs) // 10)
        if not len(lag[s]):
            continue
        print(f"{k+1:<9}{np.median(t[s])/60:>11.1f}{len(lag[s]):>5}"
              f"{np.median(lag[s]):>12.2f}{np.median(dens[s]):>12.2f}")
    print("-" * 49)
    print(f"Spearman(lag, time)          = {rho_time:+.3f}   <- drift?")
    print(f"Spearman(lag, onset density) = {rho_dens:+.3f}   <- detector confound?")
    print("\nPRE-REGISTERED READ")
    print("  |rho_time| < 0.2 and |rho_dens| < 0.2  -> regular; a scalar bias suffices")
    print("  |rho_time| >= 0.2, |rho_dens| < 0.2    -> real drift; piecewise-linear warp")
    print("  |rho_dens| >= |rho_time|               -> measuring the detector; fix it first")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.scatter(t / 60, lag, s=8, alpha=.22, color="#4C72B0", label="matched clause-end → onset")
    ax.plot(mid / 60, med, lw=2.2, color="#C44E52", label=f"median, {W}-event window")
    ax.axhline(np.median(lag), ls="--", lw=1, color="#555",
               label=f"global median {np.median(lag):.2f}s")

    if args.verified:
        vp = verified_points(args.verified, words)
        if vp:
            ax.scatter([p[0] / 60 for p in vp], [p[1] for p in vp], s=55, marker="D",
                       color="#DD8452", edgecolor="k", lw=.6, zorder=5,
                       label=f"verified signs, n={len(vp)} (word→sign; different quantity)")

    ax2 = ax.twinx()
    ax2.plot(mid / 60, dmed, lw=1, color="#55A868", alpha=.65)
    ax2.set_ylabel("onset density (onsets/min)", color="#55A868")
    ax2.tick_params(axis="y", labelcolor="#55A868")

    ax.set_xlabel("video time (min)")
    ax.set_ylabel("lag (s)")
    ax.set_title(f"{args.arrays.name} — within-video lag  "
                 f"(rho_time={rho_time:+.2f}, rho_density={rho_dens:+.2f})")
    ax.legend(loc="upper left", fontsize=8, framealpha=.9)
    ax.grid(alpha=.25)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=140)

    stats = {
        "video": args.arrays.name,
        "n_matches": len(pairs),
        "n_onsets": int(len(onsets)),
        "median_lag_s": round(float(np.median(lag)), 3),
        "spread_s": round(float((np.percentile(lag, 75) - np.percentile(lag, 25)) / 2), 3),
        "rho_lag_time": round(rho_time, 4),
        "rho_lag_density": round(rho_dens, 4),
        "window_events": W,
        "decile_medians": [round(float(np.median(lag[slice(k * len(pairs) // 10,
                                                           (k + 1) * len(pairs) // 10)])), 3)
                           for k in range(10) if len(lag[slice(k * len(pairs) // 10,
                                                               (k + 1) * len(pairs) // 10)])],
    }
    jout = args.out.with_suffix(".json")
    json.dump(stats, open(jout, "w"), indent=2)
    print(f"\nwrote {args.out} and {jout}")


if __name__ == "__main__":
    main()
