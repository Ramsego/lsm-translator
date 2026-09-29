"""
Interpreter-lag estimator, v2 — detector fixes only, same measurement.

WHY THIS EXISTS
---------------
`phase2/estimate_lag.py` produces a per-video lag by matching speaker clause-ends
to interpreter motion-burst onsets. Auditing its 12 per-video outputs surfaced a
length-dependent artifact:

    motion onsets do NOT scale with video length (124-271 onsets whether the video
    is 64 min or 198 min), while speech events do (131 -> 830).

In long videos speech events outnumber onsets ~3:1, so events are forced onto more
distant onsets and the median delay inflates.  corr(lag, speech/onset ratio) = +0.78.

Three defects cause it:

  D1. GLOBAL THRESHOLDS.  `motion_onsets` sets hi/lo from the whole video's
      mean/std.  A longer, more varied recording has a wider std, so `hi` rises,
      `lo` falls, and the "genuinely paused before this burst" test gets strictly
      harder to pass.  Detection saturates.  FIX: rolling local statistics.

  D2. MANY-TO-ONE MATCHING.  `event_lag` lets any number of speech events match the
      same onset, so a run of events all resolve to one distant burst.  FIX: each
      onset serves at most one speech event (nearest-first).

  D3. CROSS-GAP MATCHING.  Speech continues over B-roll/presentations where the
      interpreter is absent; the next onset is on the far side of that gap.
      FIX: reject matches spanning an uncovered stretch (needs segments.json).

CONFOUND THIS IS MEANT TO SETTLE
--------------------------------
Mean inter-word gap also correlates with lag (+0.76), and there is a real linguistic
mechanism: a speaker who pauses more delivers propositional content more slowly, so
the interpreter waits longer before it has enough to begin.  Longer gaps ALSO produce
more `pause_events`, which feeds the artifact above -- the two are mechanically
linked and cannot be separated by correlation at n=11.

After this fix, re-run all videos and re-test corr(lag, mean_inter_word_gap):
  - correlation SURVIVES  -> the pausing explanation is real
  - correlation COLLAPSES -> it was the detector all along

Writes a NEW json (never overwrites `lag_estimate.json`).  Baseline is preserved.

Usage:
    python phase3/sea_test/estimate_lag_v2.py \
        --arrays phase3/local_drive_mirror/arrays_57TvyH9902U \
        --asr    phase3/local_drive_mirror/asr/57TvyH9902U.asr.json \
        [--segments <segments.json>] [--out <path.json>]
"""

import argparse
import json
from pathlib import Path

import numpy as np

HAND_ROWS = slice(0, 42)
DT = 0.25                       # time-grid bin (s) -- unchanged from v1
LAG_MIN, LAG_MAX = -2.0, 16.0   # search range (s)
ROLL_WIN_SEC = 120.0            # local-threshold window (s); see D1
MIN_GAP = 0.6                   # speaker pause defining a clause end (s)
MIN_LULL = 0.6                  # required quiet before a burst counts as an onset (s)
MAX_MATCH_LAG = 15.0            # furthest a match may reach (s)
GAP_TOLERANCE = 2.0             # uncovered seconds that invalidate a match (D3)


# --------------------------------------------------------------------------- #
# signals (unchanged from v1)
# --------------------------------------------------------------------------- #

def motion_series(arrays_dir: Path, dt: float):
    """Interpreter hand-motion energy on a global time grid. Identical to v1."""
    meta = json.load(open(arrays_dir / "segments.json"))
    fps, every = meta["fps"], meta["every"]
    proc_fps = fps / every
    end_t = max(s["end_sec"] for s in meta["segments"])
    grid = np.zeros(int(np.ceil(end_t / dt)) + 1, dtype=np.float64)
    counts = np.zeros_like(grid)

    for seg in meta["segments"]:
        p = arrays_dir / seg["file"]
        arr = np.load(p)["landmarks"] if p.suffix == ".npz" else np.load(p)
        hands = arr[:, HAND_ROWS, :2]
        d = np.linalg.norm(np.diff(hands, axis=0), axis=2)
        frame_motion = np.nan_to_num(np.nanmean(d, axis=1), nan=0.0)
        for i, m in enumerate(frame_motion):
            b = int((seg["start_sec"] + i / proc_fps) / dt)
            if 0 <= b < len(grid):
                grid[b] += m
                counts[b] += 1
    grid[counts > 0] /= counts[counts > 0]
    return grid


def pause_events(words, min_gap=MIN_GAP):
    """Clause ends: the moment the speaker stops before a gap > min_gap."""
    return np.array([a["end"] for a, b in zip(words, words[1:])
                     if b["start"] - a["end"] > min_gap])


# --------------------------------------------------------------------------- #
# D1 -- rolling thresholds instead of global ones
# --------------------------------------------------------------------------- #

def _rolling_mean_std(x, win_bins):
    """Centred rolling mean and std via cumulative sums. Edges use what's available."""
    n = len(x)
    half = max(1, win_bins // 2)
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    lo_i = np.clip(np.arange(n) - half, 0, n)
    hi_i = np.clip(np.arange(n) + half + 1, 0, n)
    cnt = np.maximum(hi_i - lo_i, 1)
    mean = (c1[hi_i] - c1[lo_i]) / cnt
    var = np.maximum((c2[hi_i] - c2[lo_i]) / cnt - mean ** 2, 0.0)
    return mean, np.sqrt(var)


def motion_onsets(motion, dt, smooth=4, min_lull=MIN_LULL, roll_win_sec=ROLL_WIN_SEC):
    """Burst onsets following a genuine lull, using LOCAL thresholds (D1).

    v1 used sm.mean() +/- k*sm.std() over the entire video, which saturates on long
    recordings.  Here the same rule is applied against a rolling baseline, so the
    detector's sensitivity no longer depends on how long the video is.
    """
    sm = np.convolve(motion, np.ones(smooth) / smooth, mode="same")
    win_bins = max(3, int(roll_win_sec / dt))
    mu, sd = _rolling_mean_std(sm, win_bins)

    hi = mu + 0.5 * sd
    lo = mu - 0.2 * sd
    above = sm > hi
    rises = np.where(~above[:-1] & above[1:])[0] + 1

    lull_frames = max(1, int(min_lull / dt))
    onsets = [r for r in rises
              if len(sm[max(0, r - lull_frames):r])
              and np.all(sm[max(0, r - lull_frames):r] < lo[max(0, r - lull_frames):r])]
    return np.array(onsets) * dt


# --------------------------------------------------------------------------- #
# D2 + D3 -- one-to-one matching, no matches across non-signing gaps
# --------------------------------------------------------------------------- #

def _covered_intervals(segments_json: Path):
    """Merged [start, end] spans where the interpreter is validated present."""
    segs = sorted((s["start_sec"], s["end_sec"])
                  for s in json.load(open(segments_json))["segments"])
    merged = []
    for a, b in segs:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _uncovered_between(t0, t1, covered):
    """Seconds between t0 and t1 not inside any validated segment."""
    if t1 <= t0:
        return 0.0
    total = t1 - t0
    for a, b in covered:
        if b <= t0:
            continue
        if a >= t1:
            break
        total -= min(b, t1) - max(a, t0)
    return max(total, 0.0)


def event_lag(speech_ev, motion_ev, covered=None,
              max_lag=MAX_MATCH_LAG, gap_tol=GAP_TOLERANCE):
    """Delay from each clause end to the nearest FOLLOWING onset. Matches v1 exactly,
    except that D3 may reject a match that spans a non-signing gap.

    NOTE ON D2 (many-to-one), TESTED AND REJECTED 2026-08-06.
    The original plan was to consume onsets so each serves at most one speech event.
    Measured on 57TvyH9902U it moved the lag 6.33 -> 10.35s, i.e. the wrong way, and
    the reason is substantive rather than numerical: clause-ends outnumber motion
    onsets ~3:1 because the interpreter MERGES several spoken clauses into one signing
    burst.  One-to-one matching therefore forces later events to reach past the burst
    that legitimately serves them, manufacturing long delays.  Many-to-one is correct
    behaviour, not a defect.  Only D1 (local thresholds) and D3 (gap rejection) are
    applied.
    """
    motion_ev = np.sort(motion_ev)
    delays, rejected_gap, reuse = [], 0, {}

    for t in np.sort(speech_ev):
        lo = np.searchsorted(motion_ev, t, side="left")
        hi = np.searchsorted(motion_ev, t + max_lag, side="right")
        if hi <= lo:
            continue
        j = lo
        if covered is not None and _uncovered_between(t, motion_ev[j], covered) > gap_tol:
            rejected_gap += 1
            continue
        reuse[j] = reuse.get(j, 0) + 1
        delays.append(motion_ev[j] - t)

    if not delays:
        return None, None, 0, {"rejected_gap": rejected_gap, "max_reuse": 0, "mean_reuse": 0.0}
    d = np.array(delays)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return (float(med), float((q3 - q1) / 2), len(d),
            {"rejected_gap": rejected_gap,
             "max_reuse": max(reuse.values()),
             "mean_reuse": round(sum(reuse.values()) / len(reuse), 2),
             "delays": [round(float(x), 3) for x in d]})


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--asr", type=Path, required=True)
    ap.add_argument("--segments", type=Path, default=None,
                    help="segments.json for gap rejection (D3). Defaults to <arrays>/segments.json")
    ap.add_argument("--roll-win", type=float, default=ROLL_WIN_SEC)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    words = json.load(open(args.asr))["words"]
    motion = motion_series(args.arrays, DT)
    duration_min = len(motion) * DT / 60.0

    seg_path = args.segments or (args.arrays / "segments.json")
    covered = _covered_intervals(seg_path) if seg_path.exists() else None

    speech_ev = pause_events(words)
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])
            if 0 < b["start"] - a["end"] < 5]
    mean_gap = float(np.mean(gaps)) if gaps else None

    # --- v1 detector, for side-by-side comparison -------------------------- #
    sm = np.convolve(motion, np.ones(4) / 4, mode="same")
    hi_g, lo_g = sm.mean() + 0.5 * sm.std(), sm.mean() - 0.2 * sm.std()
    above = sm > hi_g
    rises = np.where(~above[:-1] & above[1:])[0] + 1
    lull = max(1, int(MIN_LULL / DT))
    old_onsets = np.array([r for r in rises
                           if len(sm[max(0, r - lull):r])
                           and np.all(sm[max(0, r - lull):r] < lo_g)]) * DT
    old = event_lag(speech_ev, old_onsets, covered=None)

    # --- v2 detector -------------------------------------------------------- #
    new_onsets = motion_onsets(motion, DT, roll_win_sec=args.roll_win)
    new = event_lag(speech_ev, new_onsets, covered=covered)

    def row(tag, onsets, res):
        lag, spread, n, meta = res
        print(f"{tag:<12}{len(onsets):>8}{len(onsets)/duration_min:>11.2f}"
              f"{(lag if lag else float('nan')):>8.2f}{(spread if spread else float('nan')):>8.2f}"
              f"{n:>9}{meta['mean_reuse']:>8.2f}{meta['max_reuse']:>6}")

    print(f"\nvideo duration {duration_min:.1f} min | speech events {len(speech_ev)}"
          f" ({len(speech_ev)/duration_min:.2f}/min) | mean inter-word gap {mean_gap:.3f}s")
    print(f"{'detector':<12}{'onsets':>8}{'onsets/min':>11}{'lag':>8}{'spread':>8}"
          f"{'matched':>9}{'reuse':>8}{'max':>6}")
    print("-" * 68)
    row("v1 global", old_onsets, old)
    row("v2 rolling", new_onsets, new)
    print(f"\nmatches rejected for spanning a non-signing gap (D3): {new[3]['rejected_gap']}")
    print("(reuse = mean speech events sharing one onset; v1 baseline should reproduce 6.33)")

    out = args.out or (args.arrays / "lag_estimate_v2.json")
    payload = {
        "method": "events_v2",
        "roll_win_sec": args.roll_win,
        "duration_min": round(duration_min, 2),
        "mean_inter_word_gap_s": round(mean_gap, 4) if mean_gap else None,
        "n_speech_events": int(len(speech_ev)),
        "v1": {"lag_sec": old[0], "spread_sec": old[1], "n_matched": old[2],
               "n_motion_onsets": int(len(old_onsets))},
        "v2": {"lag_sec": new[0], "spread_sec": new[1], "n_matched": new[2],
               "n_motion_onsets": int(len(new_onsets)),
               "rejected_cross_gap": new[3]["rejected_gap"],
               "delays": new[3].get("delays", [])},
    }
    json.dump(payload, open(out, "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
