"""
Estimate the interpreter's lag behind the speaker — WITHOUT recognition.

The interpreter signs ~lag seconds AFTER the speaker talks. So cross-correlate two
1-D activity series on a common time axis:
  - interpreter MOTION energy: per-frame hand-landmark movement (rows 0-41), from the
    extracted .npy segments (signer/vocabulary-independent, ~99% hand coverage).
  - speaker SPEECH activity: ASR word density over time.
The shift maximizing correlation = central lag; the peak WIDTH = spread (consecutive/
batched interpreting → broad peak), which sizes the search windows downstream.

Usage:
    python phase2/estimate_lag.py --arrays <dir-with-segments.json> --asr <asr.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np

HAND_ROWS = slice(0, 42)
DT = 0.25                 # common time-grid bin (s)
LAG_MIN, LAG_MAX = -2.0, 16.0   # search range (s); interpreter lags → positive


def motion_series(arrays_dir: Path, dt: float):
    """Build interpreter hand-motion energy on a global time grid from .npy/.npz segments."""
    meta = json.load(open(arrays_dir / "segments.json"))
    fps = meta["fps"]; every = meta["every"]
    proc_fps = fps / every
    end_t = max(s["end_sec"] for s in meta["segments"])
    grid = np.zeros(int(np.ceil(end_t / dt)) + 1, dtype=np.float64)
    counts = np.zeros_like(grid)

    for seg in meta["segments"]:
        p = arrays_dir / seg["file"]
        arr = np.load(p)["landmarks"] if p.suffix == ".npz" else np.load(p)
        hands = arr[:, HAND_ROWS, :2]                      # [T,42,2]
        # per-frame motion = mean over hand points of |Δ position|; NaN-safe
        d = np.linalg.norm(np.diff(hands, axis=0), axis=2)  # [T-1,42]
        frame_motion = np.nanmean(d, axis=1)                # [T-1]
        frame_motion = np.nan_to_num(frame_motion, nan=0.0)
        for i, m in enumerate(frame_motion):
            t = seg["start_sec"] + i / proc_fps
            b = int(t / dt)
            if 0 <= b < len(grid):
                grid[b] += m; counts[b] += 1
    grid[counts > 0] /= counts[counts > 0]
    return grid


def speech_series(asr_json: Path, n_bins: int, dt: float):
    """Speaker speech activity: fraction of each time bin covered by spoken words."""
    words = json.load(open(asr_json))["words"]
    grid = np.zeros(n_bins, dtype=np.float64)
    for w in words:
        b0, b1 = int(w["start"] / dt), int(w["end"] / dt)
        for b in range(b0, b1 + 1):
            if 0 <= b < n_bins:
                grid[b] += 1.0
    return np.clip(grid, 0, 1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


def cross_correlate(speech, motion, dt):
    """corr(lag)=Σ speech(t)·motion(t+lag). Peak lag>0 ⇒ interpreter signs after speech."""
    speech, motion = _z(speech), _z(motion)
    lags = np.arange(int(LAG_MIN / dt), int(LAG_MAX / dt) + 1)
    corr = np.zeros(len(lags))
    n = len(speech)
    for i, L in enumerate(lags):
        if L >= 0:
            a, b = speech[:n - L], motion[L:]
        else:
            a, b = speech[-L:], motion[:n + L]
        corr[i] = np.dot(a, b) / len(a) if len(a) else 0.0
    return lags * dt, corr


def pause_events(asr_json: Path, min_gap=0.6):
    """Clause-boundary events = time the speaker STOPS before a gap > min_gap (s)."""
    words = json.load(open(asr_json))["words"]
    ev = []
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] > min_gap:
            ev.append(a["end"])           # speaker stopped here → clause end
    return np.array(ev)


def motion_onsets(motion, dt, smooth=4, min_lull=0.6):
    """Burst onsets that follow a genuine LULL — the interpreter paused, then started a NEW
    sign burst. Requiring a preceding lull avoids matching the tail of the PREVIOUS clause's
    signing (the bias that under-estimated the lag at 2.6s instead of the true ~6-7s).

    A lull = the smoothed motion stayed below a low threshold for >= min_lull seconds just
    before the rising crossing above the high threshold.
    """
    k = np.ones(smooth) / smooth
    sm = np.convolve(motion, k, mode="same")
    hi = sm.mean() + 0.5 * sm.std()
    lo = sm.mean() - 0.2 * sm.std()
    above = sm > hi
    rises = np.where(~above[:-1] & above[1:])[0] + 1
    lull_frames = max(1, int(min_lull / dt))
    onsets = []
    for r in rises:
        pre = sm[max(0, r - lull_frames):r]
        if len(pre) and np.all(pre < lo):      # genuinely paused before this burst
            onsets.append(r)
    return np.array(onsets) * dt


def event_lag(speech_ev, motion_ev, max_lag=15.0):
    """For each speech event, delay to nearest FOLLOWING motion onset (≤max_lag).
    Returns (median lag, IQR/2 spread, n matched)."""
    delays = []
    for t in speech_ev:
        future = motion_ev[(motion_ev >= t) & (motion_ev <= t + max_lag)]
        if len(future):
            delays.append(future[0] - t)
    if not delays:
        return None, None, 0
    d = np.array(delays)
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    return float(med), float((q3 - q1) / 2), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True, help="Dir with segments.json + clips.")
    ap.add_argument("--asr", type=Path, required=True, help="ASR word-timestamps JSON.")
    ap.add_argument("--method", choices=["events", "xcorr"], default="events",
                    help="events = pause→motion-onset delay (robust); xcorr = continuous baseline.")
    args = ap.parse_args()

    motion = motion_series(args.arrays, DT)

    if args.method == "events":
        sp_ev = pause_events(args.asr)
        mo_ev = motion_onsets(motion, DT)
        med, spread, n = event_lag(sp_ev, mo_ev)
        print(f"\nEvent-based lag: {med:+.2f}s  (spread ±{spread:.1f}s)  "
              f"from {n}/{len(sp_ev)} clause-ends matched to {len(mo_ev)} motion onsets")
        print("(positive = interpreter signs AFTER the speaker stops — expected)")
        (args.arrays / "lag_estimate.json").write_text(json.dumps({
            "method": "events", "lag_sec": round(med, 2) if med else None,
            "spread_sec": round(spread, 2) if spread else None,
            "n_matched": n, "n_speech_events": len(sp_ev), "n_motion_onsets": len(mo_ev),
        }, indent=2))
        print(f"\nSaved → {args.arrays / 'lag_estimate.json'}")
        return

    speech = speech_series(args.asr, len(motion), DT)
    lags, corr = cross_correlate(speech, motion, DT)

    peak_i = int(np.argmax(corr))
    peak_lag = lags[peak_i]
    peak_val = corr[peak_i]
    # spread = width where corr stays above half-max-above-baseline
    base = corr.min()
    half = base + 0.5 * (peak_val - base)
    above = np.where(corr >= half)[0]
    spread = (lags[above[-1]] - lags[above[0]]) / 2 if len(above) else 0.0

    print(f"\nPeak lag: {peak_lag:+.2f}s   corr={peak_val:.3f}   spread≈±{spread:.1f}s")
    print(f"(positive lag = interpreter signs AFTER the speaker — expected)\n")
    print("lag(s)  corr   " + "(bar = corr above baseline)")
    for L, c in zip(lags, corr):
        if abs(L - round(L)) < 1e-6:                 # print at 1s ticks
            bar = "#" * int(40 * (c - base) / (peak_val - base + 1e-9))
            mark = "  <-- peak" if L == peak_lag else ""
            print(f"{L:+5.0f}  {c:+.3f}  {bar}{mark}")

    out = args.arrays / "lag_estimate.json"
    out.write_text(json.dumps({
        "peak_lag_sec": round(float(peak_lag), 2),
        "peak_corr": round(float(peak_val), 3),
        "spread_sec": round(float(spread), 2),
        "dt": DT,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
