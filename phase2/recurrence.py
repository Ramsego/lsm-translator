"""
Recurrence (multiple-instance) verification — signer-independent, no word bank.

A frequent word (e.g. 'méxico' ×17) has many candidate windows; the SIGN for it
RECURS across them while the surrounding signs vary. So if a word has a real
consistent sign, its windows are more similar to EACH OTHER than to random
windows of OTHER words. We measure exactly that contrast.

Per word:
  - within_med  = median pairwise DTW among its own windows
  - cross_med   = median DTW to windows of other words (baseline)
  - contrast    = cross_med / within_med   (>1 ⇒ recurrence detected; higher = stronger)
Per window:
  - recurrence score = mean DTW to its k nearest same-word windows
    (low = sits in the recurring cluster = high-confidence exemplar)

This is the cold-start label engine: high-contrast words with low-score windows are
confident (sign→word) seeds, with NO reference signer needed.

Usage:
    python phase2/recurrence.py --arrays <dir> --aligned <aligned.json>
        [--words ...] [--min-instances 4] [--win 4]
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "spotter"))
import classify as spot

CFG = spot.DEFAULT_CFG


def norm(w):
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


STOPWORDS = {norm(w) for w in (
    "el la los las un una unos unas lo al del de a en y o u que como cuando donde quien "
    "cual cuanto porque por para con sin sobre entre yo tu te me mi nos nosotros ellos "
    "ella ellas su sus se si no ni es son ser estar haber este esta esto ese esa eso "
    "aquel aqui ahi alla muy mas menos ya pero tambien cosa cosas entonces todos todo "
    "otra tienen tenemos estan estaba hace desde").split()}


def load_timeline(arrays_dir):
    meta = json.load(open(arrays_dir / "segments.json"))
    proc_fps = meta["fps"] / meta["every"]
    end_t = max(s["end_sec"] for s in meta["segments"])
    first = arrays_dir / meta["segments"][0]["file"]
    R = (np.load(first)["landmarks"] if first.suffix == ".npz" else np.load(first)).shape[1]
    tl = np.full((int(end_t * proc_fps) + 2, R, 3), np.nan, dtype=np.float32)
    for s in meta["segments"]:
        p = arrays_dir / s["file"]
        arr = np.load(p)["landmarks"] if p.suffix == ".npz" else np.load(p)
        st = int(round(s["start_sec"] * proc_fps))
        tl[st:st + len(arr)] = arr
    return tl, proc_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--words", nargs="*", default=None)
    ap.add_argument("--min-instances", type=int, default=4)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--lag", type=float, default=None)
    args = ap.parse_args()

    if args.lag is None:
        args.lag = json.load(open(args.arrays / "lag_estimate.json")).get("lag_sec", 6.0)

    tl, proc_fps = load_timeline(args.arrays)
    aligned = json.load(open(args.aligned))["words"]

    targets = {norm(w) for w in args.words} if args.words else None
    occ = defaultdict(list)
    for w in aligned:
        k = norm(w["word"])
        if len(k) < 4 or k in STOPWORDS:
            continue
        if targets is None or k in targets:
            occ[k].append(w["end"])

    # featurize every window once, grouped by word
    feats = {}
    for k, ends in occ.items():
        if len(ends) < args.min_instances:
            continue
        fs = []
        for e in ends:
            i0 = max(0, int((e + args.lag - args.win) * proc_fps))
            i1 = min(len(tl), int((e + args.lag + args.win) * proc_fps))
            f = spot.featurize(tl[i0:i1], CFG)
            if len(f) >= 5:
                fs.append(f)
        if len(fs) >= args.min_instances:
            feats[k] = fs

    if not feats:
        print("No words with enough instances. Try --min-instances lower.")
        return

    # cross-word baseline pool: one feat per word
    pool = [(k, fs[i]) for k, fs in feats.items() for i in range(len(fs))]

    def med_pairwise(a, b=None):
        ds = []
        if b is None:
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    ds.append(spot._dist(a[i], a[j]))
        else:
            for x in a:
                for y in b:
                    ds.append(spot._dist(x, y))
        return float(np.median(ds)) if ds else np.inf

    print(f"{'word':14s} {'n':>3s}  {'within':>7s} {'cross':>7s} {'contrast':>8s}")
    results = {}
    for k, fs in sorted(feats.items(), key=lambda kv: -len(kv[1])):
        within = med_pairwise(fs)
        # baseline: distances from this word's windows to other words' windows
        others = [f for (kk, f) in pool if kk != k]
        rng = np.random.default_rng(0)
        sample = [others[i] for i in rng.choice(len(others), min(40, len(others)), replace=False)]
        cross = med_pairwise(fs, sample)
        contrast = cross / within if within > 0 else 0
        # per-window recurrence score = mean dist to k=2 nearest same-word windows
        scores = []
        for i, fi in enumerate(fs):
            d = sorted(spot._dist(fi, fj) for j, fj in enumerate(fs) if j != i)
            scores.append(float(np.mean(d[:2])) if d else np.inf)
        results[k] = {"n": len(fs), "within_med": round(within, 2),
                      "cross_med": round(cross, 2), "contrast": round(contrast, 2),
                      "window_scores": [round(s, 2) for s in scores]}
        flag = "  <-- recurrence" if contrast >= 1.25 else ""
        print(f"{k:14s} {len(fs):3d}  {within:7.2f} {cross:7.2f} {contrast:8.2f}{flag}")

    out = args.arrays / "recurrence.json"
    out.write_text(json.dumps({"lag": args.lag, "win": args.win, "words": results}, indent=2))
    print(f"\ncontrast > 1 ⇒ a word's windows are more alike than random ⇒ a consistent sign recurs.")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
