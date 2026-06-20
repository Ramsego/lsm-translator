"""
Weak-label a video: produce (sign-clip → Spanish word) seeds from the matched triple
(timed transcript + lag + landmarks), using transcript-constrained DTW spotting.

For each aligned word that is ALSO in the Phase-1 word bank, place a generous
clause-anchored window at `word_time + lag` in the landmark timeline, run DTW vs that
word's bank reference (candidate set = the single word — transcript constraint), and
emit a confidence-scored label. Precision over recall: keep only strong matches.

NOTE (test bed caveat): on the 10fps 116-schema AMLO clip there is no dense mouth, and
the bank signer ≠ the interpreter, so DTW is the WEAK cue → expect noisy output. The
point of this run is the go/no-go eyeball, not clean labels. Mouthing-primary
verification comes online on the 30fps 124-schema Sheinbaum triple.

Usage:
    python phase2/weak_label.py --arrays <dir> --aligned <aligned.json> [--lag 2.6] [--topk 30]
"""

import argparse
import json
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "spotter"))
import classify as spot          # featurize, _dist, load_bank, mirror_array, DEFAULT_CFG

CFG = spot.DEFAULT_CFG


def norm(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


# Function/deictic words are rarely signed as discrete signs → drop as candidates.
STOPWORDS = {norm(w) for w in (
    "el la los las un una unos unas lo al del de a en y o u que qué como cómo cuando cuándo "
    "donde dónde quien quién cual cuál cuanto cuánto porque por para con sin sobre entre "
    "yo tú tu te me mi mí nos nosotros ellos ella él ellas su sus se si sí no ni es son "
    "ser estar haber este esta esto ese esa eso aquel aquí ahí allá muy más menos ya "
    "pero también cosa cosas").split()}


def load_landmark_timeline(arrays_dir: Path):
    """Return (frames[N,R,3], proc_fps) — full video on one array, with a time→index map.
    Concatenates segments back onto a global timeline (gaps filled with NaN rows)."""
    meta = json.load(open(arrays_dir / "segments.json"))
    proc_fps = meta["fps"] / meta["every"]
    end_t = max(s["end_sec"] for s in meta["segments"])
    n = int(round(end_t * proc_fps)) + 2
    # infer row count from first clip
    first = arrays_dir / meta["segments"][0]["file"]
    R = (np.load(first)["landmarks"] if first.suffix == ".npz" else np.load(first)).shape[1]
    timeline = np.full((n, R, 3), np.nan, dtype=np.float32)
    for s in meta["segments"]:
        p = arrays_dir / s["file"]
        arr = np.load(p)["landmarks"] if p.suffix == ".npz" else np.load(p)
        start = int(round(s["start_sec"] * proc_fps))
        timeline[start:start + len(arr)] = arr
    return timeline, proc_fps


def build_word_refs(bank, wanted_norm):
    """Map normalized word -> list of bank ref feature dicts (the candidate set)."""
    refs = {}
    for r in bank:
        k = norm(r["label"])
        if k in wanted_norm:
            refs.setdefault(k, []).append(r)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--lag", type=float, default=None, help="Override; else read lag_estimate.json.")
    ap.add_argument("--win", type=float, default=4.0, help="Half-window (s) around lag center.")
    ap.add_argument("--topk", type=int, default=30, help="Report this many best matches.")
    args = ap.parse_args()

    if args.lag is None:
        le = json.load(open(args.arrays / "lag_estimate.json"))
        args.lag = le.get("lag_sec") or le.get("peak_lag_sec") or 2.6

    aligned = json.load(open(args.aligned))["words"]
    timeline, proc_fps = load_landmark_timeline(args.arrays)

    print("Loading word bank...")
    bank = spot.load_bank(CFG)
    wanted = {norm(w["word"]) for w in aligned} - STOPWORDS
    refs = build_word_refs(bank, wanted)
    print(f"Content bank-words present in transcript (stopwords dropped): {len(refs)}")

    results = []
    for w in aligned:
        k = norm(w["word"])
        if k in STOPWORDS or k not in refs:
            continue
        # generous clause-anchored window at word_end + lag
        t0 = w["end"] + args.lag - args.win
        t1 = w["end"] + args.lag + args.win
        i0, i1 = max(0, int(t0 * proc_fps)), min(len(timeline), int(t1 * proc_fps))
        if i1 - i0 < 5:
            continue
        clip = timeline[i0:i1]
        qf = spot.featurize(clip, CFG)
        qfm = spot.featurize(spot.mirror_array(clip), CFG)
        if len(qf) == 0:
            continue
        # transcript-constrained: distance to THIS word's bank ref(s) only
        best = min(min(spot._dist(qf, r["feat"]), spot._dist(qfm, r["feat"]))
                   for r in refs[k])
        results.append({"word": w["word"], "speaker": w["speaker"],
                        "word_time": round(w["end"], 2),
                        "window": [round(t0, 2), round(t1, 2)],
                        "dtw_distance": round(float(best), 3)})

    results.sort(key=lambda r: r["dtw_distance"])     # lower = better match
    print(f"\nScored {len(results)} candidate (window→bank-word) pairs.")
    print(f"\nTop {args.topk} by DTW (lower=better) — EYEBALL these against the video:")
    print(f"{'word':18s} {'spoken@':>8s}  {'search window':>16s}  dtw")
    for r in results[:args.topk]:
        print(f"{r['word']:18s} {r['word_time']:7.1f}s  "
              f"[{r['window'][0]:6.1f},{r['window'][1]:6.1f}]  {r['dtw_distance']:.3f}")

    out = args.arrays / "weak_labels.json"
    out.write_text(json.dumps({"lag": args.lag, "win": args.win,
                               "n": len(results), "labels": results},
                              ensure_ascii=False, indent=2))
    print(f"\nSaved → {out}")
    print("\nGo/no-go: open the video at the listed times+window and check whether the "
          "interpreter is signing that word. A few correct top hits validate the backbone.")


if __name__ == "__main__":
    main()
