"""
DTW nearest-neighbor sign spotter (offline core).

Given a landmark clip ([frames, 116, 3]), find the closest reference sign(s) in the
word bank. Hands-only distance metric, body-relative normalization (via pose
shoulders), and handedness invariance via mirror-at-query (handedness.mirror_array).

Usage:
    python scripts/05_classify.py --npy data/arrays/<id>.npy   # top-5 for one clip
    python scripts/05_classify.py --eval                        # leave-one-out accuracy
    python scripts/05_classify.py --eval --no-normalize         # measure normalization effect
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from dtaidistance import dtw_ndim

sys.path.insert(0, str(Path(__file__).parent))
from handedness import mirror_array

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lsm.db"

# Array row layout
HAND_ROWS = slice(0, 42)      # 0-20 left hand, 21-41 right hand
L_SHOULDER = 42 + 11          # MediaPipe pose idx 11 -> array row 53
R_SHOULDER = 42 + 12          # MediaPipe pose idx 12 -> array row 54

DTW_WINDOW = 10               # Sakoe-Chiba band: limits warping, speeds up, reduces drift

# Prefer the compiled C backend; fall back to pure Python if it isn't available.
try:
    dtw_ndim.distance_fast(np.zeros((2, 2)), np.zeros((2, 2)), window=1)
    _DTW = dtw_ndim.distance_fast
    _DTW_BACKEND = "C (fast)"
except Exception:
    _DTW = dtw_ndim.distance
    _DTW_BACKEND = "pure-Python (slow; eval will take a while)"


def _clip_shoulder_stats(arr):
    """Per-clip fallback shoulder midpoint (x,y) and width, from non-NaN frames."""
    ls = arr[:, L_SHOULDER, :2]
    rs = arr[:, R_SHOULDER, :2]
    ok = ~(np.isnan(ls).any(1) | np.isnan(rs).any(1))
    if not ok.any():
        return np.array([0.5, 0.5], np.float64), 1.0
    mid = ((ls[ok] + rs[ok]) / 2).mean(0)
    width = np.linalg.norm((ls[ok] - rs[ok]), axis=1).mean()
    return mid.astype(np.float64), float(width if width > 1e-6 else 1.0)


def featurize(arr, normalize=True):
    """[frames,116,3] -> [T,84] hands-only (x,y) feature sequence.

    Keeps only frames with >=1 hand present. When normalize=True, each frame is
    re-centered on the shoulder midpoint and scaled by shoulder width so different
    signers / camera distances align. A missing hand's 21 points become the body
    center (0,0 after normalization).
    """
    hands = arr[:, HAND_ROWS, :2].astype(np.float64)     # [F,42,2]
    has_hand = ~np.isnan(hands).all(axis=(1, 2))          # [F]
    if not has_hand.any():
        return np.empty((0, 84), np.float64)

    fb_mid, fb_w = _clip_shoulder_stats(arr)

    feats = []
    for f in np.where(has_hand)[0]:
        frame = hands[f].copy()                            # [42,2]
        if normalize:
            ls = arr[f, L_SHOULDER, :2]
            rs = arr[f, R_SHOULDER, :2]
            if np.isnan(ls).any() or np.isnan(rs).any():
                mid, w = fb_mid, fb_w
            else:
                mid = (ls + rs) / 2
                w = np.linalg.norm(ls - rs)
                w = float(w if w > 1e-6 else fb_w)
            frame = (frame - mid) / w
        # Absent landmarks (NaN) -> body center (0 if normalized, else fallback mid)
        fill = 0.0 if normalize else fb_mid
        frame = np.where(np.isnan(frame), fill, frame)
        feats.append(frame.reshape(-1))                    # 84-vec
    return np.ascontiguousarray(feats, dtype=np.float64)


def _dist(qf, rf):
    if len(qf) == 0 or len(rf) == 0:
        return np.inf
    return _DTW(qf, rf, window=DTW_WINDOW)


def load_bank(normalize=True):
    """Return list of dicts: youtube_id, label, source, feat, feat_mirror."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT youtube_id, label, source, array_path FROM videos"
    ).fetchall()
    conn.close()

    bank = []
    for yid, label, source, path in rows:
        arr = np.load(DATA_DIR / path)
        bank.append({
            "youtube_id": yid,
            "label": label,
            "source": source,
            "feat": featurize(arr, normalize),
            "feat_mirror": featurize(mirror_array(arr), normalize),
        })
    return bank


def classify(query_feat, query_feat_mirror, bank, k=5, exclude_id=None):
    """Return top-k (label, distance, source, youtube_id), handedness-invariant."""
    scored = []
    for ref in bank:
        if exclude_id is not None and ref["youtube_id"] == exclude_id:
            continue
        d = min(_dist(query_feat, ref["feat"]),
                _dist(query_feat_mirror, ref["feat"]))
        scored.append((d, ref["label"], ref["source"], ref["youtube_id"]))
    scored.sort(key=lambda t: t[0])
    return [(lbl, d, src, yid) for d, lbl, src, yid in scored[:k]]


def run_single(npy_path, normalize):
    arr = np.load(npy_path)
    bank = load_bank(normalize)
    qf = featurize(arr, normalize)
    qfm = featurize(mirror_array(arr), normalize)
    top = classify(qf, qfm, bank, k=5)
    print(f"\nTop-5 matches for {Path(npy_path).name}:")
    for i, (lbl, d, src, yid) in enumerate(top, 1):
        print(f"  {i}. {lbl:25s}  dist={d:8.3f}  [{src}] {yid}")


def run_eval(normalize):
    bank = load_bank(normalize)
    label_counts = Counter(r["label"] for r in bank)
    # labels present in both sources = different-signer generalization test
    src_by_label = defaultdict(set)
    for r in bank:
        src_by_label[r["label"]].add(r["source"])
    cross_labels = {l for l, s in src_by_label.items() if len(s) > 1}

    overall = {"top1": 0, "top5": 0, "n": 0}
    multi = {"top1": 0, "top5": 0, "n": 0}
    cross = {"top1": 0, "top5": 0, "n": 0}
    wrong = Counter()

    print(f"Evaluating {len(bank)} clips (normalize={normalize})... this takes a few minutes.")
    for i, q in enumerate(bank):
        top = classify(q["feat"], q["feat_mirror"], bank, k=5, exclude_id=q["youtube_id"])
        preds = [t[0] for t in top]
        t1 = preds[0] == q["label"]
        t5 = q["label"] in preds

        overall["n"] += 1
        overall["top1"] += t1; overall["top5"] += t5
        if not t1:
            wrong[(q["label"], preds[0])] += 1

        if label_counts[q["label"]] > 1:
            multi["n"] += 1; multi["top1"] += t1; multi["top5"] += t5
        if q["label"] in cross_labels:
            cross["n"] += 1; cross["top1"] += t1; cross["top5"] += t5

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(bank)}...")

    def pct(d, key):
        return 100 * d[key] / d["n"] if d["n"] else 0.0

    print("\n=== Leave-one-out accuracy ===")
    for name, d in [("Overall (all 963, many unmatchable)", overall),
                    ("Multi-example labels (fair within-bank)", multi),
                    ("Cross-source (different signers)", cross)]:
        print(f"  {name:42s}  n={d['n']:4d}  top1={pct(d,'top1'):5.1f}%  top5={pct(d,'top5'):5.1f}%")

    print("\n  Most common confusions (true -> guessed):")
    for (true, guess), n in wrong.most_common(10):
        print(f"    {true:22s} -> {guess:22s} ({n})")


def main():
    ap = argparse.ArgumentParser(description="DTW nearest-neighbor sign spotter.")
    ap.add_argument("--npy", type=Path, help="Classify a single .npy clip (top-5).")
    ap.add_argument("--eval", action="store_true", help="Leave-one-out accuracy eval.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable body-relative normalization.")
    args = ap.parse_args()
    normalize = not args.no_normalize

    if args.eval:
        run_eval(normalize)
    elif args.npy:
        run_single(args.npy, normalize)
    else:
        ap.error("pass --npy <file> or --eval")


if __name__ == "__main__":
    main()
