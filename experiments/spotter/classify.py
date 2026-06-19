"""
DTW nearest-neighbor sign spotter (offline core).

Given a landmark clip ([frames, 116, 3]), find the closest reference sign(s) in the
word bank. Hands-only distance metric, body-relative normalization (via pose
shoulders), and handedness invariance via mirror-at-query (handedness.mirror_array).

Run from the repo root (data paths are relative to CWD):
    python experiments/spotter/classify.py --npy data/arrays/<id>.npy  # top-5 for one clip
    python experiments/spotter/classify.py --eval                      # leave-one-out accuracy
    python experiments/spotter/classify.py --eval --no-normalize       # normalization effect
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from dtaidistance import dtw_ndim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from handedness import mirror_array

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "lsm.db"

# Array row layout
HAND_ROWS = slice(0, 42)      # 0-20 left hand, 21-41 right hand
L_SHOULDER = 42 + 11          # MediaPipe pose idx 11 -> array row 53
R_SHOULDER = 42 + 12          # MediaPipe pose idx 12 -> array row 54
# Arm landmarks (shoulders, elbows, wrists) carry where the hand is on the body.
POSE_ARM_ROWS = [42 + i for i in (11, 12, 13, 14, 15, 16)]  # rows 53-58

DTW_WINDOW = 10               # Sakoe-Chiba band: limits warping, speeds up, reduces drift

# A feature config. Default chosen via tune_features.py sweep: per-clip shoulder
# normalization + arm landmarks scored best (others were within noise; velocity /
# z-score hurt — they amplify MediaPipe jitter on short clips). Cross-signer 1-shot
# accuracy is inherently low regardless; this is the modest best.
DEFAULT_CFG = {
    "norm": "per_clip",    # 'per_frame' | 'per_clip' | 'none'
    "velocity": False,     # append frame-to-frame deltas to positions
    "vel_only": False,     # use deltas alone (position-invariant motion shape)
    "zscore": False,       # standardize each feature dim across the clip
    "include_pose": True,  # add arm landmarks (rows 53-58)
}

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


def featurize(arr, cfg=None):
    """[frames,116,3] -> [T, D] feature sequence per the config.

    Keeps only frames with >=1 hand present. Positions can be normalized into a
    body frame (shoulder midpoint / width), per-frame or once per clip. Optionally
    appends or replaces with velocity (frame-to-frame deltas), adds arm landmarks,
    and/or z-normalizes each feature dimension. Missing landmarks fill to the body
    center (0 when normalized, else the clip's fallback midpoint).
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    rows = list(range(0, 42)) + (POSE_ARM_ROWS if cfg["include_pose"] else [])

    hands = arr[:, HAND_ROWS, :2]
    has_hand = ~np.isnan(hands).all(axis=(1, 2))          # frames with >=1 hand
    kept = np.where(has_hand)[0]
    P = len(rows)
    if len(kept) == 0:
        return np.empty((0, P * 2), np.float64)

    fb_mid, fb_w = _clip_shoulder_stats(arr)
    pts = arr[np.ix_(kept, rows)][:, :, :2].astype(np.float64)  # [T,P,2]

    if cfg["norm"] == "none":
        fill = fb_mid
    else:
        if cfg["norm"] == "per_clip":
            mid = np.broadcast_to(fb_mid, (len(kept), 2)).copy()
            w = np.full(len(kept), fb_w)
        else:  # per_frame
            ls = arr[kept, L_SHOULDER, :2]
            rs = arr[kept, R_SHOULDER, :2]
            valid = ~(np.isnan(ls).any(1) | np.isnan(rs).any(1))
            mid = (ls + rs) / 2
            w = np.linalg.norm(ls - rs, axis=1)
            mid[~valid] = fb_mid
            w[~valid] = fb_w
            w[w <= 1e-6] = fb_w
        pts = (pts - mid[:, None, :]) / w[:, None, None]
        fill = np.array([0.0, 0.0])

    nan_mask = np.isnan(pts)
    pts[nan_mask] = np.broadcast_to(fill, pts.shape)[nan_mask]

    pos = pts.reshape(len(kept), -1)                       # [T, P*2]
    vel = np.diff(pos, axis=0, prepend=pos[:1])            # [T, P*2]
    if cfg["vel_only"]:
        feat = vel
    elif cfg["velocity"]:
        feat = np.concatenate([pos, vel], axis=1)
    else:
        feat = pos

    if cfg["zscore"]:
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)

    return np.ascontiguousarray(feat, dtype=np.float64)


def _dist(qf, rf):
    if len(qf) == 0 or len(rf) == 0:
        return np.inf
    return _DTW(qf, rf, window=DTW_WINDOW)


def load_bank(cfg=None):
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
            "feat": featurize(arr, cfg),
            "feat_mirror": featurize(mirror_array(arr), cfg),
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


def evaluate(bank, query_filter=None, verbose=False):
    """Leave-one-out over the bank. query_filter(ref)->bool restricts which clips
    are used as queries (refs are always the full bank). Returns metric dicts."""
    label_counts = Counter(r["label"] for r in bank)
    src_by_label = defaultdict(set)
    for r in bank:
        src_by_label[r["label"]].add(r["source"])
    cross_labels = {l for l, s in src_by_label.items() if len(s) > 1}

    overall = {"top1": 0, "top5": 0, "n": 0}
    multi = {"top1": 0, "top5": 0, "n": 0}
    cross = {"top1": 0, "top5": 0, "n": 0}
    wrong = Counter()

    queries = [q for q in bank if query_filter is None or query_filter(q)]
    for i, q in enumerate(queries):
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
        if verbose and (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)}...")

    return {"overall": overall, "multi": multi, "cross": cross, "wrong": wrong}


def _pct(d, key):
    return 100 * d[key] / d["n"] if d["n"] else 0.0


def run_single(npy_path, cfg):
    arr = np.load(npy_path)
    bank = load_bank(cfg)
    qf = featurize(arr, cfg)
    qfm = featurize(mirror_array(arr), cfg)
    top = classify(qf, qfm, bank, k=5)
    print(f"\nTop-5 matches for {Path(npy_path).name}:")
    for i, (lbl, d, src, yid) in enumerate(top, 1):
        print(f"  {i}. {lbl:25s}  dist={d:8.3f}  [{src}] {yid}")


def run_eval(cfg):
    print(f"Loading bank (cfg={cfg})...")
    bank = load_bank(cfg)
    print(f"Evaluating {len(bank)} clips... this takes a few minutes.")
    r = evaluate(bank, verbose=True)

    print("\n=== Leave-one-out accuracy ===")
    for name, key in [("Overall (all 963, many unmatchable)", "overall"),
                      ("Multi-example labels (fair within-bank)", "multi"),
                      ("Cross-source (different signers)", "cross")]:
        d = r[key]
        print(f"  {name:42s}  n={d['n']:4d}  top1={_pct(d,'top1'):5.1f}%  top5={_pct(d,'top5'):5.1f}%")

    print("\n  Most common confusions (true -> guessed):")
    for (true, guess), n in r["wrong"].most_common(10):
        print(f"    {true:22s} -> {guess:22s} ({n})")


def main():
    ap = argparse.ArgumentParser(description="DTW nearest-neighbor sign spotter.")
    ap.add_argument("--npy", type=Path, help="Classify a single .npy clip (top-5).")
    ap.add_argument("--eval", action="store_true", help="Leave-one-out accuracy eval.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable body-relative normalization.")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.no_normalize:
        cfg["norm"] = "none"

    if args.eval:
        run_eval(cfg)
    elif args.npy:
        run_single(args.npy, cfg)
    else:
        ap.error("pass --npy <file> or --eval")


if __name__ == "__main__":
    main()
