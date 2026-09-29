import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from burst_detect import detect_bursts
from scipy.stats import mannwhitneyu
import mediapipe as mp

LIP_IDX = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))
IDX_61, IDX_291 = LIP_IDX.index(61), LIP_IDX.index(291)

HERE = Path(__file__).parent
FEAT_DIR = HERE / "features2"
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]]
fps = 30.0

def width_trace(feats):
    x1, x2 = feats[:,2*IDX_61], feats[:,2*IDX_291]
    y1, y2 = feats[:,2*IDX_61+1], feats[:,2*IDX_291+1]
    return np.hypot(x1-x2, y1-y2)

def n_extrema(sig, on, off):
    seg = sig[on:off+1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 3: return 0
    d = np.diff(seg); signs = np.sign(d); signs = signs[signs != 0]
    if len(signs) < 2: return 0
    return int(np.sum(np.diff(signs) != 0))

snapped_w, other_w = [], []
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    sr = snapped.get(rel)
    if sr is None or sr["mouthing_detected"] != "True": continue
    name_npy = rel.replace("/", "__").replace(".mp4", ".npy")
    name_npz = rel.replace("/", "__").replace(".mp4", ".npz")
    feats = np.load(FEAT_DIR / name_npy)
    w = width_trace(feats)
    sc = np.load(HERE / "sidecar" / name_npz)
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    snapped_on = int(round(float(sr["burst_on_s"])*fps))
    for (on, off) in res["bursts"]:
        dur = (off-on)/fps
        wstd = np.nanstd(w[on:off+1])
        wrange = np.nanmax(w[on:off+1]) - np.nanmin(w[on:off+1]) if off>on else 0
        wext = n_extrema(w, on, off)
        is_snapped = abs(on - snapped_on) <= 1
        (snapped_w if is_snapped else other_w).append((wstd, wrange, wext))

snapped_w = np.array(snapped_w); other_w = np.array(other_w)
print(f"snapped bursts n={len(snapped_w)}, other bursts n={len(other_w)}\n")
for name, i in [("width_std",0), ("width_range",1), ("width_extrema",2)]:
    s, o = snapped_w[:,i], other_w[:,i]
    stat, p = mannwhitneyu(s, o, alternative='two-sided')
    print(f"{name:14s} snapped: mean={s.mean():.4f} median={np.median(s):.4f}   other: mean={o.mean():.4f} median={np.median(o):.4f}   p={p:.4f}")
