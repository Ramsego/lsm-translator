import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from burst_detect import detect_bursts
from scipy.stats import mannwhitneyu

HERE = Path(__file__).parent
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]]
fps = 30.0

def n_extrema(ap, on, off):
    seg = ap[on:off+1]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 3: return 0
    d = np.diff(seg)
    signs = np.sign(d)
    signs = signs[signs != 0]
    if len(signs) < 2: return 0
    return int(np.sum(np.diff(signs) != 0))

snapped_feats, other_feats = [], []
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    sr = snapped.get(rel)
    if sr is None or sr["mouthing_detected"] != "True": continue
    name = rel.replace("/", "__").replace(".mp4", ".npz")
    sc = np.load(HERE / "sidecar" / name)
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    snapped_on = int(round(float(sr["burst_on_s"])*fps))
    for (on, off) in res["bursts"]:
        dur = (off-on)/fps
        ext = n_extrema(sc["aperture"], on, off)
        is_snapped = abs(on - snapped_on) <= 1
        (snapped_feats if is_snapped else other_feats).append((dur, ext))

snapped_feats = np.array(snapped_feats); other_feats = np.array(other_feats)
print(f"snapped (presumed real word) bursts: n={len(snapped_feats)}")
print(f"  duration: mean={snapped_feats[:,0].mean():.2f}s median={np.median(snapped_feats[:,0]):.2f}s")
print(f"  extrema:  mean={snapped_feats[:,1].mean():.2f}   median={np.median(snapped_feats[:,1]):.1f}")
print(f"\nother (unknown: could be other real words, or fillers) bursts: n={len(other_feats)}")
print(f"  duration: mean={other_feats[:,0].mean():.2f}s median={np.median(other_feats[:,0]):.2f}s")
print(f"  extrema:  mean={other_feats[:,1].mean():.2f}   median={np.median(other_feats[:,1]):.1f}")

for name, i in [("duration",0), ("extrema",1)]:
    stat, p = mannwhitneyu(snapped_feats[:,i], other_feats[:,i], alternative='two-sided')
    print(f"\nMann-Whitney on {name}: p={p:.4f}")
