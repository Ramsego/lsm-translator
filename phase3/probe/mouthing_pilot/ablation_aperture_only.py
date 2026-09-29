import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

HERE = Path(__file__).parent
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
fps = 30.0

def valid_slice_1d(ap, i0, i1):
    seg = ap[max(0,i0):max(0,i1)]
    if len(seg)==0: return seg
    mask = ~np.isnan(seg); return seg[mask].reshape(-1,1)

def subseq_dtw(template, haystack):
    valid = ~np.isnan(haystack[:,0]); H = haystack.copy()
    n, m = len(template), len(H)
    if n < 3 or m < 3: return None
    C = np.linalg.norm(template[:,None,:] - H[None,:,:], axis=2)
    C[:, ~valid] = 1e6
    D = np.full((n+1, m+1), np.inf); D[0,:] = 0.0
    P = np.full((n+1, m+1), -1, dtype=int)
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [(D[i-1,j-1],0), (D[i-1,j],1), (D[i,j-1],2)]
            best_prev, direction = min(options, key=lambda x: x[0])
            D[i,j] = C[i-1,j-1] + best_prev; P[i,j] = direction
    best_j = int(np.argmin(D[n,1:])) + 1
    return D[n,best_j]/n

clips = {}
rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]]
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name = rel.replace("/", "__").replace(".mp4", ".npz")
    sc = np.load(HERE / "sidecar" / name)
    clips[rel] = dict(word=r["word"], aperture=sc["aperture"].reshape(-1,1))

templates = []
for rel, c in clips.items():
    sr = snapped.get(rel)
    if sr is None or sr["mouthing_detected"] != "True": continue
    bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
    templ = valid_slice_1d(c["aperture"][:,0], int(bon*fps), int(boff*fps)+1)
    if len(templ) >= 3:
        templates.append((rel, c["word"], templ))

print(f"{len(templates)} aperture-only templates\n")
ranks, top1, top3, n_trials = [], 0, 0, 0
for t_rel, t_word, templ in templates:
    costs = []
    for h_rel, h in clips.items():
        if h_rel == t_rel: continue
        cost = subseq_dtw(templ, h["aperture"])
        if cost is None: continue
        costs.append((h_rel, h["word"], cost))
    if not costs: continue
    costs.sort(key=lambda x: x[2])
    correct_positions = [i+1 for i,c in enumerate(costs) if c[1]==t_word]
    if not correct_positions: continue
    best_rank = min(correct_positions)
    ranks.append(best_rank); n_trials += 1
    if best_rank==1: top1 += 1
    if best_rank<=3: top3 += 1

ranks = np.array(ranks)
n_candidates = len(clips) - 1
chance_median = (n_candidates+1)/2
print(f"n_trials={n_trials}  chance_median_rank~={chance_median:.1f}")
print(f"APERTURE-ONLY: median rank={np.median(ranks):.1f}  mean={np.mean(ranks):.1f}")
print(f"top-1: {top1}/{n_trials}={top1/n_trials:.1%}   top-3: {top3}/{n_trials}={top3/n_trials:.1%}")
stat, p = wilcoxon(ranks - chance_median, alternative='less')
print(f"Wilcoxon p={p:.4f}")
print(f"\nFULL 80-DIM SHAPE (Step 5 result): median rank=3.0  top-1=12.5%  top-3=56.2%  p=0.0006")
