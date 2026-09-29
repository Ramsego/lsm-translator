import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
FEAT_DIR = HERE / "features2"
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]
fps = 30.0

def valid_slice(feats, i0, i1):
    seg = feats[max(0,i0):max(0,i1)]
    if len(seg)==0: return seg
    mask = ~np.isnan(seg[:,0]); return seg[mask]

def subseq_dtw_cost(template, haystack):
    valid = ~np.isnan(haystack[:,0]); H = haystack.copy()
    n, m = len(template), len(H)
    if n < 3 or m < 3: return None
    C = np.linalg.norm(template[:,None,:] - H[None,:,:], axis=2)
    C[:, ~valid] = 1e6
    D = np.full((n+1, m+1), np.inf); D[0,:] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i,j] = C[i-1,j-1] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
    return float(np.min(D[n,1:])) / n

clips = {}
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT_DIR / name)
    clips[rel] = dict(word=r["word"], feats=feats)

templates = []
for rel, c in clips.items():
    sr = snapped.get(rel)
    if sr is None or sr["mouthing_detected"] != "True": continue
    bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
    templ = valid_slice(c["feats"], int(bon*fps), int(boff*fps)+1)
    if len(templ) >= 3: templates.append((rel, c["word"], templ))

# compute full cost list per template once (25 candidates each)
all_costs = []
for t_rel, t_word, templ in templates:
    costs = []
    for h_rel, h in clips.items():
        if h_rel == t_rel: continue
        cost = subseq_dtw_cost(templ, h["feats"])
        if cost is not None: costs.append((h_word_correct := (h["word"]==t_word), cost))
    all_costs.append(costs)
    print(f"done: {t_word:12s} {t_rel}")

rng = np.random.default_rng(0)
print(f"\n{len(all_costs)} templates, 25 real candidates each\n")
print(f"{'k':>4} {'top-1':>8} {'top-3':>8} {'median_rank':>12}  (200 random subsets per template, correct answer always included)")
for k in [5, 8, 12, 25]:
    top1_hits, top3_hits, ranks, n = 0, 0, [], 0
    for costs in all_costs:
        correct_idx = [i for i,(is_c,_) in enumerate(costs) if is_c]
        wrong_idx = [i for i,(is_c,_) in enumerate(costs) if not is_c]
        if not correct_idx: continue
        reps = 200 if k < 25 else 1
        for _ in range(reps):
            if k >= len(costs):
                sample_idx = list(range(len(costs)))
            else:
                chosen_wrong = rng.choice(wrong_idx, size=min(k-1, len(wrong_idx)), replace=False)
                sample_idx = list(chosen_wrong) + [correct_idx[0]]
            sample = sorted([costs[i] for i in sample_idx], key=lambda x: x[1])
            rank = next(i+1 for i,(is_c,_) in enumerate(sample) if is_c)
            ranks.append(rank); n += 1
            if rank==1: top1_hits += 1
            if rank<=3: top3_hits += 1
    print(f"{k:>4} {top1_hits/n:>7.1%} {top3_hits/n:>7.1%} {np.median(ranks):>12.1f}   (chance top-1={1/k:.1%}, chance top-3={min(3,k)/k:.1%})")
