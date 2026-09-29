import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from burst_detect import compute_occluded
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).parent
FEAT_DIR = HERE / "features2"
SIDECAR = HERE / "sidecar"
fps = 30.0
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
rows = {r["file"].replace("../57TvyH9902U/",""): r
        for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]}

def valid_slice(feats, i0, i1):
    seg = feats[max(0,i0):max(0,i1)]
    if len(seg)==0: return seg
    mask = ~np.isnan(seg[:,0]); return seg[mask]

def subseq_dtw_masked(template, haystack, extra_valid):
    n, m = len(template), len(haystack)
    if n < 3 or m < 3: return None
    C = np.linalg.norm(template[:,None,:] - haystack[None,:,:], axis=2)
    C[:, ~extra_valid] = 1e6
    D = np.full((n+1, m+1), np.inf); D[0,:] = 0.0
    P = np.full((n+1, m+1), -1, dtype=int)
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [(D[i-1,j-1],0), (D[i-1,j],1), (D[i,j-1],2)]
            best_prev, direction = min(options, key=lambda x: x[0])
            D[i,j] = C[i-1,j-1] + best_prev; P[i,j] = direction
    best_j = int(np.argmin(D[n,1:])) + 1
    cost = D[n, best_j] / n
    i, j = n, best_j
    while i > 0:
        d = P[i,j]
        if d==0: i,j = i-1,j-1
        elif d==1: i,j = i-1,j
        else: i,j = i,j-1
        if j <= 0: break
    return j, best_j, cost

clips = {}
for rel, r in rows.items():
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT_DIR / name)
    sc = np.load(SIDECAR / (rel.replace("/", "__").replace(".mp4", ".npz")))
    occluded = compute_occluded(sc["hand_dist"], sc["face_ok"])
    n = len(feats)
    extra_valid = (~np.isnan(feats[:,0])) & (~occluded[:n] if len(occluded)>=n else np.pad(~occluded,(0,n-len(occluded)),constant_values=True))
    sr = snapped[rel]
    video_start = float(r["video_start"])
    true_sign_mid = ((float(r["sign_start"])-video_start) + (float(r["sign_end"])-video_start)) / 2
    snapped_mid = (float(sr["burst_on_s"]) + float(sr["burst_off_s"])) / 2 if sr["mouthing_detected"]=="True" else None
    clips[rel] = dict(word=r["word"], feats=feats, extra_valid=extra_valid,
                       true_sign_mid=true_sign_mid, snapped_mid=snapped_mid, snapped_row=sr)

def build_template(rel):
    sr = clips[rel]["snapped_row"]
    if sr["mouthing_detected"] != "True": return None
    bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
    return valid_slice(clips[rel]["feats"], int(bon*fps), int(boff*fps)+1)

results = []
for t_rel, tc in clips.items():
    templ = build_template(t_rel)
    if templ is None or len(templ) < 3: continue
    # full candidate cost list for this template (Test 2 style), to get target's SPECIFIC rank
    all_costs = []
    for h_rel, hc in clips.items():
        if h_rel == t_rel: continue
        res = subseq_dtw_masked(templ, hc["feats"], hc["extra_valid"])
        if res is None: continue
        sj, ej, cost = res
        all_costs.append((h_rel, cost, sj, ej))
    all_costs.sort(key=lambda x: x[1])
    rank_by_rel = {rel: i+1 for i,(rel,cost,sj,ej) in enumerate(all_costs)}

    for h_rel, hc in clips.items():
        if h_rel == t_rel or hc["word"] != tc["word"]: continue
        if hc["snapped_mid"] is None: continue
        target_rank = rank_by_rel.get(h_rel)
        match = next(x for x in all_costs if x[0]==h_rel)
        _, cost, sj, ej = match
        matched_center = (sj+ej)/2/fps
        err_a = abs(matched_center - hc["snapped_mid"])
        err_b = abs(matched_center - hc["true_sign_mid"])
        results.append(dict(word=tc["word"], templ=t_rel, target=h_rel, target_rank=target_rank,
                             cost=cost, err_a=err_a, err_b=err_b))
        print(f"{tc['word']:12s} templ={t_rel:28s} target={h_rel:28s} target_rank={target_rank:3d} cost={cost:.3f} err_a={err_a:.2f}s err_b={err_b:.2f}s")

ranks = np.array([r["target_rank"] for r in results])
costs = np.array([r["cost"] for r in results])
err_a = np.array([r["err_a"] for r in results])
err_b = np.array([r["err_b"] for r in results])

print(f"\nn={len(results)}")
rho, p = spearmanr(ranks, err_a)
print(f"Spearman(target_rank, err_a) = {rho:.3f}  p={p:.4f}   (predict: positive -- worse rank, bigger error)")
rho2, p2 = spearmanr(costs, err_a)
print(f"Spearman(match_cost, err_a) = {rho2:.3f}  p={p2:.4f}")

top3 = err_a[ranks<=3]; nontop3 = err_a[ranks>3]
print(f"\nerr_a when target_rank<=3 (n={len(top3)}): mean={top3.mean():.2f}s median={np.median(top3):.2f}s")
print(f"err_a when target_rank>3  (n={len(nontop3)}): mean={nontop3.mean():.2f}s median={np.median(nontop3):.2f}s")
if len(top3)>0 and len(nontop3)>0:
    stat,p3 = mannwhitneyu(top3, nontop3, alternative='less')
    print(f"Mann-Whitney (top3 errors < non-top3 errors): p={p3:.4f}")
