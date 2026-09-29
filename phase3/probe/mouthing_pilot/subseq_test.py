import numpy as np, csv
from pathlib import Path
from scipy.stats import wilcoxon

FEAT_DIR = Path(__file__).parent / "features2"
rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]

clips = {}
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT_DIR / name)
    video_start = float(r["video_start"])
    sign_start = float(r["sign_start"]) - video_start
    sign_end = float(r["sign_end"]) - video_start
    fps = 30.0
    clips[rel] = dict(word=r["word"], feats=feats, fps=fps,
                       t0=max(0, sign_start - 0.3), t1=sign_end + 0.3)  # small pad

def valid_slice(feats, i0, i1):
    seg = feats[i0:i1]
    mask = ~np.isnan(seg[:,0])
    return seg[mask]

def subseq_dtw(template, haystack):
    """Free-start subsequence DTW. Returns (best_end_idx, best_cost, backtrace_start_idx)."""
    valid = ~np.isnan(haystack[:,0])
    H = haystack.copy()
    n, m = len(template), len(H)
    if n < 3 or m < 3:
        return None
    cost = np.full((m,), np.inf)
    C = np.linalg.norm(template[:,None,:] - H[None,:,:], axis=2)  # n x m, nan where invalid
    C[:, ~valid] = 1e6
    D = np.full((n+1, m+1), np.inf)
    D[0, :] = 0.0  # free start anywhere in haystack
    P = np.full((n+1, m+1), -1, dtype=int)  # backpointer: 0=diag,1=up(i-1,j),2=left(i,j-1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [(D[i-1,j-1],0), (D[i-1,j],1), (D[i,j-1],2)]
            best_prev, direction = min(options, key=lambda x: x[0])
            D[i,j] = C[i-1,j-1] + best_prev
            P[i,j] = direction
    best_j = int(np.argmin(D[n, 1:])) + 1
    best_cost = D[n, best_j] / n
    # backtrack to find start
    i, j = n, best_j
    while i > 0:
        d = P[i,j]
        if d == 0: i,j = i-1,j-1
        elif d == 1: i,j = i-1,j
        else: i,j = i,j-1
        if j <= 0: break
    start_j = j
    return start_j, best_j, best_cost

# Build reference templates from tight verified windows
templates = []
for rel, c in clips.items():
    i0, i1 = int(c["t0"]*c["fps"]), int(c["t1"]*c["fps"])
    templ = valid_slice(c["feats"], i0, i1)
    if len(templ) >= 3:
        templates.append((rel, c["word"], templ, c["t0"], c["t1"]))

print(f"{len(templates)} usable templates (from {len(clips)} clips)\n")

ranks, top1, top3, overlaps = [], 0, 0, []
n_trials = 0
for t_rel, t_word, templ, _, _ in templates:
    costs = []
    for h_rel, h in clips.items():
        if h_rel == t_rel:
            continue
        res = subseq_dtw(templ, h["feats"])
        if res is None: continue
        start_j, end_j, cost = res
        costs.append((h_rel, h["word"], cost, start_j, end_j, h["fps"]))
    if not costs: continue
    costs.sort(key=lambda x: x[2])
    ranked_rels = [c[0] for c in costs]
    # find rank of correct target(s): any clip sharing the same word
    correct_positions = [i+1 for i,c in enumerate(costs) if c[1] == t_word]
    if not correct_positions: continue
    best_rank = min(correct_positions)
    ranks.append(best_rank)
    n_trials += 1
    if best_rank == 1: top1 += 1
    if best_rank <= 3: top3 += 1
    # localization check for the best-ranked same-word candidate
    match = next(c for c in costs if c[1]==t_word and (costs.index(c)+1)==best_rank)
    h_rel, h_word, cost, start_j, end_j, fps = match
    matched_t0, matched_t1 = start_j/fps, end_j/fps
    true_t0, true_t1 = clips[h_rel]["t0"], clips[h_rel]["t1"]
    overlap = max(0, min(matched_t1, true_t1) - max(matched_t0, true_t0))
    union = max(matched_t1, true_t1) - min(matched_t0, true_t0)
    iou = overlap/union if union > 0 else 0
    overlaps.append(iou)
    print(f"{t_word:12s} template={t_rel:30s} best_match_rank={best_rank:2d}/{len(costs)}  IoU_at_best_same_word={iou:.2f}")

ranks = np.array(ranks)
n_candidates = len(clips) - 1
chance_median_rank = (n_candidates+1)/2
print(f"\nn_trials={n_trials}  candidates_per_trial~={n_candidates}  chance_median_rank~={chance_median_rank:.1f}")
print(f"observed median rank of correct word = {np.median(ranks):.1f}   mean={np.mean(ranks):.1f}")
print(f"top-1 accuracy: {top1}/{n_trials} = {top1/n_trials:.1%}  (chance ~= {1/n_candidates:.1%})")
print(f"top-3 accuracy: {top3}/{n_trials} = {top3/n_trials:.1%}  (chance ~= {3/n_candidates:.1%})")
print(f"mean IoU (matched window vs target's own true window, when same-word found): {np.mean(overlaps):.2f}")

stat, p = wilcoxon(ranks - chance_median_rank, alternative='less')
print(f"\nWilcoxon signed-rank test (H1: ranks < chance median {chance_median_rank:.1f}): p={p:.4f}")
print("PRE-REGISTERED THRESHOLD: p < 0.05 for success")
print("RESULT:", "PASS" if p < 0.05 else "FAIL")
