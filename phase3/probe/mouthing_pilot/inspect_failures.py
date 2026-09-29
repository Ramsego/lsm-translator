import numpy as np, csv
from pathlib import Path

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
    clips[rel] = dict(word=r["word"], feats=feats, t0=max(0, sign_start-0.3), t1=sign_end+0.3)

def valid_slice(feats, i0, i1):
    seg = feats[i0:i1]; mask = ~np.isnan(seg[:,0]); return seg[mask]

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
    i, j = n, best_j
    while i > 0:
        d = P[i,j]
        if d==0: i,j = i-1,j-1
        elif d==1: i,j = i-1,j
        else: i,j = i,j-1
        if j <= 0: break
    return j, best_j, D[n,best_j]/n

for t_rel in ["actividad/actividad_20m20s.mp4", "familia/familia_37m06s.mp4"]:
    t = clips[t_rel]
    templ = valid_slice(t["feats"], int(t["t0"]*30), int(t["t1"]*30))
    print(f"\n=== template: {t_rel} (word={t['word']}) ===")
    costs = []
    for h_rel, h in clips.items():
        if h_rel == t_rel: continue
        res = subseq_dtw(templ, h["feats"])
        if res is None: continue
        sj, ej, cost = res
        costs.append((h_rel, h["word"], cost, sj, ej))
    costs.sort(key=lambda x: x[2])
    for rank, (h_rel, h_word, cost, sj, ej) in enumerate(costs[:5], 1):
        marker = " <-- CORRECT WORD" if h_word == t["word"] else ""
        print(f"  rank {rank}: {h_word:12s} {h_rel:35s} cost={cost:.4f}  matched_frames=[{sj},{ej}]{marker}")
    # also show where the correct word landed
    for rank,(h_rel,h_word,cost,sj,ej) in enumerate(costs,1):
        if h_word == t["word"]:
            print(f"  ... correct word actually at rank {rank}: {h_rel}  matched_frames=[{sj},{ej}]  fps=30 -> matched_time=[{sj/30:.2f},{ej/30:.2f}]s")
