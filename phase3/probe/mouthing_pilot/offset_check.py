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
    clips[rel] = dict(word=r["word"], feats=feats, fps=30.0,
                       t0=max(0, sign_start-0.3), t1=sign_end+0.3,
                       clip_dur=len(feats)/30.0)

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
    return j, best_j

templates = [(rel,c["word"],valid_slice(c["feats"],int(c["t0"]*30),int(c["t1"]*30)),c["t0"],c["t1"])
             for rel,c in clips.items()]
templates = [t for t in templates if len(t[2])>=3]

signed_offsets, abs_offsets, midpoint_errors, random_errors = [], [], [], []
rng = np.random.default_rng(0)

for t_rel, t_word, templ, _, _ in templates:
    best = None
    for h_rel, h in clips.items():
        if h_rel == t_rel or h["word"] != t_word: continue
        res = subseq_dtw(templ, h["feats"])
        if res is None: continue
        sj, ej = res
        matched_center = (sj+ej)/2/30.0
        true_center = (h["t0"]+h["t1"])/2
        # keep the match against THIS haystack (there may be multiple same-word haystacks; use each)
        offset = matched_center - true_center
        signed_offsets.append(offset)
        abs_offsets.append(abs(offset))
        midpoint_errors.append(abs(h["clip_dur"]/2 - true_center))
        random_errors.append(abs(rng.uniform(0, h["clip_dur"]) - true_center))

signed_offsets = np.array(signed_offsets)
abs_offsets = np.array(abs_offsets)
midpoint_errors = np.array(midpoint_errors)
random_errors = np.array(random_errors)

print(f"n comparisons = {len(signed_offsets)}")
print(f"\nSigned offset (matched_center - true_center): mean={signed_offsets.mean():+.2f}s  median={np.median(signed_offsets):+.2f}s")
print("  (positive = our match lands AFTER the true sign window; negative = BEFORE)")
print(f"\nOur method's mean abs error from true center: {abs_offsets.mean():.2f}s")
print(f"Naive 'always guess clip midpoint' mean abs error: {midpoint_errors.mean():.2f}s")
print(f"Naive 'guess a uniformly random point in the clip' mean abs error: {random_errors.mean():.2f}s")

from scipy.stats import wilcoxon
stat,p = wilcoxon(abs_offsets, midpoint_errors, alternative='less')
print(f"\nDoes our estimate beat 'always guess the middle'? Wilcoxon p={p:.4f} ({'PASS' if p<0.05 else 'FAIL'})")
stat2,p2 = wilcoxon(abs_offsets, random_errors, alternative='less')
print(f"Does our estimate beat 'guess randomly'?          Wilcoxon p={p2:.4f} ({'PASS' if p2<0.05 else 'FAIL'})")
