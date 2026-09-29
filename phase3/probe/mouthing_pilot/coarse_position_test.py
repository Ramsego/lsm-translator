import numpy as np, json, csv
from pathlib import Path
from scipy.stats import spearmanr

hits = json.load(open(Path(__file__).parent / "runB_hits.json"))
SPAN_FEAT = Path(__file__).parent / "span_features"
TEMPLATE_FEAT = Path(__file__).parent / "features2"

rows = {r["file"]: r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]}

def valid_slice(feats, i0, i1):
    seg = feats[max(0,i0):max(0,i1)]
    if len(seg)==0: return seg
    mask = ~np.isnan(seg[:,0]); return seg[mask]

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

# build word -> list of (rel, tight_template) from the ORIGINAL 14s clips (features2), excluding
# whichever clip is currently being used as the hit target (to keep template independent)
by_word = {}
for rel, r in rows.items():
    name = rel.replace("../57TvyH9902U/","").replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(TEMPLATE_FEAT / name)
    video_start = float(r["video_start"])
    t0 = max(0, float(r["sign_start"])-video_start-0.3)
    t1 = float(r["sign_end"])-video_start+0.3
    templ = valid_slice(feats, int(t0*30), int(t1*30))
    by_word.setdefault(r["word"], []).append((rel, templ))

true_rel, matched_rel, used = [], [], []
for i, h in enumerate(hits):
    word = h["word"]
    target_file = h["file"]
    alt_templates = [t for rel,t in by_word.get(word, []) if rel != target_file and len(t) >= 3]
    if not alt_templates:
        print(f"{i:02d} {word:12s} SKIP (no alternate-clip template)")
        continue
    templ = alt_templates[0]
    span_feats = np.load(SPAN_FEAT / f"{i:02d}_{word}.npy")
    res = subseq_dtw(templ, span_feats)
    if res is None:
        print(f"{i:02d} {word:12s} SKIP (dtw failed)"); continue
    sj, ej = res
    span_dur_frames = len(span_feats)
    matched_center_rel = ((sj+ej)/2) / span_dur_frames
    true_center_rel = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    true_rel.append(true_center_rel); matched_rel.append(matched_center_rel); used.append(word)
    print(f"{i:02d} {word:12s} true_rel_pos={true_center_rel:.2f}  matched_rel_pos={matched_center_rel:.2f}")

true_rel, matched_rel = np.array(true_rel), np.array(matched_rel)
print(f"\nn = {len(true_rel)}")
if len(true_rel) >= 3:
    rho, p = spearmanr(true_rel, matched_rel, alternative='greater')
    print(f"Spearman correlation (true vs matched relative position): rho={rho:.3f}  p={p:.4f}")
    print("PRE-REGISTERED THRESHOLD: p < 0.05, one-sided (positive correlation)")
    print("RESULT:", "PASS" if p < 0.05 else "FAIL")
    mae = np.mean(np.abs(true_rel - matched_rel))
    quartile_agree = np.mean((true_rel//0.25).astype(int) == (matched_rel//0.25).astype(int).clip(0,3))
    print(f"\nmean abs error in relative position: {mae:.2f}  (chance ~0.33 for uniform random pairs)")
    print(f"same-quartile agreement rate: {quartile_agree:.2f}  (chance = 0.25)")
else:
    print("Too few usable trials to test.")
