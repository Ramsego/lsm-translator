import numpy as np, csv, itertools
from pathlib import Path
from scipy.stats import mannwhitneyu

FEAT_DIR = Path(__file__).parent / "features"

rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]

clips = []
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT_DIR / name)
    if feats.shape[0] < 5:
        print("SKIP too few frames:", r["word"], rel); continue
    feats = feats[::2]  # downsample for speed
    clips.append((r["word"], rel, feats))

print(f"{len(clips)} clips loaded")

def dtw_dist(a, b):
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:,None,:] - b[None,:,:], axis=2)  # n x m local cost
    D = np.full((n+1, m+1), np.inf)
    D[0,0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i,j] = cost[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return D[n,m] / (n+m)  # length-normalized

same_word, diff_word = [], []
pairs_info = []
for (w1, f1, feat1), (w2, f2, feat2) in itertools.combinations(clips, 2):
    d = dtw_dist(feat1, feat2)
    if w1 == w2:
        same_word.append(d)
        pairs_info.append((w1, f1, w2, f2, d, 'SAME'))
    else:
        diff_word.append(d)

same_word = np.array(same_word)
diff_word = np.array(diff_word)

print(f"\nsame-word pairs: n={len(same_word)}  median={np.median(same_word):.4f}  mean={np.mean(same_word):.4f}")
print(f"diff-word pairs: n={len(diff_word)}  median={np.median(diff_word):.4f}  mean={np.mean(diff_word):.4f}")

stat, p = mannwhitneyu(same_word, diff_word, alternative='less')
print(f"\nMann-Whitney U test (H1: same-word distances < diff-word distances): U={stat:.1f}  p={p:.4f}")
print("PRE-REGISTERED THRESHOLD: p < 0.05 for success")
print("RESULT:", "PASS" if p < 0.05 else "FAIL")

print("\n--- same-word pair distances, individually ---")
for w1,f1,w2,f2,d,_ in sorted(pairs_info, key=lambda x: x[4]):
    print(f"  {w1:12s} {d:.4f}   ({f1} vs {f2})")

# unambiguous effect size: P(a random same-word pair is closer than a random diff-word pair)
wins = sum(1 for s in same_word for d in diff_word if s < d)
total = len(same_word) * len(diff_word)
print(f"\nP(same-word pair closer than diff-word pair) = {wins/total:.3f}  (0.5 = no separation, 1.0 = perfect)")
