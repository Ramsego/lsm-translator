import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from collections import Counter

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

def valid_slice_1d(ap, i0, i1):
    seg = ap[max(0,i0):max(0,i1)]
    if len(seg)==0: return seg
    mask = ~np.isnan(seg); return seg[mask].reshape(-1,1)

# load both feature types
clips_shape, clips_ap = {}, {}
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name_npy = rel.replace("/", "__").replace(".mp4", ".npy")
    name_npz = rel.replace("/", "__").replace(".mp4", ".npz")
    clips_shape[rel] = dict(word=r["word"], feats=np.load(FEAT_DIR / name_npy))
    sc = np.load(HERE / "sidecar" / name_npz)
    clips_ap[rel] = dict(word=r["word"], feats=sc["aperture"].reshape(-1,1))

word_count = Counter(c["word"] for c in clips_shape.values())

def build_templates(clips):
    templates = []
    for rel, c in clips.items():
        sr = snapped.get(rel)
        if sr is None or sr["mouthing_detected"] != "True": continue
        bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
        if c["feats"].shape[1] == 1:
            templ = valid_slice_1d(c["feats"][:,0], int(bon*fps), int(boff*fps)+1)
        else:
            templ = valid_slice(c["feats"], int(bon*fps), int(boff*fps)+1)
        if len(templ) >= 3: templates.append((rel, c["word"], templ))
    return templates

def get_ranks(clips, templates):
    out = {}
    for t_rel, t_word, templ in templates:
        costs = []
        for h_rel, h in clips.items():
            if h_rel == t_rel: continue
            cost = subseq_dtw_cost(templ, h["feats"])
            if cost is not None: costs.append((h_rel, h["word"], cost))
        if not costs: continue
        costs.sort(key=lambda x: x[2])
        correct_positions = [i+1 for i,c in enumerate(costs) if c[1]==t_word]
        if not correct_positions: continue
        out[t_rel] = min(correct_positions)
    return out

templ_shape = build_templates(clips_shape)
templ_ap = build_templates(clips_ap)

ranks_shape = get_ranks(clips_shape, templ_shape)
ranks_ap = get_ranks(clips_ap, templ_ap)

print("per-template: word, n_correct_in_pool(=count-1), shape_rank, aperture_rank, closed_form_null_mean")
common = sorted(set(ranks_shape) & set(ranks_ap))
rows_out = []
for rel in common:
    word = clips_shape[rel]["word"]
    c = word_count[word] - 1
    null_mean = 26/(c+1)
    rows_out.append((rel, word, c, ranks_shape[rel], ranks_ap[rel], null_mean))
    print(f"  {word:12s} n_correct={c}  shape_rank={ranks_shape[rel]:2d}  aperture_rank={ranks_ap[rel]:2d}  null_mean={null_mean:.1f}")

# Monte Carlo per-template null + aggregate test
rng = np.random.default_rng(0)
def mc_null_ranks(c_list, N=25, trials=20000):
    """for each template's c, simulate min-rank under null; return matrix (trials, n_templates)"""
    sims = np.zeros((trials, len(c_list)))
    for j,c in enumerate(c_list):
        perm_mins = rng.integers(0, N, size=(trials, c)).min(axis=1) + 1 if False else None
    # proper: sample without replacement c ranks from 1..N, take min, repeat
    for j,c in enumerate(c_list):
        mins = np.empty(trials)
        for t in range(trials):
            mins[t] = rng.choice(N, size=c, replace=False).min() + 1
        sims[:,j] = mins
    return sims

c_list = [r[2] for r in rows_out]
null_medians = [ (26/(c+1)) for c in c_list ]  # use closed-form mean as null center (matches Fable's framing)

obs_shape = np.array([r[3] for r in rows_out])
obs_ap = np.array([r[4] for r in rows_out])
null_med_arr = np.array(null_medians)

print(f"\n--- SHAPE vs per-template null ---")
stat, p = wilcoxon(obs_shape - null_med_arr, alternative='less')
print(f"Wilcoxon (obs < per-template null mean): p={p:.4f}")

print(f"\n--- APERTURE vs per-template null ---")
stat, p = wilcoxon(obs_ap - null_med_arr, alternative='less')
print(f"Wilcoxon (obs < per-template null mean): p={p:.4f}")

print(f"\n--- PAIRED: shape vs aperture (does shape beat aperture per-template?) ---")
diff = obs_ap - obs_shape  # positive = shape rank is lower/better
wins = np.sum(diff > 0); losses = np.sum(diff < 0); ties = np.sum(diff==0)
print(f"shape better: {wins}, aperture better: {losses}, ties: {ties}")
stat, p = wilcoxon(obs_shape, obs_ap, alternative='less')
print(f"Wilcoxon paired (shape < aperture): p={p:.4f}")
