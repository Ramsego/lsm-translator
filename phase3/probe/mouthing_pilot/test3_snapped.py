import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from burst_detect import compute_occluded

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
    """extra_valid: bool array, True = usable frame (not nan, not occluded)."""
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
    i, j = n, best_j
    while i > 0:
        d = P[i,j]
        if d==0: i,j = i-1,j-1
        elif d==1: i,j = i-1,j
        else: i,j = i,j-1
        if j <= 0: break
    return j, best_j

# preload clip data: features + occlusion mask + snapped window + sign window
clips = {}
for rel, r in rows.items():
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT_DIR / name)
    sc = np.load(SIDECAR / (rel.replace("/", "__").replace(".mp4", ".npz")))
    occluded = compute_occluded(sc["hand_dist"], sc["face_ok"])
    extra_valid = (~np.isnan(feats[:,0])) & (~occluded[:len(feats)] if len(occluded)>=len(feats) else np.pad(~occluded, (0,len(feats)-len(occluded)), constant_values=True))
    sr = snapped[rel]
    video_start = float(r["video_start"])
    true_sign_mid = ((float(r["sign_start"])-video_start) + (float(r["sign_end"])-video_start)) / 2
    snapped_mid = None
    if sr["mouthing_detected"] == "True":
        snapped_mid = (float(sr["burst_on_s"]) + float(sr["burst_off_s"])) / 2
    clips[rel] = dict(word=r["word"], feats=feats, extra_valid=extra_valid,
                       true_sign_mid=true_sign_mid, snapped_mid=snapped_mid,
                       clip_dur=len(feats)/fps, snapped_row=sr)

def build_template(rel):
    sr = clips[rel]["snapped_row"]
    if sr["mouthing_detected"] != "True": return None
    bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
    return valid_slice(clips[rel]["feats"], int(bon*fps), int(boff*fps)+1)

rng = np.random.default_rng(0)
results = {"err_a": [], "err_b": [], "mid_base_a": [], "mid_base_b": [], "rand_base_a": [], "rand_base_b": []}
n_pairs = 0
for t_rel, tc in clips.items():
    templ = build_template(t_rel)
    if templ is None or len(templ) < 3: continue
    for h_rel, hc in clips.items():
        if h_rel == t_rel or hc["word"] != tc["word"]: continue
        if hc["snapped_mid"] is None: continue  # need truth (a)
        res = subseq_dtw_masked(templ, hc["feats"], hc["extra_valid"])
        if res is None: continue
        sj, ej = res
        matched_center = (sj+ej)/2/fps
        err_a = abs(matched_center - hc["snapped_mid"])
        err_b = abs(matched_center - hc["true_sign_mid"])
        mid_base_a = abs(hc["clip_dur"]/2 - hc["snapped_mid"])
        mid_base_b = abs(hc["clip_dur"]/2 - hc["true_sign_mid"])
        rand_draws = rng.uniform(0, hc["clip_dur"], 100)
        rand_base_a = np.mean(np.abs(rand_draws - hc["snapped_mid"]))
        rand_base_b = np.mean(np.abs(rand_draws - hc["true_sign_mid"]))
        results["err_a"].append(err_a); results["err_b"].append(err_b)
        results["mid_base_a"].append(mid_base_a); results["mid_base_b"].append(mid_base_b)
        results["rand_base_a"].append(rand_base_a); results["rand_base_b"].append(rand_base_b)
        n_pairs += 1
        print(f"{tc['word']:12s} templ={t_rel:32s} -> target={h_rel:32s} err_a={err_a:.2f}s err_b={err_b:.2f}s")

for k in results: results[k] = np.array(results[k])
print(f"\nn_pairs = {n_pairs}\n")

for truth_name, err_key, base_mid_key, base_rand_key in [("(a) snapped mouthing midpoint","err_a","mid_base_a","rand_base_a"),
                                                            ("(b) human sign midpoint","err_b","mid_base_b","rand_base_b")]:
    err = results[err_key]; mid_base = results[base_mid_key]; rand_base = results[base_rand_key]
    print(f"--- Ground truth {truth_name} ---")
    print(f"  ours:   mean={err.mean():.2f}s  median={np.median(err):.2f}s")
    print(f"  midpoint baseline: mean={mid_base.mean():.2f}s")
    print(f"  random baseline:   mean={rand_base.mean():.2f}s")
    for thr in [1.0, 1.5, 3.0]:
        frac = np.mean(err <= thr)
        print(f"  fraction within +/-{thr}s: {frac:.1%}")
    stat, p = wilcoxon(err, mid_base, alternative='less')
    print(f"  Wilcoxon (ours < midpoint baseline): p={p:.4f}  {'PASS beats baseline' if p<0.05 else 'does not beat baseline'}")
    print()

print("PILOT REFERENCE (old mistimed windows): ours 5.39s vs midpoint 4.97s, p=0.65 (FAIL)")
med_a = np.median(results["err_a"])
print(f"\nPRE-REGISTERED DECISION based on Test3 truth(a) cross-clip median AE = {med_a:.2f}s:")
if med_a <= 1.0: print("  <=1.0s -> localization WORKS")
elif med_a <= 3.0: print("  1.0-3.0s -> MARGINAL, soft prior only")
else: print("  >3.0s -> NO useful positional signal")
