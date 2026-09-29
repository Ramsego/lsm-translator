import sys, csv, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from burst_detect import detect_bursts, compute_occluded

HERE = Path(__file__).parent
fps = 30.0
SEARCH_PRE, SEARCH_POST = 2.5, 1.0

hits = json.load(open(HERE / "runB_hits.json"))
snapped = {r["file"]: r for r in csv.DictReader(open(HERE / "snapped_windows.csv"))}
FEAT2 = HERE / "features2"

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
    i, j = n, best_j
    while i > 0:
        d = P[i,j]
        if d==0: i,j = i-1,j-1
        elif d==1: i,j = i-1,j
        else: i,j = i,j-1
        if j <= 0: break
    return j, best_j

# templates: snapped windows from the ORIGINAL 26 clips (features2), excluding the hit's own file
by_word_templates = {}
for rel, sr in snapped.items():
    if sr["mouthing_detected"] != "True": continue
    name = rel.replace("/", "__").replace(".mp4", ".npy")
    feats = np.load(FEAT2 / name)
    bon, boff = float(sr["burst_on_s"]), float(sr["burst_off_s"])
    templ = valid_slice(feats, int(bon*fps), int(boff*fps)+1)
    if len(templ) >= 3:
        by_word_templates.setdefault(sr["word"], []).append((rel, templ))

true_rel, matched_rel = [], []
err_a_list, err_b_list = [], []
used = []
for i, h in enumerate(hits):
    word = h["word"]
    target_file = h["file"]
    alt = [t for rel,t in by_word_templates.get(word, []) if rel != target_file and len(t) >= 3]
    if not alt:
        print(f"{i:02d} {word:12s} SKIP (no alternate snapped template)"); continue
    templ = alt[0]

    span_feats = np.load(HERE / "span_features" / f"{i:02d}_{word}.npy")
    sc = np.load(HERE / "sidecar_spans" / f"{i:02d}_{word}.npz")
    occluded = compute_occluded(sc["hand_dist"], sc["face_ok"])
    n = len(span_feats)
    extra_valid = (~np.isnan(span_feats[:,0])) & (~occluded[:n] if len(occluded)>=n else np.pad(~occluded,(0,n-len(occluded)),constant_values=True))

    res = subseq_dtw_masked(templ, span_feats, extra_valid)
    if res is None:
        print(f"{i:02d} {word:12s} SKIP (dtw failed)"); continue
    sj, ej = res
    span_dur_s = n / fps
    matched_center_s = (sj+ej)/2/fps
    matched_center_rel = matched_center_s / span_dur_s

    true_center_rel = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    true_center_s = true_center_rel * span_dur_s  # == h['sign_local'] - (cue_start-0.2) approx; use rel for consistency

    # truth (a): burst-detected mouthing within this span, nearest the true sign point
    span_res = detect_bursts(np.load(HERE/"span_features"/f"{i:02d}_{word}.npy")[:,1] if False else None, None, None) if False else None
    ap = None
    # recompute aperture directly from sidecar for the span (already have aperture there)
    span_bursts = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    true_frame = true_center_s*fps
    lo, hi = true_frame - SEARCH_PRE*fps, true_frame + SEARCH_POST*fps
    best_b, best_ov = None, -1
    for (bs,be) in span_bursts["bursts"]:
        ov = max(0, min(be,hi)-max(bs,lo))
        if ov > best_ov: best_ov, best_b = ov, (bs,be)
    truth_a_s = ((best_b[0]+best_b[1])/2)/fps if best_b and best_ov>0 else None

    err_b = abs(matched_center_s - true_center_s)
    err_a = abs(matched_center_s - truth_a_s) if truth_a_s is not None else None

    true_rel.append(true_center_rel); matched_rel.append(matched_center_rel)
    err_b_list.append(err_b)
    if err_a is not None: err_a_list.append(err_a)
    used.append(word)
    print(f"{i:02d} {word:12s} true_rel={true_center_rel:.2f} matched_rel={matched_center_rel:.2f}  err_b={err_b:.2f}s  err_a={'%.2fs'%err_a if err_a is not None else 'n/a'}")

true_rel, matched_rel = np.array(true_rel), np.array(matched_rel)
print(f"\nn = {len(true_rel)}")
if len(true_rel) >= 3:
    rho, p = spearmanr(true_rel, matched_rel, alternative='greater')
    print(f"Spearman rho={rho:.3f} p={p:.4f}  (PILOT REFERENCE: rho=-0.267, p=0.756)")
    err_b_arr = np.array(err_b_list)
    print(f"\nabs error vs truth (b) human sign point: mean={err_b_arr.mean():.2f}s median={np.median(err_b_arr):.2f}s")
    if err_a_list:
        err_a_arr = np.array(err_a_list)
        print(f"abs error vs truth (a) span-internal burst: n={len(err_a_arr)} mean={err_a_arr.mean():.2f}s median={np.median(err_a_arr):.2f}s")
