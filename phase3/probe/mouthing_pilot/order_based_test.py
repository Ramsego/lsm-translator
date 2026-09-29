import sys, csv, json, re
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from burst_detect import detect_bursts

HERE = Path(__file__).parent
fps = 30.0
hits = json.load(open(HERE / "runB_hits.json"))

orig_ctx = {r["file"]: r["context"] for r in csv.DictReader(open("/Volumes/Crucial X8/LSM_Translator/review/57TvyH9902U_slice_verify/review.csv"))}
rows = {r["file"].replace("../57TvyH9902U/",""): r
        for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]}

def word_order_position(context, target_word):
    """Return (index_of_bolded_word, total_words) using simple whitespace tokenization."""
    m = re.search(r"\*\*(.+?)\*\*", context)
    if not m: return None
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", context)
    tokens = re.findall(r"[\wÀ-ÿ]+", clean.lower())
    target_clean = re.sub(r"[^\wÀ-ÿ]", "", m.group(1).lower())
    if target_clean not in tokens: return None
    idx = tokens.index(target_clean)
    return idx, len(tokens)

true_rel_order, burst_rel_time, true_rel_time = [], [], []
loc_error_s = []
used = []

for i, h in enumerate(hits):
    rel = h["file"]
    r = rows.get(rel.replace("../57TvyH9902U/", ""))
    if r is None: continue
    ctx = orig_ctx.get(rel)
    if not ctx: print(f"{i:02d} {h['word']:12s} SKIP no context"); continue
    pos = word_order_position(ctx, h["word"])
    if pos is None: print(f"{i:02d} {h['word']:12s} SKIP word not found in tokenized context"); continue
    idx, total = pos
    word_rel = idx / max(1, total-1)

    sc = np.load(HERE / "sidecar_spans" / f"{i:02d}_{h['word']}.npz")
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    if not res["bursts"]:
        print(f"{i:02d} {h['word']:12s} SKIP no bursts detected in span"); continue
    # pick the burst with the largest peak |z| (most prominent activity) as "the mouthing event"
    best_burst = max(res["bursts"], key=lambda b: np.nanmax(np.abs(res["z"][b[0]:b[1]+1])) if b[1]>b[0] else 0)
    span_dur = len(sc["aperture"])
    burst_rel = ((best_burst[0]+best_burst[1])/2) / span_dur

    true_rel_t = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    true_time_s = true_rel_t * (span_dur/fps)
    burst_time_s = ((best_burst[0]+best_burst[1])/2) / fps
    err = abs(burst_time_s - true_time_s)

    true_rel_order.append(word_rel); burst_rel_time.append(burst_rel); true_rel_time.append(true_rel_t)
    loc_error_s.append(err); used.append(h["word"])
    print(f"{i:02d} {h['word']:12s} word_order_rel={word_rel:.2f} (pos {idx}/{total})  strongest_burst_rel_time={burst_rel:.2f}  true_rel_time={true_rel_t:.2f}  loc_error={err:.2f}s")

true_rel_order = np.array(true_rel_order); burst_rel_time = np.array(burst_rel_time)
loc_error_s = np.array(loc_error_s)
print(f"\nn = {len(true_rel_order)}")

rho1, p1 = spearmanr(true_rel_order, burst_rel_time, alternative='greater')
print(f"\n[Core hypothesis] Spearman(word's order-position in sentence, strongest-burst's time-position in span) = {rho1:.3f}  p={p1:.4f}")

print(f"\n[Sanity: does just picking the single strongest burst in the span already localize well?]")
print(f"loc_error vs TRUE sign time: mean={loc_error_s.mean():.2f}s  median={np.median(loc_error_s):.2f}s")
print(f"(pilot/Route1 comparison points: naive span-midpoint guesses were in the 4-5s range; DTW shape-matching Test 4 median AE was 1.19-1.49s)")
