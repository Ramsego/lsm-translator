import sys, re, json, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from burst_detect import detect_bursts
from viseme_table_v2 import word_to_viseme_sequence

HERE = Path(__file__).parent
fps = 30.0
STOPWORDS = {w for w in (
    "el la los las un una unos unas lo al del de a en y o u que como cuando donde quien "
    "cual cuanto porque por para con sin sobre entre yo tu te me mi nos nosotros ellos "
    "ella ellas su sus se si no ni es son ser estar haber este esta esto ese esa eso "
    "aquel aqui ahi alla muy mas menos ya pero tambien").split()}

def norm(w): return re.sub(r"[^\wÀ-ÿ]", "", w.lower())
def _match(a,b):
    a,b = norm(a), norm(b)
    return a==b or a.startswith(b) or b.startswith(a)

def parse_vtt(path):
    cues, cur = [], None
    tpat = re.compile(r"(\d\d):(\d\d):(\d\d(?:\.\d+)?)\s*-->\s*(\d\d):(\d\d):(\d\d(?:\.\d+)?)")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = tpat.search(line)
        if m:
            h1,m1,s1,h2,m2,s2 = m.groups()
            st = int(h1)*3600+int(m1)*60+float(s1); en = int(h2)*3600+int(m2)*60+float(s2)
            cur=[st,en,[]]; cues.append(cur)
        elif cur is not None and line.strip() and line.strip()!="WEBVTT":
            cur[2].append(line.strip())
    return [(c[0],c[1]," ".join(c[2])) for c in cues if c[2]]

def resample_seq(seq, T):
    if len(seq) == 0: return None
    if len(seq) == 1: seq = seq*2
    xs_src = np.linspace(0, 1, len(seq)); xs_dst = np.linspace(0, 1, T)
    op = np.interp(xs_dst, xs_src, [s[0] for s in seq])
    ro = np.interp(xs_dst, xs_src, [s[1] for s in seq])
    return op, ro

def score_word(word, aperture, width):
    seq = word_to_viseme_sequence(word)
    if len(seq) < 1: return None
    T = len(aperture)
    if T < 3: return None
    op, ro = resample_seq(seq, T)
    valid = ~np.isnan(aperture) & ~np.isnan(width)
    if valid.sum() < 3: return None
    a, w = aperture[valid], width[valid]; op, ro = op[valid], ro[valid]
    c1 = 0.0 if np.std(a)<1e-9 or np.std(op)<1e-9 else np.corrcoef(a,op)[0,1]
    c2 = 0.0 if np.std(w)<1e-9 or np.std(ro)<1e-9 else -np.corrcoef(w,ro)[0,1]
    c1 = 0.0 if np.isnan(c1) else c1
    c2 = 0.0 if np.isnan(c2) else c2
    return c1 + c2

B = parse_vtt(str(Path("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/sea_test/out/aligned_B/57TvyH9902U_slice.vtt")))
hits = json.load(open(HERE / "runB_hits.json"))

ranks, ns = [], []
for i, h in enumerate(hits):
    st, en = h["cue_start"], h["cue_end"]
    match = [tx for cst,cen,tx in B if abs(cst-st)<0.5 and abs(cen-en)<0.5]
    if not match: continue
    tokens_raw = re.findall(r"[\wÀ-ÿ]+", match[0].lower())
    content_words = [t for t in tokens_raw if t not in STOPWORDS]
    if not any(_match(h["word"], t) for t in content_words): continue
    true_idx = next(j for j,t in enumerate(content_words) if _match(h["word"], t))
    N = len(content_words)

    sc = np.load(HERE / "sidecar_spans_v3" / f"{i:02d}_{h['word']}.npz")
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    span_len = len(sc["aperture"])
    true_rel_time = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    # account for the 1.5s pre-pad added when v2 spans were cut
    pad_pre = 1.5
    orig_span_dur = h["cue_end"] - h["cue_start"]
    true_time_s = pad_pre + true_rel_time * orig_span_dur

    true_burst = min(res["bursts"], key=lambda b: abs(((b[0]+b[1])/2)/fps - true_time_s), default=None)
    if true_burst is None:
        print(f"{i:02d} {h['word']:12s} SKIP no bursts at all"); continue
    on, off = true_burst
    burst_time = ((on+off)/2)/fps
    if abs(burst_time - true_time_s) > 2.0:
        print(f"{i:02d} {h['word']:12s} SKIP nearest burst too far from true time ({burst_time:.2f} vs {true_time_s:.2f})"); continue

    ap_seg = sc["aperture"][on:off+1]; wd_seg = sc["width"][on:off+1]
    scores = []
    for j, cw in enumerate(content_words):
        s = score_word(cw, ap_seg, wd_seg)
        if s is not None: scores.append((j, s))
    scores.sort(key=lambda x: -x[1])
    rank = next((k+1 for k,(j,s) in enumerate(scores) if j==true_idx), None)
    if rank is None:
        print(f"{i:02d} {h['word']:12s} SKIP true word not scoreable"); continue
    ranks.append(rank); ns.append(N)
    print(f"{i:02d} {h['word']:12s} N={N:3d} true_idx={true_idx:2d} rank={rank:3d}/{len(scores)}  top3={[content_words[j] for j,s in scores[:3]]}")

ranks = np.array(ranks); ns = np.array(ns)
chance = (ns+1)/2
print(f"\nn={len(ranks)}")
print(f"median rank={np.median(ranks):.1f}  median chance={np.median(chance):.1f}")
stat,p = wilcoxon(ranks-chance, alternative='less')
print(f"Wilcoxon vs per-sentence chance: p={p:.4f}  RESULT: {'PASS' if p<0.05 else 'FAIL'}")
top1 = np.mean(ranks==1); top3 = np.mean(ranks<=3)
print(f"top-1={top1:.1%}  top-3={top3:.1%}")
