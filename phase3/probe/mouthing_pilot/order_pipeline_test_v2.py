"""Primary timing/order-based identification, no viseme yet. Pre-registered thresholds
(midpoint between real-word and other-burst medians measured earlier):
  DURATION_THRESH = 0.55s, EXTREMA_THRESH = 8
"""
import sys, re, json, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from burst_detect import detect_bursts

HERE = Path(__file__).parent
fps = 30.0
DURATION_THRESH = 0.55
EXTREMA_THRESH = 8

STOPWORDS = {w for w in (
    "el la los las un una unos unas lo al del de a en y o u que como cuando donde quien "
    "cual cuanto porque por para con sin sobre entre yo tu te me mi nos nosotros ellos "
    "ella ellas su sus se si no ni es son ser estar haber este esta esto ese esa eso "
    "aquel aqui ahi alla muy mas menos ya pero tambien").split()}

def norm(w):
    return re.sub(r"[^\wÀ-ÿ]", "", w.lower())

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

def n_extrema(sig, on, off):
    seg = sig[on:off+1]; seg = seg[~np.isnan(seg)]
    if len(seg) < 3: return 0
    d = np.diff(seg); signs = np.sign(d); signs = signs[signs != 0]
    if len(signs) < 2: return 0
    return int(np.sum(np.diff(signs) != 0))

B = parse_vtt(str(Path("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/sea_test/out/aligned_B/57TvyH9902U_slice.vtt")))
hits = json.load(open(HERE / "runB_hits.json"))

results = []
for i, h in enumerate(hits):
    st, en = h["cue_start"], h["cue_end"]
    match = [tx for cst,cen,tx in B if abs(cst-st)<0.5 and abs(cen-en)<0.5]
    if not match:
        print(f"{i:02d} {h['word']:12s} SKIP no cue text match"); continue
    tokens_raw = re.findall(r"[\wÀ-ÿ]+", match[0].lower())
    content = [(j,t) for j,t in enumerate(tokens_raw) if t not in STOPWORDS and norm(h["word"]) not in ("",)]
    content_words = [t for j,t in content]
    def _match(a,b):
        a,b=norm(a),norm(b)
        return a==b or a.startswith(b) or b.startswith(a)
    if not any(_match(h["word"], t) for t in content_words):
        print(f"{i:02d} {h['word']:12s} SKIP target not in content-word list (tokens: {tokens_raw})"); continue
    true_idx = next(j for j,t in enumerate(content_words) if _match(h["word"], t))
    N = len(content_words)
    true_expected_pos = true_idx / max(1, N-1)

    sc = np.load(HERE / "sidecar_spans_v2" / f"{i:02d}_{h['word']}.npz")
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    span_len = len(sc["aperture"])
    true_rel_time = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    true_time_s = true_rel_time * span_len / fps

    word_bursts = []
    for (on,off) in res["bursts"]:
        dur = (off-on)/fps
        ext = n_extrema(sc["aperture"], on, off)
        if dur >= DURATION_THRESH and ext >= EXTREMA_THRESH:
            word_bursts.append((on,off))

    if not word_bursts:
        print(f"{i:02d} {h['word']:12s} N={N:2d} true_idx={true_idx:2d} NO WORD-LIKE BURSTS DETECTED -> miss")
        results.append(dict(word=h["word"], found=False, err=None))
        continue

    # assign each word-like burst to nearest candidate word by order proximity
    assignments = []
    for (on,off) in word_bursts:
        burst_rel = ((on+off)/2) / span_len
        best_j = min(range(N), key=lambda j: abs(j/max(1,N-1) - burst_rel))
        assignments.append((on,off,best_j))

    claimed = [(on,off) for on,off,j in assignments if j == true_idx]
    if not claimed:
        print(f"{i:02d} {h['word']:12s} N={N:2d} true_idx={true_idx:2d} true_expected_pos={true_expected_pos:.2f} "
              f"-> {len(word_bursts)} word-bursts, NONE assigned to true word -> miss")
        results.append(dict(word=h["word"], found=False, err=None))
        continue

    on,off = min(claimed, key=lambda b: abs(((b[0]+b[1])/2)/fps - true_time_s))
    burst_time_s = ((on+off)/2)/fps
    err = abs(burst_time_s - true_time_s)
    print(f"{i:02d} {h['word']:12s} N={N:2d} true_idx={true_idx:2d} true_expected_pos={true_expected_pos:.2f} "
          f"-> ASSIGNED, err={err:.2f}s")
    results.append(dict(word=h["word"], found=True, err=err))

n_total = len(results)
n_found = sum(1 for r in results if r["found"])
errs = [r["err"] for r in results if r["found"]]
print(f"\nn={n_total}  found(claimed true word)={n_found} ({n_found/n_total:.0%})")
if errs:
    print(f"localization error among found: mean={np.mean(errs):.2f}s median={np.median(errs):.2f}s")
