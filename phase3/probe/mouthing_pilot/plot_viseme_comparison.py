import sys, re, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
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
    if len(seq) == 1: seq = seq*2
    xs_src = np.linspace(0,1,len(seq)); xs_dst = np.linspace(0,1,T)
    op = np.interp(xs_dst, xs_src, [s[0] for s in seq])
    ro = np.interp(xs_dst, xs_src, [s[1] for s in seq])
    return op, ro

B = parse_vtt(str(Path("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/sea_test/out/aligned_B/57TvyH9902U_slice.vtt")))
hits = json.load(open(HERE / "runB_hits.json"))

def get_case(i):
    h = hits[i]
    st, en = h["cue_start"], h["cue_end"]
    match = [tx for cst,cen,tx in B if abs(cst-st)<0.5 and abs(cen-en)<0.5][0]
    tokens_raw = re.findall(r"[\wÀ-ÿ]+", match.lower())
    content_words = [t for t in tokens_raw if t not in STOPWORDS]
    true_idx = next(j for j,t in enumerate(content_words) if _match(h["word"], t))
    sc = np.load(HERE / "sidecar_spans_v3" / f"{i:02d}_{h['word']}.npz")
    res = detect_bursts(sc["aperture"], sc["hand_dist"], sc["face_ok"])
    span_len = len(sc["aperture"])
    true_rel_time = (h["sign_local"] - h["cue_start"]) / (h["cue_end"] - h["cue_start"])
    true_time_s = 1.5 + true_rel_time * (h["cue_end"] - h["cue_start"])
    on,off = min(res["bursts"], key=lambda b: abs(((b[0]+b[1])/2)/fps - true_time_s))
    return h, content_words, true_idx, sc, on, off

def plot_case(ax_ap, ax_wd, i, wrong_idxs):
    h, content_words, true_idx, sc, on, off = get_case(i)
    ap = sc["aperture"][on:off+1]; wd = sc["width"][on:off+1]
    t = np.arange(len(ap))/fps
    ax_ap.plot(t, ap, 'k-', lw=2, label="REAL aperture (observed)")
    ax_wd.plot(t, wd, 'k-', lw=2, label="REAL width (observed)")
    colors = {"true": "green"}
    op, ro = resample_seq(word_to_viseme_sequence(content_words[true_idx]), len(ap))
    ax_ap.plot(t, op, '--', color="green", lw=2, label=f"expected: '{content_words[true_idx]}' (TRUE word)")
    ax_wd.plot(t, ro, '--', color="green", lw=2)
    for k, wi in enumerate(wrong_idxs):
        seq = word_to_viseme_sequence(content_words[wi])
        if not seq: continue
        op, ro = resample_seq(seq, len(ap))
        c = ["red","orange"][k%2]
        ax_ap.plot(t, op, ':', color=c, lw=1.5, label=f"expected: '{content_words[wi]}' (wrong)")
        ax_wd.plot(t, ro, ':', color=c, lw=1.5)
    ax_ap.set_title(f"'{h['word']}' — aperture (openness)")
    ax_wd.set_title(f"'{h['word']}' — width (roundedness, inverted)")
    ax_ap.legend(fontsize=7, loc="upper right")
    ax_ap.set_xlabel("seconds"); ax_wd.set_xlabel("seconds")

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
# case 00 gracias: rank 2/3 (decent). wrong candidates from earlier printout: presidenta
h0, cw0, ti0, *_ = get_case(0)
wrong0 = [j for j,w in enumerate(cw0) if w != cw0[ti0]][:1]
plot_case(axes[0][0], axes[1][0], 0, wrong0)

# case 01 actividad: rank 17/22 (clear failure). wrong top candidates: realizamos, fútbol
h1, cw1, ti1, *_ = get_case(1)
wrong_words = ["realizamos", "fútbol"]
wrong1 = [cw1.index(w) for w in wrong_words if w in cw1]
plot_case(axes[0][1], axes[1][1], 1, wrong1)

plt.tight_layout()
out = HERE / "viseme_comparison.png"
plt.savefig(out, dpi=130)
print("saved", out)
