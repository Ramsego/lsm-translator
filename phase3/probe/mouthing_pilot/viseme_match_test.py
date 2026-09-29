import sys, re, json, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from burst_detect import detect_bursts
from viseme_table import word_to_viseme_sequence
import mediapipe as mp

LIP_IDX = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))
IDX_61, IDX_291 = LIP_IDX.index(61), LIP_IDX.index(291)

HERE = Path(__file__).parent
FEAT_DIR = HERE / "span_features_v2"  # will build below if missing

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
    xs_src = np.linspace(0, 1, len(seq))
    xs_dst = np.linspace(0, 1, T)
    op = np.interp(xs_dst, xs_src, [s[0] for s in seq])
    ro = np.interp(xs_dst, xs_src, [s[1] for s in seq])
    return op, ro

def score_word(word, aperture, width):
    seq = word_to_viseme_sequence(word)
    if len(seq) < 1: return None
    T = len(aperture)
    if T < 3: return None
    op, ro = resample_seq(seq, T)
    a, w = aperture.copy(), width.copy()
    valid = ~np.isnan(a) & ~np.isnan(w)
    if valid.sum() < 3: return None
    if np.std(a[valid])<1e-9 or np.std(op[valid])<1e-9: c1 = 0.0
    else: c1 = np.corrcoef(a[valid], op[valid])[0,1]
    if np.std(w[valid])<1e-9 or np.std(ro[valid])<1e-9: c2 = 0.0
    else: c2 = -np.corrcoef(w[valid], ro[valid])[0,1]
    if np.isnan(c1): c1 = 0.0
    if np.isnan(c2): c2 = 0.0
    return c1 + c2

B = parse_vtt(str(Path("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/sea_test/out/aligned_B/57TvyH9902U_slice.vtt")))
hits = json.load(open(HERE / "runB_hits.json"))

# need span_features (80-dim) for the v2 spans to get width; only have aperture/hand/face in sidecar_spans_v2
# reuse sidecar_spans_v2 aperture; compute width by re-decoding needed columns -- but we didn't save width there.
# Quick fix: recompute width directly from sidecar_spans_v2 raw landmarks -- not saved. Use extract to add width.
print("NOTE: need width trace for v2 spans -- checking if saved")
import os
print(os.path.exists(HERE / "sidecar_spans_v2" / "00_gracias.npz"))
d = np.load(HERE / "sidecar_spans_v2" / "00_gracias.npz")
print(list(d.keys()))
