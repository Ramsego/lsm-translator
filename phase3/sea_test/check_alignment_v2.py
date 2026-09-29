"""
Workstream B1: continuous-metric alignment checker. Companion to check_alignment.py
(baseline, do not edit), not a replacement -- run both and compare.

The binary check ("does the span contain the sign moment") dichotomises a continuous
quantity and, at n=26, cannot reliably detect anything below a ~18-25 point improvement.
This adds, per verified sign instance, against the aligned cue whose text contains that
word and is temporally nearest the true sign moment:

  contains      - the original binary (kept for continuity)
  signed_dist_s - sign_time - span_center, seconds (signed, so systematic bias is visible)
  abs_dist_s    - |signed_dist_s|
  norm_dist     - abs_dist_s / (span_duration/2); <1.0 means inside the span
  iou           - overlap([sign_start,sign_end], [span_start,span_end]) / union
  span_dur_s    - span duration (a method that "wins" by widening spans must be visible)

Usage:
    python phase3/sea_test/check_alignment_v2.py \
        --aligned phase3/sea_test/out/aligned_B/57TvyH9902U_slice.vtt \
        --slice-start 1200 --video-id 57TvyH9902U \
        --review-csv phase3/probe/review_2.csv
"""
import argparse
import csv
import re
import statistics as stats
import unicodedata
from pathlib import Path


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# Explicit accepted-forms map, one entry per gold word type in phase3/probe/review_2.csv
# with a verdict=y instance. Replaces an earlier stem-prefix heuristic (word's first
# min(6,len) chars matched anywhere at a token boundary) that was checked against the
# actual 17 gold word types on 2026-07-31 and found on both sides: it still missed
# stem-changing/irregular conjugations (e.g. encontrar->encuentra, not present in this
# transcript but a live gap in the heuristic) and it over-matched unrelated words sharing
# a prefix (director's stem also matched directamente, an unrelated adverb -- didn't flip
# a result here only because the spurious cue was ~200s away, not because the heuristic
# was safe). Forms below are the exhaustive set each word actually takes across
# subtitles/57TvyH9902U_slice.vtt and all four aligned_*/57TvyH9902U_slice.vtt variants
# (verified identical across variants) -- exact by construction. Extend by hand when new
# instances/words are verified into review_2.csv; do not fall back to stemming.
FORMS = {
    "gracias": ["gracias"],
    "joven": ["jovenes"],
    "actividad": ["actividades"],
    "atender": ["atendemos", "atender"],
    "equipo": ["equipo", "equipos"],
    "decidir": ["decidido", "decidio", "decidir"],
    "derecho": ["derechos"],
    "experiencia": ["experiencia", "experiencias"],
    "ganar": ["ganarnos", "ganaron"],
    "familia": ["familia", "familiares", "familias"],
    "encontrar": ["encontrar", "encontraran", "encontraras", "encontrarse"],
    "edad": ["edad"],
    "director": ["director"],
    "aprovechar": ["aprovecho"],
    "cambiar": ["cambiado", "cambiar"],
    "futuro": ["futuro"],
    "compartir": ["compartir"],
}


def matched_form(word, text):
    """Returns the specific accepted form found in text, or None. Accent-insensitive,
    exact token-boundary match against FORMS -- see module-level comment for rationale."""
    forms = FORMS.get(word.lower())
    if forms is None:
        raise KeyError(f"no FORMS entry for {word!r} -- add its accepted surface forms")
    t = _strip_accents(text.lower())
    for f in forms:
        if re.search(r"\b" + re.escape(_strip_accents(f)) + r"\b", t):
            return f
    return None


def word_in_text(word, text):
    return matched_form(word, text) is not None


def parse_vtt(path):
    cues, cur = [], None
    tpat = re.compile(
        r"(\d\d):(\d\d):(\d\d(?:\.\d+)?)\s*-->\s*(\d\d):(\d\d):(\d\d(?:\.\d+)?)"
    )
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = tpat.search(line)
        if m:
            h1, m1, s1, h2, m2, s2 = m.groups()
            st = int(h1) * 3600 + int(m1) * 60 + float(s1)
            en = int(h2) * 3600 + int(m2) * 60 + float(s2)
            cur = [st, en, []]
            cues.append(cur)
        elif cur is not None and line.strip() and line.strip() != "WEBVTT":
            cur[2].append(line.strip())
    return [(c[0], c[1], " ".join(c[2])) for c in cues if c[2]]


def recover_sign_times(rows, args):
    """Same recovery logic as check_alignment.py: gold-pipeline (video_start>0, sign_start
    already absolute) vs probe-pipeline (video_start==0 placeholder, needs manifest join)."""
    mf = Path(args.interp_manifest)
    clip_abs_by_id = {}
    if mf.exists():
        for r in csv.DictReader(open(mf)):
            m = re.search(r"_(\d+)m(\d+)s", r["src_path"])
            if m:
                clip_abs_by_id[r["clip_id"]] = int(m.group(1)) * 60 + int(m.group(2))

    out = []
    for r in rows:
        word = r["word"]
        video_start = float(r.get("video_start") or 0)
        if video_start > 0:
            sign_start_abs = float(r["sign_start"])
            sign_end_abs = float(r["sign_end"]) if r.get("sign_end") else sign_start_abs
        else:
            clip_id = Path(r["file"]).stem
            clip_abs = clip_abs_by_id.get(clip_id)
            if clip_abs is None:
                continue
            sign_start_abs = clip_abs + float(r["sign_start"])
            sign_end_abs = clip_abs + float(r["sign_end"]) if r.get("sign_end") else sign_start_abs
        out.append(dict(word=word, file=r["file"],
                         sign_start=sign_start_abs - args.slice_start,
                         sign_end=sign_end_abs - args.slice_start))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--slice-start", type=float, default=0.0)
    ap.add_argument("--slice-duration", type=float, default=1800.0)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--review-csv", default="phase3/probe/review_2.csv")
    ap.add_argument("--interp-manifest", default="phase3/probe/clips_interpreter/manifest.csv")
    ap.add_argument("--label", default=None, help="name for this run, for the results table")
    ap.add_argument("--dump-csv", default=None, help="write per-clip metrics to this CSV path")
    args = ap.parse_args()

    algn = parse_vtt(args.aligned)

    rows = [
        r for r in csv.DictReader(open(args.review_csv))
        if r.get("verdict") == "y" and r.get("sign_start")
        and args.video_id in r.get("file", "")
    ]
    instances = recover_sign_times(rows, args)

    results = []
    for inst in instances:
        word, sign_start, sign_end = inst["word"], inst["sign_start"], inst["sign_end"]
        if not (0 <= sign_start <= args.slice_duration):
            continue
        sign_mid = (sign_start + sign_end) / 2

        # candidates: any aligned cue containing an accepted surface form of the word
        cands = [(st, en, tx, matched_form(word, tx)) for st, en, tx in algn]
        cands = [c for c in cands if c[3] is not None]
        if not cands:
            print(f"  {word:12s} -> SKIP (word not found in any aligned cue text)")
            continue
        # nearest by center to the true sign moment -- this is the span under test,
        # hit or miss, so distance is always measured against something meaningful
        st, en, tx, form = min(cands, key=lambda c: abs((c[0] + c[1]) / 2 - sign_mid))

        span_center = (st + en) / 2
        span_dur = en - st
        contains = (st - 1.0 <= sign_mid <= en + 1.0)
        signed_dist = sign_mid - span_center
        abs_dist = abs(signed_dist)
        norm_dist = abs_dist / (span_dur / 2) if span_dur > 0 else float("inf")
        overlap = max(0.0, min(sign_end, en) - max(sign_start, st))
        union = max(sign_end, en) - min(sign_start, st)
        iou = overlap / union if union > 0 else 0.0

        results.append(dict(word=word, file=inst["file"], contains=contains,
                             signed_dist_s=signed_dist, abs_dist_s=abs_dist,
                             norm_dist=norm_dist, iou=iou, span_dur_s=span_dur))
        print(f"  {word:12s} matched={form:14s} sign_mid={sign_mid:8.1f}  span=[{st:8.1f},{en:8.1f}] dur={span_dur:6.1f}s"
              f"  dist={signed_dist:+7.2f}s  norm={norm_dist:5.2f}  iou={iou:.2f}"
              f"  {'HIT' if contains else 'miss'}")

    n = len(results)
    label = args.label or Path(args.aligned).parent.name
    print(f"\n=== {label}: n={n} ===")
    if n == 0:
        return

    if args.dump_csv:
        with open(args.dump_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["word", "file", "contains", "signed_dist_s",
                                               "abs_dist_s", "norm_dist", "iou", "span_dur_s"])
            w.writeheader()
            w.writerows(results)
    binary_rate = sum(r["contains"] for r in results) / n
    print(f"binary contains rate : {binary_rate:.1%} ({sum(r['contains'] for r in results)}/{n})")
    for metric in ["abs_dist_s", "signed_dist_s", "norm_dist", "iou", "span_dur_s"]:
        vals = [r[metric] for r in results]
        print(f"{metric:14s}: mean={stats.fmean(vals):7.3f}  median={stats.median(vals):7.3f}")

    # machine-readable line for aggregation across runs
    print(f"\nCSV,{label},{n},{binary_rate:.4f},"
          f"{stats.median([r['abs_dist_s'] for r in results]):.4f},"
          f"{stats.fmean([r['abs_dist_s'] for r in results]):.4f},"
          f"{stats.median([r['iou'] for r in results]):.4f},"
          f"{stats.median([r['span_dur_s'] for r in results]):.4f}")


if __name__ == "__main__":
    main()
