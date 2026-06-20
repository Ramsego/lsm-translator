"""
Cut short interpreter-only clips for human verification of weak labels.

For each occurrence of a target word in the timed transcript, cut the interpreter crop
over its lag-anchored window into clips/<word>/<word>_<mmss>.mp4 (2× upscaled, no audio).
Reviewer watches a folder of clips for one word back-to-back and marks each y/n in
review.csv. Survivors become GOLD labels (clean, human-verified).

Why per-word: you see the recurring sign across all instances → fast + validates recurrence.
Labels are video-time based → durable across 10fps↔30fps re-extraction.

Usage:
    python phase2/make_review_clips.py --video <mp4> --aligned <aligned.json> \\
        --arrays <dir> --out <clipdir> [--words gobierno dinero ...] [--win 4] [--max-per-word 25]
"""

import argparse
import csv
import json
import re
import subprocess
import unicodedata
from pathlib import Path

try:
    import spacy as _spacy
    _nlp = _spacy.load("es_core_news_sm", disable=["ner", "parser"])
    def _lemma(w: str) -> str:
        return _nlp(w)[0].lemma_.lower().strip()
except Exception:
    _nlp = None
    def _lemma(w: str) -> str:
        return w.lower()


def norm(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


def key(w: str) -> str:
    """Canonical key for grouping: lemma, then norm. Groups conjugations under one sign."""
    cleaned = w.strip().strip(".,;:¿?¡!()")
    if not cleaned:
        return ""
    lemma = _lemma(cleaned)
    return norm(lemma)


STOPWORDS = {norm(w) for w in (
    "el la los las un una unos unas lo al del de a en y o u que como cuando donde quien "
    "cual cuanto porque por para con sin sobre entre yo tu te me mi nos nosotros ellos "
    "ella el ellas su sus se si no ni es son ser estar haber este esta esto ese esa eso "
    "aquel aqui ahi alla muy mas menos ya pero tambien cosa cosas").split()}


def crop_at(crop_changes, t):
    """Interpreter crop active at video time t (last change with at_sec <= t)."""
    active = crop_changes[0]["crop"]
    for c in crop_changes:
        if c["at_sec"] <= t:
            active = c["crop"]
    return active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, required=True, help="Dir with segments.json + lag_estimate.json.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--words", nargs="*", default=None,
                    help="Target words; default = all content words present.")
    ap.add_argument("--win", type=float, default=4.0, help="Seconds BEFORE lag center (sets clip start).")
    ap.add_argument("--win-post", type=float, default=None,
                    help="Seconds AFTER lag center (extends clip end). Default = --win.")
    ap.add_argument("--pad", type=float, default=0.25, help="Crop padding fraction.")
    ap.add_argument("--max-per-word", type=int, default=25)
    args = ap.parse_args()

    meta = json.load(open(args.arrays / "segments.json"))
    crop_changes = meta["crop_changes"]
    lag = json.load(open(args.arrays / "lag_estimate.json")).get("lag_sec", 6.0)
    aligned = json.load(open(args.aligned))["words"]

    targets = {norm(w) for w in args.words} if args.words else None

    # group occurrences by lemma key
    from collections import defaultdict
    occ = defaultdict(list)
    for w in aligned:
        k = key(w["word"])
        if len(k) < 3:
            continue
        if targets is not None and k not in targets:
            continue
        if targets is None and k in STOPWORDS:
            continue
        occ[k].append(w)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, ws in sorted(occ.items(), key=lambda kv: -len(kv[1])):
        wdir = args.out / k
        wdir.mkdir(exist_ok=True)
        win_post = args.win_post if args.win_post is not None else args.win
        for w in ws[:args.max_per_word]:
            t0 = max(0, w["end"] + lag - args.win)   # clip START — keep stable to preserve filenames/progress
            dur = args.win + win_post                # extend the END only
            x, y, cw, ch = crop_at(crop_changes, t0)
            px, py = int(cw * args.pad), int(ch * args.pad)
            cx, cy = max(0, x - px), max(0, y - py)
            cw2, ch2 = cw + 2 * px, ch + 2 * py
            mmss = f"{int(t0)//60:02d}m{int(t0)%60:02d}s"
            out = wdir / f"{k}_{mmss}.mp4"
            cmd = ["ffmpeg", "-y", "-ss", f"{t0:.2f}", "-i", str(args.video),
                   "-t", f"{dur:.2f}",
                   "-filter:v", f"crop={cw2}:{ch2}:{cx}:{cy},scale=2*iw:2*ih",
                   "-an", "-loglevel", "error", str(out)]
            subprocess.run(cmd, check=False)
            rows.append({"file": str(out.relative_to(args.out)), "word": k,
                         "video_start": round(t0, 1), "verdict": ""})
        print(f"  {k}: {min(len(ws), args.max_per_word)} clips")

    csv_path = args.out / "review.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "word", "video_start", "verdict"])
        wr.writeheader(); wr.writerows(rows)

    print(f"\n{len(rows)} clips → {args.out}")
    print(f"Review sheet → {csv_path}")
    print("Workflow: open each <word>/ folder, watch the clips, and put y (interpreter signs "
          "this word) or n in the 'verdict' column of review.csv. y's become gold labels.")


if __name__ == "__main__":
    main()
