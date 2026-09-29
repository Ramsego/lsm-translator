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
import shutil
import subprocess
import unicodedata
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}


def find_source_video(videos_root: Path, youtube_id: str):
    """First real video file (not a .info.json sidecar) for a dictionary youtube_id."""
    for p in videos_root.rglob(f"{youtube_id}.*"):
        if p.suffix.lower() in VIDEO_EXTS:
            return p
    return None

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


SENT_END = (".", "!", "?", "…")


def get_context(aligned, idx, max_words=30, fallback=12):
    """The sentence around idx (split on . ! ? …), capped, with the target wrapped in **...**.
    Falls back to a fixed ±fallback window if no sentence punctuation is nearby."""
    n = len(aligned)
    half = max_words // 2
    # extend left to the start of the current sentence (stop after a prior sentence-ending word)
    left = idx
    while left > 0 and (idx - left) < half:
        prev = aligned[left - 1]["word"].strip()
        if prev and prev[-1] in SENT_END:
            break
        left -= 1
    # extend right through the sentence-ending word
    right = idx
    while right < n - 1 and (right - idx) < half:
        cur = aligned[right]["word"].strip()
        if cur and cur[-1] in SENT_END:
            break
        right += 1
    if right - left < 3:                       # no punctuation found nearby → fixed window
        left = max(0, idx - fallback)
        right = min(n - 1, idx + fallback)
    parts = []
    for i in range(left, right + 1):
        w = aligned[i]["word"].strip()
        parts.append(f"**{w}**" if i == idx else w)
    return " ".join(parts)


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
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--metadata", type=Path, default=repo / "data" / "metadata.csv",
                    help="Dictionary metadata → reference citation clip per word.")
    ap.add_argument("--videos-root", type=Path, default=None,
                    help="Root that holds dictionary source videos (default: <video drive>/videos).")
    ap.add_argument("--ref-seconds", type=float, default=8.0,
                    help="Max length of the reference citation clip shown in the form.")
    args = ap.parse_args()

    # word(lemma key) -> dictionary youtube_id, for the side-by-side reference clip.
    # Uses the SAME key() as the occurrence grouping so cutter keys and dict keys match.
    dict_ref = {}
    if args.metadata.exists():
        for r in csv.DictReader(open(args.metadata)):
            lab = r["label"].strip()
            if not lab or "(" in lab or " " in lab:
                continue
            dict_ref.setdefault(key(lab), r["youtube_id"])
    videos_root = args.videos_root or (args.video.parents[2] if len(args.video.parents) >= 3 else args.video.parent)

    meta = json.load(open(args.arrays / "segments.json"))
    crop_changes = meta["crop_changes"]
    lag_file = args.arrays / "lag_estimate.json"
    if lag_file.exists():
        lag = json.load(open(lag_file)).get("lag_sec", 6.0)
    else:
        lag = 6.0
        print(f"  (no lag_estimate.json in {args.arrays} — using default lag {lag:.1f}s)")
    aligned = json.load(open(args.aligned))["words"]

    # pre-build index: object id → position in aligned list (for context lookup)
    word_index = {id(w): i for i, w in enumerate(aligned)}

    # lemmatize targets the same way occurrences are keyed, so "gracias" matches its lemma key
    targets = {key(w) for w in args.words} if args.words else None

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
    ref_dir = args.out / "_ref"
    ref_dir.mkdir(exist_ok=True)
    rows = []
    for k, ws in sorted(occ.items(), key=lambda kv: -len(kv[1])):
        wdir = args.out / k
        wdir.mkdir(exist_ok=True)
        win_post = args.win_post if args.win_post is not None else args.win

        # Build the dictionary citation clip once per word (side-by-side reference in the form).
        dict_yt = dict_ref.get(k, "")
        ref_rel = ""
        if dict_yt:
            ref_path = ref_dir / f"{k}.mp4"
            if not ref_path.exists():
                src = find_source_video(videos_root, dict_yt)
                if src:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(src), "-t", f"{args.ref_seconds:.1f}",
                         "-vf", "scale=-2:360", "-an", "-loglevel", "error", str(ref_path)],
                        check=False)
            if ref_path.exists():
                ref_rel = str(ref_path.relative_to(args.out))

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
            idx = word_index[id(w)]
            ctx = get_context(aligned, idx)
            rows.append({"file": str(out.relative_to(args.out)), "word": k,
                         "video_start": round(t0, 1), "context": ctx, "verdict": "",
                         "dict_youtube_id": dict_yt, "ref_clip": ref_rel})
        print(f"  {k}: {min(len(ws), args.max_per_word)} clips"
              f"{'' if ref_rel else '  (no ref clip)'}")

    csv_path = args.out / "review.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "word", "video_start", "context",
                                           "verdict", "dict_youtube_id", "ref_clip"])
        wr.writeheader(); wr.writerows(rows)

    print(f"\n{len(rows)} clips → {args.out}")
    print(f"Review sheet → {csv_path}")
    print("Workflow: open each <word>/ folder, watch the clips, and put y (interpreter signs "
          "this word) or n in the 'verdict' column of review.csv. y's become gold labels.")


if __name__ == "__main__":
    main()
