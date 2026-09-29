"""
Print the real transcript context around every candidate hit for a word.

This is the input to the sense-splitting step: before cutting review clips for a word,
look at HOW the word is actually used in the mañaneras. Yesterday's review showed that
words like `mesa` and `amigo` almost never carry their literal dictionary meaning in
political speech (mesa de diálogo, amigas y amigos) — so their candidates can't succeed
no matter how good the model is, and reviewing them wastes annotation time.

Usage:
    python phase3/gold/show_contexts.py mesa
    python phase3/gold/show_contexts.py amigo --window 12
"""
import argparse
import json
import os
from pathlib import Path

DRIVE = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator"))
REPO = Path(__file__).resolve().parents[2]
CANDIDATES = REPO / "phase3" / "gold" / "out" / "candidates.jsonl"


def load_asr_words(video_id):
    """Return the ASR word list for a video, or None if not extracted."""
    p = DRIVE / "videos" / "mananera" / video_id / f"{video_id}.asr.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return data.get("words") or data


def context_for(words, t, window):
    """Words within +-`window` positions of the entry nearest timestamp `t`."""
    idx = min(range(len(words)), key=lambda i: abs(float(words[i]["start"]) - t))
    lo, hi = max(0, idx - window), min(len(words), idx + window + 1)
    parts = []
    for i in range(lo, hi):
        w = words[i]["word"].strip()
        parts.append(f"**{w}**" if i == idx else w)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("word")
    ap.add_argument("--window", type=int, default=10, help="words of context each side")
    ap.add_argument("--max", type=int, default=40, help="max hits to print")
    args = ap.parse_args()

    entry = None
    with open(CANDIDATES) as f:
        for line in f:
            d = json.loads(line)
            if d["word"] == args.word:
                entry = d
                break
    if entry is None:
        raise SystemExit(f"'{args.word}' not found in {CANDIDATES}")

    print(f"=== {args.word} — {entry['n_total']} hits across {entry['n_videos']} videos ===\n")

    asr_cache = {}
    shown = 0
    for h in entry["hits"]:
        if shown >= args.max:
            break
        vid = h["video"]
        if vid not in asr_cache:
            asr_cache[vid] = load_asr_words(vid)
        words = asr_cache[vid]
        if not words:
            continue
        ctx = context_for(words, float(h["start"]), args.window)
        print(f"[{vid} @ {h['start']:.0f}s]  ...{ctx}...\n")
        shown += 1

    missing = [v for v, w in asr_cache.items() if not w]
    if missing:
        print(f"(no ASR found for: {', '.join(missing)})")


if __name__ == "__main__":
    main()
