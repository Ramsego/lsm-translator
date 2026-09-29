"""
Step 1 of the gold-label scheme: pick the ~50 target words for SignCLIP fine-tuning.

What it does
------------
1. Reads every mañanera ASR transcript (<ID>.asr.json: flat {start,end,word} list).
2. Lemmatizes each transcript IN CONTEXT with spaCy es_core_news_sm (so conjugated
   verbs/plurals collapse to the dictionary's canonical form), keeping NOUN/VERB/ADJ
   content words and dropping stopwords + proper nouns.
3. Intersects those lemmas with the Phase-1 dictionary's single-word labels — every kept
   word therefore has a citation-form reference clip (for training AND the annotator).
4. Flags negated occurrences (a negator in the preceding 2 tokens) so they can be excluded
   from the affirmative set later, and ranks words by affirmative mañanera frequency.

Outputs (to --out-dir, default phase3/gold/out):
  ranking.csv     — word, n_affirmative, n_negated, n_total, n_videos, dict_youtube_id
  candidates.jsonl— one row per word: {word, dict_youtube_id, counts, hits:[{video,start,end,negated,surface}]}

The ranking.csv is the verification gate: it tells us whether 50 words × ~20-25 instances
spanning several videos is actually achievable BEFORE we build the cutting/annotation tools.

Run (base env has spaCy 3.8 + es_core_news_sm; mediapipe not needed here):
  /opt/miniconda3/bin/python phase3/gold/select_words.py
  # optional: --data-root "/Volumes/Crucial X8/LSM_Translator" --top 80
"""

import argparse
import bisect
import csv
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path

import spacy

# Negation cues: an affirmative hit is reclassified "negated" if any of these appear in the
# 2 tokens before it (LSM negation ≠ "no"+verb, so these instances are routed out of v1).
NEGATORS = {"no", "nunca", "nada", "ni", "tampoco", "sin", "jamas"}
CONTENT_POS = {"NOUN", "VERB", "ADJ"}
LANG_TAG = "<es> <mfs>"  # carried into the manifest later; recorded here for reference


def norm(s: str) -> str:
    """Lowercase + strip accents (NFKD). Applied to BOTH ASR lemmas and dict labels so
    the intersection matches. Mirrors phase3/probe/signclip_probe.py:_norm."""
    s = unicodedata.normalize("NFKD", s.lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


def load_dictionary(metadata_csv: Path):
    """Single-word dictionary labels -> youtube_id of the citation clip.
    Drops phrases and variant markers ("aceite (a)") exactly like the probe's selector."""
    vocab = {}
    for r in csv.DictReader(open(metadata_csv)):
        lab = r["label"].strip()
        if not lab or "(" in lab or " " in lab:
            continue
        key = norm(lab)
        # keep the first occurrence; metadata is the reference-clip source
        vocab.setdefault(key, {"label": lab, "youtube_id": r["youtube_id"]})
    return vocab


def iter_transcripts(data_root: Path):
    base = data_root / "videos" / "mananera"
    for f in sorted(base.rglob("*.asr.json")):
        if f.name.startswith("._"):   # macOS AppleDouble sidecar, not a real transcript
            continue
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            print(f"  ! skipping unreadable {f.name}: {e}")
            continue
        vid = Path(d.get("video", f.stem)).stem.replace(".asr", "")
        words = d.get("words", [])
        if words:
            yield vid, words


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")))
    ap.add_argument("--metadata", type=Path, default=repo / "data" / "metadata.csv")
    ap.add_argument("--out-dir", type=Path, default=repo / "phase3" / "gold" / "out")
    ap.add_argument("--top", type=int, default=80, help="rows to print to the console")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dict_vocab = load_dictionary(args.metadata)
    print(f"Dictionary: {len(dict_vocab)} single-word labels (reference clips).")

    print("Loading spaCy es_core_news_sm ...")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    stop = {norm(w) for w in nlp.Defaults.stop_words}

    # word(normalized dict key) -> aggregated record
    hits = defaultdict(list)        # key -> list of (video, start, end, negated, surface)
    n_videos = 0

    for vid, words in iter_transcripts(args.data_root):
        n_videos += 1
        surfaces = [w["word"] for w in words]
        norm_surfaces = [norm(s) for s in surfaces]

        # Join into text and remember each ASR word's char span, so spaCy tokens map back
        # to a timestamp by offset (robust to spaCy splitting/merging tokens).
        text_parts, spans, pos = [], [], 0
        for i, s in enumerate(surfaces):
            text_parts.append(s)
            spans.append(pos)               # char offset where ASR word i starts
            pos += len(s) + 1               # +1 for the joining space
        text = " ".join(text_parts)
        span_starts = spans  # sorted ascending by construction

        doc = nlp(text)
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.is_stop:
                continue
            if tok.pos_ not in CONTENT_POS:
                continue
            lemma = norm(tok.lemma_)
            surf = norm(tok.text)
            if lemma in stop or len(lemma) < 3:
                continue
            # match against dictionary by lemma OR raw surface (maximizes recall)
            key = lemma if lemma in dict_vocab else (surf if surf in dict_vocab else None)
            if key is None:
                continue
            # which ASR word (timestamp) does this token belong to?
            ai = bisect.bisect_right(span_starts, tok.idx) - 1
            if ai < 0 or ai >= len(words):
                continue
            w = words[ai]
            negated = any(norm_surfaces[j] in NEGATORS for j in (ai - 1, ai - 2) if j >= 0)
            hits[key].append((vid, w["start"], w["end"], negated, surfaces[ai]))

    # ---- aggregate + write -------------------------------------------------------------
    rows = []
    cand_path = args.out_dir / "candidates.jsonl"
    with open(cand_path, "w") as cf:
        for key, hs in hits.items():
            # dedupe by (video, start)
            seen, uniq = set(), []
            for h in hs:
                k = (h[0], round(h[1], 2))
                if k not in seen:
                    seen.add(k); uniq.append(h)
            n_aff = sum(1 for h in uniq if not h[3])
            n_neg = sum(1 for h in uniq if h[3])
            aff_videos = {h[0] for h in uniq if not h[3]}
            rec = {
                "word": dict_vocab[key]["label"],
                "dict_youtube_id": dict_vocab[key]["youtube_id"],
                "lang_tag": LANG_TAG,
                "n_affirmative": n_aff,
                "n_negated": n_neg,
                "n_total": len(uniq),
                "n_videos": len(aff_videos),
                "hits": [{"video": h[0], "start": h[1], "end": h[2],
                          "negated": h[3], "surface": h[4]} for h in uniq],
            }
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)

    rows.sort(key=lambda r: (r["n_affirmative"], r["n_videos"]), reverse=True)
    rank_path = args.out_dir / "ranking.csv"
    with open(rank_path, "w", newline="") as rf:
        wr = csv.writer(rf)
        wr.writerow(["word", "n_affirmative", "n_negated", "n_total", "n_videos", "dict_youtube_id"])
        for r in rows:
            wr.writerow([r["word"], r["n_affirmative"], r["n_negated"],
                         r["n_total"], r["n_videos"], r["dict_youtube_id"]])

    # ---- console summary (the achievability gate) --------------------------------------
    achievable = [r for r in rows if r["n_affirmative"] >= 20 and r["n_videos"] >= 3]
    print(f"\nTranscripts scanned: {n_videos}")
    print(f"Dictionary words found in mañaneras: {len(rows)}")
    print(f"Words with >=20 affirmative hits across >=3 videos: {len(achievable)}  "
          f"(target = ~50)")
    print(f"\nTop {min(args.top, len(rows))} by affirmative frequency:")
    print(f"  {'word':22s} {'aff':>4} {'neg':>4} {'vids':>4}")
    for r in rows[:args.top]:
        print(f"  {r['word'][:22]:22s} {r['n_affirmative']:>4} {r['n_negated']:>4} {r['n_videos']:>4}")
    print(f"\nWrote:\n  {rank_path}\n  {cand_path}")


if __name__ == "__main__":
    main()
