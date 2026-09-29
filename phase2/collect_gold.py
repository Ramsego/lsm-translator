"""
Collect human-verified review.csv verdicts into the central gold-label set.

review.csv (from make_review_clips.py) has columns:
  file, word, video_start, verdict, signer, sign_start, sign_end

Verdict routing:
  y / yes / 1 / si → gold_labels under the word key (training data)
  neg              → gold_labels under "{word}:neg" (negated form = distinct sign class)
  n / no           → implicit_signs (sign not present; linguistically noted, not trained on)
  '' / skip        → ignored

Usage:
    python phase2/collect_gold.py --review <review.csv> --video-id 2XU5NTX4Xfs \\
        --signer amlo_interp_A [--win 4] [--gold <gold_labels.json>]
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_GOLD = Path("phase2/gold_labels.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--signer", default=None,
                    help="Fallback signer id if a row has no per-clip signer tag.")
    ap.add_argument("--win", type=float, default=4.0, help="Half-window used when clips were cut.")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = ap.parse_args()

    # load existing gold (merge across batches/signers); de-dup by (video,word,start)
    existing = {"labels": [], "implicit_signs": []}
    if args.gold.exists():
        existing = json.load(open(args.gold))
        if "implicit_signs" not in existing:
            existing["implicit_signs"] = []

    gold = existing["labels"]
    implicit = existing["implicit_signs"]

    seen_gold = {(g["video_id"], g["word"], g["start_sec"]) for g in gold}
    seen_implicit = {(g["video_id"], g["word"], g["start_sec"]) for g in implicit}

    added_gold = added_implicit = kept = 0
    per_word: dict = {}

    with open(args.review) as f:
        for row in csv.DictReader(f):
            kept += 1
            v = row.get("verdict", "").strip().lower()
            if not v:
                continue

            clip_start = float(row["video_start"])
            ss = row.get("sign_start", "").strip()
            se = row.get("sign_end", "").strip()
            sig = (row.get("signer", "").strip() or args.signer or "1")
            signer_id = f"{args.video_id}:{sig}"

            if v in ("y", "yes", "1", "si", "sí", "neg"):
                # gold: confirmed sign (y) or negated form (neg → word:neg class)
                word_key = row["word"] if v != "neg" else f"{row['word']}:neg"
                if ss and se:
                    start, end, localized = float(ss), float(se), True
                else:
                    start, end, localized = clip_start, clip_start + 2 * args.win, False
                key = (args.video_id, word_key, round(clip_start, 1))
                if key in seen_gold:
                    continue
                seen_gold.add(key)
                gold.append({
                    "word": word_key, "video_id": args.video_id, "signer": signer_id,
                    "start_sec": round(start, 1), "end_sec": round(end, 1),
                    "localized": localized, "tier": "gold", "source": "human",
                })
                added_gold += 1
                per_word[word_key] = per_word.get(word_key, 0) + 1

            elif v in ("n", "no"):
                # implicit: sign was not present; signer conveyed meaning another way
                key = (args.video_id, row["word"], round(clip_start, 1))
                if key in seen_implicit:
                    continue
                seen_implicit.add(key)
                implicit.append({
                    "word": row["word"], "video_id": args.video_id, "signer": signer_id,
                    "start_sec": round(clip_start, 1),
                })
                added_implicit += 1

    args.gold.parent.mkdir(parents=True, exist_ok=True)
    signers = sorted({g["signer"] for g in gold})
    words = sorted({g["word"] for g in gold})
    args.gold.write_text(json.dumps({
        "n_labels": len(gold), "signers": signers, "n_words": len(words),
        "labels": gold,
        "implicit_signs": implicit,
    }, ensure_ascii=False, indent=2))

    print(f"Reviewed rows: {kept}   gold added: {added_gold}   implicit added: {added_implicit}")
    if per_word:
        print(f"This pass by word: {per_word}")
    print(f"\nGOLD now: {len(gold)} labels, {len(words)} words, signers={signers}")
    print(f"Implicit signs: {len(implicit)} total")
    print(f"Saved → {args.gold}")


if __name__ == "__main__":
    main()
