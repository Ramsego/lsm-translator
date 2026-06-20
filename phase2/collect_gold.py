"""
Collect human-verified review.csv verdicts into the central gold-label set.

review.csv (from make_review_clips.py) has columns: file, word, video_start, verdict.
The reviewer fills 'verdict' with y/n. This keeps the y's, attaches video + signer +
the clip window, and merges them into gold_labels.json (the GOLD tier — clean,
human-verified seeds + the leave-one-signer-out eval set).

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
    gold = []
    if args.gold.exists():
        gold = json.load(open(args.gold)).get("labels", [])
    seen = {(g["video_id"], g["word"], g["start_sec"]) for g in gold}

    added = kept = 0
    per_word = {}
    with open(args.review) as f:
        for row in csv.DictReader(f):
            kept += 1
            v = row.get("verdict", "").strip().lower()
            if v not in ("y", "yes", "1", "si", "sí"):
                continue
            clip_start = float(row["video_start"])
            # prefer the human-marked sign window; else fall back to the generous clip span
            ss, se = row.get("sign_start", "").strip(), row.get("sign_end", "").strip()
            if ss and se:
                start, end, localized = float(ss), float(se), True
            else:
                start, end, localized = clip_start, clip_start + 2 * args.win, False
            key = (args.video_id, row["word"], round(clip_start, 1))
            if key in seen:
                continue
            seen.add(key)
            # per-clip signer (rotating interpreters within a video); namespaced by video
            sig = (row.get("signer", "").strip() or args.signer or "1")
            signer_id = f"{args.video_id}:{sig}"
            gold.append({
                "word": row["word"], "video_id": args.video_id, "signer": signer_id,
                "start_sec": round(start, 1), "end_sec": round(end, 1),
                "localized": localized, "tier": "gold", "source": "human",
            })
            added += 1
            per_word[row["word"]] = per_word.get(row["word"], 0) + 1

    args.gold.parent.mkdir(parents=True, exist_ok=True)
    signers = sorted({g["signer"] for g in gold})
    words = sorted({g["word"] for g in gold})
    args.gold.write_text(json.dumps({
        "n_labels": len(gold), "signers": signers, "n_words": len(words),
        "labels": gold,
    }, ensure_ascii=False, indent=2))

    print(f"Reviewed rows: {kept}   added this pass: {added}")
    print(f"This pass by word: {per_word}")
    print(f"\nGOLD now: {len(gold)} labels, {len(words)} words, signers={signers}")
    print(f"Saved → {args.gold}")


if __name__ == "__main__":
    main()
