"""
Build a review.csv (in build_review_form.py's expected format) for the 32 interpreter
clips selected by extract_interpreter_poses.sh, so we can reuse the existing review UI
to verify each clip before trusting it in the SignCLIP cross-signer test.

Two failure modes make a weak-label clip untrustworthy as ground truth (both flagged
2026-07-29, before running Cell 8):
  1. The interpreter used a different/related sign for the same concept (e.g. dinero ->
     RECURSOS) — the target word was never actually signed here.
  2. The lag-based timing window is loose (~14s) and only a soft estimate (+-3.5s) — the
     sign may be off-center, clipped, or the window may have caught a neighboring sign.

Reviewing here produces (a) a y/n/neg verdict per clip and (b) an exact sign_start/
sign_end within the clip, which cut_verified_interp_clips.py then uses to make a TIGHT
clip before pose extraction — instead of blindly keeping the first ~8.5s of the loose
14s window (what Cell 8 currently does).

Usage: python phase3/probe/make_interp_review_csv.py
Then:  python phase2/build_review_form.py --dir phase3/probe
       open phase3/probe/review.html, review all 32, Download review.csv,
       save it to phase3/probe/review.csv (overwrite this generated one).
"""
import csv
from pathlib import Path

WORK = Path(__file__).parent

dict_by_word = {}
with open(WORK / "clips" / "manifest.csv") as f:
    for r in csv.DictReader(f):
        dict_by_word[r["label"]] = r["clip"]   # e.g. agua -> TT15eicsCrQ.mp4

rows = []
with open(WORK / "clips_interpreter" / "manifest.csv") as f:
    for r in csv.DictReader(f):
        word = r["word"]
        rows.append({
            "file": f"clips_interpreter/{r['clip_id']}.mp4",
            "word": word,
            "video_start": "0",   # these are already-cut clips; times below are clip-relative
            "context": f"signer: {r['signer']}",
            "ref_clip": f"clips/{dict_by_word[word]}" if word in dict_by_word else "",
            "dict_youtube_id": "",
        })

out = WORK / "review.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file", "word", "video_start", "context", "ref_clip", "dict_youtube_id"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {out} ({len(rows)} clips). Next: python phase2/build_review_form.py --dir {WORK}")
