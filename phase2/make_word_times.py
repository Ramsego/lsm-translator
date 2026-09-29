"""
Build a word-time sidecar for a review batch.

WHY A SIDECAR AND NOT A COLUMN
------------------------------
`review.html` writes back only these columns on download:

    file,word,video_start,verdict,signer,sign_start,sign_end

Every other column in `review.csv` (context, surface, video, lag, ...) is DROPPED the
moment the reviewer exports.  So adding `word_start` to review.csv would silently
disappear.  This writes a separate file keyed on `file`, which survives the export and
is joined at analysis time.

WHY IT MATTERS
--------------
The headline product of a review batch is the pair

    (when the word was spoken, when it was signed)

`sign_start` comes from the reviewer.  The spoken time has to come from the ASR, and it
is currently NOT stored anywhere.  The manifest says clips begin `window_pre` seconds
before the word, but measured against the ASR the real offset is 0.28-0.82s rather than
a clean 1.0 -- something in the cutting shifts it.  So `video_start + window_pre` is not
safe, and every lag derived that way inherits a half-second of doubt.  That is the same
class of error as the derived +3.43s figure in WORD_SELECTION.md, which did not survive
recomputation.

This resolves it by going back to the ASR and recording the actual word interval.

Matching uses the `surface` column -- the exact inflected form the cutter matched -- not
a stem or prefix.  Prefix matching has produced a false result in this project four
separate times (actividad/actividades, joven/jovenes, encontrar/encuentra,
gusta/gustar).  Rows where the surface form appears more than once in the search window
are marked ambiguous rather than guessed at.

Usage:
    python phase2/make_word_times.py \
        --batch "/Volumes/Crucial X8/LSM_Translator/review/batch2" \
        --asr-dir phase3/local_drive_mirror/asr
"""

import argparse
import csv
import json
import unicodedata
from pathlib import Path

SEARCH_LO = -0.5    # seconds relative to video_start to start looking
SEARCH_HI = 4.0     # ... and stop looking


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.strip(".,;:!?¿¡\"'()")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--asr-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.batch / "review.csv")))
    asr_cache = {}

    def words_for(video):
        if video not in asr_cache:
            p = args.asr_dir / f"{video}.asr.json"
            asr_cache[video] = (sorted(json.load(open(p))["words"], key=lambda w: w["start"])
                                if p.exists() else None)
        return asr_cache[video]

    out_rows, stats = [], {"ok": 0, "ambiguous": 0, "no_match": 0, "no_asr": 0}
    for r in rows:
        rec = {"file": r["file"], "video": r["video"], "word": r["word"],
               "surface": r["surface"], "video_start": r["video_start"],
               "word_start": "", "word_end": "", "match": ""}
        w = words_for(r["video"])
        if w is None:
            rec["match"] = "no_asr"; stats["no_asr"] += 1
            out_rows.append(rec); continue

        vs = float(r["video_start"])
        target = norm(r["surface"])
        hits = [x for x in w
                if vs + SEARCH_LO <= x["start"] <= vs + SEARCH_HI and norm(x["word"]) == target]
        if not hits:
            rec["match"] = "no_match"; stats["no_match"] += 1
        else:
            best = min(hits, key=lambda x: x["start"])
            rec["word_start"] = round(best["start"], 3)
            rec["word_end"] = round(best["end"], 3)
            if len(hits) > 1:
                rec["match"] = f"ambiguous_x{len(hits)}"; stats["ambiguous"] += 1
            else:
                rec["match"] = "ok"; stats["ok"] += 1
        out_rows.append(rec)

    out = args.out or (args.batch / "word_times.csv")
    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        wtr.writeheader(); wtr.writerows(out_rows)

    n = len(out_rows)
    print(f"{n} clips")
    for k, v in stats.items():
        print(f"  {k:<10} {v:>4}  ({v/n:.0%})")
    got = [r for r in out_rows if r["word_start"] != ""]
    if got:
        offs = [float(r["word_start"]) - float(r["video_start"]) for r in got]
        offs.sort()
        print(f"\nword_start - video_start: min {offs[0]:.2f}  median {offs[len(offs)//2]:.2f}"
              f"  max {offs[-1]:.2f}   (manifest window_pre claims 1.00)")
    print(f"\nwrote {out}")
    print("Join to the exported review.csv on `file` when computing lag:")
    print("    lag = sign_start - word_end")


if __name__ == "__main__":
    main()
