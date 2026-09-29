"""
Build a WebVTT subtitle file (SEA's expected subtitle input) from our ASR word timings.

SEA aligns subtitle *sentences* to signing spans, so we group ASR words into sentences
on terminal punctuation and emit one cue per sentence, with times in the SLICE's local
timeline (i.e. minus --start).

Note these are AUDIO times (when the speaker said it). The interpreter lags by ~6s, and
correcting that lag is exactly what SEA is supposed to figure out — so we deliberately do
NOT pre-shift the timings here. Comparing SEA's output shift against our measured lag
(6.33s +- 3.48s for this video) is the sanity check.

Usage:
    python phase3/sea_test/make_vtt.py --video-id 57TvyH9902U --start 1200 --duration 1800
"""
import argparse
import json
import os
import re
from pathlib import Path

DRIVE = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator"))


def ts(seconds):
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="slice start in source video (s)")
    ap.add_argument("--duration", type=float, default=None, help="slice length (s)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    asr_path = DRIVE / "videos" / "mananera" / args.video_id / f"{args.video_id}.asr.json"
    data = json.loads(asr_path.read_text())
    words = data.get("words") or data

    end_limit = args.start + args.duration if args.duration else float("inf")

    sentences, cur = [], []
    for w in words:
        st, en = float(w["start"]), float(w["end"])
        if en < args.start or st > end_limit:
            if cur and st > end_limit:
                break
            continue
        cur.append(w)
        if re.search(r"[.!?]$", w["word"].strip()):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    out = args.out or Path(__file__).parent / "subtitles" / f"{args.video_id}_slice.vtt"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]
    n = 0
    for sent in sentences:
        text = " ".join(w["word"].strip() for w in sent).strip()
        if not text:
            continue
        s = float(sent[0]["start"]) - args.start
        e = float(sent[-1]["end"]) - args.start
        if e <= 0 or s >= (args.duration or float("inf")):
            continue
        n += 1
        lines.append(f"{ts(s)} --> {ts(e)}")
        lines.append(text)
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} — {n} sentence cues")
    if n:
        print("First 3 cues:")
        print("\n".join(lines[2:11]))


if __name__ == "__main__":
    main()
