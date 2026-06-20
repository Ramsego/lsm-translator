"""
Audio->word alignment bridge: run faster-whisper on a video's audio to get
word-level timestamps. These anchor spoken Spanish words in video time; the
interpreter signs them with a short lag (the basis for weak labeling).

Usage:
    python phase2/align_audio.py <video.mp4> [--seconds N] [--model small] [--out file.json]
    # omit --seconds to transcribe the whole video
"""

import argparse
import json
from pathlib import Path
from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--seconds", type=float, default=None,
                    help="Transcribe only the first N seconds (default: whole video).")
    ap.add_argument("--model", default="small", help="faster-whisper model size.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Save word timestamps as JSON (default: <video_stem>.asr.json beside video).")
    args = ap.parse_args()

    print(f"Loading faster-whisper '{args.model}' (CPU int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    span = f"first {args.seconds:.0f}s" if args.seconds else "whole video"
    print(f"Transcribing {span} of {args.video.name} (es, word timestamps)...")
    kwargs = dict(language="es", word_timestamps=True)
    if args.seconds:
        kwargs["clip_timestamps"] = [0, args.seconds]
    segments, info = model.transcribe(str(args.video), **kwargs)

    words = []
    for seg in segments:
        for wd in (seg.words or []):
            words.append({"start": round(wd.start, 3),
                          "end": round(wd.end, 3),
                          "word": wd.word.strip()})

    print("\n--- word-level timestamps (first 40) ---")
    for w in words[:40]:
        print(f"  {w['start']:7.2f}-{w['end']:6.2f}  {w['word']}")
    print(f"\nTotal words timestamped: {len(words)}")

    out = args.out or args.video.with_suffix(".asr.json")
    out.write_text(json.dumps({"video": str(args.video), "model": args.model,
                               "words": words}, ensure_ascii=False, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
