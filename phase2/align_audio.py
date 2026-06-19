"""
Test the audio->word alignment bridge: run faster-whisper on the section's audio to
get word-level timestamps. These anchor transcript words in video time (the
interpreter then signs them with a short lag).

Usage:
    python phase2/align_audio.py <video.mp4> [--seconds 120] [--model small]
"""

import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--model", default="small", help="faster-whisper model size.")
    args = ap.parse_args()

    print(f"Loading faster-whisper '{args.model}' (CPU int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"Transcribing first {args.seconds:.0f}s of {args.video.name} (es, word timestamps)...")
    segments, info = model.transcribe(
        str(args.video), language="es", word_timestamps=True,
        clip_timestamps=[0, args.seconds] if args.seconds else None,
    )

    n_words = 0
    print("\n--- word-level timestamps (first 40) ---")
    for seg in segments:
        for wd in (seg.words or []):
            if n_words < 40:
                print(f"  {wd.start:7.2f}-{wd.end:6.2f}  {wd.word.strip()}")
            n_words += 1
    print(f"\nTotal words timestamped: {n_words}")


if __name__ == "__main__":
    main()
