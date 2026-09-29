"""
Audio -> word alignment via faster-whisper (Spanish, word timestamps).

Decodes the audio in fixed-length CHUNKS (default 10 min) so long videos do not
OOM — the previous whole-file decode got Killed:9 on the 148-min mañanera.

Usage:
    python phase2/align_audio.py <video.mp4> [--model medium] [--out file.json] [--chunk-minutes 10]
    # --chunk-minutes 0 restores the old whole-file behavior
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel


def chunk_status(returncode: int, wav_size: int, min_bytes: int = 2000) -> str:
    """Classify one ffmpeg chunk-extraction attempt.

    "ok"          -- ffmpeg succeeded and produced real audio.
    "end_of_audio" -- ffmpeg succeeded but produced a near-empty wav, which
                      only means anything past the real end of the file
                      (ffmpeg exiting 0 on a -ss past EOF writes a tiny/empty
                      wav rather than erroring).
    "error"       -- ffmpeg failed (nonzero return code). A tiny/missing wav
                      here is NOT proof of end-of-audio -- it's a transient
                      failure, and treating it as "past the end" would
                      silently truncate the transcript.
    """
    if returncode != 0:
        return "error"
    if wav_size < min_bytes:
        return "end_of_audio"
    return "ok"


def _transcribe_whole(model, video, seconds=None):
    kwargs = dict(language="es", word_timestamps=True)
    if seconds:
        kwargs["clip_timestamps"] = [0, seconds]
    segments, _ = model.transcribe(str(video), **kwargs)
    words = []
    for seg in segments:
        for wd in (seg.words or []):
            words.append({"start": round(wd.start, 3), "end": round(wd.end, 3), "word": wd.word.strip()})
    return words


def _extract_chunk_wav(video, t, chunk_s):
    """Run ffmpeg once for one chunk. Returns (returncode, wav_path, wav_size, stderr)."""
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", str(t), "-t", str(chunk_s), "-i", str(video),
         "-ac", "1", "-ar", "16000", "-vn", wav],
        check=False, capture_output=True, text=True,
    )
    size = os.path.getsize(wav) if os.path.exists(wav) else 0
    return proc.returncode, wav, size, proc.stderr


def _transcribe_chunked(model, video, chunk_s):
    """Extract each chunk's audio with ffmpeg, transcribe it, offset times back to absolute.

    A transient ffmpeg failure (nonzero return code) is NOT treated as
    end-of-audio -- that used to silently truncate the transcript. On error
    the chunk is retried once; if it fails again this exits the process
    without writing an output JSON, so a truncated transcript is never
    silently produced.
    """
    words, t = [], 0.0
    while True:
        rc, wav, size, stderr = _extract_chunk_wav(video, t, chunk_s)
        status = chunk_status(rc, size)

        if status == "error":
            print(f"  ffmpeg failed on chunk at {t/60:.0f} min (rc={rc}); retrying once...")
            if os.path.exists(wav):
                os.unlink(wav)
            rc, wav, size, stderr = _extract_chunk_wav(video, t, chunk_s)
            status = chunk_status(rc, size)

        if status == "error":
            if os.path.exists(wav):
                os.unlink(wav)
            print(f"ERROR: ffmpeg failed twice on chunk at {t/60:.0f} min (rc={rc}). "
                  f"Aborting WITHOUT writing output (a truncated transcript would be worse "
                  f"than none). ffmpeg stderr:\n{stderr}")
            sys.exit(1)

        if status == "end_of_audio":
            if os.path.exists(wav):
                os.unlink(wav)
            break

        segs, _ = model.transcribe(wav, language="es", word_timestamps=True)
        n0 = len(words)
        for seg in segs:
            for wd in (seg.words or []):
                words.append({"start": round(wd.start + t, 3),
                              "end": round(wd.end + t, 3),
                              "word": wd.word.strip()})
        os.unlink(wav)
        print(f"  {t/60:5.0f}-{(t+chunk_s)/60:>4.0f} min   +{len(words)-n0} words   ({len(words)} total)")
        t += chunk_s
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default="medium", help="faster-whisper model size.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: <video_stem>.asr.json beside the video.")
    ap.add_argument("--chunk-minutes", type=float, default=10.0,
                    help="Decode audio in chunks this long (0 = whole file, may OOM on long videos).")
    ap.add_argument("--seconds", type=float, default=None, help="Spike: transcribe only first N seconds.")
    args = ap.parse_args()

    print(f"Loading faster-whisper '{args.model}' (CPU int8)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    if args.chunk_minutes and not args.seconds:
        print(f"Transcribing {args.video.name} in {args.chunk_minutes:.0f}-min chunks (es, word timestamps)...")
        words = _transcribe_chunked(model, args.video, args.chunk_minutes * 60)
    else:
        print(f"Transcribing {args.video.name} (es, whole file)...")
        words = _transcribe_whole(model, args.video, args.seconds)

    print(f"Total words timestamped: {len(words)}")
    out = args.out or args.video.with_suffix(".asr.json")
    out.write_text(json.dumps({"video": str(args.video), "model": args.model, "words": words},
                              ensure_ascii=False, indent=2))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
