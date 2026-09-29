"""
Align the speaker-tagged estenográfica transcript to the timed ASR word stream.

We have two half-useful word sources:
  - ASR (faster-whisper): has TIMESTAMPS, but recognition errors.
  - estenográfica transcript: AUTHORITATIVE wording + SPEAKER tags, but no times.

This matches them (difflib sequence alignment over normalized tokens) so each
transcript word inherits an ASR timestamp and each kept word carries its speaker.
Words ASR missed get times interpolated from neighbors; ASR errors/hallucinations are
dropped in favor of the transcript wording.

Output: timed, speaker-attributed, authoritative words →
    [{word, start, end, speaker, matched}]

Usage:
    python phase2/align_transcript.py --transcript <txt> --asr <asr.json> --out <json>
"""

import argparse
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

# speaker tag at line start: "PRESIDENTE ANDRÉS MANUEL LÓPEZ OBRADOR:", "PREGUNTA:", etc.
SPEAKER_RE = re.compile(r"^([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ .,'\-]{2,}?):\s")


def norm(w: str) -> str:
    """lowercase, strip accents/punct — for matching only (not the stored word)."""
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


def parse_transcript(path: Path):
    """Return [(word, speaker)] in order, tracking the current speaker by tag lines."""
    out = []
    speaker = "UNKNOWN"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            speaker = m.group(1).strip()
            line = line[m.end():]
        for tok in line.split():
            w = tok.strip()
            if w:
                out.append((w, speaker))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--asr", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    asr = json.load(open(args.asr))["words"]            # [{word,start,end}]
    tr = parse_transcript(args.transcript)              # [(word, speaker)]

    asr_keys = [norm(w["word"]) for w in asr]
    tr_keys = [norm(w) for w, _ in tr]

    sm = SequenceMatcher(a=tr_keys, b=asr_keys, autojunk=False)
    aligned = [None] * len(tr)                          # per transcript word: time or None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                a = asr[j1 + k]
                aligned[i1 + k] = (a["start"], a["end"], True)

    # interpolate times for unmatched transcript words from nearest matched neighbors
    last = None
    for i in range(len(tr)):
        if aligned[i]:
            last = aligned[i][1]
    # forward fill start times, backward fill — simple linear interpolation between anchors
    anchors = [(i, aligned[i][0]) for i in range(len(tr)) if aligned[i]]
    def interp_time(i):
        if not anchors:
            return 0.0
        prev = max([a for a in anchors if a[0] <= i], default=anchors[0], key=lambda x: x[0])
        nxt = min([a for a in anchors if a[0] >= i], default=anchors[-1], key=lambda x: x[0])
        if prev[0] == nxt[0]:
            return prev[1]
        frac = (i - prev[0]) / (nxt[0] - prev[0])
        return prev[1] + frac * (nxt[1] - prev[1])

    words = []
    for i, (w, spk) in enumerate(tr):
        if aligned[i]:
            s, e, matched = aligned[i]
        else:
            t = interp_time(i)
            s, e, matched = t, t, False
        words.append({"word": w, "start": round(float(s), 3),
                      "end": round(float(e), 3), "speaker": spk, "matched": matched})

    n_matched = sum(w["matched"] for w in words)
    speakers = sorted(set(w["speaker"] for w in words))
    args.out.write_text(json.dumps({
        "transcript": str(args.transcript), "asr": str(args.asr),
        "n_words": len(words), "n_matched": n_matched,
        "match_rate": round(n_matched / len(words), 3) if words else 0,
        "speakers": speakers, "words": words,
    }, ensure_ascii=False, indent=2))

    print(f"Transcript words: {len(words)}   ASR words: {len(asr)}")
    print(f"Matched (timed directly): {n_matched} ({100*n_matched/len(words):.0f}%); "
          f"rest interpolated.")
    print(f"Speakers found: {speakers}")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
