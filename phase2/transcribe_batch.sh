#!/usr/bin/env bash
# Run faster-whisper (Spanish, word timestamps) on every mañanera video that
# lacks a transcript. Saves <ID>.asr.json beside the video, then rebuilds the
# manifest so transcript_source flips to "whisper".
#
# Usage:
#   caffeinate -i bash phase2/transcribe_batch.sh [--model medium]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
VIDEOS_BASE="$DATA_ROOT/videos/mananera"
MODEL="medium"
if [[ "${1:-}" == "--model" && -n "${2:-}" ]]; then
    MODEL="$2"
fi

# IDs with transcript_source == "none"
MISSING=(
    57TvyH9902U
    Ff4d82K1DJg
    cjpPmeg-Doo
    lrcmIZFvFgY
    oExqlI7iZU4
    pMpjBLT7J_g
    tQymWk0YJqc
    u3Gu2gSxUVY
    wXHZn-8b28M
    zMjQwChNKHQ
    z_tL847-RQk
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for ID in "${MISSING[@]}"; do
    VIDEO="$VIDEOS_BASE/$ID/$ID.mp4"
    OUT="$VIDEOS_BASE/$ID/$ID.asr.json"

    if [ -f "$OUT" ]; then
        log "Already transcribed: $ID — skipping"
        continue
    fi

    if [ ! -f "$VIDEO" ]; then
        log "WARNING: video not found, skipping: $VIDEO"
        continue
    fi

    log "▶ Transcribing $ID (model=$MODEL)..."
    python3 "$PROJECT_ROOT/phase2/align_audio.py" "$VIDEO" --model "$MODEL" --out "$OUT"
    log "✓ $ID done → $OUT"
done

log "All transcriptions done — rebuilding manifest..."
python3 "$PROJECT_ROOT/phase2/manifest.py" --rebuild
log "Manifest updated."
