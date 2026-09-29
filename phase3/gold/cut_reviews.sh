#!/usr/bin/env bash
# Cut the review clip set for the SignCLIP gold labeling, across all transcribed mañaneras.
# For each video: make_review_clips.py (interpreter crop, lag-anchored, dict reference clip)
# then build_review_form.py → review/<ID>/review.html.
#
# cap = clips per word PER VIDEO (low, so each word spreads across interpreters).
#
# Usage:
#   caffeinate -i bash phase3/gold/cut_reviews.sh
set -uo pipefail

export PATH=/opt/miniconda3/bin:$PATH          # ffmpeg/ffprobe live here
PY=/opt/miniconda3/bin/python                   # base env has spaCy es_core_news_sm
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DR="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
WORDS_FILE="$REPO/phase3/gold/words.txt"
CAP="${CAP:-4}"
WIN="${WIN:-5}"          # seconds before lag center (sets clip start)
WINPOST="${WINPOST:-9}"  # seconds after lag center (extends clip tail)
OUT_ROOT="$DR/review"

WORDS=()
while IFS= read -r w; do [ -n "$w" ] && WORDS+=("$w"); done < "$WORDS_FILE"
IDS=(57TvyH9902U Ff4d82K1DJg cjpPmeg-Doo lrcmIZFvFgY oExqlI7iZU4 pMpjBLT7J_g
     tQymWk0YJqc u3Gu2gSxUVY wXHZn-8b28M zMjQwChNKHQ z_tL847-RQk)

log(){ echo "[$(date '+%H:%M:%S')] $*"; }
log "Cutting ${#WORDS[@]} words × ${#IDS[@]} videos  (cap=$CAP/word/video)  → $OUT_ROOT"

for ID in "${IDS[@]}"; do
    VIDEO="$DR/videos/mananera/$ID/$ID.mp4"
    ASR="$DR/videos/mananera/$ID/$ID.asr.json"
    ARRAYS="$DR/arrays/mananera/$ID"
    OUT="$OUT_ROOT/$ID"
    if [ ! -f "$VIDEO" ] || [ ! -f "$ASR" ] || [ ! -d "$ARRAYS" ]; then
        log "SKIP $ID (missing video/asr/arrays)"; continue
    fi
    log "▶ $ID"
    rm -rf "$OUT"                      # clip filenames change with the window → avoid orphans
    "$PY" "$REPO/phase2/make_review_clips.py" \
        --video "$VIDEO" --aligned "$ASR" --arrays "$ARRAYS" --out "$OUT" \
        --words "${WORDS[@]}" --max-per-word "$CAP" --win "$WIN" --win-post "$WINPOST" \
        && "$PY" "$REPO/phase2/build_review_form.py" --dir "$OUT" \
        && log "✓ $ID → $OUT/review.html" \
        || log "✗ $ID failed"
done
log "Done. One review.html per video under $OUT_ROOT/"
