#!/usr/bin/env bash
# Download a batch of mañanera videos robustly.
# Retries each video up to 5 times with backoff.
# Prints a clear FAILED notice and exits non-zero if any video fails.
#
# Usage:  bash phase2/download_batch.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG="$PROJECT_ROOT/phase2/download_batch.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

download_video() {
    local video_id="$1"
    local extra_args="${2:-}"
    local attempt=1
    local max=5

    while [ $attempt -le $max ]; do
        log "▶ $video_id  attempt $attempt/$max"
        if python3 "$PROJECT_ROOT/phase2/download_mananera.py" $video_id $extra_args; then
            log "✓ $video_id  done"
            DOWNLOADED+=("$video_id")
            return 0
        fi
        log "  failed (attempt $attempt/$max)"
        attempt=$((attempt + 1))
        [ $attempt -le $max ] && sleep $((attempt * 10))
    done

    log "✗✗✗ FAILED after $max attempts: $video_id ✗✗✗"
    return 1
}

log "=== Batch download started ==="
FAILED=()
DOWNLOADED=()

# Full video — 2 new CDMX translators
download_video "LLvV-Arlokg" || FAILED+=("LLvV-Arlokg")

# Signing starts at 1:10:00 — skip the first 70 minutes (3h31m total)
download_video "cjpPmeg-Doo" "--start 01:10:00 --end 03:31:30" || FAILED+=("cjpPmeg-Doo")

if [ ${#FAILED[@]} -gt 0 ]; then
    log ""
    log "══════════════════════════════════════"
    log "BATCH FAILED — these videos did not download:"
    for v in "${FAILED[@]}"; do
        log "  ✗  $v  (https://www.youtube.com/watch?v=$v)"
    done
    log "══════════════════════════════════════"
    exit 1
fi

log "══════════════════════════════════════"
log "Downloads complete — starting extraction"
log "══════════════════════════════════════"

# Extract landmarks for all successfully downloaded videos
EXTRACT_IDS=()
for v in "${DOWNLOADED[@]}"; do
    EXTRACT_IDS+=("$v")
done

if [ ${#EXTRACT_IDS[@]} -gt 0 ]; then
    bash "$PROJECT_ROOT/phase2/overnight_run.sh" "${EXTRACT_IDS[@]}" 2>&1 | tee -a "$LOG"
    log "══════════════════════════════════════"
    log "BATCH COMPLETE — downloaded + extracted"
    log "══════════════════════════════════════"
fi
