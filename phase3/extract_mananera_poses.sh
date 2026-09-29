#!/usr/bin/env bash
# Re-extract ALL mañanera videos to MediaPipe Holistic .pose files (for SignCLIP / SEA).
# Resumable (skips finished), CPU-bound, MULTI-DAY → always run under caffeinate.
#
# Must run in the probe env (Python 3.11, mediapipe<0.10.30):
#   conda activate lsm-probe
#   caffeinate -i bash phase3/extract_mananera_poses.sh
#
# Output: $LSM_DATA_ROOT/poses/mananera/<ID>.pose
set -uo pipefail

DRIVE="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
VIDEOS="$DRIVE/videos/mananera"
POSES="$DRIVE/poses/mananera"
LOG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/extract_mananera_poses.log"
mkdir -p "$POSES"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ── env sanity: legacy Holistic must import (this is the lsm-probe env, not base) ──
python3 -c "from pose_format.utils.holistic import load_holistic" 2>/dev/null || {
    log "✗ Holistic not importable — wrong env. Run:  conda activate lsm-probe"
    log "  (needs pose-format + mediapipe<0.10.30; base/3.13 will NOT work)"
    exit 1; }

# pick whichever extractor the install provides
if command -v video_to_pose >/dev/null; then
    extract(){ video_to_pose -i "$1" --format mediapipe -o "$2"; }   # per-file, no copy
elif command -v videos_to_poses >/dev/null; then
    extract(){ local t; t="$(mktemp -d)"; cp "$1" "$t/"; \
               videos_to_poses --format mediapipe --directory "$t" && mv "$t"/*.pose "$2"; \
               rm -rf "$t"; }
else
    log "✗ neither video_to_pose nor videos_to_poses found — is pose-format installed?"; exit 1
fi

mapfile -t VIDS < <(find "$VIDEOS" -type f \( -name '*.mp4' -o -name '*.mkv' -o -name '*.webm' \) | sort)
log "════════════════════════════════════════════"
log "Found ${#VIDS[@]} mañanera videos. Output → $POSES"
log "Heads-up: ~21h of video on CPU ≈ a couple of DAYS. It's resumable; safe to stop/restart."
log "════════════════════════════════════════════"

done=0; skip=0; fail=0; i=0
for V in "${VIDS[@]}"; do
    i=$((i+1))
    ID="$(basename "${V%.*}")"
    OUT="$POSES/$ID.pose"
    if [ -s "$OUT" ]; then
        log "[$i/${#VIDS[@]}] ✓ already done: $ID — skipping"; skip=$((skip+1)); continue
    fi
    log "[$i/${#VIDS[@]}] ▶ extracting $ID ..."
    if extract "$V" "$OUT"; then
        log "[$i/${#VIDS[@]}] ✓ $ID → $OUT"; done=$((done+1))
    else
        log "[$i/${#VIDS[@]}] ✗ FAILED: $ID (left for retry)"; rm -f "$OUT"; fail=$((fail+1))
    fi
done

log "════════════════════════════════════════════"
log "DONE.  extracted=$done  skipped=$skip  failed=$fail"
log "Poses in: $POSES"
[ "$fail" -gt 0 ] && log "Re-run the same command to retry the $fail failed video(s)." || true
