#!/bin/bash
# Extract landmarks at NATIVE 30fps from all mañanera videos on disk.
# Streaming-to-disk + crash-safe; safe to re-run (skips videos already done at 30fps).
#
# Run (wrap in caffeinate so a multi-day run survives idle sleep):
#   caffeinate -i bash phase2/overnight_run.sh <VIDEO_ID> [VIDEO_ID ...]
# With no args, uses the default set below.
set -e

DATA_ROOT="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
VIDEOS_BASE="$DATA_ROOT/videos/mananera"
ARRAYS_BASE="$DATA_ROOT/arrays/mananera"
if [ "$#" -gt 0 ]; then
    IDS=("$@")
else
    IDS=(Z-pG_c9HoT8 pMpjBLT7J_g 2XU5NTX4Xfs)
fi
echo "Videos to process: ${IDS[*]}"

# ── 1. Ensure full videos are downloaded (skips any already present) ───────────
python3 phase2/download_mananera.py "${IDS[@]}"

# ── 2. Extract each at 30fps (--every 1, the script default) ──────────────────
for ID in "${IDS[@]}"; do
    VIDEO="$VIDEOS_BASE/$ID/$ID.mp4"
    OUT="$ARRAYS_BASE/$ID"
    if [ ! -f "$VIDEO" ]; then
        echo "WARNING: $VIDEO not found, skipping."
        continue
    fi

    # skip only if already extracted at native fps (every==1); otherwise (re)do it
    DONE=$(python3 - "$OUT/segments.json" <<'PY'
import json, sys
try:
    print(1 if json.load(open(sys.argv[1])).get("every") == 1 else 0)
except Exception:
    print(0)
PY
)
    if [ "$DONE" = "1" ]; then
        echo "Already extracted at 30fps: $ID, skipping."
        continue
    fi

    echo "=========================================="
    echo "Extracting (30fps): $ID"
    echo "=========================================="
    rm -rf "$OUT"            # clear any prior (e.g. 10fps) extraction
    python3 phase2/extract_continuous.py "$VIDEO" --every 1 --debug --out "$OUT"
done

echo ""
echo "All done."
