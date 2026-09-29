#!/usr/bin/env bash
# Run SEA (Segment + Align, no SignCLIP embeddings) on the LSM mañanera slice.
#
# Two runs, because the pre-alignment bias interacts with how we validate:
#
#   RUN A (bias 0): SEA must discover the whole speaker->interpreter offset itself.
#     This is the clean, independent test — if the shift it applies clusters near our
#     independently measured lag (6.33s from phase2/estimate_lag.py), it is genuinely
#     locating the signing rather than echoing the input timings.
#
#   RUN B (bias = measured lag): the realistic configuration — give it the correct prior,
#     as BOBSL does with its own ~2.6s offset, and let DP refine from there. Judged by the
#     verified-clip check, which is independent of the bias.
#
# Usage:  bash phase3/sea_test/run_sea.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEA="$REPO/phase3/sea_src"
TEST="$REPO/phase3/sea_test"
PY=/opt/miniconda3/envs/sea/bin/python
export PATH="/opt/miniconda3/envs/sea/bin:$PATH"

VID="57TvyH9902U_slice"
LAG=6.33

mkdir -p "$TEST/out/segmentation" "$TEST/out/aligned_A" "$TEST/out/aligned_B" "$TEST/ids"
echo "$VID" > "$TEST/ids/videos.txt"

echo "=============== 1. SEGMENT ==============="
cd "$SEA"
$PY segmentation.py --overwrite \
    --video_ids "$TEST/ids/videos.txt" \
    --pose_dir "$TEST/video" \
    --video_dir "$TEST/video" \
    --subtitle_dir "$TEST/subtitles" \
    --subtitle_dir_corrected "$TEST/subtitles" \
    --save_dir "$TEST/out/segmentation" \
    --sign-b-threshold 30 --sign-o-threshold 50 \
    --num_workers 1

SEG_DIR="$TEST/out/segmentation/E4s-1_30_50"
echo "segments produced:"
$PY - <<PY
import xml.etree.ElementTree as ET, glob
for f in sorted(glob.glob("$SEG_DIR/*.eaf")):
    t = ET.parse(f); r = t.getroot()
    for tier in r.iter('TIER'):
        n = len(list(tier.iter('ALIGNABLE_ANNOTATION')))
        if n: print(f"  {f.split('/')[-1]}  tier={tier.get('TIER_ID')}  n={n}")
PY

echo "=============== 2a. ALIGN (bias 0 — discover the offset) ==============="
$PY align.py --overwrite --mode=inference \
    --video_ids "$TEST/ids/videos.txt" \
    --num_workers 1 \
    --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 \
    --dp_max_gap 10 --dp_window_size 50 \
    --sign-b-threshold 30 --sign-o-threshold 50 \
    --pr_subs_delta_bias_start 0 --pr_subs_delta_bias_end 0 \
    --similarity_measure none \
    --pr_sub_path "$TEST/subtitles" \
    --gt_sub_path "$TEST/subtitles" \
    --segmentation_dir "$TEST/out/segmentation" \
    --save_dir "$TEST/out/aligned_A" || echo "RUN A FAILED"

echo "=============== 2b. ALIGN (bias = measured lag ${LAG}s) ==============="
$PY align.py --overwrite --mode=inference \
    --video_ids "$TEST/ids/videos.txt" \
    --num_workers 1 \
    --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 \
    --dp_max_gap 10 --dp_window_size 50 \
    --sign-b-threshold 30 --sign-o-threshold 50 \
    --pr_subs_delta_bias_start $LAG --pr_subs_delta_bias_end $LAG \
    --similarity_measure none \
    --pr_sub_path "$TEST/subtitles" \
    --gt_sub_path "$TEST/subtitles" \
    --segmentation_dir "$TEST/out/segmentation" \
    --save_dir "$TEST/out/aligned_B" || echo "RUN B FAILED"

echo "=============== 3. CHECK ==============="
cd "$REPO"
for R in A B; do
  OUT=$(find "$TEST/out/aligned_$R" -name "*.vtt" | head -1)
  if [ -n "$OUT" ]; then
    echo "--- RUN $R : $OUT ---"
    $PY phase3/sea_test/check_alignment.py \
        --original "$TEST/subtitles/$VID.vtt" \
        --aligned "$OUT" \
        --lag $LAG --slice-start 1200 --video-id 57TvyH9902U || true
  else
    echo "--- RUN $R : no output vtt found ---"
    find "$TEST/out/aligned_$R" -type f | head
  fi
done
