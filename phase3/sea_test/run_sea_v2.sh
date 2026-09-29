#!/usr/bin/env bash
# Workstream A3: re-run SEA align with A1's phantom-segment-filtered segmentation.
# A2 (subtitle re-segmentation) turned out to be a no-op -- the existing subtitles are
# already sentence-segmented (232/233 cues end in terminal punctuation; the one exception
# is the slice boundary cutting off the ASR transcript mid-sentence, not a bug). So this
# only changes --segmentation_dir; everything else is identical to run_sea.sh.
#
# Does NOT re-run segmentation.py -- filter_segments.py already produced the filtered
# .eaf from the existing segmentation output.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEA="$REPO/phase3/sea_src"
TEST="$REPO/phase3/sea_test"
PY=/opt/miniconda3/envs/sea/bin/python
export PATH="/opt/miniconda3/envs/sea/bin:$PATH"

VID="57TvyH9902U_slice"
LAG=6.33

mkdir -p "$TEST/out/aligned_A2" "$TEST/out/aligned_B2" "$TEST/ids"
echo "$VID" > "$TEST/ids/videos.txt"

cd "$SEA"

echo "=============== 2a. ALIGN (bias 0, filtered segmentation) ==============="
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
    --segmentation_dir "$TEST/out/segmentation_filtered" \
    --save_dir "$TEST/out/aligned_A2" || echo "RUN A2 FAILED"

echo "=============== 2b. ALIGN (bias = measured lag ${LAG}s, filtered segmentation) ==============="
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
    --segmentation_dir "$TEST/out/segmentation_filtered" \
    --save_dir "$TEST/out/aligned_B2" || echo "RUN B2 FAILED"

echo "=============== 3. CHECK (old binary metric, for continuity) ==============="
cd "$REPO"
for R in A2 B2; do
  OUT=$(find "$TEST/out/aligned_$R" -name "*.vtt" | head -1)
  if [ -n "$OUT" ]; then
    echo "--- RUN $R : $OUT ---"
    $PY phase3/sea_test/check_alignment.py \
        --original "$TEST/subtitles/$VID.vtt" \
        --aligned "$OUT" \
        --lag $LAG --slice-start 1200 --video-id 57TvyH9902U \
        --review-csv phase3/probe/review_2.csv || true
  else
    echo "--- RUN $R : no output vtt found ---"
    find "$TEST/out/aligned_$R" -type f | head
  fi
done
