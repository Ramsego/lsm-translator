#!/usr/bin/env bash
# Bias sweep — does a better scalar shift help, and does SEA keep its margin?
#
# WHY
# ---
# `run_sea.sh` hardcodes LAG=6.33, which came from the saturating onset detector.
# The fixed detector (estimate_lag_v2.py) says 5.62 for this video.  Two questions:
#
#   1. Which scalar does SEA's cost function actually want?
#   2. As the shift improves, does SEA KEEP its margin over a plain shift?
#      - margin holds/grows at a higher baseline -> SEA fixes clause-to-clause wobble
#        that no scalar can reach.  Strong case for keeping the DP.
#      - margin shrinks -> part of SEA's advantage was just cleaning up a bad scalar.
#
# Every arm is paired: the same bias is scored as a plain shift AND through SEA, so
# the comparison never confounds "better scalar" with "SEA is working".
#
# Segmentation is NOT re-run (it does not depend on the bias); out/segmentation is reused.
#
# Usage:  bash phase3/sea_test/bias_sweep.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEA="$REPO/phase3/sea_src"
TEST="$REPO/phase3/sea_test"
PY=/opt/miniconda3/envs/sea/bin/python
VID="57TvyH9902U_slice"
SWEEP="$TEST/out/sweep"
mkdir -p "$SWEEP"
[ -f "$TEST/ids/videos.txt" ] || { mkdir -p "$TEST/ids"; echo "$VID" > "$TEST/ids/videos.txt"; }

# bias_start bias_end label
ARMS=(
  "0.00 0.00 bias0"
  "4.50 4.50 bias4.50"
  "5.00 5.00 bias5.00"
  "5.62 5.62 bias5.62_v2detector"
  "6.33 6.33 bias6.33_v1detector"
  "7.00 7.00 bias7.00"
  "5.62 5.12 bias5.62-5.12_startNEend"
)

RESULTS="$SWEEP/results.csv"
echo "arm,method,n,binary,median_abs_dist,mean_abs_dist,median_iou,median_span_dur" > "$RESULTS"

for arm in "${ARMS[@]}"; do
  read -r BS BE LABEL <<< "$arm"
  echo "=============== $LABEL  (start $BS / end $BE) ==============="

  # ---- naive flat shift: move every cue by the same amount -----------------
  NAIVE="$SWEEP/naive_$LABEL.vtt"
  $PY - "$TEST/subtitles/$VID.vtt" "$NAIVE" "$BS" "$BE" <<'PYEOF'
import re, sys
src, dst, bs, be = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def to_s(t):
    h, m, rest = t.split(":"); return int(h)*3600 + int(m)*60 + float(rest)
def to_t(s):
    s = max(s, 0.0); h = int(s//3600); m = int((s%3600)//60)
    return f"{h:02d}:{m:02d}:{s%60:06.3f}"
out = []
for line in open(src):
    m = re.match(r"^\s*([\d:.]+)\s*-->\s*([\d:.]+)(.*)$", line)
    out.append(f"{to_t(to_s(m.group(1))+bs)} --> {to_t(to_s(m.group(2))+be)}{m.group(3)}\n"
               if m else line)
open(dst, "w").writelines(out)
PYEOF

  $PY "$TEST/check_alignment_v2.py" --aligned "$NAIVE" --slice-start 1200 \
      --video-id 57TvyH9902U --label "naive_$LABEL" 2>/dev/null \
      | grep '^CSV,' | sed "s/^CSV,naive_$LABEL/$LABEL,naive/" >> "$RESULTS"

  # ---- SEA: same bias, DP on top ------------------------------------------
  SAVE="$SWEEP/aligned_$LABEL"
  rm -rf "$SAVE"; mkdir -p "$SAVE"
  ( cd "$SEA" && $PY align.py --overwrite --mode=inference \
      --video_ids "$TEST/ids/videos.txt" --num_workers 1 \
      --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 \
      --dp_max_gap 10 --dp_window_size 50 \
      --sign-b-threshold 30 --sign-o-threshold 50 \
      --pr_subs_delta_bias_start "$BS" --pr_subs_delta_bias_end "$BE" \
      --similarity_measure none \
      --pr_sub_path "$TEST/subtitles" --gt_sub_path "$TEST/subtitles" \
      --segmentation_dir "$TEST/out/segmentation" \
      --save_dir "$SAVE" ) >/dev/null 2>&1

  AV=$(find "$SAVE" -name "*.vtt" | head -1)
  if [ -n "$AV" ]; then
    $PY "$TEST/check_alignment_v2.py" --aligned "$AV" --slice-start 1200 \
        --video-id 57TvyH9902U --label "sea_$LABEL" 2>/dev/null \
        | grep '^CSV,' | sed "s/^CSV,sea_$LABEL/$LABEL,sea/" >> "$RESULTS"
  else
    echo "$LABEL,sea,FAILED,,,,," >> "$RESULTS"
  fi
done

echo; echo "=== $RESULTS ==="; column -s, -t "$RESULTS"
