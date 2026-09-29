#!/usr/bin/env bash
# ── SignCLIP-LSM probe, part 1: set up tools + turn 30 dictionary videos into .pose ──
#
# Run from repo root:   bash phase3/probe/extract_poses.sh
# Needs: the dictionary SOURCE VIDEOS on the external drive (we have them).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVE="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
WORK="$REPO/phase3/probe"
CLIPS="$WORK/clips"      # the 30 selected source videos
POSES="$WORK/poses"      # the .pose files SignCLIP eats
rm -rf "$CLIPS" "$POSES"   # fresh each run so re-selecting a new word set doesn't mix old clips
mkdir -p "$CLIPS" "$POSES"

echo "── 1/3  Checking tools ──────────────────────────────────────────"
# One environment for the whole project now (see requirements.txt): Python 3.11,
# mediapipe==0.10.21 (has legacy solutions.Holistic AND the Tasks API), pose-format
# pinned there too. `pip install -r requirements.txt` already gives you everything
# this script needs -- this just verifies that happened instead of reinstalling.
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYVER" != "3.11" ]; then
  echo "  ✗ Python $PYVER — this project's environment is pinned to 3.11"
  echo "    (mediapipe==0.10.21 needs it for legacy Holistic; see requirements.txt)."
  exit 1
fi
python3 -c "import pose_format, mediapipe" || {
  echo "  ✗ pose-format/mediapipe not installed — run: pip install -r requirements.txt"
  exit 1; }
# SignCLIP: weights download separately — the python script's two bridge fns load it.
# Not on pip — clone per its README: https://github.com/J22Melody/fairseq

echo "── 2/3  Selecting 30 signs + copying their source videos ───────"
# Python picks 30 signs that HAVE a source video on the drive and writes clips/ + a manifest.
python3 "$WORK/signclip_probe.py" select --drive "$DRIVE" --clips "$CLIPS" --n 30

echo "── 3/3  Video → .pose (MediaPipe Holistic via pose-format) ─────"
# videos_to_poses runs over the whole clips/ folder at once.
videos_to_poses --format mediapipe --directory "$CLIPS" || {
    echo "videos_to_poses failed — check pose-format[mediapipe] install"; exit 1; }
# Move the produced .pose files next to the eval.
find "$CLIPS" -name '*.pose' -exec mv {} "$POSES"/ \;

ZIP="$WORK/signclip_probe_data.zip"
rm -f "$ZIP"
zip -j "$ZIP" "$POSES"/*.pose "$CLIPS"/manifest.csv >/dev/null

echo ""
echo "✓ Poses ready + bundled → $ZIP"
echo "Re-upload that zip to Colab (Cell 3) and re-run Cell 4 / Cell 5."
