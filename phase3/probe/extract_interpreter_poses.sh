#!/usr/bin/env bash
# ── SignCLIP-LSM probe, part 2: interpreter-footage side of the cross-domain test ──
#
# The 30 dictionary poses (phase3/probe/poses/) are clean citation-form signs, one
# signer each — that's what the June/July Colab runs tested. This script builds the
# OTHER half: real mañanera interpreter clips, multiple signers per word, for the
# 8 words that exist in BOTH the dictionary set and the gold review-clip set
# (agua, amigo, gracias, hablar, hombre, mesa, mujer, verde).
#
# Source: already-cut review clips from phase3/gold/cut_reviews.sh (June), sitting on
# the external drive at $LSM_DATA_ROOT/review/<video_id>/<word>/*.mp4. These are
# WEAK labels (transcript+lag windows, never human-verified) — good enough for a
# first cheap signal check, not for a paper claim.
#
# Run from repo root:   bash phase3/probe/extract_interpreter_poses.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DRIVE="${LSM_DATA_ROOT:-/Volumes/Crucial X8/LSM_Translator}"
WORK="$REPO/phase3/probe"
CLIPS="$WORK/clips_interpreter"
POSES="$WORK/poses_interpreter"
CAP_PER_WORD="${CAP_PER_WORD:-4}"   # distinct signers per word
WORDS=(agua amigo gracias hablar hombre mesa mujer verde)

rm -rf "$CLIPS" "$POSES"
mkdir -p "$CLIPS" "$POSES"

echo "── 1/3  Selecting up to $CAP_PER_WORD signers per word ─────────"
MANIFEST="$CLIPS/manifest.csv"
echo "word,signer,clip_id,src_path" > "$MANIFEST"
for w in "${WORDS[@]}"; do
    n=0
    # one clip per distinct video(=signer) directory, first match, sorted for reproducibility
    # NOTE: $DRIVE contains a space ("Crucial X8") — must use -print0/read -d '', not
    # `for x in $(find ...)`, which word-splits the path and silently finds nothing.
    while IFS= read -r -d '' vid_dir; do
        [ "$n" -ge "$CAP_PER_WORD" ] && break
        vid="$(basename "$vid_dir")"
        wdir="$vid_dir/$w"
        [ -d "$wdir" ] || continue
        clip="$(find "$wdir" -name '*.mp4' -not -name '._*' | sort | head -1)"
        [ -n "$clip" ] || continue
        n=$((n+1))
        clip_id="${w}__${vid}"
        cp "$clip" "$CLIPS/${clip_id}.mp4"
        echo "$w,$vid,$clip_id,$clip" >> "$MANIFEST"
    done < <(find "$DRIVE/review" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
    echo "  $w: $n signers"
done

echo "── 2/3  Video → .pose (MediaPipe Holistic via pose-format) ─────"
# Same tool + environment as the dictionary side (phase3/probe/extract_poses.sh) so
# the two pose sets are directly comparable — one project environment now, see
# requirements.txt (Python 3.11, mediapipe==0.10.21).
if ! command -v videos_to_poses >/dev/null 2>&1; then
    echo "  ✗ videos_to_poses not on PATH — run: pip install -r requirements.txt"
    exit 1
fi
videos_to_poses --format mediapipe --directory "$CLIPS" || {
    echo "videos_to_poses failed — check pose-format[mediapipe] install"; exit 1; }
find "$CLIPS" -name '*.pose' -exec mv {} "$POSES"/ \;

echo "── 3/3  Bundling for Colab ──────────────────────────────────────"
ZIP="$WORK/signclip_probe_data_interpreter.zip"
rm -f "$ZIP"
zip -j "$ZIP" "$POSES"/*.pose "$MANIFEST" >/dev/null

n_poses=$(find "$POSES" -name '*.pose' | wc -l | tr -d ' ')
echo ""
echo "✓ $n_poses interpreter poses ready → $ZIP"
echo "Upload alongside signclip_probe_data.zip in the same Colab session (Cell 8)."
