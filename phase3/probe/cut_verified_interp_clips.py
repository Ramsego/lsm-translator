"""
Consume the reviewed phase3/probe/review.csv (downloaded from review.html after
watching all 32 interpreter clips) and produce a clean pose set for Cell 8:

  1. Drop clips verdict != 'y' (either the interpreter used a different/substituted
     sign, or the sign never appears here) — these would silently corrupt the
     retrieval test as false misses if left in.
  2. For the clips that ARE verdict 'y', cut a TIGHT clip using the human-marked
     sign_start/sign_end (+- 0.3s buffer) instead of the loose 14s candidate window —
     fixes the truncation problem where Cell 8 was blindly keeping the first ~8.5s.
  3. Re-run videos_to_poses (Holistic, lsm-probe env) on the tight clips only, and
     rebuild signclip_probe_data_interpreter.zip.

Usage (after downloading + overwriting phase3/probe/review.csv, with lsm-probe active):
    conda activate lsm-probe
    python phase3/probe/cut_verified_interp_clips.py
"""
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path(__file__).parent
BUFFER = 0.6  # seconds of padding around the marked sign span
# NOTE (2026-07-29): some marked spans came out as short as 0.2-0.3s — likely s/e tapped
# back-to-back rather than bracketing the full gesture. 0.3s buffer wasn't enough to
# recover a full sign in those cases (retrieval collapsed onto a few generic wrong
# answers). Widened here as a free first attempt; if that doesn't help, the real fix is
# re-marking those specific clips with s at first visible movement, e at return-to-rest.

def main():
    review = WORK / "review.csv"
    rows = list(csv.DictReader(open(review)))

    kept, dropped = [], []
    for r in rows:
        if r["verdict"] == "y" and r["sign_start"] and r["sign_end"]:
            kept.append(r)
        else:
            dropped.append(r)

    print(f"Reviewed {len(rows)} clips: {len(kept)} verified 'y' with timing, "
          f"{len(dropped)} dropped (wrong sign / absent / unmarked timing).")
    if dropped:
        by_verdict = {}
        for r in dropped:
            by_verdict.setdefault(r["verdict"] or "(unmarked)", []).append(f"{r['word']}[{r['file'].split('__')[-1].replace('.mp4','')}]")
        for v, items in by_verdict.items():
            print(f"  {v}: {', '.join(items)}")

    if not kept:
        print("Nothing verified as usable — nothing to extract.")
        sys.exit(1)

    tight_dir = WORK / "clips_interpreter_verified"
    tight_dir.mkdir(exist_ok=True)
    for f in list(tight_dir.glob("*.mp4")) + list(tight_dir.glob("*.pose")):
        f.unlink()   # stale .pose with a matching name makes videos_to_poses skip re-cutting silently

    manifest_rows = []
    for r in kept:
        clip_id = Path(r["file"]).stem   # e.g. agua__57TvyH9902U
        src = WORK / r["file"]
        start = max(0.0, float(r["sign_start"]) - BUFFER)
        end = float(r["sign_end"]) + BUFFER
        dur = end - start
        out = tight_dir / f"{clip_id}.mp4"
        # -ss AFTER -i + re-encode = frame-accurate. "-c copy" (stream copy) snaps to
        # keyframes and silently overshoots on clips this short (seen 2026-07-29: a
        # requested ~1.4s cut came out at ~6s) — precision matters more than speed here.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
            check=True,
        )
        signer = clip_id.split("__", 1)[1]
        manifest_rows.append({"word": r["word"], "signer": signer, "clip_id": clip_id})

    manifest_path = tight_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["word", "signer", "clip_id"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nCut {len(manifest_rows)} tight clips -> {tight_dir}")
    print("Running videos_to_poses (Holistic)...")
    subprocess.run(["videos_to_poses", "--format", "mediapipe", "--directory", str(tight_dir)], check=True)

    zip_path = WORK / "signclip_probe_data_interpreter.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        for pf in tight_dir.glob("*.pose"):
            z.write(pf, pf.name)
        z.write(manifest_path, "manifest.csv")

    n_poses = len(list(tight_dir.glob("*.pose")))
    print(f"\n{n_poses} verified, tightly-cropped interpreter poses -> {zip_path}")
    print("Upload this to Colab (Cell 8) — it replaces the earlier unverified version.")


if __name__ == "__main__":
    main()
