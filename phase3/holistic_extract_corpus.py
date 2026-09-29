"""
Corpus-scale Holistic .pose extraction, resumable across multiple nights.

Directly reuses extract_continuous.py's proven structure -- one video opened
once, sequential frame reads, crop_at() for the correct region per timestamp --
instead of the external-CLI-batch-directory design this replaced. That design
cut every segment to its own clip file, wrote them all into one staging
directory, and handed the whole directory to pose-format's `videos_to_poses`
CLI tool. It hit two real, corpus-fatal bugs the first time it ran: the X8's
exFAT filesystem silently writes a `._<name>` AppleDouble sidecar for every
file created on it, which doubled videos_to_poses's file count and crashed the
entire batch (an unhandled exception) the moment it tried to open one as a
video; and `videos_to_poses` resolved via PATH to the wrong conda environment's
copy. Both problems come from routing through an external directory-scanning
tool at all -- this version never does. It calls pose-format's own
`load_holistic()` directly, in-process, one segment-chunk at a time, and writes
exactly the file it means to write. No staging directory, no batch scan, no
subprocess.

Reads segments_and_gate.json (built by build_and_gate_segments.py -- hand AND
pose validated) for each already-extracted video. Long segments are split into
<=60s chunks before processing -- load_holistic wants the whole chunk's frames
in memory as a list at once, and some AND-gate segments run 400+ seconds,
which at a few hundred KB/frame would be multiple GB for one segment on an
8GB machine. Chunking bounds that regardless of segment length.

RESUMABLE BY DESIGN: every chunk's expected output filename is checked before
any work is done on it. Kill this at any point and re-running it later picks
up exactly where it left off.

Usage:
    caffeinate -i /opt/miniconda3/envs/lsm-probe/bin/python3 \
        phase3/holistic_extract_corpus.py \
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera" \
        --videos-root "/Volumes/Crucial X8/LSM_Translator/videos/mananera" \
        --out "/Volumes/Crucial X8/LSM_Translator/poses/mananera"
"""
import argparse
import json
from pathlib import Path

import cv2
from pose_format.utils.holistic import load_holistic

MAX_CHUNK_SEC = 60.0
HOLISTIC_CONFIG = dict(model_complexity=2, smooth_landmarks=False,
                       refine_face_landmarks=True)


def crop_at(crop_changes, t):
    prior = [c for c in crop_changes if c["at_sec"] <= t]
    return (prior[-1] if prior else crop_changes[0])["crop"]


def chunk_segment(start, end, max_sec=MAX_CHUNK_SEC):
    """Split one AND-gate segment into <=max_sec pieces, bounding memory per call."""
    chunks = []
    t = start
    while t < end:
        t1 = min(t + max_sec, end)
        chunks.append((round(t, 2), round(t1, 2)))
        t = t1
    return chunks


def collect_chunks(arrays_root: Path, videos_root: Path):
    """One entry per (video, chunk) across every already-extracted video."""
    chunks = []
    for vdir in sorted(arrays_root.iterdir()):
        if not vdir.is_dir() or vdir.name.startswith("."):
            continue
        and_gate_path = vdir / "segments_and_gate.json"
        if not and_gate_path.exists():
            continue
        meta = json.load(open(and_gate_path))
        crops = sorted(meta["crop_changes"], key=lambda c: c["at_sec"])
        vid = vdir.name
        src = videos_root / vid / f"{vid}.mp4"
        if not src.exists():
            print(f"  WARN: {vid} has segments_and_gate.json but no source video, skipping")
            continue
        for seg_i, s in enumerate(meta["segments"]):
            for chunk_i, (c_start, c_end) in enumerate(chunk_segment(s["start_sec"], s["end_sec"])):
                chunks.append(dict(
                    video=vid, src=src, seg_idx=seg_i, chunk_idx=chunk_i,
                    start=c_start, end=c_end,
                    crop=crop_at(crops, c_start),
                ))
    return chunks


def chunk_name(c):
    return f"{c['video']}__seg{c['seg_idx']:04d}_chunk{c['chunk_idx']:02d}_{int(c['start'])}s"


def read_cropped_frames(src, start, end, crop, fps):
    x, y, w, h = crop
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    n_frames = int(round((end - start) * fps))
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        patch = frame[y:y + h, x:x + w]
        frames.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose-workers", type=int, default=3,
                    help="Internal thread pool inside load_holistic (not separate OS "
                         "processes -- one Python process, one model load, cheaper on RAM "
                         "than the old --num-workers subprocess design).")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    chunks = collect_chunks(args.arrays_root, args.videos_root)
    total = len(chunks)
    todo = [c for c in chunks if not (args.out / f"{chunk_name(c)}.pose").exists()]
    done_at_start = total - len(todo)
    print(f"{total} chunks total across the corpus (<={MAX_CHUNK_SEC:.0f}s each)")
    print(f"{done_at_start} already have a .pose file -- skipping")
    print(f"{len(todo)} remaining tonight\n", flush=True)

    if not todo:
        print("Nothing left to do. Corpus-scale Holistic extraction is complete.")
        return

    fps_cache = {}
    done = 0
    for c in todo:
        if c["video"] not in fps_cache:
            cap = cv2.VideoCapture(str(c["src"]))
            fps_cache[c["video"]] = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
        fps = fps_cache[c["video"]]

        frames, w, h = read_cropped_frames(c["src"], c["start"], c["end"], c["crop"], fps)
        if not frames:
            print(f"  WARN: 0 frames read for {chunk_name(c)}, skipping", flush=True)
            continue

        pose = load_holistic(frames, fps=fps, width=w, height=h,
                             additional_holistic_config=HOLISTIC_CONFIG,
                             pose_workers=args.pose_workers, progress=False)
        out_path = args.out / f"{chunk_name(c)}.pose"
        tmp_path = out_path.with_suffix(".pose.tmp")
        with open(tmp_path, "wb") as f:
            pose.write(f)
        tmp_path.rename(out_path)  # atomic-ish: a killed run never leaves a half-written .pose

        done += 1
        if done % 10 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)} this run "
                  f"({done_at_start + done}/{total} overall, "
                  f"{100*(done_at_start+done)/total:.1f}%)", flush=True)

    print(f"\nDone this run: {done}/{len(todo)}")
    total_done = done_at_start + done
    print(f"Progress: {total_done}/{total} chunks extracted ({100*total_done/total:.1f}%)")
    if total_done < total:
        print("Re-run this same command tomorrow night to continue.")
    else:
        print("Corpus-scale Holistic extraction is complete.")


if __name__ == "__main__":
    main()
