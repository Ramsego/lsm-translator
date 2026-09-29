"""
Build SEA inputs (cropped+upscaled interpreter clips -> .pose, + matching VTT) for the
whole corpus. Until now this existed only as a hand-made 30-minute slice
(video/57TvyH9902U_slice.{mp4,pose}, 900x1030 = the interpreter crop upscaled ~3.5x).

Unit of work is ONE REVALIDATED SEGMENT, not one video, for three reasons:
  * segments_revalidated.json already guarantees the interpreter is present throughout
    (see revalidate_segments.py) -- so no SEA span can land on B-roll;
  * a whole-video .pose would be up to ~2.3 GB held in memory to write (pose-format
    needs the full array), which does not fit comfortably in 8 GB RAM;
  * it mirrors how the working slice was handled -- SEA treats each unit as a "video"
    with its own VTT.
Crops are stable inside segments (measured: only 3/236 have an internal crop change),
so one crop per segment is safe; those 3 get sub-split at the crop boundary.

Outputs, per unit:
    <out>/video/<vid>_s<NNNN>.mp4     cropped+upscaled interpreter
    <out>/video/<vid>_s<NNNN>.pose    MediaPipe Holistic, what SEA consumes
    <out>/subtitles/<vid>_s<NNNN>.vtt sentence-grouped ASR, times rebased to unit-local
    <out>/units.csv                   vid, unit, abs_start, abs_end, crop, duration

Usage:
    /opt/miniconda3/envs/lsm-probe/bin/python phase3/sea_test/build_sea_inputs.py \
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera" \
        --videos-root "/Volumes/Crucial X8/LSM_Translator/videos/mananera" \
        --out phase3/sea_test/corpus --only 57TvyH9902U --limit-units 1
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

TARGET_W = 900          # match the hand-made slice's upscale convention


def crop_at(crop_changes, t):
    prior = [c for c in crop_changes if c["at_sec"] <= t]
    return (prior[-1] if prior else crop_changes[0])["crop"]


def split_on_crop_changes(seg, crop_changes):
    """One (start, end, crop) per contiguous run with a constant crop."""
    a, b = seg["start_sec"], seg["end_sec"]
    bounds = [a] + [c["at_sec"] for c in crop_changes if a < c["at_sec"] < b] + [b]
    return [(bounds[i], bounds[i + 1], crop_at(crop_changes, bounds[i]))
            for i in range(len(bounds) - 1)]


def load_words(video_dir, vid):
    """ASR word list -> [(start, end, word)], absolute seconds. Handles .asr.json or .vtt."""
    aj = video_dir / f"{vid}.asr.json"
    if aj.exists():
        j = json.load(open(aj))
        return [(w["start"], w["end"], w["word"].strip()) for w in j["words"]]
    vt = next(iter(video_dir.glob("*.vtt")), None)
    if vt is None:
        return None
    # caption VTT: cue-level, treat each cue as one "word" block
    out, cur = [], None
    tpat = re.compile(r"(\d\d):(\d\d):(\d\d[.,]\d+)\s*-->\s*(\d\d):(\d\d):(\d\d[.,]\d+)")
    for line in vt.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = tpat.search(line)
        if m:
            g = [x.replace(",", ".") for x in m.groups()]
            st = int(g[0]) * 3600 + int(g[1]) * 60 + float(g[2])
            en = int(g[3]) * 3600 + int(g[4]) * 60 + float(g[5])
            cur = [st, en]
        elif cur and line.strip() and line.strip() != "WEBVTT":
            out.append((cur[0], cur[1], line.strip()))
            cur = None
    return out or None


def write_vtt(words, t0, t1, path):
    """Group words into sentences on terminal punctuation; rebase to unit-local time."""
    sel = [(s - t0, e - t0, w) for s, e, w in words if t0 <= s < t1]
    if not sel:
        return 0
    cues, cur = [], []
    for s, e, w in sel:
        cur.append((s, e, w))
        if re.search(r"[.!?]$", w):
            cues.append(cur); cur = []
    if cur:
        cues.append(cur)

    def ts(x):
        h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for c in cues:
            f.write(f"{ts(c[0][0])} --> {ts(c[-1][1])}\n{' '.join(w for _, _, w in c)}\n\n")
    return len(cues)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays-root", type=Path, required=True)
    ap.add_argument("--videos-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit-units", type=int, default=None, help="stop after N units (smoke test)")
    ap.add_argument("--min-duration", type=float, default=30.0,
                    help="skip units shorter than this (too short to align usefully)")
    ap.add_argument("--skip-pose", action="store_true", help="cut clips + VTT only")
    args = ap.parse_args()

    (args.out / "video").mkdir(parents=True, exist_ok=True)
    (args.out / "subtitles").mkdir(parents=True, exist_ok=True)

    rows, made = [], 0
    for d in sorted(args.arrays_root.iterdir()):
        f = d / "segments_revalidated.json"
        if not f.exists() or (args.only and d.name not in args.only):
            continue
        vid = d.name
        m = json.load(open(f))
        crops = sorted(m["crop_changes"], key=lambda c: c["at_sec"])
        src = args.videos_root / vid / f"{vid}.mp4"
        words = load_words(args.videos_root / vid, vid)
        if words is None:
            print(f"{vid}: SKIP -- no transcript", flush=True)
            continue

        idx = 0
        for seg in sorted(m["segments"], key=lambda s: s["start_sec"]):
            for a, b, crop in split_on_crop_changes(seg, crops):
                if b - a < args.min_duration:
                    continue
                unit = f"{vid}_s{idx:04d}"; idx += 1
                x, y, w, h = crop
                oh = int(round(TARGET_W * h / w)) // 2 * 2
                mp4 = args.out / "video" / f"{unit}.mp4"
                if not mp4.exists():
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                        "-ss", str(a), "-t", str(b - a), "-i", str(src),
                        "-vf", f"crop={w}:{h}:{x}:{y},scale={TARGET_W}:{oh}:flags=lanczos",
                        "-an", "-c:v", "libx264", "-preset", "veryfast", str(mp4)], check=True)
                ncues = write_vtt(words, a, b, args.out / "subtitles" / f"{unit}.vtt")
                rows.append(dict(video=vid, unit=unit, abs_start=round(a, 2),
                                 abs_end=round(b, 2), duration=round(b - a, 1),
                                 crop=f"{x},{y},{w},{h}", scaled=f"{TARGET_W}x{oh}", cues=ncues))
                print(f"  {unit}  {a:8.1f}-{b:8.1f}s ({b-a:6.1f}s)  {w}x{h}->{TARGET_W}x{oh}  {ncues} cues", flush=True)
                made += 1
                if args.limit_units and made >= args.limit_units:
                    break
            if args.limit_units and made >= args.limit_units:
                break
        if args.limit_units and made >= args.limit_units:
            break

    with open(args.out / "units.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["video", "unit", "abs_start", "abs_end",
                                            "duration", "crop", "scaled", "cues"])
        wr.writeheader(); wr.writerows(rows)
    print(f"\n{len(rows)} units -> {args.out}/units.csv")

    if not args.skip_pose:
        # resolve videos_to_poses from THIS interpreter's env -- the base env also has a
        # copy on PATH whose mediapipe import is broken (py3.13), and PATH wins otherwise
        v2p = Path(sys.executable).parent / "videos_to_poses"
        if not v2p.exists():
            sys.exit(f"videos_to_poses not found at {v2p} -- run this with the lsm-probe python")
        print(f"extracting poses via {v2p} ...", flush=True)
        subprocess.run([str(v2p), "--format", "mediapipe",
                        "--directory", str(args.out / "video")], check=False)
        n = len(list((args.out / "video").glob("*.pose")))
        print(f"{n}/{len(rows)} .pose files present")


if __name__ == "__main__":
    main()
