"""
Extract continuous landmark arrays from a mañanera video.

Auto-detects the interpreter recuadro by scanning the first 60s on the FULL frame
and anchoring the crop on POSE (body envelope), padded generously so wide signs are
not clipped. The per-frame hand detector acts as the presence gate; pose+face run
only when hands are present. On a sustained absence (interpreter gone / box moved)
it re-detects, constrained to frame corners and requiring hands AND pose, so the
main speaker is never mistaken for the interpreter.

Saves one .npy per active signing segment (124-landmark schema) + segments.json.
Coordinates are stored RAW (crop-relative); downstream MUST normalize by shoulders.

Usage:
    # auto-detect crop (recommended)
    python phase2/extract_continuous.py <video.mp4> --out <dir>/

    # short validation slice with debug frames
    python phase2/extract_continuous.py <video.mp4> --out <dir>/ --seconds 180 --debug

    # manual crop override (disables auto-detect)
    python phase2/extract_continuous.py <video.mp4> --crop 1010 400 270 320 --out <dir>/
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_root = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "extract02", _root / "scripts" / "02_extract.py")
extract02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract02)

FACE_LANDMARKS = extract02.FACE_LANDMARKS
make_detectors = extract02.make_detectors


def make_image_detectors():
    """IMAGE-mode hand+pose detectors for crop detection (arbitrary frame jumps;
    no monotonic-timestamp constraint, unlike VIDEO mode)."""
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(extract02.HAND_MODEL)),
        num_hands=2, running_mode=vision.RunningMode.IMAGE)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(extract02.POSE_MODEL)),
        running_mode=vision.RunningMode.IMAGE)
    return (vision.HandLandmarker.create_from_options(hand_opts),
            vision.PoseLandmarker.create_from_options(pose_opts))

N_LEFT  = 21
N_RIGHT = 21
N_POSE  = 33
N_FACE  = len(FACE_LANDMARKS)          # 49 after dense-mouth change
N_ROWS  = N_LEFT + N_RIGHT + N_POSE + N_FACE  # 124

HAND_ROWS = slice(0, 42)

# pose landmark indices (within the 33-pt pose block)
POSE_L_SHOULDER = 11
POSE_R_SHOULDER = 12
POSE_NOSE       = 0


# ── crop auto-detection (pose-anchored, generous) ─────────────────────────────

# candidate corner regions as fractions (x0, y0, x1, y1) of the full frame.
# The interpreter PiP is always in a corner; the main speaker fills the center,
# so cropping a corner makes the small interpreter large enough to detect while
# excluding the speaker. Generous so the whole PiP fits.
_CORNER_REGIONS = {
    "bottom-right": (0.72, 0.45, 1.00, 1.00),
    "bottom-left":  (0.00, 0.45, 0.28, 1.00),
    "top-right":    (0.72, 0.00, 1.00, 0.55),
    "top-left":     (0.00, 0.00, 0.28, 0.55),
}


def _detect_in_region(cap, hand_det, pose_det, region, sample_frames):
    """
    Crop the given corner region, run hand+pose on the CROP (where the interpreter
    is large enough) at each frame index in `sample_frames`, and build a pose-anchored
    generous bbox in FULL-FRAME coords. Returns (bbox, hit_rate) where hit_rate is the
    fraction of sampled frames with BOTH hands and pose. bbox is None if no hits.
    """
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rx0, ry0 = int(region[0] * W), int(region[1] * H)
    rx1, ry1 = int(region[2] * W), int(region[3] * H)
    rw, rh = rx1 - rx0, ry1 - ry0

    # Anchor the crop on HAND points only — hands are reliably the interpreter's,
    # whereas a pose skeleton fit inside a region crop can sprawl onto background/
    # podium and blow the box up (making the interpreter small → face undetectable).
    # Pose is used only for: (a) the presence gate, (b) shoulder-width = body scale,
    # (c) nose y = head top for headroom.
    hxs, hys, nys, sws = [], [], [], []
    hits = 0
    n = 0
    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        patch = frame[ry0:ry1, rx0:rx1]
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr = hand_det.detect(img)
        pr = pose_det.detect(img)
        if not (hr.hand_landmarks and pr.pose_landmarks):
            continue
        hits += 1
        pose = pr.pose_landmarks[0]
        lsh, rsh = pose[POSE_L_SHOULDER], pose[POSE_R_SHOULDER]
        for hand in hr.hand_landmarks:
            for lm in hand:
                hxs.append(rx0 + lm.x * rw); hys.append(ry0 + lm.y * rh)
        nys.append(ry0 + pose[POSE_NOSE].y * rh)
        sws.append(abs(lsh.x - rsh.x) * rw)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not hxs or n == 0:
        return None, 0.0

    sw = np.median(sws) if sws else 0.10 * rw     # body scale (px)
    nose_y = np.median(nys)
    # pad relative to body size, not hand spread (hands may be still in the sample)
    pad_x    = 0.6 * sw
    pad_up   = 0.8 * sw       # headroom above the face for high signs
    pad_down = 0.4 * sw

    # percentiles (not min/max) so a rare spurious hand detection can't blow up the box
    hx_lo, hx_hi = np.percentile(hxs, [2, 98])
    hy_lo, hy_hi = np.percentile(hys, [2, 98])

    x1 = max(0, int(hx_lo - pad_x))
    x2 = min(W, int(hx_hi + pad_x))
    # top = above the head (nose minus headroom), but never below the highest hand
    y1 = max(0, int(min(hy_lo, nose_y) - pad_up))
    y2 = min(H, int(hy_hi + pad_down))
    return (x1, y1, x2 - x1, y2 - y1), hits / n


def _interpreter_bbox_full(cap, hand_det, pose_det, fps,
                           start_frame=0, end_frame=None, n_samples=40, prefer=None):
    """
    Find the interpreter by scanning each corner region across the frame range
    [start_frame, end_frame] (defaults to the whole video). Samples are spread evenly
    so long no-signing intros don't cause a miss. Picks the corner with the highest
    hit-rate. Returns (x,y,w,h) or None. `prefer` gets a small tie-break bonus (used
    during relocation so the box tends to return to where it was).
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if end_frame is None or end_frame > total:
        end_frame = total if total > 0 else start_frame + int(3600 * fps)
    span = max(1, end_frame - start_frame)
    step = max(1, span // n_samples)
    sample_frames = list(range(start_frame, end_frame, step))[:n_samples]

    best_bbox, best_score, best_name = None, 0.0, None
    for name, region in _CORNER_REGIONS.items():
        bbox, rate = _detect_in_region(cap, hand_det, pose_det, region, sample_frames)
        score = rate + (0.1 if name == prefer else 0.0)
        if bbox is not None and score > best_score:
            best_bbox, best_score, best_name = bbox, score, name
    if best_bbox is not None:
        print(f"    corner={best_name} hit-rate={best_score:.2f}")
    return best_bbox


def _hands_in_crop(frame, hand_det, crop, ts):
    x, y, w, h = crop
    patch = frame[y:y+h, x:x+w]
    if patch.size == 0:
        return False
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return bool(hand_det.detect_for_video(img, ts).hand_landmarks)


# ── landmark packing ──────────────────────────────────────────────────────────

def frame_to_row(hand_result, pose_result, face_result) -> np.ndarray:
    row = np.full((N_ROWS, 3), np.nan, dtype=np.float32)

    if hand_result.hand_landmarks:
        for idx, hand_lms in enumerate(hand_result.hand_landmarks):
            side   = hand_result.handedness[idx][0].category_name.lower()
            offset = 0 if side == "left" else N_LEFT
            for i, lm in enumerate(hand_lms):
                row[offset + i] = [lm.x, lm.y, lm.z]

    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            row[N_LEFT + N_RIGHT + i] = [lm.x, lm.y, lm.z]

    if face_result.face_landmarks:
        face_lms = face_result.face_landmarks[0]
        for pos, mesh_idx in enumerate(FACE_LANDMARKS):
            lm = face_lms[mesh_idx]
            row[N_LEFT + N_RIGHT + N_POSE + pos] = [lm.x, lm.y, lm.z]

    return row


# ── segmentation ──────────────────────────────────────────────────────────────

def segments_from_presence(presence, gap_threshold, min_frames):
    """Split on runs of >= gap_threshold no-hand frames; keep groups >= min_frames.
    Operates on the tiny in-RAM bool array, not the full landmark array."""
    hand_idx = np.where(presence)[0]
    if len(hand_idx) == 0:
        return []
    splits = np.where(np.diff(hand_idx) > gap_threshold)[0]
    groups = np.split(hand_idx, splits + 1)
    segs = []
    for g in groups:
        if len(g) < min_frames:
            continue
        start = max(0, int(g[0]) - 5)
        end   = min(len(presence), int(g[-1]) + 6)
        cov   = presence[start:end].mean()
        segs.append((start, end, float(cov)))
    return segs


# ── debug overlay ─────────────────────────────────────────────────────────────

def _save_debug_frame(frame, crop, hr, out_path):
    """Draw the crop box + detected hand/pose points on the full frame."""
    vis = frame.copy()
    x, y, w, h = crop
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    H, W = frame.shape[:2]
    if hr.hand_landmarks:
        for hand in hr.hand_landmarks:
            for lm in hand:
                px, py = int((x + lm.x * w)), int((y + lm.y * h))
                cv2.circle(vis, (px, py), 2, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), vis)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    default=None, help="Manual crop override; auto-detected if omitted.")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1,
                    help="Process every Nth frame (default 1 = native fps; use 3 for ~10fps test runs).")
    ap.add_argument("--keep-full", action="store_true",
                    help="Keep the full continuous .npy memmap (default: delete after segmenting).")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--relocate-threshold", type=float, default=12,
                    help="Seconds of no-hand before triggering a full-frame re-detect.")
    ap.add_argument("--gap", type=int, default=20,
                    help="Processed no-hand frames before splitting a segment.")
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--debug", action="store_true",
                    help="Save annotated frames to <out>/debug/ for visual QA.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem

    hand_det, pose_det, face_det = make_detectors()
    det_hand, det_pose = make_image_detectors()   # IMAGE-mode, for crop detection
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(args.seconds * fps) if args.seconds else int(1e9)

    # absence is measured in PROCESSED frames; convert thresholds to that unit
    proc_fps = fps / args.every
    relocate_after = int(args.relocate_threshold * proc_fps)

    # ── crop detection ────────────────────────────────────────────────────────
    if args.crop:
        active_crop = tuple(args.crop)
        print(f"Using manual crop: {active_crop}")
    else:
        print("Auto-detecting interpreter crop (corner scan across whole video)...")
        active_crop = _interpreter_bbox_full(cap, det_hand, det_pose, fps)
        if active_crop is None:
            print("ERROR: no hands+pose detected anywhere in the video. Use --crop manually.")
            return
        print(f"Detected crop: x={active_crop[0]} y={active_crop[1]} "
              f"w={active_crop[2]} h={active_crop[3]}")

    print(f"Extracting: {args.video.name}  scale={args.scale}  every={args.every}")

    debug_dir = args.out / "debug"
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    # ── pre-allocate on-disk memmap (stream rows straight to disk: crash-safe, O(1) RAM) ──
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_frames = min(nframes, total_frames) if nframes > 0 else total_frames
    if cap_frames >= int(1e9):
        print("WARNING: unknown frame count; allocating 4h buffer.")
        cap_frames = int(4 * 3600 * fps)
    n_alloc = cap_frames // args.every + 4

    full_path = args.out / f"{stem}_full.npy"
    arr = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float32,
                                    shape=(n_alloc, N_ROWS, 3))
    presence = np.zeros(n_alloc, dtype=bool)   # tiny in-RAM hand-presence timeline
    nan_row = np.full((N_ROWS, 3), np.nan, dtype=np.float32)

    crop_changes = [{"at_sec": 0.0, "crop": list(active_crop)}]
    fi        = 0
    processed = 0
    absence   = 0
    debug_saved = 0

    while cap.isOpened() and fi < total_frames and processed < n_alloc:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.every != 0:
            fi += 1
            continue

        ts = int(fi * 1000 / fps)
        x, y, w, h = active_crop
        patch = frame[y:y+h, x:x+w]
        if patch.size == 0:
            fi += 1
            continue
        if args.scale != 1.0:
            patch = cv2.resize(patch, None, fx=args.scale, fy=args.scale,
                               interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr  = hand_det.detect_for_video(img, ts)

        if hr.hand_landmarks:
            pr = pose_det.detect_for_video(img, ts)
            fr = face_det.detect_for_video(img, ts)
            arr[processed] = frame_to_row(hr, pr, fr)
            presence[processed] = True
            absence = 0
            if args.debug and debug_saved < 3 and processed % 50 == 0:
                _save_debug_frame(frame, active_crop, hr,
                                  debug_dir / f"{stem}_dbg{debug_saved}.jpg")
                debug_saved += 1
        else:
            arr[processed] = nan_row     # keeps the timeline; segmentation drops it
            presence[processed] = False
            absence += 1
            if absence >= relocate_after:
                print(f"  [{fi/fps:.0f}s] interpreter lost {args.relocate_threshold}s "
                      f"→ re-detecting...")
                new_crop = _interpreter_bbox_full(
                    cap, det_hand, det_pose, fps,
                    start_frame=fi, end_frame=fi + int(300 * fps), n_samples=12)
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi + 1)
                if new_crop:
                    active_crop = new_crop
                    crop_changes.append({"at_sec": round(fi/fps, 1),
                                         "crop": list(active_crop)})
                    print(f"    → crop updated: {active_crop}")
                else:
                    print(f"    → no interpreter found; continuing to scan.")
                absence = 0

        fi        += 1
        processed += 1
        if processed % 600 == 0:
            arr.flush()   # periodic flush so a crash loses only the tail
            print(f"  {fi} frames ({fi/fps:.0f}s)  |  {processed} processed...")

    arr.flush()
    cap.release()
    hand_det.close(); pose_det.close(); face_det.close()
    det_hand.close(); det_pose.close()

    if processed == 0:
        print("No frames processed.")
        del arr
        full_path.unlink(missing_ok=True)
        return

    presence = presence[:processed]
    print(f"\nTotal processed: {processed} frames ({processed/proc_fps:.1f}s effective at {fps:.0f}fps)")
    print(f"  hand/interpreter present: {100*presence.mean():.1f}%")

    segs = segments_from_presence(presence, args.gap, args.min_frames)
    print(f"Active signing segments: {len(segs)}")

    seg_meta = []
    for i, (start, end, cov) in enumerate(segs):
        clip = np.array(arr[start:end])   # copy this segment's slice from the memmap
        out_path = args.out / f"{stem}_seg{i:04d}.npz"
        np.savez_compressed(out_path, landmarks=clip)
        meta = {
            "file": out_path.name,
            "start_frame": start, "end_frame": end,
            "start_sec": round(start / proc_fps, 2),
            "end_sec":   round(end / proc_fps, 2),
            "n_frames":  end - start,
            "hand_coverage": round(cov, 3),
        }
        seg_meta.append(meta)
        print(f"  seg{i:04d}: {meta['start_sec']:.1f}s – {meta['end_sec']:.1f}s  "
              f"({end-start} frames, hand {100*cov:.0f}%)")

    meta_path = args.out / "segments.json"
    with open(meta_path, "w") as f:
        json.dump({
            "video": str(args.video),
            "schema_rows": N_ROWS,
            "crop_changes": crop_changes,
            "scale": args.scale, "every": args.every, "fps": fps,
            "segments": seg_meta,
        }, f, indent=2)

    # the full memmap is scratch (crash-safety during the run); segments are the deliverable
    del arr
    if args.keep_full:
        print(f"Kept full array → {full_path}")
    else:
        full_path.unlink(missing_ok=True)

    print(f"\nSaved {len(segs)} segments (.npz) → {args.out}")
    print(f"Metadata → {meta_path}")
    if args.debug and debug_saved:
        print(f"Debug frames → {debug_dir}")


if __name__ == "__main__":
    main()
