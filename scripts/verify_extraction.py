"""
Overlay extracted landmarks on the REAL video frames to verify tracking.

Unlike visualize.py (blank canvas) and inspect_trim.py (array only), this samples
frames evenly across the active signing window — including frames where no hand
was detected — and draws the skeleton on top of the actual video. A tracking miss
shows up as a visible hand in the frame with no skeleton on it.

Usage:
    python scripts/verify_extraction.py pbZ3BQKhZnI [EeqHGXo5pBA ...]
    python scripts/verify_extraction.py --suspects   # built-in worst-gap list
"""

import argparse
import json
import sys
import cv2
import numpy as np
from pathlib import Path

# A video may live in either source folder; search both.
VIDEO_DIRS = [
    Path("/Volumes/Crucial X8/LSM_Translator/videos/isolated"),
    Path("/Volumes/Crucial X8/LSM_Translator/videos/wikisigns"),
]
OUT_DIR = Path("data/viz/verify")


def find_folder(youtube_id: str) -> Path | None:
    for base in VIDEO_DIRS:
        folder = base / youtube_id
        if folder.is_dir():
            return folder
    return None

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
]

# Videos with the largest interior hand-dropout gaps (post-trim diagnosis)
SUSPECTS = [
    "pbZ3BQKhZnI",  # vaso
    "EeqHGXo5pBA",  # tenedor
    "RqEV20epiak",  # tapete
    "AlxYBw0KqRY",  # telefono
    "1F000COwrok",  # escalera
    "dupvirW7_TU",  # silla
    "KgFCJeYjXYs",  # horno
    "Ehxdj47rxyc",  # papel de bano
]

N_SAMPLES = 8


def draw_overlay(frame: np.ndarray, by_source: dict):
    """Draw hand + pose skeleton onto a real video frame (in place)."""
    h, w = frame.shape[:2]

    def pt(lm):
        return (int(lm["x"] * w), int(lm["y"] * h))

    for source, color in [("left_hand", (255, 100, 100)),
                          ("right_hand", (100, 100, 255))]:
        lms = by_source.get(source, {})
        for a, b in HAND_CONNECTIONS:
            if a in lms and b in lms:
                cv2.line(frame, pt(lms[a]), pt(lms[b]), color, 2)
        for lm in lms.values():
            cv2.circle(frame, pt(lm), 3, color, -1)

    pose = by_source.get("pose", {})
    for a, b in POSE_CONNECTIONS:
        if a in pose and b in pose:
            cv2.line(frame, pt(pose[a]), pt(pose[b]), (80, 220, 80), 2)


def index_landmarks(landmarks):
    """frame -> {source -> {landmark_index -> lm}}"""
    by_frame = {}
    for lm in landmarks:
        by_frame.setdefault(lm["frame"], {}).setdefault(lm["source"], {})[lm["landmark_index"]] = lm
    return by_frame


def verify(youtube_id: str):
    folder = find_folder(youtube_id)
    if folder is None:
        print(f"  {youtube_id}: folder not found in any video dir, skipping.")
        return
    landmark_file = folder / "landmarks.json"
    mp4s = [p for p in folder.glob("*.mp4") if not p.name.startswith("._")]

    if not landmark_file.exists() or not mp4s:
        print(f"  {youtube_id}: missing landmarks.json or .mp4, skipping.")
        return

    with open(landmark_file) as f:
        data = json.load(f)
    landmarks = data["landmarks"]
    by_frame = index_landmarks(landmarks)

    hand_frames = sorted({lm["frame"] for lm in landmarks if "hand" in lm["source"]})
    if not hand_frames:
        print(f"  {youtube_id}: no hand detections at all, skipping.")
        return

    # Sample evenly across the active window (first to last hand frame),
    # INCLUDING frames with no hand so dropouts are visible.
    start, end = hand_frames[0], hand_frames[-1]
    sample_idx = np.linspace(start, end, N_SAMPLES, dtype=int)

    cap = cv2.VideoCapture(str(mp4s[0]))
    tiles = []
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (320, 240))
        srcs = by_frame.get(int(idx), {})
        draw_overlay(frame, srcs)
        has_hand = any("hand" in s for s in srcs)
        tag = f"f{idx}" + ("" if has_hand else "  NO-HAND")
        color = (255, 255, 255) if has_hand else (0, 220, 255)
        cv2.putText(frame, tag, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        tiles.append(frame)
    cap.release()

    # 2 rows of 4
    row1 = np.hstack(tiles[:4])
    row2 = np.hstack(tiles[4:8])
    grid = np.vstack([row1, row2])

    header = np.ones((30, grid.shape[1], 3), dtype=np.uint8) * 255
    cov = len(hand_frames) / (end - start + 1)
    cv2.putText(header, f"{youtube_id}  |  hand-coverage in window: {cov:.0%}  |  yellow = MediaPipe found no hand",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    final = np.vstack([header, grid])

    out_path = OUT_DIR / f"{youtube_id}.png"
    cv2.imwrite(str(out_path), final)
    print(f"  Saved {out_path.name}  (window {start}-{end}, {cov:.0%} hand coverage)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs="*", help="youtube_ids to verify.")
    parser.add_argument("--suspects", action="store_true",
                        help="Use the built-in worst-gap suspect list.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = SUSPECTS if args.suspects else args.ids
    if not ids:
        print("Pass youtube_ids or --suspects.")
        sys.exit(1)

    print(f"Verifying {len(ids)} videos → {OUT_DIR}")
    for yid in ids:
        verify(yid)
    print(f"\nDone. Open {OUT_DIR}/ — yellow NO-HAND tiles with a visible hand = tracking miss.")


if __name__ == "__main__":
    main()
