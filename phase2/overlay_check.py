"""
Render a skeleton-overlay video from the interpreter crop.

Usage:
    python phase2/overlay_check.py <video.mp4> --crop X Y W H [--scale 3] [--seconds 120] [-o out.mp4]
"""

import argparse
import importlib.util
import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

_spec = importlib.util.spec_from_file_location(
    "extract02", Path(__file__).resolve().parents[1] / "scripts" / "02_extract.py")
extract02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract02)

HAND_COLOR  = (0, 255, 120)
POSE_COLOR  = (255, 180, 0)
FACE_COLOR  = (0, 180, 255)
DOT_R       = 3
CONN_T      = 1

# MediaPipe hand skeleton connections (21 landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17),                           # palm
]


def draw_landmarks(frame, results_hand, results_pose, results_face, h, w):
    # hands
    for hand_lms in results_hand.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for p in pts:
            cv2.circle(frame, p, DOT_R, HAND_COLOR, -1)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], HAND_COLOR, CONN_T)

    # pose (upper body — skip feet/ankles below index 25)
    for pose_lms in results_pose.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_lms]
        for i, p in enumerate(pts[:25]):
            cv2.circle(frame, p, DOT_R, POSE_COLOR, -1)

    # face — just the curated subset
    for face_lms in results_face.face_landmarks:
        all_pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]
        for idx in extract02.FACE_LANDMARKS:
            if idx < len(all_pts):
                cv2.circle(frame, all_pts[idx], 2, FACE_COLOR, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("-o", "--out", type=Path, default=Path("/tmp/recuadro_test/overlay.mp4"))
    args = ap.parse_args()

    hand_det, pose_det, face_det = extract02.make_detectors()
    x, y, cw, ch = args.crop

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(args.seconds * fps)

    out_w = int(cw * args.scale)
    out_h = int(ch * args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (out_w, out_h))

    fi = 0
    while cap.isOpened() and fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y:y+ch, x:x+cw]
        if args.scale != 1.0:
            crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(fi * 1000 / fps)

        hr = hand_det.detect_for_video(img, ts)
        pr = pose_det.detect_for_video(img, ts)
        fr = face_det.detect_for_video(img, ts)

        draw_landmarks(crop, hr, pr, fr, out_h, out_w)
        writer.write(crop)

        if fi % 300 == 0:
            print(f"  {fi/fps:.0f}s / {args.seconds:.0f}s", flush=True)
        fi += 1

    cap.release()
    writer.release()
    hand_det.close(); pose_det.close(); face_det.close()
    print(f"\nSaved → {args.out}  ({fi} frames)")


if __name__ == "__main__":
    main()
