"""
Run MediaPipe on the cropped interpreter recuadro and report detection coverage.

This is the key go/no-go metric: can MediaPipe track hands/face on the small PiP
interpreter (optionally upscaled)? Reuses the detector setup from scripts/02_extract.py.

Usage:
    python phase2/extract_interpreter.py <video.mp4> --crop X Y W H [--scale 2] [--seconds 120]
"""

import argparse
import importlib.util
import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

# Import make_detectors + FACE_LANDMARKS from scripts/02_extract.py (numeric name).
_spec = importlib.util.spec_from_file_location(
    "extract02", Path(__file__).resolve().parents[1] / "scripts" / "02_extract.py")
extract02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="Upscale factor for the crop.")
    ap.add_argument("--seconds", type=float, default=120, help="How many seconds to process.")
    ap.add_argument("--face-confidence", type=float, default=0.5,
                    help="min_face_detection_confidence for FaceLandmarker (default 0.5).")
    args = ap.parse_args()

    hand_det, pose_det, face_det = extract02.make_detectors(
        face_min_detection_confidence=args.face_confidence)
    x, y, w, h = args.crop

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    max_frames = int(args.seconds * fps)

    n = hand_f = pose_f = face_f = 0
    fi = 0
    while cap.isOpened() and fi < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y:y+h, x:x+w]
        if args.scale != 1.0:
            crop = cv2.resize(crop, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(fi * 1000 / fps)

        hr = hand_det.detect_for_video(img, ts)
        pr = pose_det.detect_for_video(img, ts)
        fr = face_det.detect_for_video(img, ts)

        n += 1
        hand_f += bool(hr.hand_landmarks)
        pose_f += bool(pr.pose_landmarks)
        face_f += bool(fr.face_landmarks)
        fi += 1

    cap.release()
    hand_det.close(); pose_det.close(); face_det.close()

    if n == 0:
        print("No frames processed.")
        return
    print(f"\nProcessed {n} frames ({n/fps:.0f}s), crop={args.crop} scale={args.scale}")
    print(f"  hand detected: {100*hand_f/n:5.1f}% of frames")
    print(f"  pose detected: {100*pose_f/n:5.1f}% of frames")
    print(f"  face detected: {100*face_f/n:5.1f}% of frames")


if __name__ == "__main__":
    main()
