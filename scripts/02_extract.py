import argparse
import json
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from handedness import motion_dominant

VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated"
HAND_MODEL = Path("scripts/hand_landmarker.task")
POSE_MODEL = Path("scripts/pose_landmarker.task")
FACE_MODEL = Path("scripts/face_landmarker.task")

# Curated subset of MediaPipe Face Mesh points (478-pt model). We keep only the
# ~41 landmarks that carry linguistic non-manual signal in LSM, not the full mesh,
# to keep the input dense and the downstream model small/cheap to train.
# Array row for a face point = 75 + its position in this list.
FACE_LANDMARKS = [
    # eyebrows (10)
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # eyes / openness (12)
    33, 133, 159, 145, 160, 144, 362, 263, 386, 374, 387, 373,
    # iris / gaze (2)
    468, 473,
    # mouth — dense (20): interpreters mouth Spanish words continuously, a strong
    # disambiguation cue, so we sample both lip rings for viseme-level detail.
    # outer lip ring (10)
    61, 291, 0, 17, 37, 267, 84, 314, 40, 270,
    # inner lip ring (10)
    78, 308, 13, 14, 82, 312, 87, 317, 81, 311,
    # head-orientation anchors (5)
    10, 152, 234, 454, 1,
]

def make_detectors(face_min_detection_confidence: float = 0.5):
    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL)),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
    )
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=face_min_detection_confidence,
    )
    return (
        vision.HandLandmarker.create_from_options(hand_opts),
        vision.PoseLandmarker.create_from_options(pose_opts),
        vision.FaceLandmarker.create_from_options(face_opts),
    )

def extract_landmarks(video_path: Path) -> tuple:
    hand_detector, pose_detector, face_detector = make_detectors()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_number * 1000 / fps)

        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)

        frame_data = []

        if hand_result.hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks):
                handedness = hand_result.handedness[hand_idx][0].category_name.lower()
                source = f"{handedness}_hand"
                for i, lm in enumerate(hand_landmarks):
                    frame_data.append({"frame": frame_number, "source": source, "landmark_index": i, "x": lm.x, "y": lm.y, "z": lm.z})

        if pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                frame_data.append({"frame": frame_number, "source": "pose", "landmark_index": i, "x": lm.x, "y": lm.y, "z": lm.z})

        if face_result.face_landmarks:
            face_landmarks = face_result.face_landmarks[0]
            for pos, mesh_idx in enumerate(FACE_LANDMARKS):
                lm = face_landmarks[mesh_idx]
                frame_data.append({"frame": frame_number, "source": "face", "landmark_index": pos, "x": lm.x, "y": lm.y, "z": lm.z})

        frames.append(frame_data)
        frame_number += 1

    cap.release()
    hand_detector.close()
    pose_detector.close()
    face_detector.close()

    all_landmarks = [lm for frame in frames for lm in frame]
    # Store RAW, as-detected coordinates. Handedness invariance is handled at DTW
    # query time (handedness.mirror_array); dominant_hand is informational only.
    dominant_hand = motion_dominant(all_landmarks)

    return all_landmarks, dominant_hand

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand+pose+face landmarks from sign videos.")
    parser.add_argument("--videos-dir", type=Path, default=VIDEOS_DIR,
                        help="Folder of per-video subfolders to process.")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if landmarks.json already exists.")
    args = parser.parse_args()

    for video_folder in sorted(args.videos_dir.iterdir()):
        if not video_folder.is_dir() or video_folder.name.startswith("._"):
            continue

        landmark_file = video_folder / "landmarks.json"
        if landmark_file.exists() and not args.force:
            continue

        video_files = [p for p in video_folder.glob("*.mp4") if not p.name.startswith("._")]
        if not video_files:
            continue

        video_path = video_files[0]
        print(f"Extracting: {video_path.name}")
        landmarks, dominant_hand = extract_landmarks(video_path)

        with open(landmark_file, "w") as f:
            json.dump({"dominant_hand": dominant_hand, "orientation": "raw",
                       "landmarks": landmarks}, f)

        # Source video is intentionally retained (we no longer delete it) so the
        # dataset can be re-extracted without re-downloading.
        print(f"  → {len(landmarks)} records | dominant hand: {dominant_hand}")