import argparse
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

VIDEOS_DIR = Path("/Volumes/Crucial X8/LSM_Translator/videos/isolated")
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
    # mouth shape (12)
    61, 291, 0, 17, 13, 14, 78, 308, 82, 312, 87, 317,
    # head-orientation anchors (5)
    10, 152, 234, 454, 1,
]

def make_detectors():
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
    )
    return (
        vision.HandLandmarker.create_from_options(hand_opts),
        vision.PoseLandmarker.create_from_options(pose_opts),
        vision.FaceLandmarker.create_from_options(face_opts),
    )

def infer_dominant_hand(frames: list) -> str:
    left_only = 0
    right_only = 0
    for frame in frames:
        sources = {lm["source"] for lm in frame}
        has_left = "left_hand" in sources
        has_right = "right_hand" in sources
        if has_left and not has_right:
            left_only += 1
        elif has_right and not has_left:
            right_only += 1
    return "left" if left_only > right_only else "right"

def normalize_handedness(landmarks: list, dominant_hand: str) -> list:
    if dominant_hand == "right":
        return landmarks
    for lm in landmarks:
        lm["x"] = 1 - lm["x"]
    return landmarks

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
    dominant_hand = infer_dominant_hand(frames)
    all_landmarks = normalize_handedness(all_landmarks, dominant_hand)

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
            json.dump({"dominant_hand": dominant_hand, "landmarks": landmarks}, f)

        # Source video is intentionally retained (we no longer delete it) so the
        # dataset can be re-extracted without re-downloading.
        print(f"  → {len(landmarks)} records | dominant hand: {dominant_hand}")