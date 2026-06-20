import json
import cv2
import numpy as np
import os
from pathlib import Path
import sys

VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated"
VIZ_DIR = Path("data/viz")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

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

def draw_landmarks(landmarks: list, frame_number: int, width=640, height=480) -> np.ndarray:
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 240

    frame_lms = [lm for lm in landmarks if lm["frame"] == frame_number]

    by_source = {}
    for lm in frame_lms:
        by_source.setdefault(lm["source"], {})[lm["landmark_index"]] = lm

    for source, color, connections in [
        ("left_hand", (255, 100, 100), HAND_CONNECTIONS),
        ("right_hand", (100, 100, 255), HAND_CONNECTIONS),
        ("pose", (100, 200, 100), POSE_CONNECTIONS),
    ]:
        lms = by_source.get(source, {})
        for a, b in connections:
            if a in lms and b in lms:
                pt1 = (int(lms[a]["x"] * width), int(lms[a]["y"] * height))
                pt2 = (int(lms[b]["x"] * width), int(lms[b]["y"] * height))
                cv2.line(canvas, pt1, pt2, color, 2)
        for lm in lms.values():
            pt = (int(lm["x"] * width), int(lm["y"] * height))
            cv2.circle(canvas, pt, 4, color, -1)

    cv2.putText(canvas, f"frame {frame_number}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return canvas

def visualize(youtube_id: str):
    folder = VIDEOS_DIR / youtube_id
    landmark_file = folder / "landmarks.json"
    info_files = list(folder.glob("*.info.json"))

    if not landmark_file.exists():
        print(f"No landmarks found for {youtube_id}")
        return

    with open(landmark_file) as f:
        data = json.load(f)

    landmarks = data["landmarks"]
    dominant_hand = data["dominant_hand"]
    title = ""
    if info_files:
        with open(info_files[0], encoding="utf-8", errors="ignore") as f:
            import json as j
            info = j.load(f)
            title = info.get("title", "")

    hand_frames = sorted({lm["frame"] for lm in landmarks if "hand" in lm["source"]})

    if not hand_frames:
        print(f"No hand detections in {youtube_id}")
        return

    sample_frames = hand_frames[::max(1, len(hand_frames)//6)][:6]
    canvases = [draw_landmarks(landmarks, f) for f in sample_frames]

    rows = [np.hstack(canvases[:3])]
    if len(canvases) > 3:
        row2 = canvases[3:]
        while len(row2) < 3:
            row2.append(np.ones_like(canvases[0]) * 240)
        rows.append(np.hstack(row2))

    grid = np.vstack(rows)

    label_bar = np.ones((40, grid.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(label_bar, f"{youtube_id} | {title} | dominant: {dominant_hand}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    final = np.vstack([label_bar, grid])

    out_path = VIZ_DIR / f"{youtube_id}.png"
    cv2.imwrite(str(out_path), final)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <youtube_id>")
        sys.exit(1)
    visualize(sys.argv[1])