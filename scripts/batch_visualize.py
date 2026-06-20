import json
import os
from pathlib import Path
import cv2
import numpy as np

VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated"
VIZ_DIR = Path("data/viz")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE = {
    "JFYEdvq3kNU", "hR4BZNxbi1Y", "EX9SPT1ytx8", "Dm-neOnO0-E",
    "K56zs2zaIQE", "gH1vhh-V6oE", "FgThSMNNPpg", "Ra-0o-PKcJk",
    "19qD4s4ZsZU", "YiWJTcitkRc", "BP3R8czviC8"
}

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

def draw_landmarks(landmarks, frame_number, width=320, height=240):
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
                cv2.line(canvas, pt1, pt2, color, 1)
        for lm in lms.values():
            pt = (int(lm["x"] * width), int(lm["y"] * height))
            cv2.circle(canvas, pt, 3, color, -1)
    return canvas

def visualize_one(folder):
    youtube_id = folder.name
    landmark_file = folder / "landmarks.json"
    info_files = list(folder.glob("*.info.json"))

    if not landmark_file.exists():
        return

    with open(landmark_file) as f:
        data = json.load(f)

    landmarks = data["landmarks"]
    dominant_hand = data["dominant_hand"]
    title = ""
    if info_files:
        try:
            with open(info_files[0], encoding="utf-8", errors="ignore") as f:
                info = json.load(f)
                title = info.get("title", "")
        except:
            pass

    hand_frames = sorted({lm["frame"] for lm in landmarks if "hand" in lm["source"]})
    if not hand_frames:
        return

    sample_frames = hand_frames[::max(1, len(hand_frames) // 3)][:3]
    canvases = [draw_landmarks(landmarks, f) for f in sample_frames]
    while len(canvases) < 3:
        canvases.append(np.ones((240, 320, 3), dtype=np.uint8) * 240)

    grid = np.hstack(canvases)
    label_bar = np.ones((30, grid.shape[1], 3), dtype=np.uint8) * 255
    label = f"{title} | {dominant_hand}"[:80]
    cv2.putText(label_bar, label, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    final = np.vstack([label_bar, grid])

    out_path = VIZ_DIR / f"{youtube_id}.png"
    cv2.imwrite(str(out_path), final)

if __name__ == "__main__":
    folders = sorted(VIDEOS_DIR.iterdir())
    total = len([f for f in folders if f.name not in EXCLUDE and f.is_dir()])
    done = 0
    for folder in folders:
        if not folder.is_dir() or folder.name in EXCLUDE:
            continue
        visualize_one(folder)
        done += 1
        if done % 50 == 0:
            print(f"{done}/{total} done...")
    print(f"All done. PNGs saved to {VIZ_DIR}")