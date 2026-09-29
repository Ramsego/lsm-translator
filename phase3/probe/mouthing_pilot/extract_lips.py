import cv2, numpy as np, csv, sys
from pathlib import Path
import mediapipe as mp

mp_face = mp.solutions.face_mesh
LIP_IDX = sorted(set(i for pair in mp_face.FACEMESH_LIPS for i in pair))
LEFT_EYE, RIGHT_EYE = 33, 263  # outer eye corners, for scale normalization

DRIVE = Path("/Volumes/Crucial X8/LSM_Translator/review/57TvyH9902U")
OUT = Path(__file__).parent / "features"
OUT.mkdir(exist_ok=True)

rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]

import mediapipe.python.solutions.holistic as mp_holistic

results_summary = []
with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                           refine_face_landmarks=True) as holistic:
    for r in rows:
        rel = r["file"].replace("../57TvyH9902U/", "")
        path = DRIVE / rel
        if not path.exists():
            print("MISSING", path); continue
        cap = cv2.VideoCapture(str(path))
        feats = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.face_landmarks is None:
                continue
            lm = res.face_landmarks.landmark
            eye_dist = np.hypot(lm[LEFT_EYE].x - lm[RIGHT_EYE].x, lm[LEFT_EYE].y - lm[RIGHT_EYE].y)
            if eye_dist < 1e-6:
                continue
            cx = np.mean([lm[i].x for i in LIP_IDX])
            cy = np.mean([lm[i].y for i in LIP_IDX])
            vec = []
            for i in LIP_IDX:
                vec.append((lm[i].x - cx) / eye_dist)
                vec.append((lm[i].y - cy) / eye_dist)
            feats.append(vec)
        cap.release()
        feats = np.array(feats)
        out_name = rel.replace("/", "__").replace(".mp4", ".npy")
        np.save(OUT / out_name, feats)
        results_summary.append((r["word"], rel, feats.shape))
        print(f"{r['word']:12s} {rel:35s} frames_with_face={feats.shape[0]}")

print("\nDone.", len(results_summary), "clips processed.")
