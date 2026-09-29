import cv2, numpy as np, json
from pathlib import Path
import mediapipe as mp
import mediapipe.python.solutions.holistic as mp_holistic

mp_face = mp.solutions.face_mesh
LIP_IDX = sorted(set(i for pair in mp_face.FACEMESH_LIPS for i in pair))
LEFT_EYE, RIGHT_EYE = 33, 263

IN = Path(__file__).parent / "span_clips"
OUT = Path(__file__).parent / "span_features"
OUT.mkdir(exist_ok=True)

hits = json.load(open(Path(__file__).parent / "runB_hits.json"))

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                           refine_face_landmarks=True) as holistic:
    for i, h in enumerate(hits):
        path = IN / f"{i:02d}_{h['word']}.mp4"
        cap = cv2.VideoCapture(str(path))
        n_dims = len(LIP_IDX) * 2
        feats = []
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.face_landmarks is None:
                feats.append(np.full(n_dims, np.nan)); continue
            lm = res.face_landmarks.landmark
            eye_dist = np.hypot(lm[LEFT_EYE].x - lm[RIGHT_EYE].x, lm[LEFT_EYE].y - lm[RIGHT_EYE].y)
            if eye_dist < 1e-6:
                feats.append(np.full(n_dims, np.nan)); continue
            cx = np.mean([lm[i2].x for i2 in LIP_IDX]); cy = np.mean([lm[i2].y for i2 in LIP_IDX])
            vec = []
            for i2 in LIP_IDX:
                vec.append((lm[i2].x - cx) / eye_dist); vec.append((lm[i2].y - cy) / eye_dist)
            feats.append(np.array(vec))
        cap.release()
        feats = np.array(feats)
        np.save(OUT / f"{i:02d}_{h['word']}.npy", feats)
        n_valid = np.sum(~np.isnan(feats[:,0])) if len(feats) else 0
        print(f"{i:02d} {h['word']:12s} frames={feats.shape[0]:4d} valid={n_valid}")
