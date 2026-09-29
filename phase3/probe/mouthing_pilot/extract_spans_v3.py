"""Route 1 Step 1: occlusion + aperture sidecars for the 16 Run-B hit spans."""
import cv2, numpy as np, json
from pathlib import Path
import mediapipe as mp
import mediapipe.python.solutions.holistic as mp_holistic

LEFT_EYE, RIGHT_EYE = 33, 263
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14
LEFT_CORNER, RIGHT_CORNER = 61, 291
LIP_IDX = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))

IN = Path(__file__).parent / "span_clips_v2"
OUT = Path(__file__).parent / "sidecar_spans_v3"
OUT.mkdir(exist_ok=True)

hits = json.load(open(Path(__file__).parent / "runB_hits.json"))

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                           refine_face_landmarks=True) as holistic:
    for i, h in enumerate(hits):
        path = IN / f"{i:02d}_{h['word']}.mp4"
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        aperture, width, hand_dist, face_ok = [], [], [], []
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.face_landmarks is None:
                aperture.append(np.nan); width.append(np.nan); hand_dist.append(np.nan); face_ok.append(False)
                continue
            lm = res.face_landmarks.landmark
            eye_dist = np.hypot(lm[LEFT_EYE].x - lm[RIGHT_EYE].x, lm[LEFT_EYE].y - lm[RIGHT_EYE].y)
            if eye_dist < 1e-6:
                aperture.append(np.nan); width.append(np.nan); hand_dist.append(np.nan); face_ok.append(False)
                continue
            ap = (lm[LOWER_INNER_LIP].y - lm[UPPER_INNER_LIP].y) / eye_dist
            wd = np.hypot(lm[LEFT_CORNER].x - lm[RIGHT_CORNER].x, lm[LEFT_CORNER].y - lm[RIGHT_CORNER].y) / eye_dist
            aperture.append(ap); width.append(wd); face_ok.append(True)
            mcx = np.mean([lm[j].x for j in LIP_IDX]); mcy = np.mean([lm[j].y for j in LIP_IDX])
            dmin = np.inf
            for hand_lms in (res.left_hand_landmarks, res.right_hand_landmarks):
                if hand_lms is None: continue
                for hl in hand_lms.landmark:
                    d = np.hypot(hl.x - mcx, hl.y - mcy) / eye_dist
                    if d < dmin: dmin = d
            hand_dist.append(dmin)
        cap.release()
        aperture = np.array(aperture); width = np.array(width); hand_dist = np.array(hand_dist); face_ok = np.array(face_ok)
        np.savez(OUT / f"{i:02d}_{h['word']}.npz", aperture=aperture, width=width, hand_dist=hand_dist, face_ok=face_ok, fps=fps)
        n_occ = np.sum((hand_dist < 0.9))
        print(f"{i:02d} {h['word']:12s} frames={len(aperture):4d} fps={fps:.1f} occluded_frames(<0.9)={n_occ}")
