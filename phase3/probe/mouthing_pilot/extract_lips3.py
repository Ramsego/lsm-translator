"""Route 1 Step 1: occlusion + aperture sidecars for the 26 verified clips.
Does NOT touch features2/*.npy. Writes sidecar/<clipname>.npz with aperture, hand_dist, face_ok.
"""
import cv2, numpy as np, csv
from pathlib import Path
import mediapipe as mp
import mediapipe.python.solutions.holistic as mp_holistic

LEFT_EYE, RIGHT_EYE = 33, 263
UPPER_INNER_LIP, LOWER_INNER_LIP = 13, 14

DRIVE = Path("/Volumes/Crucial X8/LSM_Translator/review/57TvyH9902U")
OUT = Path(__file__).parent / "sidecar"
OUT.mkdir(exist_ok=True)

rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                           refine_face_landmarks=True) as holistic:
    for r in rows:
        rel = r["file"].replace("../57TvyH9902U/", "")
        path = DRIVE / rel
        if not path.exists():
            print("MISSING", path); continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        aperture, hand_dist, face_ok = [], [], []
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)
            if res.face_landmarks is None:
                aperture.append(np.nan); hand_dist.append(np.nan); face_ok.append(False)
                continue
            lm = res.face_landmarks.landmark
            eye_dist = np.hypot(lm[LEFT_EYE].x - lm[RIGHT_EYE].x, lm[LEFT_EYE].y - lm[RIGHT_EYE].y)
            if eye_dist < 1e-6:
                aperture.append(np.nan); hand_dist.append(np.nan); face_ok.append(False)
                continue
            ap = (lm[LOWER_INNER_LIP].y - lm[UPPER_INNER_LIP].y) / eye_dist
            aperture.append(ap); face_ok.append(True)
            cx = np.mean([lm[i].x for i in range(468)]) if False else None
            # mouth centroid in raw image-normalized coords, same definition as LIP_IDX centroid
            LIP_IDX_LOCAL = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))
            mcx = np.mean([lm[i].x for i in LIP_IDX_LOCAL]); mcy = np.mean([lm[i].y for i in LIP_IDX_LOCAL])
            dmin = np.inf
            for hand_lms in (res.left_hand_landmarks, res.right_hand_landmarks):
                if hand_lms is None: continue
                for hl in hand_lms.landmark:
                    d = np.hypot(hl.x - mcx, hl.y - mcy) / eye_dist
                    if d < dmin: dmin = d
            hand_dist.append(dmin)
        cap.release()
        aperture = np.array(aperture); hand_dist = np.array(hand_dist); face_ok = np.array(face_ok)
        out_name = rel.replace("/", "__").replace(".mp4", ".npz")
        np.savez(OUT / out_name, aperture=aperture, hand_dist=hand_dist, face_ok=face_ok, fps=fps)
        n_occ = np.sum((hand_dist < 0.9))
        print(f"{r['word']:12s} {rel:35s} frames={len(aperture):4d} fps={fps:.1f} occluded_frames(<0.9)={n_occ}")
