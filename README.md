# LSM Landmark Dataset & Recognition Pipeline

A reproducible pipeline that builds a landmark dataset for **Lengua de Señas Mexicana (LSM)** —
Mexican Sign Language — from public YouTube videos, and (work in progress) a sign-recognition
classifier on top of it.

> **Status:** Phase 1 complete — 490 isolated signs extracted, validated, and stored as
> ML-ready landmark arrays. Classifier and continuous-signing (mañanera) work are in progress.

There is currently no public LSM landmark dataset; this is an attempt to fill that gap.

## What's in the dataset

- **490 isolated LSM signs**, one Spanish label each (from the source video title).
- Per sign: a fixed-shape **MediaPipe Holistic** landmark array of shape `[num_frames, 75, 3]`
  (`float32`), where the 75 landmarks are ordered:
  - rows `0–20`  → left hand (21 points)
  - rows `21–41` → right hand (21 points)
  - rows `42–74` → pose (33 points)
  - `x, y, z` normalized to `[0, 1]`; **`NaN` marks a landmark not detected in that frame**
    (zero is a valid coordinate, so missing data is explicitly NaN, never 0).
- Left-handed signers are mirrored (`x → 1 − x`) so all signs are stored right-handed.
- Metadata (label, signer/channel, dominant hand, frame counts) in both SQLite and CSV.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/01_scrape.py` | Download all videos + metadata from a YouTube channel with `yt-dlp` |
| 2 | `scripts/02_extract.py` | Extract hand + pose landmarks per frame (MediaPipe Tasks API) |
| 3 | `scripts/03_ingest.py` | Build dense `[frames, 75, 3]` arrays + metadata DB/CSV |
| — | `scripts/validate.py` | Dataset-wide quality report → `data/dataset_stats.json` |
| — | `scripts/visualize.py` / `batch_visualize.py` | Render landmarks over frames for visual QA |
| — | `scripts/verify_ingest.py` | Automated integrity checks on the ingested dataset |
| 4 | `scripts/04_classify.py` | DTW nearest-neighbor sign spotter *(in progress)* |

## Setup

```bash
pip install -r requirements.txt
```

Download the MediaPipe model files (not redistributed here — © Google, Apache-2.0):

```bash
curl -o scripts/hand_landmarker.task -L \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -o scripts/pose_landmarker.task -L \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Reproduce the dataset

```bash
python scripts/01_scrape.py "https://www.youtube.com/@chnt1204/videos"
python scripts/02_extract.py
python scripts/03_ingest.py
python scripts/verify_ingest.py
```

Paths to the video/output directory are set at the top of each script (`VIDEOS_DIR`, etc.) —
edit them for your machine.

## Loading the data

```python
import numpy as np, pandas as pd

meta = pd.read_csv("data/metadata.csv")
row  = meta.iloc[0]
arr  = np.load(f"data/{row.array_path}")   # shape [num_frames, 75, 3]
print(row.label, arr.shape)
```

## Known limitations

- **Single signer** — all signs come from one YouTube channel; the model would learn one signing style.
- **One example per sign** — ~1 video per label, so a trained classifier isn't viable yet;
  DTW nearest-neighbor is the appropriate method.
- **Dominant-hand heuristic** — inferred from single-hand frames; it can flip on two-handed signs
  when the non-dominant hand is briefly more active (affects a minority of signs).
- **Source quality** — landmarks depend on MediaPipe; fast motion or hands near the face
  occasionally drop detections. 13 videos were excluded after manual review
  (11 multi-sign compilations, 2 mistracked).

## License

Code: TBD. Dataset labels derive from publicly posted YouTube titles; raw videos are not redistributed.
MediaPipe models © Google (Apache-2.0), downloaded separately.
