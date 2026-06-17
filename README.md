# LSM Landmark Dataset & Recognition Pipeline

A reproducible pipeline for **Lengua de Señas Mexicana (LSM)** — Mexican Sign Language. It builds a
landmark **word bank** of isolated signs from public YouTube videos, to be used as a reference
library for the project's actual goal: recognizing **continuous** signing from a broadcast
interpreter (Mexican presidential press conferences, the *mañanera*) aligned to public transcripts.

> **Status:** Phase 1 complete — 490 isolated signs extracted, validated, and stored as
> ML-ready landmark arrays. This is a **single-signer, ~1-example-per-sign reference set**, not a
> competitive standalone isolated-sign dataset (see Related work). The classifier and the
> continuous-signing (mañanera) work are in progress.

**Scope, honestly:** isolated LSM sign recognition with MediaPipe is already an established area
(see Related work). Phase 1 here is deliberately a lightweight *word bank* to bootstrap labeling of
continuous signing — where the real, underexplored gap for LSM lies. Building a continuous-signing
dataset from a broadcast interpreter + transcripts has been done for German, Uruguayan, and Chinese
sign languages, but not, as far as we found, for LSM.

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

## Related work

Isolated LSM recognition with MediaPipe landmarks is an active area; this project does **not** claim
novelty there. Prior public LSM resources include:

- **MSL-150** — public keypoint-only dataset, 150 LSM signs from a *native signer*, MediaPipe
  Holistic. Closest comparison to Phase 1 here, and stronger as a standalone dataset.
  <https://github.com/armandobecerril/MSL-150-Dataset>
- **Mexican Sign Language Recognition: Dataset Creation and Performance Evaluation Using MediaPipe
  and Machine Learning** (MDPI Electronics, 2025). <https://www.mdpi.com/2079-9292/14/7/1423>
- **MX-ITESO-100** — 100 signs, 5,000 videos, 3 signers (cited in the Frontiers 2026 review of
  dynamic LSM recognition).
- **Spanish → LSM gloss corpus** (text only) — Nature Scientific Data, 2025.
  <https://www.nature.com/articles/s41597-025-04871-7>

What appears underexplored for LSM, and what this project targets, is **continuous** signing from a
broadcast interpreter aligned to transcripts — analogous to RWTH-PHOENIX (German) or iLSU-T
(Uruguayan), but for LSM.

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
