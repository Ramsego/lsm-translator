# LSM Landmark Dataset & Recognition Pipeline

A reproducible pipeline for **Lengua de Señas Mexicana (LSM)** — Mexican Sign Language. It builds a
landmark **word bank** of isolated signs from public YouTube videos, to be used as a reference
library for the project's actual goal: recognizing **continuous** signing from a broadcast
interpreter (Mexican presidential press conferences, the *mañanera*) aligned to public transcripts.

> **Status:** Phase 1 complete — **963 isolated signs (886 distinct labels)** from two sources,
> extracted, temporally trimmed, validated, and stored as ML-ready landmark arrays. This is a
> **mostly single-example-per-sign reference set** (one signer for the core 490; community-contributed
> signers for the rest), not a competitive standalone isolated-sign dataset (see Related work). The
> classifier and the continuous-signing (mañanera) work are in progress.

**Scope, honestly:** isolated LSM sign recognition with MediaPipe is already an established area
(see Related work). Phase 1 here is deliberately a lightweight *word bank* to bootstrap labeling of
continuous signing — where the real, underexplored gap for LSM lies. Building a continuous-signing
dataset from a broadcast interpreter + transcripts has been done for German, Uruguayan, and Chinese
sign languages, but not, as far as we found, for LSM.

## What's in the dataset

- **963 isolated LSM signs** (886 distinct Spanish labels), from two sources: the original
  single-signer word bank (`chnt`, 490) and community-contributed signs from wikisigns.org
  (`wikisigns`, 473, including "Señas Internacional" country/city signs).
- Per sign: a fixed-shape **MediaPipe Tasks** landmark array of shape `[num_frames, 116, 3]`
  (`float32`), where the 116 landmarks are ordered:
  - rows `0–20`  → left hand (21 points)
  - rows `21–41` → right hand (21 points)
  - rows `42–74` → pose (33 points)
  - rows `75–115` → curated 41-point face subset (brows, eyes, iris, mouth, head anchors —
    not the full 478 mesh, to keep input dense and the model cheap to train)
  - `x, y, z` normalized to `[0, 1]`; **`NaN` marks a landmark not detected in that frame**
    (zero is a valid coordinate, so missing data is explicitly NaN, never 0).
- **Temporally trimmed**: each array is cropped to the active signing window (largest contiguous
  hand-present segment), removing silent lead/trail padding and embedded outro cards. This dropped
  the overall NaN fraction from ~0.53 to ~0.18 and makes DTW alignment far more reliable.
- Left-handed signers are mirrored (`x → 1 − x`) so all signs are stored right-handed.
- Metadata (label, source, aliases, signer/channel, dominant hand, frame counts) in both SQLite and CSV.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `scripts/01_scrape.py` | Download all videos + metadata from a YouTube channel with `yt-dlp` |
| 2 | `scripts/02_extract.py` | Extract hand + pose + face landmarks per frame (MediaPipe Tasks API) |
| 3 | `scripts/03_ingest.py` | Build dense `[frames, 116, 3]` arrays + metadata DB/CSV |
| 4 | `scripts/trim_arrays.py` | Crop each array to its active signing window (drops padding + outro cards) |
| — | `scripts/quality_report.py` | Per-video coverage + flags → `data/quality_report.csv` |
| — | `scripts/verify_extraction.py` | Overlay landmarks on real video frames to spot tracking misses |
| — | `scripts/inspect_trim.py` | Before/after PNGs of trimmed arrays for visual QA |
| — | `scripts/visualize.py` / `batch_visualize.py` | Render landmarks over frames for visual QA |
| — | `scripts/verify_ingest.py` | Automated integrity checks on the ingested dataset |
| 5 | `scripts/05_classify.py` | DTW nearest-neighbor sign spotter *(in progress)* |

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
curl -o scripts/face_landmarker.task -L \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Reproduce the dataset

```bash
python scripts/01_scrape.py "https://www.youtube.com/@chnt1204/videos"
python scripts/02_extract.py
python scripts/03_ingest.py
python scripts/trim_arrays.py        # crop to active signing window
python scripts/quality_report.py     # per-video QA → data/quality_report.csv
python scripts/verify_ingest.py
```

Paths to the video/output directory are set at the top of each script (`VIDEOS_DIR`, etc.) —
edit them for your machine.

## Loading the data

```python
import numpy as np, pandas as pd

meta = pd.read_csv("data/metadata.csv")
row  = meta.iloc[0]
arr  = np.load(f"data/{row.array_path}")   # shape [num_frames, 116, 3]
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

- **Few examples per sign** — the core `chnt` set is one signer, ~1 video per label; `wikisigns`
  adds different signers but the two halves don't overlap, so a trained classifier isn't viable yet
  and DTW nearest-neighbor is the appropriate method.
- **Dominant-hand heuristic** — inferred from single-hand frames; it can flip on two-handed signs
  when the non-dominant hand is briefly more active (affects a minority of signs).
- **Labels unverified by a fluent signer** — labels derive from YouTube titles / wikisigns pages,
  normalized programmatically but not checked by a native LSM signer.
- **Source quality** — landmarks depend on MediaPipe; fast motion or hands near the face
  occasionally drop detections. 13 `chnt` videos were excluded after manual review
  (11 multi-sign compilations, 2 mistracked). Manual visual QA (`verify_extraction.py`) confirmed
  the remaining tracking is sound — apparent mid-sign dropouts were uploader editing artifacts
  (outro cards, static tails), now removed by trimming.

## License

Code: TBD. Dataset labels derive from publicly posted YouTube titles; raw videos are not redistributed.
MediaPipe models © Google (Apache-2.0), downloaded separately.
