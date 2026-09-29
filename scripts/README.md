# Phase 1 — isolated-sign word bank

A landmark **word bank** of isolated LSM signs built from public YouTube videos. It serves as a
dictionary of candidate sign forms for the corpus pipeline (see the [top-level README](../README.md)),
**not** as training data.

**Status:** complete — **963 isolated signs (886 distinct labels)** from two sources, extracted,
temporally trimmed, validated, and stored as ML-ready landmark arrays. This is a **mostly
single-example-per-sign reference set** (819 labels have one example), not a competitive standalone
isolated-sign dataset — isolated LSM recognition with MediaPipe is already an established area.

## What's in the dataset

- **963 isolated LSM signs** (886 distinct Spanish labels), from two sources: the original
  single-signer word bank (`chnt`, 490) and community-contributed signs from wikisigns.org
  (`wikisigns`, 473, including "Señas Internacional" country/city signs).
- Per sign: a fixed-shape **MediaPipe Tasks** landmark array of shape `[num_frames, 116, 3]`
  (`float32`), where the 116 landmarks are ordered:
  - rows `0–20`  → left hand (21 points)
  - rows `21–41` → right hand (21 points)
  - rows `42–74` → pose (33 points)
  - rows `75–115` → curated 41-point face subset: eyebrows (10), eyes (12), iris (2),
    mouth outer ring (10), head-orientation anchors (5) — not the full 478-point mesh,
    to keep input dense and the downstream model cheap to train.
    *Note: the continuous-signing pipeline uses an expanded 124-row schema (49-pt face,
    adding a dense inner lip ring for mouthing detection). Phase 1 arrays on disk remain 116.*
  - `x, y, z` normalized to `[0, 1]`; **`NaN` marks a landmark not detected in that frame**
    (zero is a valid coordinate, so missing data is explicitly NaN, never 0).
- **Temporally trimmed**: each array is cropped to the active signing window (largest contiguous
  hand-present segment), removing silent lead/trail padding and embedded outro cards. This dropped
  the overall NaN fraction from ~0.53 to ~0.18 and makes DTW alignment far more reliable.
- **Stored in raw, as-detected orientation** — coordinates are *not* canonicalized for handedness.
  Handedness invariance is handled at match time: the DTW classifier compares a query against both
  itself and its mirror (`handedness.mirror_array`, which flips `x` *and* swaps the hand channels)
  and takes the smaller distance. `dominant_hand` in the metadata is informational only.
- Metadata (label, source, aliases, signer/channel, dominant hand, frame counts) in both SQLite and CSV.

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_scrape.py` | Download all videos + metadata from a YouTube channel with `yt-dlp` |
| 2 | `02_extract.py` | Extract hand + pose + face landmarks per frame (MediaPipe Tasks API), raw orientation |
| 3 | `03_ingest.py` | Build dense `[frames, 116, 3]` arrays + metadata DB/CSV |
| 4 | `trim_arrays.py` | Crop each array to its active signing window (drops padding + outro cards) |
| — | `handedness.py` | Shared handedness helpers (`mirror_array`, motion-based dominance) |
| — | `unmirror.py` | One-time pass: restore raw orientation from old canonicalized data |
| — | `quality_report.py` | Per-video coverage + flags → `data/quality_report.csv` |
| — | `verify_extraction.py` | Overlay landmarks on real video frames to spot tracking misses |
| — | `inspect_trim.py` | Before/after PNGs of trimmed arrays for visual QA |
| — | `visualize.py` / `batch_visualize.py` | Render landmarks over frames for visual QA |
| — | `verify_ingest.py` | Automated integrity checks on the ingested dataset |

> An exploratory DTW nearest-neighbor sign spotter built on this dataset lives in
> [`experiments/spotter/`](../experiments/spotter/). It is a baseline, not part of the dataset
> pipeline — open-vocabulary cross-signer accuracy is low (see its README).

## Reproduce the dataset

Run from the repo root, after the setup steps in the [top-level README](../README.md#setup):

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

## Known limitations

- **Few examples per sign** — the core `chnt` set is one signer, ~1 video per label; `wikisigns`
  adds different signers but the two halves don't overlap, so a trained classifier isn't viable
  and DTW nearest-neighbor is the appropriate method.
- **Labels unverified by a fluent signer** — labels derive from YouTube titles / wikisigns pages,
  normalized programmatically but not checked by a native LSM signer.
- **Source quality** — landmarks depend on MediaPipe; fast motion or hands near the face
  occasionally drop detections. 13 `chnt` videos were excluded after manual review
  (11 multi-sign compilations, 2 mistracked). Manual visual QA (`verify_extraction.py`) confirmed
  the remaining tracking is sound — apparent mid-sign dropouts were uploader editing artifacts
  (outro cards, static tails), now removed by trimming.
