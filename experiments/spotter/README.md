# DTW sign spotter (exploratory baseline)

A Dynamic Time Warping nearest-neighbor matcher over the project's landmark word bank. Given a
landmark clip (`[frames, 116, 3]`), it returns the closest reference sign(s).

**This is an exploratory baseline, not the project's headline contribution** (that's the dataset +
pipeline in the repo root). It is kept here as an honest baseline and a potential Phase-2 bootstrapping
aid.

## How it works
- **Feature:** hand + arm landmarks, `(x, y)`, per-clip shoulder normalization (config in `DEFAULT_CFG`).
- **Handedness:** invariant via mirror-at-query — compares a clip against both itself and its mirror
  (`scripts/handedness.py: mirror_array`) and takes the smaller distance, since the dataset is stored
  raw (not canonicalized).
- **Distance:** `dtaidistance.dtw_ndim` with a Sakoe-Chiba band. Needs the compiled C backend
  (`pip install --force-reinstall --no-binary :all: dtaidistance`); falls back to pure Python.

## Results (honest)
Leave-one-out over the 963-clip bank:

| Slice | top-1 | top-5 |
|---|---|---|
| Multi-example labels (fair within-bank) | 4.9% | 13.9% |
| Cross-source (different signers) | 2.9% | 5.7% |

Cross-signer, open-vocabulary (886 classes), 1-shot matching is **inherently weak** — a feature sweep
(`tune_features.py`) confirmed the ceiling; velocity and z-score features actively hurt (they amplify
MediaPipe jitter on short clips). The engine itself is correct: self-match is exact and neighbors
cluster sensibly (food signs match food signs).

**Where it would pay off:** within-signer matching and transcript-constrained candidate sets (a handful
of words, not 886) — i.e. Phase 2 mañanera bootstrapping, where a transcript narrows the choices and the
interpreter's own signs become the references after seeding.

## Run (from repo root)
```bash
python experiments/spotter/classify.py --npy data/arrays/<id>.npy   # top-5 for one clip
python experiments/spotter/classify.py --eval                       # leave-one-out accuracy
python experiments/spotter/tune_features.py                         # feature-variant sweep
python experiments/spotter/test_classify.py                         # unit tests (7/7)
```
