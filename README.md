# LSM Broadcast Corpus — mining Mexican Sign Language from government broadcasts

A weakly-supervised pipeline that turns unlabeled, interpreted government broadcasts into a
corpus of continuous **Lengua de Señas Mexicana (LSM)** — Mexican Sign Language — aligned to
Spanish text.

**The deliverable** is a corpus + baselines + the pipeline that mines it, released as a
resource/methods contribution. **It is not a translator.** An LSM→Spanish translator is the
north star that motivates the design, but no work is currently pointed at it: a translator is
trained on (video, text) pairs, and producing those pairs is the problem this project solves.

> The single authoritative statement of project scope and status is the **CANON block** at the
> top of [`PIPELINE_MAP.md`](PIPELINE_MAP.md). If this README and that block ever disagree, the
> block wins.

## Why

LSM has almost no labeled data of **continuous** signing — the kind needed to train recognition
or translation models. Other sign languages got theirs by mining interpreted broadcasts against
their subtitles or transcripts: German (RWTH-PHOENIX), British (BSL-1K / BOBSL, Oxford VGG), and
Uruguayan (iLSU-T). To our knowledge, this has not been done for LSM.

Mexico has the raw material. The President's daily press conferences (*mañaneras*) are broadcast
with a sign-language interpreter in a picture-in-picture inset, and the government publishes an
official transcript of each one (*versión estenográfica*). The archive runs to **thousands of
hours**. The obstacle: **none of it is labeled.** The transcript has no timestamps, and nothing
says which stretch of signing corresponds to which sentence or word.

So far a **21-hour slice (16 videos)** has been processed. Compute is the limit, not the source —
the pipeline is built to scale to the rest of the archive.

## How — the pipeline

The method is the BSL-1K / BOBSL weak-supervision recipe: don't hand-annotate; auto-annotate
interpreter footage using the transcript, a model of interpreter lag, mouthing cues, and
dictionary lookups, then train on the mined labels and use the trained model to mine better ones.

```
Stage 0  RAW → CLEAN SIGNAL                                   done
         download → locate interpreter inset → MediaPipe landmarks (124 pts/frame)
         → presence gate (is the interpreter actually on screen and signing?)

Stage 1  SENTENCE-LEVEL ALIGNMENT                             working, preliminary
         which ~10–15 s span of signing ↔ which transcript sentence?
         ASR word times + transcript + lag estimate + SEA segmenter + DP alignment

Stage 2  WORD-LEVEL LOCALIZATION                              probes run, none passed yet
         inside an aligned span, where is each word signed?
         mouthing + LSM word-order prior + visual scorer + Phase 1 dictionary

Stage 3  TRAINING, ROUND 1                                    not started
         fine-tune SignCLIP on mined (clip, word) pairs → in-domain scorer
         → feed back into Stage 2 → better labels → re-mine (bootstrap loop)

Stage 4  RELEASE                                              future
         corpus + baselines + pipeline; translator remains the north star
```

| stage | key code | what it does |
|---|---|---|
| 0 | [`phase2/download_mananera.py`](phase2/download_mananera.py), [`phase2/locate_recuadro.py`](phase2/locate_recuadro.py) | fetch broadcasts, find the interpreter inset |
| 0 | [`phase2/extract_continuous.py`](phase2/extract_continuous.py) | 124-row landmark arrays: 21+21 hand, 33 pose, 49 face incl. inner lip ring |
| 0 | [`phase3/sea_test/revalidate_segments.py`](phase3/sea_test/revalidate_segments.py) | pose-based presence gate (drops B-roll / interpreter-absent stretches) |
| 1 | [`phase2/fetch_estenografica.py`](phase2/fetch_estenografica.py), [`phase2/align_audio.py`](phase2/align_audio.py) | official transcript + word-level ASR timings |
| 1 | [`phase2/estimate_lag.py`](phase2/estimate_lag.py) | how far the interpreter signs behind the speaker |
| 1 | [`phase3/sea_test/build_sea_inputs.py`](phase3/sea_test/build_sea_inputs.py) | package inputs for the SEA segmenter + DP aligner ([Jiang et al.](https://arxiv.org/abs/2512.08094)) |
| eval | [`phase3/sea_test/check_alignment_v2.py`](phase3/sea_test/check_alignment_v2.py) | alignment metrics against hand-verified sign locations |
| QC | [`qc/run_gates.py`](qc/run_gates.py) | integrity / presence / temporal / activity / review gates over pipeline outputs |
| ref | [`scripts/`](scripts/) | Phase 1 isolated-sign dictionary (see [its README](scripts/README.md)) |

## What has been done

| component | result |
|---|---|
| Broadcast processing (Stage 0) | 16 videos, 21.15 h of signing extracted; presence gate keeps **236 of 549 segments (20.79 h)**. Face/mouth landmarks present in 88–99.8% of frames |
| Transcripts + ASR | word-level ASR on 16/16 videos, paired with the official transcripts |
| Interpreter lag | **+6.33 s** behind speech (spread ±3.5 s) on the test slice |
| Sentence alignment (Stage 1) | **73.1% (19/26)** of verified words land in the correct aligned span, vs **65.4%** for a fixed-lag baseline; median error **3.46 s vs 4.81 s**. *Preliminary:* one interpreter, one 30-minute slice, n=26, improvement not yet significant (Wilcoxon p=0.089) |
| Word order (Stage 2 prior) | measured on a 3,000-pair Spanish↔LSM gloss corpus: 42% of pairs differ in order; time markers are fronted 87% of the time. Not yet validated on held-out data |
| Mouthing (Stage 2 cue) | interpreter mouthing present in 16/16 videos; mouth-shape DTW discriminates between words above chance (Monte Carlo p≈0.0006) |
| Phase 1 dictionary | **963 isolated signs / 886 labels** as MediaPipe arrays — a reference of candidate sign forms, not training data |
| Quality control | automated gate suite + tests over pipeline outputs ([`qc/`](qc/)) |

### Paths tested and closed

Negative results, each of which ruled out a branch:

- **Cross-signer DTW template matching** — ~1% accuracy. Template matching can't generalize across
  signers, which is why a trained scorer is needed.
- **Off-the-shelf SignCLIP as the scorer** — weak but real on dictionary-style signing, at chance
  on interpreter footage (domain gap measured, p=0.038). The pipeline design survives; the
  pretrained scorer doesn't — hence Stage 3 fine-tuning.
- **Duration / grammar-length prior** — measured signing-to-speech ratio is 0.966; no headroom.
- **Hand-built mouthing features** — aperture-only, word-order-position and viseme-table
  features are null. (Mouth-shape DTW is not — see above.)

## What is missing

- **A word-level localizer that passes its test (Stage 2).** Nothing has passed yet; this is the
  main open problem.
- **Per-block lag.** Replace the single lag estimate with one per block, removing the ±3.5 s spread
  the aligner currently has to absorb.
- **Scaling.** Run Stage 1 over the full processed 21 h, and extend extraction beyond the 21 h
  slice to more of the archive (compute-bound).
- **Training.** None has run. Stage 3 (SignCLIP fine-tuning) is gated on Stages 1+2 producing a
  first batch of high-confidence pairs.
- **Verification by fluent signers.** Labels and evaluation are not yet checked by native LSM
  signers.
- **Release.** Corpus packaging, documentation, and licensing.
- **Test coverage** on the Phase 2/3 code beyond the QC gates.

## Methodology

- **Pre-registered success criteria**: each experiment's pass/fail threshold is fixed before it
  runs and not adjusted after seeing results; a failed gate stops the line of work.
- **Metrics chosen by power analysis**: binary hit/miss scoring can't detect realistic
  improvements at this sample size, so alignment is judged on continuous distance metrics with
  paired tests.
- **Results are never overwritten**: every run writes to a new output directory so baselines stay
  comparable.
- **Instrument bugs are documented, not buried** — e.g. a Spanish-inflection matching bug that had
  been understating alignment accuracy was found, fixed, and all affected numbers re-derived.

## Repository layout

| path | contents |
|---|---|
| [`scripts/`](scripts/) | Phase 1: isolated-sign dictionary pipeline ([README](scripts/README.md)) |
| [`phase2/`](phase2/) | broadcast download, interpreter extraction, transcripts, ASR, lag, review tooling |
| [`phase3/`](phase3/) | sentence alignment (`sea_test/`), gold-set tooling (`gold/`), review server, gloss corpus |
| [`qc/`](qc/) | quality gates + tests |
| [`experiments/spotter/`](experiments/spotter/) | exploratory DTW sign spotter (closed baseline) |
| [`PIPELINE_MAP.md`](PIPELINE_MAP.md) | canonical project statement; every component and its status |
| [`phase3/NEXT_STEPS_SPEC.md`](phase3/NEXT_STEPS_SPEC.md) | current workstreams |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm   # Spanish lemmatization (Phase 2)
playwright install chromium                  # transcript fetcher (Phase 2)
```

Two further environments are used for Phase 3: one for the SEA segmenter (Python 3.12, from the
[SEA repo](https://arxiv.org/abs/2512.08094)'s environment) and one pinned to `mediapipe==0.10.21`
(Python 3.11), since mainline MediaPipe removed the legacy Holistic API.

Data location: videos and landmark arrays live outside the repo (they're large and git-ignored).
Point the pipeline at wherever you keep them via an environment variable (defaults to the
maintainer's drive if unset):

```bash
export LSM_DATA_ROOT="/path/to/LSM_Translator"   # contains videos/ and arrays/
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

## Related work

**Method lineage**
- **BSL-1K** (Albanie et al., 2020) and **BOBSL** (Albanie et al., 2021) — mining British Sign
  Language from interpreted TV using subtitles, mouthing, and dictionaries. This project follows
  their recipe. <https://arxiv.org/abs/2007.12131>, <https://arxiv.org/abs/2111.03635>
- **SEA: Segment, Embed, and Align** (Jiang et al.) — sign segmentation + subtitle alignment used
  in Stage 1. <https://arxiv.org/abs/2512.08094>
- **SignCLIP** (Jiang et al., 2024) — multilingual sign/text embedding; tested as the Stage 2 scorer
  and the planned Stage 3 fine-tuning target. <https://arxiv.org/abs/2407.01264>
- **RWTH-PHOENIX-Weather** (German) and **iLSU-T** (Uruguayan) — continuous-signing corpora from
  interpreted broadcasts.

**LSM resources** — all isolated-sign or text-only; none are continuous signing aligned to text:
- **MSL-150** — keypoint-only dataset, 150 signs from a native signer, MediaPipe Holistic.
  <https://github.com/armandobecerril/MSL-150-Dataset>
- **Mexican Sign Language Recognition: Dataset Creation and Performance Evaluation Using MediaPipe
  and Machine Learning** (MDPI Electronics, 2025). <https://www.mdpi.com/2079-9292/14/7/1423>
- **MX-ITESO-100** — 100 signs, 5,000 videos, 3 signers.
- **Spanish → LSM gloss corpus** (text only) — Nature Scientific Data, 2025. Used here for the
  word-order prior. <https://www.nature.com/articles/s41597-025-04871-7>

## Scope and claims

This is a solo, unfunded technical project. Its claims are technical and resource-level only; it
makes no claim of community impact. Deaf-community involvement is planned once there is funding
to pay for it.

## License

Code: TBD. Source broadcasts and transcripts are publicly posted by the Mexican government; raw
videos are not redistributed here. Phase 1 labels derive from publicly posted YouTube titles and wikisigns.org.
MediaPipe models © Google (Apache-2.0), downloaded separately. The gloss corpus is CC BY 4.0
(figshare DOI 10.6084/m9.figshare.28519580).
