# LSM Interpreter Corpus — Project Overview

*A scalable pipeline for mining continuous Mexican Sign Language (Lengua de Señas Mexicana)
from government broadcast video.*

Last updated: 2026-08-13 (title and §1 now defer to the CANON block; §9 added 2026-08-01;
§7 kept for history but superseded by §9)

> **`PIPELINE_MAP.md` holds the CANON block — the single authoritative statement of what
> this project is. Read it first, and quote it rather than this document when describing
> the project.** This document holds the fuller background, data-source rationale, schema,
> and model sketch. Where the two disagree, CANON wins.
>
> *(The old title of this file was "LSM → Spanish Translator", which contradicted its own
> §1. The translator is the north star, not the deliverable.)*

---

## 1. Goal

**See the CANON block in `PIPELINE_MAP.md`. It is authoritative; this section only
elaborates on the north star and does not restate the deliverable.**

The north star — what motivates the design, and explicitly *not* the next milestone — is a
working **translator MVP**: given a press-conference clip from an *unseen* interpreter,
output approximate Spanish sentences.

```
Signing input → CTC → "gobierno medidas salud personas semana"
              → decoder → "El gobierno tomará medidas de salud para las personas esta semana"
```

Approximate, not literary. Gaps where signs fall outside the vocabulary. The target is a
followable gist on cross-signer data — **not** an isolated 50-word bank.

**Scope — of the north star, not of the current deliverable:**
- Vocabulary: 150–200 content words
- Signers: 8–12 distinct interpreters
- Model: CTC sequence output + mT5/mBART Spanish decoder (gloss-free)
- Training budget: tens of dollars (landmarks are cheap to train on)
- Evaluation: **leave-one-signer-out** — the honest cross-signer metric

---

## 2. Why this might be novel / useful

Purpose-built sign datasets (e.g. MX-ITESO-100, MSL-150) have a human cost that scales
**linearly** with vocabulary: every new sign must be recorded with paid signers in a studio.
That does not scale.

The contribution here is a **scalable, automatic data pipeline** with a *fixed* human cost.
Adding vocabulary = feeding more already-existing broadcast video; the marginal cost is
roughly compute. This mirrors the weak-labeling literature's own thesis (Oxford VGG /
BSL-1K: hand annotation doesn't scale → auto-annotate using mouthing cues + transcripts).

**The data source is the key insight.** Mexican government press conferences (the daily
federal *mañanera*, COVID-era health briefings, and conferences from the 23 MORENA-governed
states) are broadcast with a burned-in sign-language interpreter **and** an official
time-unaligned Spanish transcript (*versión estenográfica*). That's a large, growing,
multi-interpreter, weakly-paired corpus that nobody has systematically mined for LSM.

**Defensible claim (we try not to overclaim):** this cheaply covers the *high-frequency*
vocabulary that dominates real usage. The rare long tail (Zipf) and true any-signer
generalization still need much more data and signer diversity. The honest headline is
"generalizes to held-out *interpreters in this pool*," not "any signer on Earth."

---

## 3. Data sources

| Source | Role | Notes |
|--------|------|-------|
| Federal *mañanera* (Sheinbaum, 2024–) | Vocabulary + signers | Most colloquial/transferable Q&A vocab |
| Federal *mañanera* (AMLO, 2019–2024) | Signing dynamics | Monologue; weaker vocab transfer; batched interpreting |
| COVID-era health briefings | Health vocabulary | salud, enfermedad, tratamiento — transfers well |
| MORENA state conferences (23 states) | **Signer diversity** | 40–60+ potential distinct interpreters — the scaling story |

A single conference contains **multiple interpreters** who rotate in shifts — free signer
diversity inside one video, but it forces **per-clip** signer tagging (not per-video) so the
leave-one-signer-out evaluation isn't corrupted.

---

## 4. Methodology — the pipeline

```
                       ┌─────────────────────────────────────────────────┐
 YouTube broadcast ──► │ 1. download (yt-dlp)                             │
                       └─────────────────────────────────────────────────┘
                                          │ video.mp4
                       ┌─────────────────────────────────────────────────┐
 gob.mx estenográfica ─►│ 2. fetch transcript (Playwright, bot-challenge) │
                       └─────────────────────────────────────────────────┘
                                          │ transcript.txt (speaker-tagged, no timestamps)
                       ┌─────────────────────────────────────────────────┐
                       │ 3. extract_continuous.py                        │
                       │    auto-detect interpreter PiP (pose-anchored)   │
                       │    MediaPipe Tasks → 124-landmark rows           │
                       │    stream to on-disk memmap, segment by hands    │
                       └─────────────────────────────────────────────────┘
                                          │ per-segment .npz [T,124,3] + segments.json
                       ┌─────────────────────────────────────────────────┐
                       │ 4. align_audio.py (faster-whisper word times)    │
                       │    align_transcript.py (estenográfica ↔ ASR)     │
                       └─────────────────────────────────────────────────┘
                                          │ word stream with video timestamps
                       ┌─────────────────────────────────────────────────┐
                       │ 5. estimate_lag.py (interpreter lags speech)     │
                       └─────────────────────────────────────────────────┘
                                          │ per-video lag estimate
                       ┌─────────────────────────────────────────────────┐
                       │ 6. make_review_clips.py → per-word clips         │
                       │    build_review_form.py → browser review UI      │
                       │    collect_gold.py → human-verified gold labels  │
                       └─────────────────────────────────────────────────┘
                                          │ gold_labels.json (clip → word, per-signer)
                       ┌─────────────────────────────────────────────────┐
                       │ 7. phase3/dataset.py → train (CTC + decoder)     │
                       └─────────────────────────────────────────────────┘
```

### 4.1 Landmark schema (124 rows)

We use the **MediaPipe Tasks API** (HandLandmarker + PoseLandmarker + FaceLandmarker in
VIDEO mode), not the legacy Holistic. Each frame is a `[124, 3]` array (x, y, z):

| Rows | Content |
|------|---------|
| 0–20 | Left hand (21 pts) |
| 21–41 | Right hand (21 pts) |
| 42–74 | Pose (33 pts) |
| 75–123 | Face (49 pts): eyebrows 10 + eyes 12 + iris 2 + **dense mouth 20** + head anchors 5 |

The **dense 20-point mouth** is deliberate: interpreters mouth Spanish words continuously,
and mouthing is a **signer-independent** cue (the basis of BSL-1K's auto-annotation). It is
*stored always* but *never required* for recognition — at training we randomly mask the mouth
stream (SHuBERT-style) so a sign stays recognizable from hands+pose alone.

### 4.2 The alignment problem (the crux)

The transcript has **no timestamps**, and the interpreter signs with a **variable lag** behind
the speaker (we observe batched/consecutive interpreting: the interpreter waits for a clause,
then signs it as a burst — lag ≈ 6–7s for AMLO, less for faster speakers).

The bridge is **audio**: ASR (faster-whisper) gives word-level timestamps in video time; the
estenográfica is aligned to the ASR stream via `difflib.SequenceMatcher` (~94% match); a
per-video lag estimate then anchors a **generous, clause-scaled window** in the landmark
stream where the sign should appear. Mouthing localizes *within* that window; CTC learns the
exact alignment at training time. We never need exact per-word placement.

### 4.3 Weak labels vs gold labels

- **Weak labels** (transcript + lag + window) are the *primary* training signal — scalable,
  noisy, generated automatically.
- **Gold labels** (human-verified clips) are for *evaluation* and *seeds* — clean, small.

Human review is the real bottleneck (~60–80 min/video). The review UI shows per-word clips
back-to-back with surrounding sentence context, a per-clip signer tag, and `s`/`e` keys to
mark the exact sign moment. Verdicts: `y` (sign present), `n` (implicit — signer conveyed it
without a discrete sign), `neg` (negated form, a distinct sign — LSM incorporates negation, so
"no tener" ≠ "no" + "tener").

### 4.4 Why both gold AND weak labels?

Pure transcript+landmark training with no anchoring doesn't converge cross-signer. The weak
labels provide coarse windows that CTC turns into alignment; the gold labels provide a clean
held-out eval and within-signer reference seeds that bootstrap the noisy labels.

---

## 5. Example code

### 5.1 Landmark normalization (shoulder-relative, schema-agnostic)

The same `featurize()` is used by the DTW spotter *and* the training dataloader, so
train/inference normalization is identical. MediaPipe normalizes coordinates to whatever crop
it's fed, and crops differ across videos and after interpreter relocation — so raw coordinates
are **not** comparable. We normalize every clip into a body frame (shoulder midpoint / width):

```python
# experiments/spotter/classify.py
HAND_ROWS = slice(0, 42)            # 0-20 left hand, 21-41 right hand
L_SHOULDER, R_SHOULDER = 53, 54     # pose shoulders → array rows
POSE_ARM_ROWS = [42 + i for i in (11, 12, 13, 14, 15, 16)]  # shoulders, elbows, wrists

def featurize(arr, cfg=None):
    """[frames,124,3] -> [T, D]: keep hand-present frames, normalize into the body frame."""
    rows = list(range(0, 42)) + POSE_ARM_ROWS
    hands = arr[:, HAND_ROWS, :2]
    has_hand = ~np.isnan(hands).all(axis=(1, 2))
    kept = np.where(has_hand)[0]
    pts = arr[np.ix_(kept, rows)][:, :, :2].astype(np.float64)   # [T,P,2]

    mid, width = _clip_shoulder_stats(arr)        # body center + scale
    pts = (pts - mid) / width                      # crop-size invariant
    return pts.reshape(len(kept), -1)              # [T, P*2]
```

### 5.2 Streaming, crash-safe extraction (the gate is the hand detector)

Extraction runs MediaPipe on the auto-detected interpreter crop. The per-frame hand detector
doubles as a presence gate (pose+face run only when hands are present), and rows stream
straight to an on-disk memmap so a crash/unmount loses only the tail:

```python
# phase2/extract_continuous.py  (core loop, abridged)
arr = np.lib.format.open_memmap(full_path, mode="w+", dtype=np.float32,
                                shape=(n_alloc, 124, 3))      # O(1) RAM, crash-safe
presence = np.zeros(n_alloc, dtype=bool)

hr = hand_det.detect_for_video(img, ts)
if hr.hand_landmarks:                          # interpreter present
    pr = pose_det.detect_for_video(img, ts)
    fr = face_det.detect_for_video(img, ts)
    arr[processed] = frame_to_row(hr, pr, fr)  # pack into the 124-row schema
    presence[processed] = True
else:                                          # absent: maybe the PiP moved
    absence += 1
    if absence >= relocate_after:              # re-detect, corner-constrained
        active_crop = _interpreter_bbox_full(cap, det_hand, det_pose, fps, ...)
```

### 5.3 Whole-video ASR for word timestamps

```python
# phase2/align_audio.py
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(str(video), language="es", word_timestamps=True)
words = [{"start": w.start, "end": w.end, "word": w.word.strip()}
         for seg in segments for w in (seg.words or [])]
```

### 5.4 Model-agnostic training dataset

`gold_labels.json` stores each sign in *video time*; the dataset maps that window into the
right extraction segment, slices the frames, and reuses `featurize()`. Output is padded
batches `[B, Tmax, D]` + lengths — consumable by a classifier now or a CTC head later:

```python
# phase3/dataset.py
def _locate(meta, proc_fps, start_sec, end_sec):
    """Video-time window -> (segment file, frame_lo, frame_hi), or None if it fell in a gap."""
    mid = 0.5 * (start_sec + end_sec)
    for s in meta["segments"]:
        if s["start_sec"] <= mid <= s["end_sec"]:
            lo = round(start_sec * proc_fps) - s["start_frame"]
            hi = round(end_sec   * proc_fps) - s["start_frame"]
            return s["file"], max(0, lo), min(hi, s["n_frames"])
    return None

class GoldClipDataset(Dataset):
    def __getitem__(self, i):
        _, npz_path, lo, hi, word_idx, signer = self.index[i]
        arr = np.load(npz_path)["landmarks"][lo:hi]        # [T,124,3]
        if self.augment and random.random() < self.mirror_p:
            arr = mirror_array(arr)                         # handedness aug
        return torch.from_numpy(featurize(arr, self.cfg)).float(), word_idx, signer
```

---

## 6. Planned model (Phase 3)

Based on a review of 2024–2026 CSLR/SLT SOTA (gloss-free is winning; CTC-alone fails on
weak/limited data; decoder-only transformers do well at small scale):

```
124-pt landmark sequences (hands + pose + face)
   │  [optional] SHuBERT-style multi-stream masked pre-training
   ▼
1D CNN temporal compression (4–8×)
   ▼
Transformer encoder
   ├─ Auxiliary loss: CTC over the word vocabulary (weight ~0.2)
   └─ Primary loss:  cross-entropy on Spanish tokens via mT5/mBART decoder (~0.8)
   ▼
Spanish text (gloss-free)
```

CTC is used as an *alignment tool / auxiliary loss*, not the final classifier. The first
milestone is a Stage-A proof-of-concept: ~50 high-frequency signs recognized **cross-signer**,
leave-one-signer-out, on a budget of tens of dollars (landmarks are tiny to train on; the real
cost is CPU extraction time and human review hours).

---

## 7. Current status (2026-06-22)

**Working & validated:**
- End-to-end pipeline validated on a COVID briefing: extract → ASR (8,556 words) →
  transcript align (94%) → lag estimate (+5.77s) → 503 review clips → review UI.
- Eyeball validation of the timing→window bridge on AMLO data (signs land in-window with
  visible mouthing).
- `phase3/dataset.py` built + self-tested (time→frame mapping, normalization, padded
  batches, leave-one-signer-out splits).
- **MediaPipe skeleton quality confirmed on corner-box footage** (Sinaloa Gobierno del Estado
  press conferences): finger-level tracking is clean on upscaled PiP, hand detection 90.7%.
  This is the format planned for held-out test videos.
- **Cross-signer DTW baseline confirmed at ~1%** on the Phase-1 word bank. Root cause: 963
  clips, 886 unique labels, 819 with only a single example — the model never sees the same
  sign from two different people. Direct template matching cannot generalize. This closes the
  DTW path and confirms the training approach is necessary, not optional.

**Revised near-term plan (anchor-first):**
The original "mine continuous footage → train CTC" sequence has been reordered. Direct
cross-signer matching failed, so we first need a trained recognizer before mining is useful.
The anchor set is the gate:

1. **100-word anchor set** — 100 high-frequency signs × ≥50 examples × 6–8 signers, pulled
   from the continuous footage using the audio as a candidate-window generator, confirmed
   by eye. This is the first real training data (not a reference dictionary).
2. **Cross-signer recognizer** — trained on the anchor set, evaluated on held-out Sinaloa
   signers (leave-one-source-out). This is the viability milestone.
3. **Bootstrap mining** — the trained recognizer (not the broken DTW matcher) becomes the
   auto-confirmation step for mining more examples from the 20h corpus.

**Continuous footage assembled so far:**
- 14 videos, ~20h signing time (15h confirmed + 2 in download), across 8 identified training
  signers: 2 AMLO-era mañanera, 2 Sheinbaum-era, 2 COVID health briefings, 2 Sonora.
- **Held out (never to be trained on):** 2 Sinaloa Gobierno del Estado signers — a
  geographically and institutionally distinct source, stronger test than same-pool held-out.

**Not started:**
- Word frequency analysis across transcripts → 100-word selection
- Anchor set clip collection (semi-manual, audio-guided)
- `phase3/model.py`, `train.py`, `evaluate.py` (deliberately deferred until the anchor
  recognizer is built and real gold exists)

---

## 8. Questions I'd like opinions on

1. **Lag as a soft prior vs hard alignment.** We treat lag as a generous, clause-scaled
   window and let CTC learn placement. Is that the right call vs investing in tighter
   per-word alignment (e.g. forced alignment on the mouthing stream)?
2. **Concept-not-word.** LSM signs are concepts, not 1:1 Spanish words (e.g. *dinero* →
   RECURSOS). We plan to group synonyms to one concept label and rely on the decoder to pick
   the surface word. Is gloss-free SLT genuinely the right end goal here, or should we keep an
   explicit gloss layer?
3. **Cross-signer ceiling.** DTW on the Phase-1 word bank gave ~1% cross-signer accuracy —
   confirmed the template-matching baseline is too weak to use as a mining filter. The planned
   recognizer (trained on 100-word anchor set, 50 examples × 6–8 signers) should clear this.
   Honest generalization claim will be "interpreters in this pool" until more state-conference
   data is added. Initial test: held-out Sinaloa signers (different state, different institution).
4. **Pre-training.** Worth the effort to pre-train (SHuBERT-style) on the 490-sign Phase-1
   isolated set + Mendeley (249 signs, 11 signers) before fine-tuning on continuous data, or
   diminishing returns at this scale?
5. **Weak-label cleanliness.** Is ~94% transcript-alignment + a 6–7s lag window clean enough
   to train CTC, or will the noise floor dominate before vocabulary coverage does?
```

---

## 9. Status update (2026-08-01) — supersedes §7 where they conflict

**Corpus.** 16 videos, ~21.1h raw interpreter footage, all extracted to the 124-landmark
schema (face landmarks present 88–99.8% — an earlier "mostly NaN" claim was stale).
Signer-presence regated on **pose, not hands** after a B-roll bug (hands anywhere in the
inset opened spurious windows): 20.79h validated signing across 236 segments. ASR
transcripts: 16/16. Full-corpus arrays live on the external drive (mirror of the test
slice is local at `phase3/local_drive_mirror/`).

**The anchor-first plan in §7 is superseded.** The current sequence (see
`PIPELINE_MAP.md` and `phase3/NEXT_STEPS_SPEC.md`):

1. **Stage-1 sentence alignment** (SEA segmenter + DP + lag model): best config Run B =
   73.1% binary / 3.46s median error on the 26-clip verified instrument. Next: D1
   per-block lag; grow the verified eval set to 60–80.
2. **Stage-2 word localization** inside aligned spans: LSM ordering prior (Workstream E,
   gloss corpus measured: 42% of pairs reorder, time-marker fronting 87%), mouthing
   (Workstream C: presence 16/16; **shape-DTW clip-to-clip discrimination confirmed
   above chance p≈0.0006 in the 2026-08-01 audit** — AV-HuBERT retest now aims to beat
   that baseline, not merely beat chance), and eventually a visual scorer.
3. **First training event: fine-tune SignCLIP on our own mined pairs.** Measured
   2026-07-31: off-the-shelf SignCLIP is unusable as a scorer on interpreter footage
   (flat at chance; domain gap vs dictionary clips p=0.038) though it retains weak
   signal on citation-form clips — and LSM is very likely in its Spreadthesign
   pretraining (es.mx edition exists), so this is a register/domain gap, not language
   novelty. Fine-tuning is a prerequisite, not an option.
4. **Candidate generation is open-vocabulary** (settled 2026-08-01): the LLM proposes
   candidate signs freely, ranked; the 886 inventory is the *verification seed and eval
   anchor*, not a ceiling. Precedents: BOBSL dense annotation (arXiv 2208.02802,
   synonym expansion + novel-class pseudo-labeling), SignAgent (arXiv 2603.19059),
   pseudo-gloss LLM generation. Contrastive training can acquire out-of-inventory signs
   from recurrence once the encoder is fine-tuned in-domain — open the vocabulary
   *after* the encoder works, not before.

**Paths closed by measurement** (do not reopen without new evidence): cross-signer DTW
template matching (~1%); duration rescale / grammar-duration prior (ratio 0.966, no
headroom); aperture-only, order-position, and viseme-table mouthing word-ID; padding as
the explanation for SignCLIP's weak control; off-the-shelf SignCLIP as ranking stage.

**End products, in order:** (1) the corpus + alignment pipeline + baselines paper;
(2) fine-tuned in-domain sign encoder enabling scaled mining; (3) the translator MVP
(§6 architecture) trained on the mined corpus.
