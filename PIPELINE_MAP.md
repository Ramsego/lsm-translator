# Pipeline map — what every piece is actually for

Written 2026-08-01 to answer one question: **why does each thing we've built or tested
exist, and what does it feed?** If a future experiment doesn't slot into this map, that's
a flag to stop and ask why we're running it.

## CANON — quote this block, do not paraphrase it

**This is the single authoritative statement of what the project is.** Every other doc in
this repo defers to it. If any file — including the rest of this one — disagrees with this
block, that file is stale and this block wins. Anyone (human or model) describing the
project externally should copy these lines verbatim rather than restate them.
Last verified: 2026-08-13.

- **Deliverable:** an LSM broadcast corpus + baselines + the pipeline that mines it,
  released as a resource/methods contribution. **Not a translator.**
- **North star:** an LSM→Spanish translator. It motivates the design. It is not a
  milestone and no work is currently pointed at it.
- **Source:** ~21 h of Mexican presidential press conferences (*mañaneras*), 16 videos,
  each with a burned-in interpreter inset plus an official time-unaligned Spanish
  transcript (*versión estenográfica*). Presence-gated to 236 segments / 20.79 h.
- **Central obstacle:** none of it is labeled. Every component built or tested is either
  part of the machine that manufactures (video span, text) pairs, or part of the
  instrument that measures whether that machine works. None of it is the translator.
- **Method lineage:** the BSL-1K / BOBSL weak-supervision recipe (Oxford VGG), executed
  for LSM, where nobody has done it.
- **Landmarks:** MediaPipe. Phase 2 continuous extraction uses a **124-row** schema
  (21+21 hand, 33 pose, 49 face incl. inner lip ring). Phase-1 arrays on disk remain the
  older **116-row** schema (41 face points). Face/mouth landmarks are present at 88–99.8%.
- **Stage 1 (sentence alignment) status:** SEA (Jiang et al., "Segment, Embed, and Align",
  arXiv 2512.08094) run in `--similarity_measure none` mode — segmentation + DP, no
  visual embeddings. Best config: **73.1% binary, 3.46 s median error, n=26**, one
  interpreter, one 30-minute slice. This supersedes the earlier "n=1, inconclusive" result.
- **Stage 2 (word localization) status:** no localizer has passed its gate.
- **Training:** none has run. Gated on Stage 1+2 emitting a first batch of pairs.
- **Phase-1 word bank:** 963 clips / 886 labels, mostly one example per label. It is a
  dictionary of candidate forms, **not** training data.
- **Community:** no deaf-community involvement until there is funding to pay for it.
  Claims are technical and resource-level only — no community-impact claim.

## The one paragraph that organizes everything

The deliverable is a **dataset + baselines + pipeline** — the first mineable LSM broadcast
corpus (mañanera interpreter insets), credible enough to attract researchers. A working
translator is the north star, not the near-term deliverable. The central obstacle is that
**none of the 21 hours is labeled**: we have continuous signing video and a Spanish
transcript, but nothing that says *this stretch of signing = this sentence/word*. A
translator (or a fine-tuned encoder) is trained on (video span, text) pairs — so until
pairs exist, there is nothing to train on. **Every single thing we have built or tested is
part of the machine that manufactures those pairs, or part of the instrument that measures
whether the machine works.** None of it is the translator itself.

This is the weak-supervision recipe from BSL-1K/BOBSL (Oxford VGG): hand-annotating sign
language doesn't scale, so auto-annotate interpreter TV footage using subtitles, lag
models, mouthing cues, and dictionary lookups — then train on the mined labels. We are
executing that recipe for LSM, where nobody has.

## The stages

```
Stage 0  RAW → CLEAN SIGNAL          (done, recently re-fixed)
         download → find interpreter inset → MediaPipe 124-landmark arrays
         → presence gate (pose-based revalidation: 236 segments, 20.79 h)

Stage 1  SENTENCE-LEVEL ALIGNMENT    (working, ~73% at n=26; the current front)
         "which ~10–15 s span of signing corresponds to which spoken sentence?"
         ASR word times + estenográfica + lag estimate + SEA segmenter + DP

Stage 2  WORD-LEVEL LOCALIZATION     (probes run; nothing passed decisively yet)
         "inside an aligned span, which segment is which word?"
         ordering prior (E) + mouthing (C) + visual scorer (SignCLIP) + sense cloud

Stage 3  TRAINING, ROUND 1           (not started — gated on Stage 1+2 output)
         fine-tune SignCLIP on mined (clip, word) pairs → in-domain scorer
         → feed it back into Stage 2 → better labels → re-mine (bootstrap loop)

Stage 4  TRAINING, ROUND 2 / END PRODUCT   (future)
         corpus + baselines released = the benchmark deliverable
         translator (CTC + mT5/mBART decoder) trained on the corpus = north star
```

Stages 1 and 2 are the "data engine." Stage 3 is where training begins — and its first
job is not translation, it's **making the data engine better** (a scorer that works on
our footage). That loop is standard practice (BSL-1K iterated exactly this way).

## Every component, its objective, and where it stands

| component | stage | the question it exists to answer | status |
|---|---|---|---|
| Phase-1 word bank (963 clips / 886 labels) | reference | dictionary of what signs look like — candidate side for any matcher; **not** training data (819 labels have a single example) | done |
| `extract_continuous.py` + presence gate | 0 | clean landmark arrays; is the interpreter even on screen? | done; B-roll bug found and fixed via pose-based `revalidate_segments.py` (549→236 segments) |
| ASR (`align_audio.py`) + estenográfica alignment | 1 | put the transcript's words on the video clock | done, 16/16 videos transcribed |
| `estimate_lag.py` | 1 | how far behind speech does the interpreter sign? (+6.33 s ± 3.5 on the test slice) | done; per-block version = D1, next up |
| SEA segmenter + DP alignment (Run A/B) | 1 | carve continuous signing into SIGN/SENTENCE units and align sentence cues to them | best config: 73.1% binary, 3.46 s median error, n=26 |
| `check_alignment_v2.py` + FORMS map | instrument | measure Stage 1 honestly (continuous metrics, no matcher bias) | done; this is the ruler, not the pipeline |
| D1 per-block lag | 1 | remove the ±3.5 s lag spread the DP currently has to discover unaided | ready, unblocked |
| Gloss corpus + ordering prior (Workstream E) | 2 | in what *order* do signs appear inside a span? (LSM ≠ Spanish order: 42% of pairs differ; time-markers front 87%) | measured, rules not yet validated on held-out |
| Mouthing (Workstream C) | 2 | independent, signer-independent anchor: the interpreter mouths the Spanish word while signing it | presence 16/16; **audit 2026-08-01: shape-DTW clip-to-clip discrimination IS above chance (MC p≈0.0006)** — aperture/order/viseme routes are the true nulls; AV-HuBERT retest now aims to *beat* the shape-DTW baseline |
| SignCLIP (off-the-shelf) | 2 | can a pretrained model score "does this clip show sign X?" | **not on interpreter footage** — flat at chance there, weak-but-real on citation-form; domain gap measured (p=0.038). LSM is very likely IN its pretraining (Spreadthesign runs an es.mx edition), so this is a register gap, not language novelty |
| Concept:sign cloud | 2 | word-sense disambiguation before matching (*dejar* in "déjame comer" = ALLOW, not LEAVE) — stops us scoring against wrong answer keys | deferred until a consumer (Stage-3 mining) exists; **when built: open-vocabulary** — the 886 is a verification seed, not a constraint |
| SignCLIP fine-tuning | 3 | turn the failed scorer into a working in-domain one, using our own mined pairs | **the next training event**; prerequisite now evidence-backed |
| CTC + mT5/mBART translator | 4 | the actual translator | deliberately not started |

## Paths tested and closed (these were not wasted — each killed a branch)

- **Cross-signer DTW template matching** — ~1% accuracy; template matching can't
  generalize across signers. Closed. This is *why* a trained recognizer is necessary.
- **Duration rescaling / grammar-duration prior (D2/D3)** — measured signing/speech
  ratio is 0.966; no headroom. Closed without building anything.
- **Hand-built mouthing features** — CORRECTED 2026-08-01: only three of the four are
  true nulls (aperture-only, order-position, viseme tables). Cross-instance shape DTW
  passes a proper Monte Carlo null (p≈0.0006, within-signer) — the earlier "all at
  chance" claim came from a null-model bug (dead-code Monte Carlo, unreported p). The
  AV-HuBERT retest's bar is now "beat shape-DTW," not "beat chance."
- **Mouthing localization — NOT closed** (this line previously said it was). Test 3 fails,
  but the failure was traced to the search picking a real-but-wrong burst, not to timing:
  measured mouthing-to-sign lead is +0.12s (sd 1.09s). And discrimination scales steeply
  with candidate-set size — top-3 56%→86% going from 25 candidates to a realistic 5. The
  pipeline configuration (burst detect → discriminate among that sentence's 3–8 content
  words → sign is at the burst) has never been tested. Blocked on verified instances of
  competing words from the same sentences, which review batch 2 can supply.
- **Padding as explanation for SignCLIP's weak control** — clean vs padded identical.
  Closed; saved a full re-trim of 144 clips.
- **Off-the-shelf SignCLIP as the ranking stage** — the headline negative result.
  The pipeline design survives; the pretrained scorer at the end of it doesn't.

## Why there has been no training yet

Training is not a step you unlock by wanting it; it consumes labeled pairs, and the pairs
are what don't exist yet. The measured state of the label factory: sentence-level
alignment puts the right sentence in a ~13 s span 73% of the time (n=26, one interpreter,
one slice), and no word-level localizer has passed its gate. Training a translator on
that today would burn money to learn noise — and worse, we'd have no instrument to tell
us whether a bad result meant bad model or bad labels. The probes exist precisely to
pin down the label noise *before* any loss function ever sees it. The first training run
(SignCLIP fine-tuning) becomes rational the moment Stage 1+2 can emit a first batch of
high-confidence pairs — that is what D1, E, and C are for.

## Extraction status — no re-extract needed (answered 2026-08-01)

The full 21h was extracted once, with the 124-landmark schema (21+21 hand + 33 pose +
49 face points, `phase2/extract_continuous.py`). The SEA segmenter was verified to use
**hands and pose only** (blanking the face changed 0/982 SIGN and 0/140 SENTENCE
boundaries), and those points are all present in the existing `.npz` arrays. Scaling SEA
to the corpus is therefore a **repackaging job (.npz → .pose format), minutes not
~20h of re-inference**, with two pending checks: coordinate normalization against
varying crop sizes, and re-indexing to the pose-revalidated segment boundaries.
Blocker: the full-corpus arrays live on the Crucial X8, which is still not mounted
(the "Untitled" volume visible on 2026-08-01 is a near-empty 52 GB partition, not it).

## Known weaknesses of the current position (kept here honestly)

1. **Everything is measured against one 26-clip instrument** — one interpreter, one
   30-minute slice, 16 word types. Many sequential hypotheses have been scored on it;
   borderline numbers (e.g. p=0.0888) may not survive a larger eval set.
   **In progress 2026-08-01:** review batch 2 generated at `<drive>/review/batch2` —
   203 clips, 29 words, 13 videos, presence-verified 203/203 by pose probe. Waiting on
   human review. See `phase2/WORD_SELECTION.md`. Two defects it fixes are worth knowing
   about independently: the old cutter never gated on interpreter presence at all, and
   its window started ~4s too late (measured: signs land a median 2.90s *earlier* than
   the lag model predicts, with 7/26 pinned at the clip's leading edge — a censored
   distribution, meaning **some existing "sign doesn't appear" verdicts are probably
   clips that started after the sign**, not genuine omissions).
2. ~~PROJECT_OVERVIEW.md still frames the deliverable as a translator MVP.~~
   **Resolved 2026-08-13.** This entry was stale the day it was written: PROJECT_OVERVIEW.md
   §1 was reframed on 2026-08-01, the same day as this file. All docs now defer to the
   CANON block above instead of restating the goal.
3. **The bootstrap loop is designed but unproven** — fine-tuning SignCLIP on weak pairs
   assumes the pairs are clean enough to improve it. That is exactly the question the
   current alignment work is quantifying, but it remains an assumption until Round 1 runs.
