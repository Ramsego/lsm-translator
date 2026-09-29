# SEA feasibility test on LSM mañanera footage

**Question:** does SEA's subtitle↔signing alignment work on our data, using the base
(un-fine-tuned) SignCLIP — or without SignCLIP at all?

## Why this test, and why now

We measured SignCLIP on 1-of-30 dictionary sign retrieval and got 11% top-1 cross-signer
(chance 3.3%), and concluded the embedder was too weak to build on. That conclusion was
drawn from the wrong experiment. Alignment is a substantially easier task than isolated
sign recognition:

- it is monotonic (signs arrive in transcript order),
- the search is time-constrained (we have ASR word times + a measured ~6s interpreter lag),
- dynamic programming absorbs local errors into a global solution.

**SEA's own published numbers make the point:** on BOBSL, alignment *without* SignCLIP
embeddings scores F1@0.25 = 79.3 / frame-accuracy 80.7. *With* embeddings: 82.9 / 82.5.
SignCLIP contributes ~3 points; segmentation + DP alignment do the rest. So a weak
embedder is far less damaging here than in retrieval, and there is a
`--similarity_measure none` mode that skips SignCLIP entirely.

That no-embedding mode is what we test first.

## Setup

- **Slice:** `57TvyH9902U`, 1200–3000s (30 min), the interpreter region cropped to
  `crop=360:412:919:308` and upscaled to 900x1030.
  The crop is the union of the 43 stable crop boxes in `segments.json`; the 9 outlier
  boxes all occur after 7100s (broadcast sign-off layout change) and are excluded.
- **Poses:** `videos_to_poses --format mediapipe` with SEA's recommended
  `model_complexity=2, smooth_landmarks=false, refine_face_landmarks=true`.
- **Subtitles:** `make_vtt.py` groups ASR words into 233 sentence cues, in slice-local
  time. Times are the SPEAKER's audio times, deliberately **not** lag-corrected — undoing
  that lag is precisely what SEA is supposed to do, and is what we measure.
- **Env:** `sea` conda env (CPU-only subset of SEA's `environment.yml`; the GPU-only
  Embed stage is skipped).

## How we judge the result without reading LSM

`check_alignment.py` applies two independent checks:

1. **Shift vs. measured lag.** We know from `phase2/estimate_lag.py` that this video's
   interpreter lags the speaker by **6.33s (spread 3.48s)**, measured empirically from
   motion onsets. If SEA is genuinely locating the signing, the shift it applies to each
   subtitle should cluster near that value. If it shifts nothing, it is a no-op. If the
   shifts scatter, it is noise.
2. **Verified-clip coverage.** For clips where a human confirmed word W is signed at time
   T (`phase3/probe/review.csv`), check whether the re-timed subtitle containing W now
   covers T. Reuses existing labels; needs no LSM knowledge.

## Interpreting the outcome

- **Working** → the user's proposed ordering is right: run SEA over the 21h to harvest
  (span ↔ sentence) pairs, fine-tune SignCLIP on those pairs, re-run SEA with the better
  embedder, repeat. That loop is the self-improvement cycle, and it produces exactly the
  training data the translation model needs.
### Downstream: how signs get localized INSIDE an aligned span

Sentence alignment does not identify signs — it shrinks the search. Inside a ~3s aligned
span, three ordered signals combine (user's design, 2026-07-29):

1. **Segmenter** → N candidate sign boundaries (validated: it runs on LSM poses and
   produced 4 sign-sized segments in a 2.1s clip, without knowing any LSM).
2. **Grammar prior** → the M content words of the sentence, reordered into expected LSM
   order using the Spanish→LSM gloss corpus (3,000 pairs, Scientific Data 2025). NOT for
   translation — only for the *ordering/deletion rules* (time markers front, articles
   dropped). Structural rules survive the corpus's tiny 820-token vocabulary in a way
   lexical translation would not.
3. **Monotonic DP** aligns the N segments to the M expected glosses, since signing follows
   sentence order. Every segment gets a candidate label *before any recognition happens*.

SignCLIP then only has to validate/re-rank those candidates — a far easier job than
identifying a sign cold, and one our measured 44% top-5 is plausibly adequate for.
**Mouthing** enters here too, as a temporal anchor confirming which segment carries which
word.

Cheapest first step for (2): extract explicit rules from the 3,000 pairs by diffing each
Spanish sentence against its gloss (article deletion rate, adverbial position, adjective
order) rather than training a seq2seq model — more robust to unseen vocabulary and
inspectable. Known risk: the corpus encodes *prescriptive* LSM from grammar books, while
live interpreters compress and improvise, so measure the gap once verified spans exist.

- **No-op / broken** → determine whether the failure is the BSL-tuned segmenter (does it
  find plausible sign boundaries on LSM?) or the embeddings, and fall back to the
  multi-cue spotter (transcript timing + SignCLIP ranking + mouthing anchor) to bootstrap
  before retrying alignment.

## Files

| File | Purpose |
|---|---|
| `make_vtt.py` | ASR word timings → sentence-level `.vtt` in slice-local time |
| `check_alignment.py` | the two verdict checks above |
| `video/` | cropped interpreter slice + extracted `.pose` |
| `subtitles/` | generated `.vtt` |
| `out/` | SEA segmentation + alignment output |
