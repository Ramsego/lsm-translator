# Mouthing as a localization anchor for LSM — final report

Consolidates the whole mouthing investigation (supersedes `MOUTHING_FINDINGS.md`,
`ROUTE1_RESULTS.md`, `ROUTE1_FOLLOWUP.md` for decision purposes; those retain detail).
Corpus: 21h mañanera broadcast, interpreter inset. Test data: 26 human-verified sign
instances in one video, 16 of which fall inside SEA Run-B aligned spans. No new manual
labeling was used anywhere in this work.

## TLDR (read this first — the conclusion reversed late)

> **⚠️ AUDIT CORRECTION 2026-08-01 — one "at chance" claim below is wrong.** A re-run
> with a proper Monte Carlo null (see final section) shows **cross-instance shape DTW is
> significantly above chance: p≈0.0006** (p≈0.0025 with `decidir` excluded). "Failed at
> chance across four mechanisms" should read "three of four": aperture-only,
> order-position, and both viseme tables are genuine nulls; shape DTW is not.

Presence detection works (16/16) and mouthing-to-sign timing is tight (mean +0.12s).
Word *identification* failed at chance across four mechanisms — but a late-found paper
(SignMouth, arXiv 2509.10266) shows a frozen AV-HuBERT extracts real word-level signal
from mouth crops at **essentially our resolution** (PHOENIX14T ≈ 15–17 px mouth vs. our
14.5–16.9 px). So the failures are attributable to our hand-built feature extractor, not
to the data. **The recommendation therefore flipped from "drop mouthing" to "drop the
hand-built features and re-run in AV-HuBERT embedding space."** Sections below are in
chronological order; the "What fails" section should be read with this caveat.

## The question

SEA gives us (sentence ↔ span) pairs at ~62% accuracy with spans averaging 15.3s
(range 3.6–31.6s). Can interpreter mouthing narrow a span to a specific sign, i.e.
serve as a temporal anchor for harvesting (clip ↔ word) training pairs?

## What works: presence detection

Burst detector on mouth aperture (MediaPipe Holistic landmarks 13/14, inter-eye
normalized, robust z-score on frame-differences, occlusion-masked via hand-to-mouth
distance). **Detected a real word-like burst in 16/16 aligned spans.** Bursts
corresponding to verified signs are separable from other bursts in the same clip:
duration median 0.80s vs 0.33s (p=0.0009), aperture direction-changes median 10 vs 5
(p=0.0043). Mouth width adds nothing here (p=0.74 magnitude; its extrema signal is
redundant with aperture's).

**Mouthing-to-sign timing is tight.** Measured within-clip, no matching involved:
mean +0.12s, median −0.03s, stdev 1.09s, max 2.43s. So the interpreter mouths
essentially concurrent with the sign — the user's direct visual impression was correct,
and this is *not* where the difficulty lies.

## What fails: identifying which word

Four independent mechanisms, all evaluated against per-sentence or per-candidate-set
chance baselines:

| mechanism | result |
|---|---|
| cross-instance shape DTW (80-dim lip contour) | median rank 3/25, top-1 12.5%, top-3 56.2% — real but weak signal |
| aperture-only ablation | **not significant under corrected null** (p=0.34) |
| order-position assignment in sentence | 3/16 = 19% correct |
| hand-built viseme table | median rank 9.5 vs 9.8 chance, p=0.67, **top-1 0%** |
| G2P viseme table (epitran, real Spanish IPA) | median rank 9.0 vs 9.8 chance, p=0.68, **top-1 0%** |

Upgrading from a hand-built letter table to validated Spanish G2P changed nothing,
which rules out "the phonetic model was too crude" as the explanation.

Template distinguishability was checked independently of video: across 2,061 pairs of
different candidate words, expected-template correlation averaged 0.12 with only 5.4%
near-duplicates. **The templates are discriminable in principle** — the failure is that
real observed traces don't match their own word's template better than a random one's.

**Localization fails too.** Fine localization (Test 3): median absolute error 5.02s,
does not beat "guess the clip midpoint" (p=0.69). Coarse position in full spans
(Test 4): rho +0.191, n=14, n.s.

**A prediction that failed.** Fable predicted localization error would be small when
discrimination succeeded and catastrophic when it didn't. Tested per-pair: targets
ranked top-3 had **median error 8.18s**, targets ranked worse had **2.58s** —
opposite direction, Mann-Whitney p=0.97. One case ranked #1 and localized 11.9s off.
Implication: DTW match cost does not distinguish "found the true occurrence" from
"found a coincidentally cheap alignment elsewhere," so ranking well does not imply the
match is semantically meaningful.

## Resolution measurements (corrected — an earlier overclaim)

Measured precisely, un-upscaled to native source:

| | mouth width |
|---|---|
| our interpreter, 720p source (what we have) | **14.5 px** (1.14% of frame width; head 46.7 px) |
| our interpreter, 1080p source (measured, not extrapolated) | **16.9 px** (head 64 px) |
| BBC/BOBSL signer (eyeball estimate from a screenshot, NOT measured) | ~1.7× larger |

I initially claimed a "96 px requirement" and concluded resolution was the decisive
blocker. **That was wrong.** 96 px is the training crop format of a Spanish VSR
checkpoint, not an empirical requirement, and BSL-1K reports no mouth pixel size at all —
it says only that the signer occupies a fixed screen region cropped from footage,
structurally the same as ours. The real deficit vs. BOBSL is ~1.7–2×, a headwind, not a
wall. 1080p is available (we capped downloads at 720p by choice) and recovers ~1.5×.

## The decisive finding: BSL-1K already did this, and the economics don't scale down

[BSL-1K, Albanie/Varol/Momeni/Afouras/Chung/Fox/Zisserman, ECCV 2020, arXiv 2007.12131]
used weakly-aligned broadcast subtitles + a visual keyword spotter on interpreter
mouthing to auto-localize 1,000 signs across 1,000 hours of BSL. This is our exact
proposed pipeline. Gül Varol co-authors both BSL-1K and SEA.

**Two facts from that paper drive the decision:**

1. **Yield.** From **1,060 hours**, the spotter produced **5,826 predictions** at high
   confidence (>0.9) ≈ 5.5/hour. Their human-verified test set was **2,103 annotations
   covering 334 signs** of a 1,064-word vocabulary. Scaled to our 21 hours: **~115
   high-confidence instances.** We hand-verified 26 in an afternoon.

2. **Lip reading does not work for this.** They report that a state-of-the-art lip
   reading model achieved **zero recall** on 500 randomly sampled sentences of signer
   mouthings, and state that existing lip reading approaches cannot be used directly.
   This kills the cheap path (run Spanish VSR on bursts, edit-distance the output).
   Only a purpose-built keyword spotter worked — and no Spanish/LSM one exists.

They also confirm signers mouth inconsistently and often partially ("fsh" for "finish").

## Paths considered for making it work, and why each is blocked

- **Fine-tune KWS-Net (English, MIT, pretrained) to Spanish.** Needs Spanish AV data
  with word-level alignments; KWS-Net also consumes features from a separate pretrained
  lipreading front-end. A project, not an experiment.
- **Train a Spanish spotter on the speaker's own face in the same footage** (free
  in-domain supervision from ASR). **Not viable for this layout** — sampled 1080p frames
  show the main feed is a wide room shot with a distant podium, not a broadcast
  close-up. Speaker faces would be small or absent. (Single-frame observation; worth a
  systematic check before fully discarding.)
- **More footage.** Mañaneras are produced daily; the corpus is expandable. This is the
  only variable that actually changes the yield math, and it is a data-collection
  problem, not a research one. Cost: prior full-corpus pose extraction took ~60 hours.
- **Mouthing as a weak term in SEA's similarity-matrix slot rather than a hard anchor.**
  The DP already accepts an arbitrary (cues × segments) similarity matrix. Bar for a
  useful soft term is lower than for an anchor. (Earlier objection — "our best content
  signal is at chance" — **no longer holds**; see the SignMouth section: our features
  were the problem, not the signal.)
- **Bursts as a segmentation cue rather than an identification cue.** Untested. Nearly
  free with existing data: do detected bursts correlate with sign boundaries?

## Bugs found and fixed along the way (relevant to trusting earlier numbers)

1. `check_alignment.py` read clip timestamps from the wrong CSV column, silently
   skipping every row (earlier session) — invalidated an earlier "0/3" result.
2. `make_review_clips.py:get_context()` caps its sentence walk at ±15 words, silently
   truncating. Made word-position features meaningless for 9/16 cases. Fixed by taking
   sentence text from SEA Run-B cue text instead.
3. Span clips were cut with only 0.2s pre-padding, truncating the real burst's rising
   edge to 5 frames — one below `MIN_BURST=6`, so it was discarded. Re-cutting with 1.5s
   padding recovered it and took presence detection from 0% to 16/16. This alone flipped
   the order-position pipeline from 0/10 to 3/16.
4. Exact-string word matching missed inflected forms (`actividad` vs `actividades`),
   silently skipping 7/16 cases.
5. Key-format mismatch causing silent skips (n=0 result).

**My own analytical errors, for calibration:** (a) I claimed amplitude compression
explained the viseme failure, but the scoring used Pearson correlation, which is
scale-invariant — the claim was incoherent. (b) I asserted the 96 px requirement and
built a "resolution is the wall" conclusion on it without checking what BSL-1K actually
had. (c) I never measured mouth pixel resolution before running four experiments that
depended on it.

## ⚠️ Late addition (SignMouth, arXiv 2509.10266) — refutes one of my conclusions

Wu, Yuan, Li, Wang, Fu — "SignMouth: Leveraging Mouthing Cues for Sign Language
Translation by Multimodal Contrastive Fusion" (Sept 2025). Uses mouthing as an auxiliary
**feature stream for translation**, not as an annotation anchor.

Method: MediaPipe/IBUG-FAN landmarks → mouth crop → **frozen AV-HuBERT** + 3D CNN,
dual-stream (full frame + mouth region) with gated fusion and contrastive alignment.
Critically: **learned end-to-end, no explicit lipreading or word-level mouthing
recognition required.**

Ablation on PHOENIX14T:

| configuration | BLEU-4 | ROUGE |
|---|---|---|
| spatial (gesture) encoder only | 19.87 | 41.38 |
| **lip encoder only** | **12.73** | — |
| full fused model | 24.71 | 48.38 |
| prior SOTA (SpaMo) | 24.32 | 46.57 |

**Why this matters for us: PHOENIX14T is 210×260 full frame with the signer filling it,
so its mouth is ~15–17 px wide — essentially identical to ours (14.5 px @720p, 16.9 px
@1080p).** At that resolution, a frozen AV-HuBERT pipeline extracts enough mouthing
signal to reach BLEU-4 12.73 *on mouthing alone*.

**Therefore I retract the implication that the mouthing signal is not physically present
in our pixels.** It evidently is. Our four failures are attributable to the feature
extractor — hand-built templates and 2 scalar landmarks — not to the sensor. Extracting
this signal appears to require AV-HuBERT-class pretrained machinery.

**What still stands:** the BSL-1K yield economics for *annotation harvesting* at 21h
(~115 instances) are unaffected — SignMouth needs a fully-annotated dataset and does not
help create one. Note also they report naive fusion *degrades* performance without their
alignment module, and their gain over prior SOTA is only +0.39 BLEU-4.

**New path this opens — and it is Stage 2, not only Stage 5.** My first read of this
paper was that it relegates mouthing to a translation feature. That is too narrow, and
the user pushed back correctly. If a lip encoder alone reaches BLEU-4 12.73, those
features demonstrably carry **word identity** — a translation decoder cannot produce
correct content words in sequence from features that don't encode which word was mouthed.
AV-HuBERT features are also **frame-level**, so they support windowed matching directly.

The implication: **every Stage-2 experiment we ran is re-runnable in AV-HuBERT embedding
space instead of hand-built feature space**, without inventing anything new:

- **Test 2 (discrimination):** replace 80-dim landmark DTW / viseme templates with
  AV-HuBERT embeddings of the burst, scored against candidate words. Note this still
  needs a text→embedding bridge for candidates (the piece KWS-Net supplies via its
  P2G keyword encoder); embedding-space *clip-to-clip* matching needs no such bridge and
  is testable immediately with the 7 words that have ≥2 verified instances.
- **Test 3/4 (localization):** sliding-window AV-HuBERT similarity within a span.
- **SEA similarity-matrix slot:** the earlier objection was "our best content signal is
  at chance, and chance contributes nothing." If AV-HuBERT features are above chance,
  that objection dissolves and the slot becomes usable.

Deferred-to-Stage-5 framing therefore **withdrawn**. This is a live Stage-2 path.

## Current recommendation (revised after SignMouth)

**Do not drop mouthing. Drop the hand-built feature extractor.**

The earlier recommendation ("drop mouthing as a localization mechanism") rested on four
at-chance results that we now have good reason to attribute to our features rather than
to the data. Revised position:

1. **Next experiment: AV-HuBERT embedding-space clip-to-clip matching**, re-running Test 2
   on the 7 words with ≥2 verified instances. No text bridge, no Spanish spotter, no new
   annotation required — it is the same experiment with the feature extractor swapped.
   This is the decisive, cheap test the whole investigation has been missing.
2. If that clears chance, extend to sliding-window localization (Tests 3/4) and then to
   SEA's similarity-matrix slot.
3. Keep the burst detector, occlusion masking, and `segments.json` gating regardless —
   presence detection worked at 16/16 and gives the search windows.
4. **The yield question is genuinely reopened.** BSL-1K's 5.5/hr came from searching 1,000
   hours blind over a 1,000-word vocabulary. Our task — a ~15s span with 3–8 candidates,
   already narrowed by SEA — is far easier and should not inherit their precision/recall
   operating point. The 21h ≈ 115 instances extrapolation should be treated as a **lower
   bound under an inapplicable operating point**, not as the expected yield.

What remains true: mouthing alone will not be a hard anchor (SignMouth's own lip-only
BLEU-4 of 12.73 vs. 24.71 fused says it is a complementary cue), and interpreters mouth
partially and inconsistently.

## Questions for review

1. **Is the AV-HuBERT clip-to-clip test the right next move, and what should its
   pre-registered success criterion be?** n is small (7 words with ≥2 verified instances,
   ~14 same-word pairs) and this same setup produced a Type-I-flavoured trap earlier —
   the multi-correct-candidate null issue. What null should it be scored against?
2. **How much does BSL-1K's yield extrapolation actually transfer?** Their 5.5/hr came
   from blind search over 1,000 hours and a 1,000-word vocabulary at a >0.9 confidence
   threshold. Ours is a ~15s SEA-aligned span with 3–8 candidates. Is treating 115
   instances as a lower bound under an inapplicable operating point the right call, or
   is the binding constraint really just how often interpreters mouth detectably —
   in which case the operating point doesn't matter much?
3. **AV-HuBERT is English-pretrained (LRS3).** SignMouth applies it to German (PHOENIX14T)
   and ASL (How2Sign) as a *frozen* extractor and it works. Is there reason to expect
   Spanish mouthing to behave differently, or does frozen-extractor use largely sidestep
   the language question?
4. Note BSL-1K reports **zero recall** for a SOTA lip-reading model on signer mouthings,
   while SignMouth gets real signal from frozen AV-HuBERT features on comparable
   resolution. Are these consistent — i.e. is the distinction "don't decode words,
   just use the representations" — or is one of these results doing something we're
   misreading?
5. Is the untested burst-as-segmentation-cue idea worth the (small) effort, given
   presence detection is the one component that consistently worked?
6. For the corpus paper: what is the contribution now? The clean negative result got
   substantially weaker once SignMouth showed our features were the problem. Is it
   "presence detection + measured mouthing-to-sign timing (mean +0.12s, stdev 1.09s) for
   an LSM interpreter corpus", or does the paper need the AV-HuBERT result before there
   is anything worth reporting?
7. Corpus expansion (21h → 200h+, mañaneras are daily) remains a lever on yield. Given
   ~60h of extraction per pass for a solo unfunded researcher, does that belong ahead of
   or behind the AV-HuBERT test — and behind or ahead of the SEA/SignCLIP fine-tuning
   loop?

## Audit addendum (2026-08-01) — the shape-DTW null was never properly run, and it passes

An external audit of the grading found two defects in `verify_corrected_null.py`:

1. **The Monte Carlo null function (`mc_null_ranks`) is dead code** — defined but never
   called. The "corrected null" p-values in this report actually came from a Wilcoxon
   signed-rank test of observed ranks against per-template closed-form null *means*
   (26/(c+1)). For multi-instance words the null min-rank distribution is right-skewed
   (median < mean), making that comparison mildly anti-conservative.
2. **The corrected analysis was only reported for the aperture ablation** (p=0.34). The
   shape-DTW mechanism's corrected p was computed but never stated, leaving the table's
   "real but weak signal" hedge as the only record — while the TLDR claimed all four
   mechanisms were at chance.

Re-run 2026-08-01 with a proper Monte Carlo null (rank-sum statistic; per template, min
of c draws without replacement from 1..25; 50,000 trials):

| mechanism | observed rank-sum | null mean | MC p |
|---|---|---|---|
| shape DTW (80-dim lip contour) | 88 | 169.1 | **0.00056** |
| shape DTW, `decidir` excluded (n=12) | 77 | 143.1 | **0.0025** |
| aperture-only | 148 | 169.1 | 0.211 (null, confirmed) |

**Corrected conclusion:** within-signer clip-to-clip mouthing discrimination works even
with the hand-built lip-contour features. Genuine nulls: aperture-only, order-position
(3/16), both viseme tables (top-1 0%).

**Localization needs a more careful statement than "null" — corrected again 2026-08-01.**
Test 3 fails (median AE 5.02s, does not beat the midpoint baseline, p=0.69) and that
survived the windowing fix. But `ROUTE1_FOLLOWUP.md` traced *why*, by hand, on a specific
~7s error case: the algorithm locked onto a **genuinely real open-mouth articulation
elsewhere in the clip — the wrong word, not noise**. So Test 3's error metric conflates
two different quantities:

| quantity | measured value |
|---|---|
| real mouthing-to-sign timing (within-clip, no matching involved) | mean **+0.12s**, median −0.03s, sd 1.09s, max 2.43s |
| cross-clip search landing on the wrong burst | large, and dominates the 5.02s figure |

The first is excellent. The failure is in **search/discrimination, not timing** — which
means "mouthing cannot localize" is the wrong lesson. The right one is "cross-clip template
search over 25 unrelated words cannot pick the right burst." Those are different claims,
and the pipeline would never do the second: it would detect bursts inside an aligned span
and discriminate among *that sentence's* 3–8 content words. Measured candidate-set effect
at that scale is large and monotonic — top-1 12.5%→**47.7%** and top-3 56.2%→**85.8%**
going from k=25 to k=5 (chance 20%).

**Therefore the configuration that matters has never actually been tested.** It is not
Test 3. It is: burst detection (16/16) → discriminate against the real per-sentence
candidate set → take the sign to be at the burst (±1.09s). Building that test is blocked
only by needing verified instances of *competing* words from the same sentences — which
the 203-clip review batch (29 words × 13 videos) is the first data able to supply.

Caveats that keep this from being oversold: all clips are one interpreter in one video
(within-signer only); the 16 templates cluster on 7 words so they are not fully
independent trials; and clip-to-clip matching has no text bridge — it can say "these two
bursts are the same word," not "this burst is *dejar*."

**Implication for Workstream C:** the AV-HuBERT clip-to-clip test starts from an
already-positive baseline, not from zero. Its pre-registered criterion should therefore
be *beating the hand-built shape-DTW ranks on the same pairs* (paired comparison), not
merely clearing chance — clearing chance is now the floor, already achieved.

## Scope correction (2026-08-01): Test 3 measured a bar the pipeline never set

Raised by the user and confirmed against the original design. `PROJECT_OVERVIEW.md:145-146`
states the intended role explicitly:

> "Mouthing localizes *within* that window; CTC learns the exact alignment at training time.
> **We never need exact per-word placement.**"

Test 3 asked whether mouthing can pinpoint a sign to within ~1s by cross-clip template search.
That is a *stricter* requirement than the architecture was ever designed around, and failing it
does not indicate a failure of the intended function. The intended function is **narrowing**:
take an aligned span of ~13s with 3-8 candidate content words and reduce it to a short list
plus an approximate moment, good enough for (a) a human reviewer to confirm, or (b) a
multi-positive contrastive loss to consume directly. Measured against *that* bar:

| requirement | measured |
|---|---|
| is a mouthing burst present in the span? | 16/16 |
| is the burst at the sign? | mean +0.12s, sd 1.09s |
| does it shortlist the right word? | top-3 **85.8%** at k=5 (chance 20%) — random distractors, not real per-sentence competitors |

**So mouthing is closer to spec than the "localization failed" headline suggests.** The honest
residual doubt is the k=5 caveat (random wrong words, not the words that really compete in a
given sentence), not whether narrowing works in principle.

**What actually blocks deployment is not research, it is exemplars.** The working matcher is
shape-DTW, which is *clip-to-clip*: scoring a burst against a candidate word requires a
verified clip of that word. There is no text bridge (that is what AV-HuBERT or a keyword
spotter would supply). Coverage today is the 16 words in `review_2.csv`; review batch 2 takes
it to 29 across 13 videos and multiple signers. **Therefore reviewing batch 2 is not a detour
from the mouthing work — it is the input the mouthing work needs.**
