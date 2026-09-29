# Implementation spec — five workstreams

Written for an implementing agent. Read this whole file before starting.

| ws | what | status |
|---|---|---|
| **A** | correctness fixes (phantom segments, sentence re-segmentation) | ✅ DONE — see `phase3/sea_test/WORKSTREAM_A_RESULTS.md` |
| **B** | sharpen the evaluation metric before any hypothesis testing | ✅ DONE — see `phase3/sea_test/WORKSTREAM_B_RESULTS.md` |
| **C** | AV-HuBERT mouthing test | ready, timeboxed, own eval, hard stop |
| **D** | per-block lag (D1 only — D2/D3 measured and closed) | ready — B's metric now exists |
| **E** | LSM ordering prior for within-span localization | ready, ungated |

**A/B side-finding, important for anyone reading old numbers in this repo/memory:** a
Spanish-inflection word-matching bug was deflating every binary alignment number cited
before 2026-07-30. Corrected baseline (n=26): naive 65.4%, Run A 61.5%, **Run B 73.1%**
(was cited as 61.5%). Use `check_alignment_v2.py` going forward, not the original script.

Recommended order: **A → B**, with **C** runnable in parallel (it has its own eval and does
not depend on A or B). Then D1 and E once B's metric exists.

## Standing rules

- **Never overwrite an existing output directory.** Every run writes to a new suffixed
  dir so prior results stay comparable. Existing outputs are the baseline.
- **Pre-register before you run.** Where a success criterion is specified, do not adjust
  it after seeing results. If you must adjust anything, log the change and why.
- **No new manual labeling** without explicit approval. Nothing here requires it.
- **Stop at a failed gate.** Write up the failure. Do not improvise past it.
- **Report honestly.** Negative results are the expected outcome for at least one of
  these. Do not tune until something passes.
- Watch disk: only ~1.3 GB free at time of writing. Check before large downloads.

## Environments

| env | python | has |
|---|---|---|
| `/opt/miniconda3/envs/sea/bin/python` | 3.12 | SEA deps (numba, beartype, fastdtw, sentence_transformers, pose-format) |
| `/opt/miniconda3/envs/lsm-probe/bin/python` | 3.11 | mediapipe 0.10.21 (Holistic works), opencv, pdfplumber, epitran, scipy |

MediaPipe Holistic only runs in `lsm-probe` (mainline mediapipe removed the legacy
Holistic API; this env is pinned).

## Data availability — RESOLVED (drive was reconnected and mirrored)

Everything these workstreams need is now **local**. Do not assume the external drive is
mounted; work from `phase3/local_drive_mirror/` (438 MB, verified).

| local path | contents |
|---|---|
| `local_drive_mirror/57TvyH9902U/` | the 26 review clips (word subdirs) + `_ref/` |
| `local_drive_mirror/asr/*.asr.json` | **11 videos' word-level ASR.** `57TvyH9902U` verified: 13,827 words, schema `{video, model, words:[{start,end,word}]}` |
| `local_drive_mirror/arrays_57TvyH9902U/` | `segments.json` (40 validated segments), `lag_estimate.json`, 40 seg `.npz` |
| `local_drive_mirror/segments_all/*.json` | `segments.json` + `lag_estimate.json` for **all 11 videos** |
| `local_drive_mirror/review_csvs/*.review.csv` | review CSVs for all 11 videos (~1,409 candidate rows total, mostly unreviewed) — raw material if the eval set is grown |

Scripts default to the drive via
`DRIVE = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator"))`.
The mirror does **not** replicate that layout, so pass explicit paths rather than setting
`LSM_DATA_ROOT` blindly.

**Verified working end-to-end from the mirror:**

```
python phase2/estimate_lag.py \
  --arrays phase3/local_drive_mirror/arrays_57TvyH9902U \
  --asr    phase3/local_drive_mirror/asr/57TvyH9902U.asr.json
# → Event-based lag: +6.33s (spread ±3.5s), matching the established baseline
```

This confirms **D1 is fully unblocked** — `estimate_lag.py` needs only segments + `.npz`
landmarks + ASR, **not the source video**.

**Gloss corpus: obtained 2026-07-30**, now at `phase3/gloss_corpus/` (CC BY 4.0, from
figshare DOI 10.6084/m9.figshare.28519580 — it was never on the drive). `esp-lsm_glosses_corpus.csv`
= 3,000 rows, columns `esp`/`lsm`. See D3 for measured statistics and two significant
caveats about its usability. **Nothing is blocked on data any more.**

## Key existing assets

| path | what |
|---|---|
| `phase3/sea_test/subtitles/57TvyH9902U_slice.vtt` | 233 ASR sentence cues, slice-local time |
| `phase3/sea_test/out/segmentation/E4s-1_30_50/57TvyH9902U_slice.eaf` | SEA segmenter output: SIGN tier n=3633, SENTENCE n=527 |
| `phase3/sea_test/out/aligned_A/`, `aligned_B/` | baseline alignments (bias 0 / bias 6.33) |
| `phase3/sea_test/out/naive_flat_shift.vtt` | flat +6.33s baseline |
| `phase3/sea_test/check_alignment.py` | current (binary) verdict check |
| `phase3/probe/review_2.csv` | 26 human-verified sign instances w/ `sign_start`,`sign_end` |
| `phase3/probe/mouthing_pilot/runB_hits.json` | 16 Run-B spans containing a verified sign |
| `phase3/probe/mouthing_pilot/burst_detect.py` | mouth-activity burst detector |
| `phase3/probe/mouthing_pilot/snapped_windows.csv` | per-clip detected mouthing burst + `lead_s` |
| `phase3/local_drive_mirror/57TvyH9902U/` | the 14s review clips (external drive is disconnected; this is the local mirror) |
| `phase3/local_drive_mirror/arrays_57TvyH9902U/segments.json` | 40 hand-presence-validated segments, **absolute** video seconds |
| `phase3/sea_test/video/57TvyH9902U_slice.mp4` | 900x1030, the 1200–3000s interpreter crop |

**Slice geometry:** video `57TvyH9902U`, slice = absolute 1200–3000s.
`slice_local = absolute - 1200`. Interpreter crop `crop=360:412:919:308` from a 1280x720
source, upscaled to 900x1030. Measured interpreter lag **6.33s** (spread 3.48s).

**Current baseline to beat** (binary: does the aligned span contain the verified sign
moment): Run A 14/26, **Run B 16/26 (62%)**, naive flat-shift 15/26.

---

# Workstream A — correctness fixes (no evaluation needed)

These are defect repairs, not accuracy gambles. They are correct regardless of whether
the 26-clip check can measure them. Do not gate them on statistical significance.

## A1. Filter phantom sign segments through `segments.json`

**Problem.** The SEA segmenter ran over the whole 1800s slice including stretches where
the interpreter is not on screen (confirmed: B-roll, a QR-code sponsor graphic, a stats
infographic, a hallway shot — ~4 episodes, 0/4 of which fall inside a validated segment;
81% of the slice is validated). Segments produced in those gaps are phantoms: they
distort the gap structure the DP relies on and can steal DP mass from real signing.

**Do.** Write `phase3/sea_test/filter_segments.py`:

1. Load `phase3/local_drive_mirror/arrays_57TvyH9902U/segments.json`. Its `segments` list
   has `start_sec`/`end_sec` in **absolute** video seconds. Convert to slice-local by
   subtracting 1200; keep only those overlapping [0, 1800].
2. Parse `phase3/sea_test/out/segmentation/E4s-1_30_50/57TvyH9902U_slice.eaf` (use
   `pympi-ling`, already in the `sea` env, or `xml.etree`).
3. Drop every `SIGN`-tier annotation whose midpoint falls outside all validated windows.
   Leave the `SENTENCE`, `SUBTITLE`, `SUBTITLE_CORRECTED` tiers untouched.
4. Write to a **new** dir `phase3/sea_test/out/segmentation_filtered/E4s-1_30_50/`
   preserving the filename, so SEA's `--segmentation_dir` can point at it.
5. Print: segments before, segments after, % dropped, and the wall-clock ranges dropped.

**Sanity check (must pass):** the dropped ranges should cluster in the known-bad regions.
Slice-local ~100s (infographic), ~300s (B-roll), ~587–600s (QR ad), ~900s (hallway).
If drops are spread uniformly instead, the time-base conversion is wrong — stop and fix.

## A2. Re-segment ASR cues into real sentences

**Problem.** `make_vtt.py` grouped ASR words into 233 cues, but these are caption-length
chunks, not sentences. SEA's cost function treats each cue's boundaries and duration as
meaningful, so feeding it arbitrary chunks makes the duration and boundary terms
compare against something linguistically meaningless.

**Do.** Write `phase3/sea_test/make_vtt_sentences.py` producing
`phase3/sea_test/subtitles_sentences/57TvyH9902U_slice.vtt`.

**Primary path — the word-level ASR is now local.** Use
`phase3/local_drive_mirror/asr/57TvyH9902U.asr.json`
(schema `{video, model, words:[{start,end,word}]}`, 13,827 words, absolute video seconds —
subtract 1200 for slice-local). Read `phase3/sea_test/make_vtt.py` first and mirror its
slice-windowing logic; only the grouping rule changes.

- Group words into sentences on terminal punctuation `. ! ? …` attached to the word token.
- Cap runaway sentences at 30s; split at the largest internal inter-word gap if exceeded.
- Report: cue count before (233) and after, and duration distribution before/after.

**Cross-check (cheap, do it).** Independently derive sentences by merging adjacent cues of
the existing `subtitles/57TvyH9902U_slice.vtt` wherever a cue's text does not end in
terminal punctuation, taking the first cue's `start` and last cue's `end`. The two methods
should produce near-identical cue counts and boundaries. If they diverge sharply, one of
them is wrong — investigate before proceeding. Merging cannot create or destroy time, so
assert total covered time is unchanged in the merge version.

**Note the known trap:** `phase2/make_review_clips.py:get_context()` caps its sentence
walk at ±15 words and silently truncates. Do not reuse that function. Walk to real
punctuation with no word cap.

## A3. Re-run SEA with A1+A2 applied

New output dirs only: `out/aligned_A2/`, `out/aligned_B2/`. Copy `run_sea.sh` to
`run_sea_v2.sh` and change: `--segmentation_dir` → filtered, `--pr_sub_path` and
`--gt_sub_path` → sentence subtitles. Keep every other parameter identical
(`--dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10
--dp_window_size 50 --similarity_measure none`, bias 0 for A, 6.33 for B).

**Deliverable:** `phase3/sea_test/WORKSTREAM_A_RESULTS.md` — what was dropped/merged, and
the new alignment scored on both the old binary metric and the Workstream-B metrics.

---

# Workstream B — make the evaluation instrument sharper (do before any hypothesis testing)

**Why.** Power analysis on the current binary hit/miss check, paired, best case:

| verified clips | +7pt | +13pt | +18pt | +25pt |
|---|---|---|---|---|
| 16 | 0% | 1% | 5% | 19% |
| 26 | 2% | 11% | 32% | 66% |
| 50 | 21% | 65% | 91% | 99% |
| 100 | 82% | 99% | 100% | 100% |

At n=26 we cannot reliably detect even a 25-point improvement. Binary scoring
dichotomises a continuous quantity and throws away most of the information. Continuous
metrics recover a substantial part of that for free, on the same clips.

## B1. Add continuous metrics to the checker

Write `phase3/sea_test/check_alignment_v2.py` (do **not** edit `check_alignment.py` —
it is the baseline). Keep its existing correct behaviour:

- the `video_start > 0` vs probe-format branch for recovering absolute sign times,
- the outside-slice skip,
- the `clips_interpreter/manifest.csv` join for the older probe format.

Add, per verified sign instance, against the aligned cue containing that word:

| metric | definition |
|---|---|
| `contains` | existing binary — keep for continuity |
| `signed_dist_s` | `sign_time − span_center`, seconds (signed, so bias is visible) |
| `abs_dist_s` | absolute value of the above |
| `norm_dist` | `abs_dist_s / (span_duration/2)`; <1.0 means inside the span |
| `iou` | overlap of [sign_start, sign_end] with the aligned span / union |
| `span_dur_s` | span duration — a method that "wins" by widening spans must be visible |

Report mean and median of each, plus the binary rate. **`span_dur_s` is not optional** —
without it a method can trivially improve `contains` by producing wider spans.

## B2. Re-baseline everything on the new metrics

Score all four on the same 26 clips and write one table:
Run A, Run B, naive flat-shift, and (once A3 lands) Run A2 / Run B2.

## B3. Pre-registered decision rule for all later hypotheses

A change counts as an improvement only if it improves **median `abs_dist_s`** on a paired
Wilcoxon at p<0.05 **without** increasing median `span_dur_s` by more than 10%.
Write this into the results doc before running Workstream D.

**Deliverable:** `phase3/sea_test/WORKSTREAM_B_RESULTS.md`.

---

# Workstream C — AV-HuBERT mouthing test (timeboxed, hard stop)

**This is the last mouthing experiment on existing data.** If it does not clearly clear
chance, mouthing word-identification is written up as a characterised negative and not
reopened. Not "suggestively" — clearly. No spotter-building, no fine-tuning, no
"let's try the 1080p sources first" if it fails. Time budget: one week.

**Background.** Four hand-built approaches (cross-instance landmark DTW, order-position,
hand viseme table, G2P viseme table) all landed at chance. SignMouth (arXiv 2509.10266)
shows a **frozen AV-HuBERT** extracts real word-level signal from mouth crops at ~15–17 px
— essentially our resolution (14.5 px @720p, 16.9 px @1080p measured). So the failures are
plausibly our feature extractor, not the data. This test swaps only the feature extractor.

## C0. Availability gate (do this first, stop if it fails)

Determine whether a pretrained AV-HuBERT checkpoint is obtainable and runnable. Check the
official repo (`facebookresearch/av_hubert`, fairseq-based) and HuggingFace mirrors.
Record: checkpoint size, license, dependency chain, whether it runs CPU-only or needs GPU.

**Gate:** if no usable checkpoint is obtainable, stop and report. Do not substitute a
different model without flagging it — the whole premise is testing the specific machinery
SignMouth validated.

Note ~1.3 GB free disk. If the checkpoint is large, plan for Colab. There is a working
Colab precedent at `phase3/probe/COLAB_PROBE.md` (SignCLIP probe) — follow that pattern.
Upload is small: `phase3/probe/mouthing_pilot/span_clips_v2/` is 77 MB.

## C1. Build 96×96 mouth ROI crops

SignMouth uses 96×96 mouth crops. We already have MediaPipe face landmarks for all clips.

For each of the 26 verified clips (`phase3/local_drive_mirror/57TvyH9902U/<word>/<clip>.mp4`):
1. Per frame, get the lip landmark set
   `LIP_IDX = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))`.
2. Square ROI centred on the lip centroid, side = 1.4 × max(lip bbox width, height).
3. Crop, resize to 96×96, grayscale (check what the checkpoint expects and match it).
4. Save `phase3/probe/mouthing_pilot/mouth_rois/<clip>.npy`, shape (T, 96, 96).
5. Mark frames as invalid where no face, or where occluded per
   `burst_detect.compute_occluded` on the existing `sidecar/*.npz`.

**Sanity check:** render a few crops as PNG and look at them. If the mouth is not centred
and filling the frame, the ROI geometry is wrong — fix before embedding anything.

## C2. Embed and run retrieval

Embed each clip's **snapped mouthing burst** window (from `snapped_windows.csv`,
`burst_on_s`/`burst_off_s`, `mouthing_detected == True` → 23 of 26 clips).

**Protocol — clip-to-clip retrieval, no text bridge needed:**
For each burst, rank all other bursts by embedding similarity and ask whether same-word
bursts rank above different-word bursts. 7 words have ≥2 verified instances
(gracias, joven, actividad, atender, decidir, derecho, familia) giving ~14 same-word pairs.

**Metrics:** AUC and mean reciprocal rank.

**Null — this matters, we got it wrong once already.** Do **not** assume one correct
answer per query: `decidir` has 4 instances and `familia` 3, so a flat "chance rank =
(N+1)/2" is wrong and inflates significance. Use a **permutation null**: shuffle the word
labels across bursts, recompute AUC/MRR, repeat ≥10,000 times, and take the observed value's
percentile in that distribution. Label permutation preserves the multi-instance structure
automatically.

**Pre-registered success criterion:** observed AUC above the 95th percentile of the
permutation null.

## C3. Only if C2 passes

Sliding-window AV-HuBERT similarity within the 16 Run-B spans, scored with the Workstream-B
continuous metrics. If C2 fails, do not run this.

**Deliverable:** `phase3/probe/mouthing_pilot/WORKSTREAM_C_RESULTS.md` — checkpoint used,
crop examples, AUC/MRR vs permutation null, and a clear pass/fail against the
pre-registered criterion.

---

# Workstream D — alignment hypotheses (gated on Workstream B)

Do not start until B1–B3 are done; these need the sharper metric to be measurable at all.
Run **one at a time**, scoring each against the B3 rule. Do not stack.

## D1. Per-block interpreter lag (highest expected value)

**Rationale.** `align.py:153` applies a single scalar bias to every cue in the video
(`shift_cues(cues, pr_subs_delta_bias_start, pr_subs_delta_bias_end)`; `utils.py:81` takes
only two scalars — there is no per-block path today). The measured lag is 6.33s with
**3.48s spread**, and that spread is exactly what the DP currently has to discover
unaided. Removing it before the DP runs is free accuracy.

**Do.** Re-run `phase2/estimate_lag.py` per 10-minute block over the slice instead of once.
Then either extend `shift_cues` to accept a piecewise schedule, or (simpler, no SEA edit)
pre-shift the cues per block in the `.vtt` before handing it to SEA. Prefer the latter.

**Data: fully unblocked, verified.** `estimate_lag.py` needs only `--arrays` (a dir with
`segments.json` + the seg `.npz` landmark files) and `--asr` (word-level JSON). **No source
video.** This exact command was run and reproduces the baseline:

```
python phase2/estimate_lag.py \
  --arrays phase3/local_drive_mirror/arrays_57TvyH9902U \
  --asr    phase3/local_drive_mirror/asr/57TvyH9902U.asr.json
# → Event-based lag: +6.33s (spread ±3.5s), 291/715 clause-ends matched to 219 motion onsets
```

Note it derives lag from **clause-end events matched to interpreter motion onsets**, so a
per-block run needs enough events per block to be stable — 291 matches over 30 min is ~10
per 10-min block minimum, which is thin. **Report the per-block match counts**, and if a
block has too few matched events, fall back to the global 6.33s for that block rather than
trusting a noisy local estimate. Say so in the results doc.

## ⚠️ Scoping correction — the gloss corpus has TWO possible uses; only one is dead

An earlier version of this spec redirected the "grammar prior" from **sign ordering**
(its original purpose) to **duration prediction**, on the reasoning that SEA's cue-level DP
is monotonic so ordering cannot matter. That reasoning is correct *for Stage-1 sentence
alignment* and irrelevant to the actual intent, which was always **Stage-2: predicting what
order signs appear in inside an already-aligned span**, so that N segmenter boundaries can
be matched to M content words.

- **Duration use → dead.** See D2/D3 below; measured, ratio is 0.966, no headroom.
- **Ordering use → live, and the corpus supports it.** See E1. It does **not** belong in
  Workstream D and is **not** gated on D2.

## ⛔ D2 and D3 — MEASURED AND CLOSED 2026-07-30, do not build

The premise was that SEA's duration term is miscalibrated because signing and speech take
different amounts of time. **Measured directly on the slice and it is not:**

```
slice window            : 1800.0 s
speech, voiced only     : 1512.3 s   (84% of window, from 3,780 ASR word intervals)
signing (hand-present)  : 1460.8 s   (81% of window, from validated segments.json)
signing / voiced-speech ratio = 0.966
```

The duration term assumes a ratio of 1.0. The true ratio is **0.966** — a 3.4% error on
one of several cost terms. A global rescale by 0.966 cannot produce a detectable change,
so **D2 is a no-op and D3 (which was gated on D2 paying off) is not triggered.**

**This also explains the gloss-corpus discrepancy.** The corpus shows ~30% *token*
compression at its longest sentences (ratio 0.707), yet *duration* barely compresses at
all (0.966). Both are true: LSM drops function words (fewer items) but each sign takes
longer to articulate than a spoken word. The two effects very nearly cancel. This
confirms empirically the caveat flagged in D3 — **the corpus token ratio is not a duration
ratio and must never be substituted for one.**

**Caveats on the measurement, stated honestly:** "hand-present" overestimates active
signing (hands are visible during pauses), it is one video over 30 minutes, and a global
ratio near 1.0 does not prove every individual sentence sits at 1.0. But note that D3's
mechanism was to *predict* duration from post-deletion token count — and the DP already
has each cue's directly observed speech duration, which the measurement says is already a
good estimator. Replacing an observed quantity with a text-derived guess is a downgrade
unless per-sentence deviation is predictable, and a ≤10-token templated corpus is not
positioned to establish that.

**If you want to reopen this**, the thing to measure first is per-sentence variance of the
signing/speech ratio — not the global mean, which is now known. That needs per-sentence
signing durations, which is what alignment is trying to determine, so it is circular until
alignment is trusted.

---

## D2. Global signing/speech duration rescale — SUPERSEDED, see above

**Rationale.** The cost function's duration term (`align_dp.py:45-49`) compares cue
duration to candidate-group duration as if speech and signing take the same time. That
assumption is untested and wrong in some direction.

**Do.** Measure the corpus-wide ratio (total signing time from validated `segments.json`
vs total speech time from the ASR) and rescale cue durations by it. **Do not assume the
direction** — sign languages drop function words (fewer items) but each sign takes longer
to produce than a spoken word (slower articulator). Measure it.

**Gate for D3:** if the global rescale does not move the B3 metric, the DP does not care
about the duration term and the grammar prior — which is a *refinement of that same term*
— is not worth building. Skip D3.

## D3. Grammar-prior duration model (only if D2 pays — AND the corpus is reachable)

**Corpus is now LOCAL:** `phase3/gloss_corpus/esp-lsm_glosses_corpus.csv` (CC BY 4.0,
Lara Ortiz / Fuentes Aguilar / Chairez, figshare DOI 10.6084/m9.figshare.28519580).
3,000 rows, columns `esp` and `lsm` (plus 3 empty `Unnamed:` columns — drop them).

**Read this before deciding whether D3 is worth building.** Measured directly:

| Spanish length | n | mean gloss/Spanish token ratio |
|---|---|---|
| 1–3 tokens | 1,057 | 1.174 |
| 4–5 | 1,522 | 0.910 |
| 6–7 | 345 | 0.808 |
| 8–10 | 76 | 0.707 |

Vocab: 813 Spanish / 620 gloss tokens. **Max sentence length is 10 tokens; only 76
examples have ≥8.** Most frequently deleted: `mi, es, tu, el, está, la, gusta, a, en,
quiere, un, son` — i.e. articles, copulas, possessives, prepositions. That is a real and
linguistically sensible deletion pattern, and compression clearly increases with length.

**Two problems with using it, both of which must be stated in the writeup:**

1. **Extrapolation gap.** Our mañanera sentences run 11–31 content words. The corpus tops
   out at 10 tokens with 76 examples there. Fitting a deletion trend on ≤10-token
   textbook phrases and applying it to 30-token spontaneous broadcast speech is a large
   leap. Content is also heavily templated — e.g. `mi hijo va a la escuela de lunes a
   viernes` / `mi hija va…` / `mis hijos van…` are the same frame with slot swaps — so
   apparent "rules" may be artifacts of the template inventory.
2. **The corpus gives token-count compression, NOT duration compression.** It says nothing
   about how long a sign takes to produce. Signs are slower than spoken words, so a
   perfect deletion model still does not yield a duration without an empirical per-sign
   duration. **Therefore D3 can only refine the *count* half of D2's estimate; the
   seconds-per-sign factor must come from our own measured data.** Do not treat the
   corpus ratio as a duration ratio.

**Also available (optional, likely low value):** the authors published fine-tuned
Spanish→LSM gloss models — `VaniLara/esp-to-lsm-model` (Helsinki opus-mt-es-es) and
`vania2911/esp-to-lsm-barto-model` / `-model-split` on HuggingFace. Running one to
generate gloss sequences directly would sidestep rule-mining, but they are trained on
exactly this ≤10-token templated corpus, so expect degradation on mañanera vocabulary and
sentence length. If you try one, sanity-check its output on 5 real mañanera sentences
before building anything on it.

**Note:** word *order* is not the target. Cue-to-cue order is already monotonic by
construction in the DP; LSM reordering happens *within* a sentence, below this stage's
resolution. The duration term is the real plug-in point.

**A null result here remains a legitimate, citable finding** — "prescriptive textbook LSM
deletion rules do not transfer to live interpreting" is useful to anyone else tempted to
reach for this corpus the same way.

**Deliverable:** `phase3/sea_test/WORKSTREAM_D_RESULTS.md`, one section per hypothesis,
each scored against the pre-registered B3 rule.

---

# Workstream E — LSM ordering prior for within-span sign localization

**Not gated on anything above.** This is Stage 2 (find signs inside an aligned span), not
Stage 1 (align sentences). It has nothing to do with the duration term that D2/D3 killed.

**The job.** Given an aligned span, the segmenter proposes N sign boundaries and the
sentence supplies M content words. To match them you need the order the words appear in
*when signed*, which is not Spanish order. A monotonic DP then aligns N segments to M
expected glosses — giving every segment a candidate label before any recognition runs.

**The corpus supports this — measured 2026-07-30 on all 3,000 pairs:**

| pattern | measurement |
|---|---|
| shared-word order differs between `esp` and `lsm` | **42.2%** of pairs (n=1,859 with ≥2 shared words) |
| mean Kendall τ of shared-word positions | **0.342** (1.0 would be identical order) |
| time marker moved earlier in the gloss | **87.2%** (232/266) |
| verb is the final gloss token | 56.6% (n=1,633), p=5.7e-08 |

**Robustness check already done:** the 266 time-marker pairs span **173 distinct trailing
frames**, so the 87% is not an artifact of a few repeated templates. Representative:
`leí un libro ayer` → `ayer yo libro leer`.

**Read that as:** time-marker fronting is a strong, usable rule. Verb-final is statistically
real but only 7 points above chance — do not treat it as reliable. Overall, assuming
Spanish order would mis-order ~4 sentences in 10.

## E1. Extract and validate ordering rules

1. Mine reordering statistics from `phase3/gloss_corpus/esp-lsm_glosses_corpus.csv`
   (**train split only**; hold out 10% and never look at it until the end).
2. Express as a reordering function: Spanish content-word sequence → expected LSM order.
   Start with the two measured rules (front time markers; weak verb-final tendency).
3. **Validate on held-out corpus pairs first** — report Kendall τ of predicted vs actual
   gloss order, against the baseline of "assume Spanish order" (τ = 0.342).

**Gate:** if predicted order does not beat τ=0.342 on held-out pairs, the rules are not
capturing anything generalisable even *within* the corpus. Stop there.

## E2. Transfer check (the real risk)

**The corpus maxes out at 10 tokens and is textbook/templated; mañanera sentences run
11–31 content words and are spontaneous.** Prescriptive grammar-book LSM is not
guaranteed to describe live interpreting.

Before building E1 into any pipeline, sanity-check the rules against the verified data we
have: for the 26 verified sign instances, is the observed sign time consistent with the
predicted position of that word in the reordered sequence, more often than with its
Spanish-order position? n is small, so treat this as a smell test, not proof.

**A null result here is a citable finding** — "prescriptive textbook LSM ordering does not
transfer to live broadcast interpreting" is genuinely useful to anyone else reaching for
this corpus.

## E3. Only after E1+E2

Wire the ordering prior into the segment→gloss monotonic DP described in
`phase3/sea_test/README.md` ("Downstream" section). Score with Workstream-B metrics.

**Deliverable:** `phase3/WORKSTREAM_E_RESULTS.md`.

# Explicit non-goals

- No new manual annotation in any workstream.
- No corpus re-download or re-extraction (~60h/pass). Not needed for anything here.
- No concept:sign cloud yet — build it when its first real consumer exists (SignCLIP
  fine-tuning or candidate generation). **Design stance updated 2026-08-01: when built,
  candidate generation is OPEN-VOCABULARY** — the LLM proposes/ranks candidate signs
  freely; the 886 inventory is the verification seed and eval anchor, not a constraint
  (constraining was measured 2026-07-31 to manufacture wrong answer keys). See
  `PIPELINE_MAP.md` and memory note `concept-cloud-open-vocab`.
- No stacking of D1–D3 until each is individually measured.
- No AV-HuBERT fine-tuning, no keyword-spotter training, no Spanish spotter construction.

# Reporting

One results doc per workstream, at the paths named above. Each must state: what was run,
the pre-registered criterion, the observed numbers, pass/fail, and anything that had to be
changed mid-flight and why. If a gate fails, the doc explaining the failure **is** the
deliverable — do not push past it looking for a better answer.
