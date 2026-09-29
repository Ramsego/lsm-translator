# Workstream B results — sharper evaluation instrument

## Why

Power analysis on the binary hit/miss check, paired, best case:

| verified clips | +7pt | +13pt | +18pt | +25pt |
|---|---|---|---|---|
| 16 | 0% | 1% | 5% | 19% |
| 26 | 2% | 11% | 32% | 66% |
| 50 | 21% | 65% | 91% | 99% |
| 100 | 82% | 99% | 100% | 100% |

At n=26 we cannot reliably detect even a 25-point improvement. Binary scoring
dichotomises a continuous quantity and throws away most of the information paid for.

## B1: `check_alignment_v2.py` — DONE

New script, `check_alignment.py` untouched as baseline. Adds per verified instance,
against the nearest-by-time aligned cue whose text contains the word: `contains`
(binary, kept for continuity), `signed_dist_s`, `abs_dist_s`, `norm_dist`, `iou`,
`span_dur_s`. Supports `--dump-csv` for per-clip pairing across runs.

**Bug found and fixed while building this** — plain substring word-matching misses
Spanish inflection (`joven`/`jóvenes`, `aprovechar`/`aprovecho`), which was silently
deflating every binary number established this session. Fixed via accent-stripping +
shared-stem prefix match (`word_in_text()`). Full account and corrected historical
numbers are in `WORKSTREAM_A_RESULTS.md` — this is the same fix, documented once.

## B2: baseline table, all n=26 (fixed matching, no silent skips)

| run | binary | abs_dist median/mean (s) | iou median | span_dur median (s) |
|---|---|---|---|---|
| naive flat-shift | 65.4% | 4.81 / 6.64 | 0.019 | 12.78 |
| Run A (bias 0) | 61.5% | 5.09 / 7.34 | 0.028 | 13.33 |
| **Run B (bias 6.33)** | **73.1%** | **3.46 / 6.10** | 0.022 | 13.78 |
| Run A2 (bias 0, filtered) | 57.7% | 5.09 / 7.31 | 0.028 | 13.33 |
| Run B2 (bias 6.33, filtered) | 73.1% | 3.46 / 6.05 | 0.022 | 13.33 |

Note `iou` is low across every run (0.02–0.03) — spans average ~13s while sign windows
are ~0.3–1s, so even a "hit" by the binary/distance metrics never implies a tight span.
This is expected and consistent with everything established about span width this
session; it is not a new finding, just now quantified per-run.

## B3: pre-registered decision rule

**A change counts as an improvement only if it improves median `abs_dist_s` on a paired
Wilcoxon at p<0.05, without increasing median `span_dur_s` by more than 10%.**
Written before scoring anything against it.

**First application — Workstream A1 (segment filtering), paired on the same 26 clips:**

| comparison | median abs_dist_s | Wilcoxon p | span_dur change | passes B3? |
|---|---|---|---|---|
| Run A vs Run A2 | 5.09 → 5.09 | 0.7855 | +0.0% | **No** |
| Run B vs Run B2 | 3.46 → 3.46 | 0.5930 | −3.3% | **No** |

Confirms at the distribution level what the binary metric shows once corrected for the
word-matching bug (see `WORKSTREAM_A_RESULTS.md`): A1 flips exactly one of 26 instances
(`aprovechar`) but that single point doesn't move the median, so A1 has no detectable
effect on alignment quality by this rule. Consistent with A1 being scoped as a
correctness fix, not an accuracy bet.

**Reference — naive vs. Run B, same rule, for continuity with earlier reasoning:**

| comparison | median abs_dist_s | Wilcoxon p | span_dur change |
|---|---|---|---|
| naive vs Run B | 4.81 → 3.46 | 0.0888 | +7.9% |

Still short of p<0.05 at n=26, though closer than before (the corrected matching moved
this from noise-adjacent to borderline). Consistent with the standing read: real,
structurally-motivated signal (DP recovers per-sentence lag deviation a flat shift
cannot reach), not yet statistically decisive at this n. This is not a new test — it's
the same comparison from earlier in the investigation, now run through the sharper
instrument for the first time.

## What this changes going forward

Workstream D (per-block lag) and Workstream E (ordering prior) are scored against the
B3 rule above. Any future comparison should also use `check_alignment_v2.py`, not the
original binary-only script, and should reuse `word_in_text()` rather than plain
substring matching.
