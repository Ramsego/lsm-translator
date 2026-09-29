# Route 1 results: mouth-activity-snapped templates + occlusion mask

Executed exactly per the Route 1 spec, all gates respected, no threshold tuning against
Test 2/3/4 results. No new manual labeling used anywhere.

## Frozen constants
All used at spec defaults, unchanged: `FPS=30.0` (verified per-clip, all clips were exactly
30.0), `SMOOTH_W=5`, `Z_ON=1.0`, `MIN_BURST=6`, `MAX_GAP=5`, `MERGE_GAP=10`, `OCC_DIST=0.9`,
`PAD=0.15s`, `SEARCH_PRE=2.5s`, `SEARCH_POST=1.0s`. **No adjustments made during §3 — the
gate passed on first attempt.**

## Gates

| Gate | Result | Detail |
|---|---|---|
| §3 sanity check (visual) | **PASS** | All 5 diagnostic clips matched expected burst/occlusion behavior on first attempt — see `sanity_check.png`. `gracias_20m52s`: burst overlapping/leading the marked window. `gracias_20m40s`: only a weak burst near the marked window (~3x lower amplitude than the clip's dominant burst elsewhere). `joven_25m49s`: occlusion correctly flagged ~7.2-7.6s, matching the hand-over-mouth finding from the pilot. `actividad_20m20s`: strong burst overlapping the marked window. `futuro_45m33s`: burst activity overlaps its own window. |
| §4 mouthing-detected rate | **PASS** | 3/26 clips (`equipo_21m19s`, `derecho_43m05s`, `director_40m06s`) got no burst overlapping the search window — well under the 10/26 stop threshold. |
| §5 Test 2 regression guard | **PASS** | Median rank 3.0/25 (pilot: 3.0/25 — identical), p=0.0006 (pilot: 0.0013 — stronger), top-3 56.2% (pilot: 52.9%). Snapping templates to detected bursts did not hurt discrimination. |

## Step 4 free deliverable: lead_s distribution (mouthing vs. sign timing)
n=23 clips with detected mouthing. **Mean lead +0.12s, median −0.03s, stdev 1.09s, range
−1.67s to +2.43s. Only 11/23 (48%) show mouthing preceding the sign at all.** This is a
real finding on open question 1, and it complicates the pilot's filmstrip-based intuition:
there is no strong, consistent lead — the mouthing-to-sign offset is close to zero on
average but varies by several seconds instance-to-instance, in both directions. The
filmstrip check's two examples were not representative of a systematic bias; they were
within the range of this variance.

## Test 3 — fine localization (headline number, cross-clip, snapped + occlusion-masked)
n=26 pairs.

| | truth (a): snapped mouthing midpoint | truth (b): human sign midpoint |
|---|---|---|
| ours: mean / median AE | 4.82s / **5.02s** | 5.29s / 5.16s |
| midpoint baseline (mean AE) | 4.40s | 4.95s |
| random baseline (mean AE) | 5.18s | 5.56s |
| within ±1.0s / ±1.5s / ±3.0s | 19.2% / 23.1% / 42.3% | 15.4% / 23.1% / 38.5% |
| Wilcoxon vs. midpoint baseline | p=0.69 (does not beat it) | p=0.66 (does not beat it) |

Pilot reference (mistimed windows): 5.39s ours vs. 4.97s midpoint, p=0.65. **Fixing the
windowing bug did not change this result** — snapped, occlusion-masked templates perform
essentially identically to the old mistimed ones, and still do not beat guessing the clip's
own midpoint.

**Pre-registered decision (Test 3, truth (a), cross-clip, median AE = 5.02s): falls in the
>3.0s bucket → no useful positional signal even with corrected windows.** Deprioritize
mouthing as a hard localization anchor. Test 2's discrimination result stands independently
and is unaffected by this outcome.

## Test 4 — coarse localization in full sentence spans
n=14 pairs (of 16 hits; 2 skipped, no alternate snapped template available for `equipo`,
`director`).

- Spearman rho = **+0.191**, p=0.256 (not significant, but note the pilot's rho was
  **−0.267** — this flips to the correct direction and moves substantially closer to
  significance, though still underpowered at this n).
- Absolute error vs. truth (b) human sign point: mean 3.57s, **median 1.49s**.
- Absolute error vs. truth (a) span-internal burst (n=12): mean 3.15s, **median 1.19s**.

This is a genuinely mixed picture, not a clean win: several pairs localize well
(`edad` 0.24s, `futuro` 0.09s, `cambiar` 0.39s, `derecho`(#06) 0.77s, `decidir`(#05) 0.86s),
while a few are catastrophically wrong (`derecho`(#07) 14.08s, `familia`(#10) 11.69s,
`atender` 9.06s) and pull the mean far above the median. **The pre-registered decision rule
is defined on Test 3, not Test 4** (Test 4's Spearman test was known going in to be
"nearly powerless at n=9-14" per the spec) — so this does not override the Test 3 verdict.
But the direction-flip and the sub-1.5s median are worth carrying forward as a reason not
to fully close the door on mouthing as a *soft* prior at the sentence-span scale, even
though it fails as a hard anchor at the narrow-clip scale tested in Test 3.

## Net conclusion
- **Word discrimination (Test 2): real, reproduced, unaffected by the windowing fix** —
  median rank 3/25, p<0.001 both times.
- **Fine localization within a narrow candidate window (Test 3): no signal, confirmed
  after fixing the suspected bug.** The original failure was not just a mistimed-window
  artifact — correcting it did not change the outcome. Per the pre-registered rule,
  deprioritize mouthing as a hard localization anchor.
- **Coarse localization within a full sentence span (Test 4): weak, non-significant, but
  directionally reversed from wrong-to-right vs. the pilot**, with several genuinely close
  matches alongside a few large misses. Not strong enough to act on alone, but worth
  keeping as a candidate soft signal (e.g., fed into SEA's similarity-matrix slot as one
  weak term among several) rather than discarding entirely — that determination would need
  a larger n than this pilot supports.
- No occlusion or cross-signer normalization work was in scope here (excluded per spec);
  both remain identified, unaddressed gaps from the earlier failure-case inspection.
