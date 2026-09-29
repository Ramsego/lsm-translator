# Route 1 follow-up: execution result + two ablations

Context: `ROUTE1_RESULTS.md` in this directory has the full spec execution (all gates
passed). This note covers what came after — a debugging finding and two ablations run in
response to direct pushback on the results, not covered in that doc.

## Route 1 spec execution, in one line
All gates passed (sanity check, mouthing-detected rate 23/26, Test 2 regression guard
unchanged at median rank 3/25). **Headline result unchanged from the pilot: Test 3 fine
localization median AE = 5.02s, does not beat the naive midpoint baseline (p=0.69).**
Fixing the suspected windowing bug did not rescue localization — full writeup with all
tables in `ROUTE1_RESULTS.md`. Test 4 (coarse position in full sentence spans) flipped
from wrong-direction in the pilot (rho=-0.267) to right-direction here (rho=+0.191,
still n.s. at n=14), median AE 1.2-1.5s — not enough to override the Test 3 verdict but
flagged as worth revisiting at larger n.

## Finding: Test 3's large errors are matching confusions, not real timing gaps
User pushback: direct visual experience says mouthing tracks the sign closely, contradicting
Test 3's ~5s+ error numbers. Traced one specific ~7s "error" case by hand (frame extraction,
not just recomputation): the algorithm matched a template to a *different, genuinely real*
open-mouth moment elsewhere in the same 14s target clip — visually a real articulation, just
the wrong word, not a matching artifact from noise. The true sign moment in that clip showed
a closed mouth in the sampled frames. **This means Test 3's error metric conflates two
different things: real mouthing-to-sign timing (small — see Step 4's `lead_s`: mean +0.12s,
median -0.03s, stdev 1.09s, max 2.43s, measured within-clip, independent of any matching) vs.
cross-clip search picking the wrong candidate location entirely (large, when it happens).**
The corpus's real mouthing-timing behavior is close to what the user directly observed; the
localization test's bad numbers are mostly a search/discrimination failure, not evidence
against that observation.

## Ablation 1: is discrimination just detecting mouth-opening, or real shape?
Reran Test 2's exact protocol using aperture alone (1 number, mouth-opening amount) instead
of the full 80-dim lip-contour trajectory.

| feature | median rank | top-1 | top-3 | p |
|---|---|---|---|---|
| aperture only (1-dim) | 6.0/25 | 6.2% | 43.8% | 0.044 |
| full shape (80-dim) | 3.0/25 | 12.5% | 56.2% | 0.0006 |

Aperture alone carries some real signal (independently significant) but full shape roughly
doubles top-1 and strengthens significance by ~2 orders of magnitude. Discrimination is not
merely detecting "mouth is moving here" — lip shape contributes real, separable information.

## Ablation 2: candidate-set size effect (all prior numbers used 25 arbitrary candidates)
All discrimination numbers so far (here and in `MOUTHING_FINDINGS.md`) searched against the
full pool of 25 other verified clips spanning 16 unrelated words — harder than real
deployment, where a sentence has ~3-8 actual content words to choose among. Resampled smaller
random candidate pools (200 draws/template, correct answer always included; wrong candidates
drawn randomly, not matched for real per-sentence plausibility):

| k (candidate pool size) | top-1 | top-3 | chance top-1 |
|---|---|---|---|
| 25 (as tested throughout) | 12.5% | 56.2% | 4.0% |
| 12 | 27.9% | 59.3% | 8.3% |
| 8 | 38.4% | 70.1% | 12.5% |
| 5 (near real sentence scale) | 47.7% | 85.8% | 20.0% |

Strong, monotonic size effect. At near-realistic scale, top-1 nearly quadruples and top-3
becomes strong. Caveat: random wrong-candidate draws, not necessarily the specific words
that would really compete in a given sentence (could be easier or harder than this estimate).

## Updated net picture
- Localization (Test 3, narrow-window, cross-clip): still no usable signal after the
  windowing fix — confirmed, not an artifact.
- But Test 3's raw error numbers overstate how far off "reality" is — most of the bad
  numbers are wrong-candidate lock-on, not evidence mouthing timing itself is erratic
  (lead_s says it isn't, much).
- Discrimination (Test 2) is real shape-based signal, not a mouth-opening proxy, and looks
  meaningfully more usable at realistic candidate-set scale (top-1 ~48%, top-3 ~86% at k=5)
  than the flagship 25-candidate numbers suggested.

## Open questions for this round
1. Given discrimination looks substantially better at realistic scale, does it change the
   earlier recommendation to deprioritize mouthing-as-localization, or does localization
   remain the blocker regardless of candidate-set size (since Test 3 tests *where*, not
   *which*, and that failure is independent of candidate-set size)?
2. Is there a principled way to test candidate-set difficulty using real per-sentence
   plausible-word sets rather than random draws, given the current data only has one clip
   per word-instance (no same-sentence alternative-word clips to draw from)?
3. Given Test 4's direction-flip (wrong→right, still n.s.) — worth a dedicated larger-n
   follow-up, or is n=14 too fundamentally underpowered for this to ever be conclusive with
   the current corpus?
