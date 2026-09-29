# Mouthing pilot — findings summary for review

Goal being tested: can lip/mouth shape (MediaPipe Holistic landmarks, raw lip-contour points,
normalized by inter-eye distance, matched via DTW) serve as (a) a way to tell which candidate
word an interpreter mouthed, and (b) a way to localize *when* within a span it happened —
intended to plug into SEA's currently-unused similarity-matrix slot. All tests run on 26
human-verified clips (16 distinct words, 1 video, one signer pair) — no new manual labeling
used anywhere in this pilot.

## Test 1 (discarded): whole-clip DTW, same-word vs. different-word
Compared full ~14s clips end-to-end. FAILED pre-registered threshold (p=0.15). Diagnosed as
testing the wrong thing: whole loose candidate windows are mostly irrelevant mouth movement
outside the ~0.3-3s the target word actually occupies, diluting any real signal. Discarded,
not used to conclude anything.

## Test 2 (discrimination — PASSED): closed-set retrieval
Tight reference template (verified sign window ±0.3s) searched via subsequence DTW against
25 other clips. **Median rank of correct word: 3rd of 25 (chance ≈13th). Top-1 23.5%
(chance ~4%), top-3 53% (chance ~12%). Wilcoxon p=0.0013.** Real, non-trivial signal that
mouth shape carries usable word-identity information in a closed candidate set. Caveat: 25
largely-arbitrary candidates is likely *harder* than the real deployment case (~3-8 actual
content words per sentence) — real performance could be better, untested.

## Test 3 (fine localization — FAILED): position within an already-narrow window
Does the matched position beat naively guessing the clip's own midpoint? **No** — mean abs.
error 5.39s (ours) vs. 4.97s (guess the midpoint) vs. 5.03s (guess randomly); Wilcoxon
p=0.68-0.62, no improvement over either naive baseline. Caveat: the "midpoint" baseline is
itself informed (these clips were built around an ASR+lag estimate), so this was a harder
bar than pure chance.

## Test 4 (coarse localization — FAILED, wrong direction): position within a full SEA-aligned
sentence span (not just the narrow candidate window)
Spearman correlation between matched relative position and true relative position, n=9
(limited by how many verified words have a same-word template from a *different* clip).
**rho=-0.267, p=0.756** — not significant, and pointed the wrong way. Cross-video-source
framing mismatch (template source vs. SEA-slice source) checked visually and ruled unlikely
as the cause — scale/crop looked consistent between the two sources.

## Manual frame-by-frame check (triggered by direct disagreement: "I can see them mouthing it")
Not a statistical test. Pulled filmstrips of consecutive frames around marked windows for
specific clips. Found the ±0.3s reference window is frequently mistimed relative to the real
mouth articulation — in one clip the visible open-mouth shape occurs *before* the marked
window and has closed by the time the window starts; in another, an even wider region shows
almost no visible mouth movement at all. **This casts real doubt on whether Tests 3/4's
negative results reflect a true absence of positional signal, or reflect systematically
mistimed input windows.** Does not affect Test 2's validity (it already found signal despite
the same imperfect windows).

## Failure-case inspection (two worst Test-2 misses, ranked 20/25 and 21/25)
- `actividad` miss: correct template shows a clear, real, dynamic open-mouth articulation.
  The wrongly-top-ranked match is a **different signer** with a genuinely different but also
  real articulation — looks like cross-signer geometric confusability (generic "wide open
  mouth" shape), not a bug.
- `familia` miss: same signer both times (rules out cross-signer explanation here). The
  wrongly-matched frames show the **interpreter's own hand partially covering her mouth** —
  plausible landmark corruption from occlusion, not a linguistic confound. No occlusion-frame
  detection/filtering exists anywhere in the current pipeline (parallel to the earlier
  interpreter-absence contamination problem, same category of gap: nothing checks "is the
  signal actually clean" before trusting it).

## Value-of-information, if fixed
Current average search window (16 Run-B-aligned hit spans): 15.3s (range 3.6-31.6s); loose
candidate clips are a flat 14s. If a corrected (properly-timed) window achieves ±1s accuracy,
that's a ~7-8x reduction in search space — a real win. At ±3s, only ~2.5x — modest. **This
number has not been measured with corrected windows yet** — Tests 3/4 used the now-known-to-
be-mistimed ±0.3s windows, so we don't know which of these regimes a fixed version would
actually land in.

## Open, unresolved questions
1. Two miss categories are currently indistinguishable in the data: "word was never mouthed"
   (a real linguistic fact — interpreter mouthing is known to be inconsistent) vs. "word was
   mouthed but not detected." No ground truth exists yet for mouthing presence/timing
   separate from sign presence/timing.
2. Corrected-window localization (wider, backward-shifted per the filmstrip finding) has not
   been re-tested — the actual achievable accuracy is unknown, not just unconfirmed.
3. Occlusion detection and cross-signer normalization are unaddressed gaps surfaced by the
   two failure cases, not yet built.
4. n is small throughout (14-28 pairs depending on test, 9 for the coarse-position test) —
   Test 2's result is the most statistically solid; Tests 3/4's negative results carry real
   power caveats on top of the windowing-bug concern.

## Net status
Word-identity discrimination: real, demonstrated signal, not yet shown to transfer to the
easier real-deployment candidate-set size. Temporal localization: no positive evidence yet,
but the negative evidence is compromised by a known input-timing bug not yet corrected and
re-tested. Not a clean pass or fail — an open, partially-diagnosed result.
