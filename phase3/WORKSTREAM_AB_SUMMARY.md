# Workstream A & B — implementation summary

Executed `phase3/NEXT_STEPS_SPEC.md` Workstreams A and B (correctness fixes + sharper
eval instrument). Full detail in `phase3/sea_test/WORKSTREAM_A_RESULTS.md` and
`WORKSTREAM_B_RESULTS.md`; this is the compressed version.

## A1 — phantom-segment filtering: correct, measured zero effect

Dropped 87/3,633 SIGN-tier annotations (2.4%) whose midpoint falls outside
hand-presence-validated windows. Verified 87/87 land inside a real gap (18 gaps in this
slice, not the 4 we'd spot-checked earlier by eye — first sanity check used those 4 and
only matched 33/87; recomputed against the actual gap list and it's exact).

Effect on alignment: **near-zero, measured both ways, and reconciled.** The binary metric
was reported byte-identical pre/post-filter, but that was measured with the plain-substring
matcher then in use. Re-measured with the corrected matcher (see word-matching fix below):
Run A vs A2 differ by exactly one instance (`aprovechar`, HIT→miss) — A1 widened the gap
around that cue, shrinking its DP span 12.6s→9.3s and shifting it ~1.6s, enough to cross the
±1s tolerance boundary. Run B vs B2 stay tied. So A1 has one small, mechanistically-explained
effect at n=26, not zero, and it doesn't change any run's ranking. Workstream-B continuous
metric: paired Wilcoxon on `abs_dist_s`, p=0.7855 (Run A) / p=0.5930 (Run B) — still no
detectable change at the distribution level. Side effect worth noting: the shift-vs-lag
diagnostic got noisier (SD ~5→~14, both runs), likely because removing segments widens
neighboring gaps that the DP's gap-penalty term reacts to. Kept anyway — scoped as a
correctness fix (segments where the interpreter is provably absent shouldn't be in the
input), not an accuracy bet, and it isn't one.

## A2 — false premise, no fix built

Spec asserted the 233 subtitle cues were caption-length fragments needing sentence
re-segmentation. Checked before writing code: **232/233 already end in terminal
punctuation**; the one exception is the slice boundary cutting the ASR transcript
mid-sentence (harmless). `make_vtt.py` already does exactly the grouping A2 proposed.
This was an error in the spec — asserted a defect without checking the data — not a
finding. No code written; A3 uses the existing subtitle file unchanged.

## Side-finding that supersedes numbers cited earlier this session

Building B1 surfaced a bug in `check_alignment.py`: plain substring matching
(`word.lower() in tx.lower()`) misses Spanish inflection — `joven`/`jóvenes`,
`aprovechar`/`aprovecho`. Same class as the earlier-found `actividad`/`actividades` bug,
not fully eliminated then. Consequence is worse than a skipped instance: text-match and
time-match are ANDed into one boolean, so a silent text-match failure is indistinguishable
from a genuine timing miss and gets counted as "miss" either way.

Fixed via accent-stripping + shared-stem prefix match. **Update (2026-07-31): the
stem-prefix heuristic is superseded** by an explicit per-word accepted-forms map
(`FORMS` in `check_alignment_v2.py`) — checked against the actual 17 gold word types and
found live on both sides: it still missed stem-changing/irregular conjugations
(`encontrar`→`encuentra`, not present in this transcript but a real gap in the heuristic),
and it over-matched unrelated words sharing a prefix (`director`'s stem also matched
`directamente`, an unrelated adverb — didn't flip a result only because that spurious cue
was ~200s away). `FORMS` holds the exhaustive surface forms each gold word actually takes
across the source subtitles and all four `aligned_*` variants (verified identical across
variants). All 26 matched cues were printed and eyeballed once — no spurious matches, no
skips. Re-running the full table with the exact-forms matcher reproduces every number
below unchanged (binary rates, abs_dist mean/median, iou, span_dur, and the naive-vs-B
Wilcoxon p=0.0888), so the earlier stem heuristic's bias didn't happen to move any number
that's cited durably — but it could have, silently, and the exact map removes that risk
going forward.

Corrected binary counts, n=26, no silent skips:

| | as reported earlier this session | corrected |
|---|---|---|
| naive flat-shift | 15/26 (57.7%) | **17/26 (65.4%)** |
| Run A (bias 0) | 14/26 (53.8%) | **16/26 (61.5%)** |
| Run B (bias 6.33) | 16/26 (61.5%) | **19/26 (73.1%)** |

Relative conclusion survives and strengthens (B's margin over naive: +1→+2; over A:
+2→+3). Absolute percentages cited earlier today, including "Run B ≈ 62%", are
superseded by 73.1%.

## B — continuous metric built and applied

`check_alignment_v2.py`: adds `signed_dist_s`, `abs_dist_s`, `norm_dist`, `iou`,
`span_dur_s` per verified instance, alongside the existing binary. Rationale: power
analysis showed n=26 binary scoring can't reliably detect even a 25-point improvement
(19% power in the best case); continuous metrics recover real power for free on the same
clips.

Full baseline, n=26:

| run | binary | abs_dist median/mean (s) | iou median | span_dur median (s) |
|---|---|---|---|---|
| naive flat-shift | 65.4% | 4.81 / 6.64 | 0.019 | 12.78 |
| Run A | 61.5% | 5.09 / 7.34 | 0.028 | 13.33 |
| **Run B** | **73.1%** | **3.46 / 6.10** | 0.022 | 13.78 |
| Run A2 (filtered) | 57.7% | 5.09 / 7.31 | 0.028 | 13.33 |
| Run B2 (filtered) | 73.1% | 3.46 / 6.05 | 0.022 | 13.33 |

Note: `iou` is ~0.02–0.03 everywhere — spans average ~13s, sign windows are ~0.3–1s, so
even a "hit" never implies a tight span. Expected, now quantified.

**Pre-registered decision rule (B3), applied to naive vs. Run B for continuity:**
median abs_dist 4.81→3.46s, Wilcoxon p=0.0888. Still short of p<0.05 at n=26, but moved
from noise-adjacent to borderline once the matching bug was fixed. Same standing
interpretation as before session-end: real, structurally-motivated signal (DP recovers
per-sentence lag deviation a flat shift structurally cannot), not yet decisive at this n.

## State of the rest of the spec (not started)

- **C (AV-HuBERT mouthing test):** not started. Own eval, doesn't depend on A/B.
- **D1 (per-block lag):** ready, gated on B (now satisfied).
- **D2/D3 (duration rescale / grammar-duration-prior):** closed last session — measured
  signing/speech ratio 0.966, no headroom.
- **E (LSM ordering prior, Stage-2 within-span localization):** ready, ungated. Gloss
  corpus obtained (figshare, CC BY 4.0); order differs from Spanish in 42.2% of pairs
  (Kendall τ=0.342), time-marker fronting 87.2% across 173 distinct frames.
- **Concept:sign cloud:** deliberately deferred — scoped to wait for a real consumer
  (SignCLIP fine-tuning or candidate generation at scale), neither of which is running.
  User asked about timing this session; no change made, flagged as open below.

## Questions for review — resolved 2026-07-31 per Fable's review

1. **Keep A1 — yes, confirmed, and the null was corrected to a small measured effect.**
   Fable: filtering out provably-interpreter-absent segments is correct regardless of
   whether it moves the metric, and a fully-measured null is a finding, not a waste. Also
   flagged an internal inconsistency — the doc claimed "byte-identical" while the table
   showed 61.5%→57.7%. Traced it: "byte-identical" was measured with the pre-fix
   plain-substring matcher; with the corrected matcher, A1 flips exactly one instance
   (`aprovechar`, HIT→miss — A1 widened a gap, shrinking that cue's DP span 12.6s→9.3s and
   shifting it enough to cross the ±1s tolerance line). One paired per-block-lag comparison
   (filtered vs. unfiltered) is now on the list for D1, per Fable's suggestion, since D1
   computes those estimates anyway.
2. **Word-matching fix — was not actually complete; now is.** Fable checked the
   stem-prefix heuristic against the real 17 gold word types and found both a live false
   negative class (stem-changing verbs like `encontrar`→`encuentra`, not present in this
   transcript but a real gap) and a new false-positive class the fix itself introduced
   (loose stems over-matching, e.g. `director`→`directamente`) that would silently inflate
   scores in the flattering direction via the nearest-candidate selection. Recommendation
   (skip the morphology audit, build an explicit per-word forms map instead, twenty
   minutes at n=17) implemented as `FORMS` in `check_alignment_v2.py`. All 26 matched cues
   printed and eyeballed once — no spurious matches, no skips. Full table re-run with the
   exact map: every number is unchanged (binary rates, abs_dist, iou, span_dur, Wilcoxon
   p=0.0888) — the stem heuristic's bias didn't happen to move anything cited durably here,
   but per Fable's point it could have, and now can't.
3. **p=0.0888 stands, per the pre-registered rule, and it's now instrument-verified.**
   Not p<0.05 — no renegotiating the standing interpretation (real structural signal, not
   yet decisive). Per Fable, this number wasn't safe to quote durably until re-run past the
   matcher fix; now re-run against the exact-forms map, it reproduces exactly
   (p=0.08875), so the borderline result is not an artifact of the matcher's known upward
   bias. Fable's broader point stands independently: the highest-leverage move before
   D1/E is growing verified n (26→60-80) via the existing review pipeline, since D1/E will
   be scored against this same 26-clip instrument — that's queued as ongoing background
   work, not a one-sitting task (needs a human watching clips and judging sign timing).
4. **Concept:sign cloud — deferral held, no change made.** Fable: "no consumer yet" is
   still true (D1 is a lag model, E's inputs are already extracted, neither needs the
   cloud); the landmark-schema lesson cuts the same way — building before the consumer's
   interface is known risks a retrofit, and the cloud is cheap to build later. Revisit
   when D1 or E first emits actual pairs.

**Suggested order going in to D1:** forms-map matcher (done, this round) → re-run table +
reconcile byte-identical (done, this round) → grow verified set toward 60-80 in the
background → D1, folding in the filtered/unfiltered per-block-lag comparison from Q1.
