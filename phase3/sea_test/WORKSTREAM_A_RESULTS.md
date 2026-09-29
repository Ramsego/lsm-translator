# Workstream A results — correctness fixes

## A1: filter phantom sign segments through `segments.json` — DONE

`filter_segments.py` drops SIGN-tier annotations from the SEA segmenter output whose
midpoint falls outside any hand-presence-validated window. Applied to
`out/segmentation/E4s-1_30_50/57TvyH9902U_slice.eaf` → `out/segmentation_filtered/`.

- Validated coverage: 1,460.8s / 1800s (81%), 19 windows, 18 internal gaps.
- SIGN annotations: 3,633 → 3,546 (**87 dropped, 2.4%**).
- **Sanity check: 87/87 dropped annotations fall inside a real gap between validated
  windows — 100% match, none scattered.** (Initial check used 4 hardcoded buckets from
  earlier manual spot-checks and only matched 33/87; that was the check being too narrow,
  not the fix being wrong — there are 18 gaps in this slice, not 4. Recomputed against
  the actual gap list and it passes exactly.)

**Effect on alignment — measured, not assumed:**
- Binary 26-clip metric, measured with the plain-substring matcher then in use: **byte-for-byte
  identical** between Run A/Run A2 and Run B/Run B2. **Correction (2026-07-31):** this was an
  artifact of that matcher's word-matching bug (see below), not of A1 having zero effect.
  Re-measured with the corrected matcher (`check_alignment_v2.py`, both the original
  stem-heuristic and today's exact accepted-forms map agree): Run A vs A2 differ by exactly
  one instance — `aprovechar` (29m20s) flips HIT→miss. A1's filtering widened the gap around
  that cue, shrinking its DP-assigned span from 12.6s to 9.3s and shifting its center by
  ~1.6s, enough to cross the ±1s tolerance boundary (signed_dist -6.68s→-8.32s). Run B vs B2
  stay tied (73.1%/73.1%) — same mechanism didn't reach a boundary there. So A1 has one real,
  small, mechanistically-explained effect on the binary metric at n=26, not zero; it does not
  change the relative ranking of any run.
- Aggregate shift-vs-lag diagnostic got *noisier*: SD roughly doubled in both runs
  (Run A 5.08→13.81, Run B 4.89→13.49), flipping the auto-verdict to "NOISY" in both.
  Likely mechanism: removing segments widens the gaps around them, which the DP's
  `gap_penalty_weight` term reacts to for cues near those regions — a real, non-obvious
  side effect worth knowing about even though it didn't move the metric that matters.
- Workstream B's continuous metric (paired Wilcoxon on `abs_dist_s`) confirms this
  precisely: **p=0.7855 (Run A vs A2), p=0.5930 (Run B vs B2) — no detectable change.**

**Verdict: kept as implemented.** This was scoped as a correctness fix, not an accuracy
bet, and it's correct regardless — segments generated where the interpreter is provably
absent should not be in the input. It does not currently move the alignment metric, and
that's now measured rather than assumed.

## A2: re-segment ASR cues into real sentences — PREMISE WAS FALSE, no fix built

Before writing any code, checked whether the existing 233 cues are actually
caption-length fragments (the assumed defect). **They are not.** 232/233 cues already
end in terminal punctuation (`.!?`); the one exception is the slice boundary cutting the
ASR transcript off mid-sentence, a real and harmless edge effect, not a bug.
`make_vtt.py` already groups words into sentences via `re.search(r"[.!?]$", ...)`
— exactly the fix A2 was going to build.

This was an error in the spec, not a discovery about the data: I asserted a defect
without checking the actual VTT content first. No new script was written; A3 uses the
existing `subtitles/57TvyH9902U_slice.vtt` unchanged.

## A3: re-run SEA with A1 applied — DONE

`run_sea_v2.sh`: same parameters as `run_sea.sh` in every respect
(`--dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10
--dp_window_size 50 --similarity_measure none`), only `--segmentation_dir` changed to
the filtered output. Subtitles unchanged (A2 was a no-op). Output: `out/aligned_A2/`,
`out/aligned_B2/`.

## Significant side-finding: a word-matching bug in `check_alignment.py` affected every
binary number cited this session

While building Workstream B's continuous checker, found that plain substring matching
(`word.lower() in tx.lower()`) misses Spanish inflection — `joven` doesn't match
`jóvenes` (accent), `aprovechar` doesn't match `aprovecho` (conjugation). This is the
same class of bug found earlier with `actividad`/`actividades`, not fully eliminated.

**The consequence is worse than a skipped instance.** `check_alignment.py`'s covering
check requires text-match AND time-match in one boolean; when text-match silently fails,
the result is indistinguishable from a genuine timing miss — it's printed as "miss"
either way. So the established binary baseline (Run A 14/26, Run B 16/26, naive 15/26,
repeated throughout this session) was silently deflated by unscoreable instances counted
as failures.

Fixed in `check_alignment_v2.py` (accent-stripping + shared-stem prefix match, see B1).
Corrected binary counts, all n=26 (no more silent skips):

| | buggy matching (established this session) | fixed matching |
|---|---|---|
| naive | 15 (57.7%) | **17 (65.4%)** |
| Run A | 14 (53.8%) | **16 (61.5%)** |
| Run B | 16 (61.5%) | **19 (73.1%)** |

**The relative conclusion survives and slightly strengthens:** B's margin over naive
goes from +1 to +2; over Run A from +2 to +3. Nothing qualitative from earlier reasoning
about SEA vs. the naive heuristic needs to be retracted — but the absolute percentages
cited earlier (including "Run B ≈ 62%") should be read as measured-with-a-bug and
superseded by the 73.1% figure here.
