# Route 1 spec: mouth-activity-snapped templates + occlusion mask

## Context (read first)

`MOUTHING_FINDINGS.md` in this directory documents the pilot. Short version: Test 2
(word discrimination via subsequence DTW on lip features) passed; Tests 3/4 (temporal
localization) failed, but the manual filmstrip check found the reference windows were
built from **sign** timing (`sign_start`/`sign_end` ± 0.3s) and the actual **mouthing**
frequently happens earlier or elsewhere. So the localization tests were fed mistimed
templates and prove nothing yet.

This spec fixes that one bug and re-runs the localization tests. Nothing else.

**Goal:** replace "template = sign window ± 0.3s" with "template = detected
mouth-activity burst nearest the sign window," then re-run Tests 2/3/4 and report the
real localization accuracy.

**Hard constraints:**
- **No new manual labeling.** No human marks any timestamp. (Visual sanity checks of
  plots are allowed; transcribing what you see into training/eval data is not.)
- **No threshold tuning on test metrics.** All constants are fixed in this spec
  (§Constants). You may adjust them ONLY during the §3 sanity check, using only the two
  filmstrip clips already inspected in the pilot, and must freeze them before running
  any test in §5–§7. Record any adjustment and its justification in the results doc.
- Do not change the existing 80-dim feature layout in `features2/` and `span_features/`.
  New signals go in sidecar files.
- If an acceptance gate in §4 or §5 fails, STOP and write up the failure instead of
  proceeding. Do not improvise fixes beyond what the gate's "if it fails" note allows.

## Existing data (all paths relative to this directory)

- `features2/*.npy` — 26 verified clips, shape `(T, 80)`, 30 fps. Layout: for each
  landmark index `i` in `LIP_IDX = sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair))`
  (40 indices), columns `2k` = x, `2k+1` = y where `k = LIP_IDX.index(i)`. Coordinates
  are lip-centroid-centered and divided by inter-eye distance (landmarks 33/263). Rows
  are all-NaN when FaceMesh found no face.
- `span_features/*.npy` — 16 Run-B hit spans, same layout.
- `../review_2.csv` — columns `file, word, video_start, verdict, signer, sign_start, sign_end`.
  Verified rows: `verdict == "y" and sign_start != ""`. Times are absolute video seconds;
  clip-relative time = `sign_start - video_start`.
- `runB_hits.json` — span metadata for Test 4.
- Source videos: `/Volumes/Crucial X8/LSM_Translator/review/57TvyH9902U/` (clips) and
  `span_clips/` (spans).
- Existing test code to reuse: `subseq_test.py` (Test 2, has the `subseq_dtw`
  implementation with backtrace), `coarse_position_test.py` (Test 4), `dtw_compare.py`,
  `offset_check.py` (Test 3).

## Constants (fixed up front)

| name | value | meaning |
|---|---|---|
| `FPS` | 30.0 | verify against `cv2.CAP_PROP_FPS` per clip; if a clip differs by >0.5, use its real fps for all frame↔second conversions |
| `SMOOTH_W` | 5 frames (~167 ms) | moving-average window for activity signal |
| `Z_ON` | 1.0 | robust z-score threshold to be "active" |
| `MIN_BURST` | 6 frames (200 ms) | discard shorter bursts |
| `MAX_GAP` | 5 frames | frames below threshold allowed inside a burst |
| `MERGE_GAP` | 10 frames | merge bursts closer than this |
| `OCC_DIST` | 0.9 | hand-to-mouth distance (in inter-eye units) below which a frame is occluded |
| `PAD` | 0.15 s | padding added to snapped template window |
| `SEARCH_PRE` | 2.5 s | how far before `sign_start` to look for the mouthing burst |
| `SEARCH_POST` | 1.0 s | how far after `sign_end` to look |

The search window is asymmetric on purpose: the filmstrip check showed mouthing
*preceding* the marked sign window.

## Step 1 — Sidecar extraction (occlusion + raw signals)

Modify copies of `extract_lips2.py` and `extract_spans.py` (call them
`extract_lips3.py`, `extract_spans3.py`; do not overwrite the originals or their
outputs). Holistic already computes hand landmarks; they just weren't saved. For every
clip/span, alongside the existing features, save a sidecar
`sidecar/<same_basename>.npz` with per-frame arrays, all length T:

- `aperture`: `y(14) - y(13)` in the same normalized space (lower-inner-lip minus
  upper-inner-lip y; both landmarks are in `LIP_IDX`, so this can also be recomputed
  from existing features — column `2*LIP_IDX.index(14)+1` minus
  `2*LIP_IDX.index(13)+1`. Positive = open. Centroid-centering cancels in the
  difference.)
- `hand_dist`: min over all 42 hand landmarks (left+right, when present) of Euclidean
  distance to the mouth centroid `(cx, cy)`, divided by inter-eye distance. `inf` when
  no hand detected, NaN when no face.
- `face_ok`: bool, FaceMesh returned landmarks.

For `features2` clips only (they already exist and re-running Holistic gives identical
lips), you may recompute `aperture` from the saved `.npy` and only run video decoding
for `hand_dist`. Spans need the same treatment. Total video is ~10 min; this is cheap.

## Step 2 — Activity signal and burst detection

One function, applied identically to clips and spans:

1. `occluded[t] = (hand_dist[t] < OCC_DIST) or not face_ok[t]`
2. `d[t] = |aperture[t] - aperture[t-1]|`, with `d[t] = NaN` if either frame is
   occluded/invalid (do NOT set to 0 — an occluded mouth is unknown, not still).
3. `s = moving_average(d, SMOOTH_W)` ignoring NaNs (require ≥3 valid frames in the
   window, else NaN).
4. Robust z-score per clip: `z[t] = (s[t] - median(s)) / (1.4826 * MAD(s))`, computed
   over valid frames only. Per-clip normalization is deliberate — it absorbs
   signer/scale differences.
5. Bursts: maximal runs where `z >= Z_ON`, tolerating internal gaps ≤ `MAX_GAP` frames;
   drop bursts shorter than `MIN_BURST`; merge bursts separated by < `MERGE_GAP`.
   A burst is `(onset_frame, offset_frame)`.

## Step 3 — Sanity check (acceptance gate, before any test)

For the two clips inspected in the pilot's filmstrip check (the ones documented in
MOUTHING_FINDINGS.md §manual check — the clip where visible mouthing precedes the
marked window, and the near-motionless one) plus the `familia` occlusion clip and the
`actividad` miss pair (all identifiable from `inspect_failures.py`), produce one plot
each: aperture trace, z-trace, detected bursts as shaded regions, occluded frames as
hatched regions, and the CSV sign window as vertical lines.

**Pass criteria (by eye, no numbers recorded as data):**
- The clip with visible early mouthing → a burst overlapping where the filmstrip showed
  articulation, i.e. starting before the sign window.
- The near-motionless clip → no burst, or only weak ones, near its window.
- The `familia` occlusion clip → the hand-covering-mouth frames flagged occluded.

If these fail, adjust ONLY `Z_ON`, `SMOOTH_W`, `OCC_DIST` until they pass, log the
change, freeze. If no setting passes, stop: the activity signal itself is inadequate —
write that up.

## Step 4 — Template snapping

For each verified clip (26 rows):

1. Convert sign window to clip frames: `a = (sign_start - video_start) * fps`,
   `b = (sign_end - video_start) * fps`.
2. Search interval: `[a - SEARCH_PRE*fps, b + SEARCH_POST*fps]`, clamped to clip.
3. Choose the burst with **maximum overlap** with the search interval; tie-break by
   proximity of burst onset to `a`. If no burst overlaps, the clip gets
   `mouthing = none`.
4. Template = feature rows of the chosen burst padded ±`PAD`, with occluded rows
   dropped (same NaN-mask handling as existing `valid_slice`).
5. Emit `snapped_windows.csv`: file, word, sign_start, sign_end, burst_on_s, burst_off_s,
   lead_s = sign_start − burst_on_s, n_occluded_frames, mouthing_detected (bool).

**Free deliverable:** the distribution of `lead_s` (does mouthing consistently precede
the sign, and by how much?) and the count of `mouthing = none` clips (first real data
on open question 1: never-mouthed vs mouthed-but-missed).

**Gate:** if more than ~10 of 26 clips get `mouthing = none`, stop and report — either
the detector is too strict or mouthing is rarer than assumed; both change the plan.

## Step 5 — Re-run Test 2 (regression guard)

Re-run `subseq_test.py`'s protocol unchanged except templates come from snapped windows
(clips with `mouthing = none` are excluded as templates but stay in the haystack set;
report the resulting n). Report median rank, top-1, top-3, Wilcoxon p, next to the
pilot's numbers (median rank 3/25, top-1 23.5%, top-3 53%, p=0.0013).

**Gate:** snapped templates must not be materially worse (median rank ≤ 4). If they
are, snapping is selecting the wrong bursts — stop, inspect the worst regressions with
§3-style plots, write up. Do not proceed to Tests 3/4 with a failed gate.

## Step 6 — Re-run Test 3 (fine localization, the headline number)

Protocol as in `offset_check.py` but:

- **Templates:** snapped (Step 4). Use the cross-clip pairing (template from a
  *different* clip of the same word) as the primary condition — that is the deployment
  case. Same-clip is trivially circular; skip it.
- **Prediction:** midpoint (in seconds) of the matched subsequence returned by
  `subseq_dtw` backtrace, on the target's full 14 s feature array with occluded frames
  masked to invalid (reuse the existing `C[:, ~valid] = 1e6` mechanism, extending
  `valid` to exclude occluded frames).
- **Ground truth — report BOTH:**
  - (a) vs the target clip's own snapped mouthing window midpoint. Caveat to state in
    the writeup: truth here comes from the same detector (shared input, different
    algorithm — aperture activity vs shape-trajectory DTW — so agreement is meaningful
    but not fully independent).
  - (b) vs the human-verified sign window midpoint (fully independent truth, but
    includes the mouthing-lead offset; interpret alongside the `lead_s` distribution
    from Step 4).
- **Baselines:** clip midpoint; uniform random (100 draws, report mean).
- **Metrics:** mean and median absolute error (s); fraction of pairs within ±1.0 s,
  ±1.5 s, ±3.0 s; Wilcoxon signed-rank of |err| vs the midpoint baseline. Pilot
  reference: 5.39 s ours vs 4.97 s midpoint, p≈0.65.

## Step 7 — Re-run Test 4 (coarse localization in spans)

As in `coarse_position_test.py` (same n≈9 pairs, same Spearman protocol) but with
snapped templates and occlusion-masked span features (compute sidecars for spans in
Step 1). Additionally report absolute error in seconds against truth (a) and (b) as in
Step 6, since Spearman at n=9 is nearly powerless either way. Pilot reference:
rho = −0.267, p = 0.756.

## Step 8 — Writeup and decision

Produce `ROUTE1_RESULTS.md` in this directory: every table above, the frozen constants
(with any §3 adjustments logged), gates passed/failed, and the decision per this
pre-registered rule using **Test 3, truth (a), cross-clip, median absolute error**:

- **≤ 1.0 s** → localization works; next step is scaling word coverage
  (text→viseme matching for open vocabulary — separate spec).
- **1.0–3.0 s** → marginal; mouthing usable only as a soft prior inside SEA's
  similarity matrix, not as a hard anchor. Park the thread; note it in the benchmark
  writeup as a measured partial result.
- **> 3.0 s** → no useful positional signal even with corrected windows; deprioritize
  mouthing localization. Test 2's discrimination result still stands and is still
  reportable.

Whatever the outcome, the `lead_s` distribution and the mouthed/not-mouthed counts from
Step 4 go in the writeup — they're independently valuable for the corpus paper.

## Explicit non-goals

- No new manual annotation of any timestamp.
- No cross-signer normalization work (known gap, out of scope here).
- No changes to Test 2's protocol beyond template source.
- No expansion of the candidate set or new words.
- No tuning of any constant against Test 2/3/4 results.
