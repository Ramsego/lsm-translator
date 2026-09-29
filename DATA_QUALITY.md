# Data Quality Protocol

> **What this project is:** see the CANON block at the top of `PIPELINE_MAP.md`. This
> document is the QC layer underneath that deliverable — the corpus + pipeline is only a
> resource worth releasing if its quality claims are runnable checks, not assertions.

Written 2026-08-17 alongside `qc/`, the gate suite this document specifies. Every number
and threshold below is either measured directly from the corpus on disk or produced by
running `python qc/run_gates.py`. Where a number in this doc and a number in the code
disagree, the code is stale and should be fixed — this document is not a spec written in
advance of the data, it is a description of gates built by reading the actual data first.

## 1. Purpose & quality requirements

The corpus this pipeline mines (~21h of *mañanera* interpreter footage, weakly aligned to
an official transcript — see CANON) is only useful downstream if four things are true, and
each is checkable by a runnable test rather than an assertion in a doc:

**The signer is present.** A landmark array is worthless if the interpreter isn't in frame
for it — MediaPipe will happily return NaNs (absence) or, worse, confidently wrong
detections on B-roll that happened to land inside the interpreter's screen box (measured:
football players' arms, a trouser leg, a violin and a child's hand — see
`phase3/sea_test/revalidate_segments.py`'s docstring). Presence has to be measured
per-segment, not assumed corpus-wide.

**Timestamps are correct and alignable to transcript sentences.** The entire weak-
supervision recipe (BSL-1K/BOBSL lineage — see CANON) depends on `(video span, text)`
pairs being genuinely co-temporal. A frame index that doesn't correspond to the second it
claims to, or an ASR word whose timing is scrambled, silently poisons every alignment
built on top of it — and because the alignment stage (SEA) already tolerates some noise by
design, a bad timestamp doesn't fail loudly; it just quietly lowers the ceiling.

**Non-signing spans are excluded or flagged.** Presence (interpreter in frame) and
activity (interpreter actually moving) are different failure modes. A segment can pass a
presence gate and still be the interpreter standing still between bursts, or holding a
static B-roll false-positive. Both need a name and an audit trail, not silent inclusion.

**All of the above is verified by runnable tests, not narrative.** Every claim in this
document that isn't a raw measurement is backed by a test in `qc/test_gates.py` that
fails loudly if the claim stops being true. See `phase3/sea_test/filter_segments.py` for
the house style this suite follows: derive an invariant from the data, check it per item,
fail hard and print `*** DO NOT USE ***` rather than silently degrading.

## 2. The gate ladder

Five gates, run in this order by `qc/run_gates.py`. **integrity** and **temporal** are
HARD gates — a FAIL there is a real defect and the run exits nonzero. **presence**,
**activity**, and **review** are FLAG-only by design: the conditions they report (a
low-coverage segment, a still span, two disagreeing review exports) are known, expected
imperfections of weakly supervised broadcast mining, not process failures. Flagging them
loudly is the point; treating them as fatal would make the gate useless the first time it
ran on real data.

### integrity — does every file that should exist, exist and load?

Walks `--root` (default `phase3/local_drive_mirror`), builds a corpus-wide artifact
manifest (path, sha256, bytes, mtime, kind, and for `.npz`: shape + schema-row guess), and
cross-checks `segments.json` against the files actually on disk in both directions
(orphan `.npz` with no segment referencing it; a segment `file` field pointing at nothing).
File-existence is only checkable for videos with a local `arrays_<vid>/` mirror — for
videos known only via `segments_all/<vid>.segments.json` (arrays live on the external
drive), this check reports SKIPPED rather than guessing.

Also cross-checks the repo-root `videos_manifest.csv` against `segments_all/`: does the
manifest's claimed `n_segs` match what's on disk, and does every `id` look like a real
YouTube ID (11-char `[A-Za-z0-9_-]`)? **Hard-fail:** a locally-verifiable missing file, or
an `.npz` that fails to load or has the wrong `landmarks` shape. **Flag:** format split
(`.npy` referenced where the corpus convention is `.npz`), a stale manifest row, a garbage
manifest row.

### temporal — are timestamps internally consistent and alignable?

Per `segments.json`: boundaries must be non-overlapping and monotonic; `start_sec` must
equal `start_frame / proc_fps` within `1/proc_fps` (where **`proc_fps = fps / every`** —
never raw container fps for an `every > 1` video, the exact bug fixed in parallel in
`phase3/build_and_gate_segments.py`); `n_frames` must equal `end_frame - start_frame`
(measured exactly across every sampled segment in the corpus — no off-by-one observed, but
the check tolerates ±1 frame); durations must be positive.

Per `asr.json`: word times monotonic non-decreasing, `start < end`, no negatives. ASR
coverage is reported against `max(segment end_sec)` as a lower bound when the raw video
isn't mounted (the normal case), and the check is explicitly labeled `"partial: video not
mounted"` rather than pretending to be exact.

**ffprobe fps reconciliation** only runs when the source video file exists on disk (rare —
`local_drive_mirror` doesn't carry full videos, only word clips — but the external drive
happened to be mounted during the 2026-08-17 run, see §6). It is a **self-contained**
container check — `container nb_frames / container r_frame_rate` vs. `container duration`,
1% tolerance — deliberately independent of anything in our own `segments.json`, so it
catches source-video metadata problems (VFR mislabeling) rather than re-testing our own
extraction math.

Completed review CSVs: `sign_start < sign_end` when both present, both non-negative,
duration `< 30s` (real clip windows run ~10-15s; 30s is a sanity ceiling, not a spec).
**Hard-fail:** non-monotonic/overlapping segments, frame↔time inconsistency, inverted or
negative ASR/review timestamps. Coverage ratios and ffprobe results are informational.

### presence — is the interpreter on screen for this segment?

Coverage = fraction of frames with at least one non-NaN landmark in a row-slice, ported
from `scripts/quality_report.py`'s 116-row version to the 124-row schema: hands `0:42`,
pose `42:75`, face `75:124`.

**Corpus-eligibility: duration ≥ 10s AND hand_coverage ≥ 0.6.**

- *10s floor* — `phase3/sea_test/filter_segments.py` measured the segment-duration
  distribution as starkly bimodal (p50=7.2s, p75=81s, p90=437s). B-roll false positives
  are 55% of segment **count** but only 1.17% of total **duration** (14.8 min of 21.12h).
  A duration-weighted audit of 16 windows ≥10s came back 16/16 genuine interpreter
  footage. A 10s floor buys most of the cleanup for ~1% of the data — the same floor this
  gate uses for corpus eligibility.
- *0.6 hand-coverage floor* — real signing segments in the corpus sit ≥0.55, with long
  clean ones ≥0.9 (`57TvyH9902U/segments.json`: 0.55, 0.986, ...). The known-bad seeds in
  `R1QhGx6JdDk.segments.json` sit at 0.286-0.545. 0.6 sits in the gap between them.

This gate never hard-fails the run — it reports an exclusion list (status FLAG when
non-empty), because a segment excluded today may be re-included by a better presence
detector later (e.g. the pose-based re-gate in `revalidate_segments.py` already
supersedes the hand-only gate for some of the corpus). Reports `hours_before`/
`hours_after`, `revalidate_segments.py`-style.

### activity (FLAG-only, new signal) — is the signer actually moving?

Presence only asks "are landmarks detected"; a segment can pass that and still be the
interpreter standing still, or a static false-positive presence didn't catch. Reuses
`phase2/estimate_lag.py`'s motion-energy primitive: per-frame hand motion
(`HAND_ROWS = slice(0, 42)`), resampled to `DT = 0.25s` bins. Per segment:
`fraction_active` (bins at/above threshold) and `longest_still_s`.

**Threshold, calibrated 2026-08-17** against `57TvyH9902U` (the only video with both a
local `.npz` mirror and completed review verdicts — 37/38 `verdict=='y'` rows land inside
a locally-available segment; 35 of those resolve to a bin with real motion data):

| percentile | global bins (n=21789) | known-signing bins (n=35) |
|---|---|---|
| p5  | 0.0031 | 0.0050 |
| p10 | 0.0052 | 0.0058 |
| p25 | 0.0100 | 0.0120 |
| p50 | 0.0166 | 0.0212 |

`ACTIVE_ENERGY_THRESHOLD = 0.005` — at the known-signing 5th percentile, so a "still" call
very rarely discards a real signing bin, while still cutting the bottom of the global
distribution. Flag rule (independent of calibration): `fraction_active < 0.5` or
`longest_still_s > 8`. Never hard-fails; only videos with a local array mirror can be
scored at all (today: `57TvyH9902U`, 40 segments) — everything else reports
not-scoreable, not failed.

### review — are human review CSVs well-formed, and are disagreements surfaced?

`verdict ∈ {y, n, neg, ''}`; non-empty verdict with a blank `signer` is a warn-flag;
`sign_start`/`sign_end` must be both-present-or-both-absent and numeric. Candidate CSVs
(`review_csvs/*.review.csv`) get a lighter header + non-negative `video_start` check.

Where `review.csv` and `review.recovered.csv` coexist for the same batch, this gate
diffs them **without picking a winner** — see §4. Never hard-fails.

## 3. Reconciled constants

| constant | value | used by | justification |
|---|---|---|---|
| corpus-eligibility duration floor | **10s** | gate_presence | bimodal duration distribution, `filter_segments.py` — see §2 |
| corpus-eligibility hand-coverage floor | **0.6** | gate_presence | gap between real-segment (≥0.55) and known-bad (≤0.545) coverage — see §2 |
| activity energy threshold | **0.005** | gate_activity | known-signing p5 on 57TvyH9902U, n=35 — see §2 |
| activity fraction-active floor | **0.5** | gate_activity | default flag rule, independent of calibration |
| activity longest-still ceiling | **8s** | gate_activity | default flag rule, independent of calibration |
| frame-time tolerance | **1/proc_fps** | gate_temporal | one processed-frame's worth of rounding |
| review-timing sanity ceiling | **30s** | gate_temporal | real clip windows are ~10-15s; this is a ceiling, not a target |

**Three legacy floors exist in the repo and are pre-gate, not corpus-eligibility:**
`extract_continuous.py`'s 10-frame minimum-segment-length (extraction-time, prevents
one-frame noise blips from ever becoming a segment) and an older `3.33s` floor seen in
early review-batch tooling (a clip-cutting convenience, not a quality claim). Neither
implies a segment meeting it is corpus-eligible — only the 10s/0.6 pair in this document
does. If you see a 10-frame or 3.33s reference elsewhere in the repo, it predates this
protocol and should not be read as a quality bar.

## 4. Authoritative-file rules

Per video, segment-boundary files have a precedence order as re-gating passes are added:

```
segments.json  <  segments_revalidated.json  <  segments_and_gate.json
(raw extraction)   (pose re-gate, revalidate_    (hand AND pose, build_and_
                     segments.py)                 gate_segments.py)
```

Later files are stricter re-gates of earlier ones and should be preferred when present
for a given video; earlier files are not deleted (see the "never overwrite" rule below) so
you can always compare a re-gate against its baseline. Today only some videos have the
later files — `qc/gate_presence.py` scores whatever is present; it does not assume a
specific one is available corpus-wide.

**`review.csv` vs `review.recovered.csv` (57TvyH9902U): UNRESOLVED, on purpose.** Measured
2026-08-17: 164 rows in common, 34 disagree; verdict counts `a={y:38, n:88, '':38}` vs.
`b={y:31, n:61, '':72}` — `.recovered.csv` has systematically *more* blanks, consistent
with a partial-recovery-from-a-crash story, but that is a hypothesis, not a resolution.
`qc/gate_review.py` reports this diff every run and refuses to merge or pick a winner.
Anything consuming these verdicts today must pick one file explicitly and say which.

**Rule going forward: any new gate or re-gate writes a NEW file, never overwrites.** This
is the `estimate_lag_v2.py` pattern (`lag_estimate.json` is never touched;
`lag_estimate_v2.json` is written alongside it) applied corpus-wide. It is what makes the
provenance chain in §5 possible at all — an overwritten file destroys the ability to audit
what changed and why.

## 5. Provenance requirements

Every artifact this pipeline produces should be traceable to: **what produced it**
(script + git rev), **when**, **with what parameters**, and **what schema it claims**.

**Implemented today**, via `qc/`:
- `qc/out/artifact_manifest.csv` — sha256, byte count, mtime, kind, and (for `.npz`)
  shape + schema-row guess for every file under a QC root.
- `qc/out/qc_report.json` — a provenance block per run: date, `git rev-parse --short
  HEAD`, root, and the exact params (thresholds, tolerances) each gate ran with.
- `revalidate_segments.py` and `estimate_lag_v2.py` already stamp their own output with a
  provenance sub-object (criterion, params, hours_before/after) — this predates `qc/` and
  is the pattern `qc/` generalizes.

**Aspirational, not yet true:** individual `.npz` files carry no internal provenance
(no producer/git-rev/param record inside the array itself) — that only exists at the
manifest level, external to the file. A `.npz` copied out of this repo loses its
provenance trail. Fixing this would mean either a sidecar-per-array or an npz metadata
field written at extraction time; neither exists yet.

## 6. Known-defect ledger

**Producer bugs fixed in parallel with this protocol** (see `qc/test_producers.py`,
written by a concurrent workstream — not this one):
- `build_and_gate_segments.py` was mixing raw container fps with `proc_fps` when
  reconstructing segment times for `every > 1` videos — the exact class of error §2's
  temporal gate is built to catch generically going forward.
- `extract_continuous.py` silently skipped frames on an empty crop patch (a crop
  relocation landing off-frame) rather than recording an explicit absence, which would
  have shifted the timeline for every subsequent frame without any error.
- `align_audio.py` treated any small ffmpeg output as end-of-audio without checking the
  return code — a transient ffmpeg failure mid-file would have silently truncated the
  transcript. No confirmed instance of it firing; fixed because the failure would have
  been invisible if it had.

**Accepted, uncovered risks** (not fixed by this protocol, listed so they're not
mistaken for closed):
- **VFR / container fps trust.** `gate_temporal`'s ffprobe reconciliation is a real check,
  but only runs when the source video is mounted locally — the normal case in this repo
  is that it isn't. When the external drive *was* mounted during the 2026-08-17 run, the
  check immediately found a live instance: `cjpPmeg-Doo`'s container reports
  `nb_frames=135796` at `30fps` (expected duration 4526.5s) against an actual container
  duration of 8499.7s — roughly 1.9x off. **Root-caused 2026-08-17** (per-stream ffprobe):
  the *video stream* is 4521.0s / 135796 frames — internally consistent at 30fps — but the
  *audio stream* runs 8489.7s. The video track is truncated at ~75 min of a ~142-min
  broadcast, almost certainly a partial download/merge. Not VFR, not an fps mislabel; our
  extraction correctly stops at 4520.67s (max segment `end_sec`), but the ASR covers the
  full audio (last word at 8468.4s), so **~66 min of this mañanera has transcript and audio
  but no video** — any Stage-1 alignment for this video must be restricted to t < 4521s,
  and the video is a re-download candidate (same remediation as the earlier ChNt
  re-download). The other 6 mounted videos reconciled cleanly.
- **Crop relocations clipping hands.** The interpreter's screen box moves during a
  broadcast (see `crop_changes` in every `segments.json`); a relocation that clips too
  tight can silently lower coverage without tripping the presence floor if the clipping
  is partial rather than total. Not separately instrumented.
- **Single-annotator review, no inter-annotator agreement.** Every completed review CSV
  in this corpus has one reviewer. IAA is future work, gated on funding to pay a second
  annotator (see the community-involvement stance — no unpaid/tokenistic review).
- **Presence ≠ activity, now partially covered.** `gate_activity` (§2) is new, FLAG-only,
  and calibrated on a single video — it narrows this gap but does not close it corpus-wide,
  since only videos with a local array mirror can be scored.

## 7. How to run

```bash
# full ladder against the local mirror (default root)
python qc/run_gates.py

# a specific root, or a single gate
python qc/run_gates.py --root phase3/local_drive_mirror
python qc/run_gates.py --root phase3/local_drive_mirror --gate presence

# activity-gate calibration (regenerates the numbers in §2)
python qc/gate_activity.py --root phase3/local_drive_mirror --calibrate

# dependency-free test suite (this workstream's qc/test_gates.py, plus
# qc/test_producers.py from the parallel workstream if present)
bash qc/run_tests.sh
```

**Exit codes:** `run_gates.py` exits nonzero **iff** `integrity` or `temporal` reports
FAIL — a real defect, not a known imperfection. `presence`, `activity`, and `review` never
affect the exit code; their findings live in `qc/out/*.csv` and the FLAG status in
`qc/out/qc_report.json` is the signal to go read them, not a build-breaking condition.
`run_tests.sh` aggregates every `qc/test_*.py` file's exit code and exits nonzero if any
failed.

**Where reports land:** `qc/out/qc_report.json` (provenance + per-gate summary),
`qc/out/artifact_manifest.csv`, `qc/out/integrity_issues.csv`,
`qc/out/temporal_issues.csv`, `qc/out/temporal_asr_coverage.csv`,
`qc/out/presence_audit.csv`, `qc/out/activity_audit.csv`, `qc/out/review_issues.csv`,
`qc/out/review_file_diffs.json`. All are gitignored working output, not committed
artifacts — regenerate by running the commands above.
