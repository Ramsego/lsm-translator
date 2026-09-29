"""
QC gate 2/5: TEMPORAL — are timestamps internally consistent and alignable?

Checks, cheapest/most-load-bearing first:

  1. segments.json boundaries: monotonic & non-overlapping, start_sec consistent
     with start_frame/proc_fps (proc_fps = fps/every — NOT raw container fps, see
     phase3/build_and_gate_segments.py's docstring for why that distinction bit us
     once already), n_frames consistent with end_frame-start_frame, durations positive.
  2. asr.json word timestamps: monotonic non-decreasing, start<end, no negatives.
     ASR coverage vs video duration — the raw video is usually not mounted, so we
     fall back to max(segment end_sec) as a lower bound and mark the check "partial".
  3. ffprobe fps reconciliation — only runs when the source video file actually
     exists on disk; otherwise SKIPPED per video (this is the common case here).
  4. Completed review CSVs: sign_start < sign_end when both present, both within a
     [0, 30s) sanity window (real clip windows are ~10-15s).

Hard-fail: non-monotonic/overlapping segment timestamps, frame<->time inconsistency,
non-monotonic or inverted ASR word times. Coverage/ffprobe checks are informational.

Run standalone:
    python qc/gate_temporal.py --root phase3/local_drive_mirror
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_HEADER = {"file", "word", "video_start", "verdict", "signer", "sign_start", "sign_end"}
REVIEW_SANITY_MAX_S = 30.0


# --------------------------------------------------------------------------- #
# core, testable functions
# --------------------------------------------------------------------------- #

def check_segment_boundaries(segments: list, proc_fps: float, frame_tol: int = 1):
    """Per-segment + pairwise checks. Returns list of {severity, seg, kind, detail}."""
    issues = []
    time_tol = 1.0 / proc_fps + 1e-6 if proc_fps else 0.05

    for s in segments:
        tag = s.get("file", "<unknown>")
        dur = s["end_sec"] - s["start_sec"]
        if dur <= 0:
            issues.append({"severity": "HARD", "seg": tag, "kind": "non_positive_duration",
                            "detail": f"start_sec={s['start_sec']} end_sec={s['end_sec']}"})

        frame_diff = s["end_frame"] - s["start_frame"]
        if abs(frame_diff - s["n_frames"]) > frame_tol:
            issues.append({"severity": "HARD", "seg": tag, "kind": "frame_count_mismatch",
                            "detail": f"end_frame-start_frame={frame_diff} but n_frames={s['n_frames']}"})

        expected_start = s["start_frame"] / proc_fps if proc_fps else None
        if expected_start is not None and abs(expected_start - s["start_sec"]) > time_tol:
            issues.append({"severity": "HARD", "seg": tag, "kind": "frame_time_mismatch",
                            "detail": f"start_frame/proc_fps={expected_start:.3f} but "
                                      f"start_sec={s['start_sec']} (tol={time_tol:.3f})"})

    ordered = sorted(segments, key=lambda s: s["start_sec"])
    for a, b in zip(ordered, ordered[1:]):
        if b["start_sec"] < a["end_sec"] - 1e-6:
            issues.append({"severity": "HARD", "seg": f"{a.get('file')}/{b.get('file')}",
                            "kind": "overlapping_segments",
                            "detail": f"{a.get('file')} ends {a['end_sec']}, "
                                      f"{b.get('file')} starts {b['start_sec']}"})
    return issues


def check_asr_words(words: list):
    """Monotonic non-decreasing starts, start<end, no negatives. Returns issue list."""
    issues = []
    for i, w in enumerate(words):
        if w["start"] < 0 or w["end"] < 0:
            issues.append({"severity": "HARD", "seg": f"word[{i}]", "kind": "negative_time",
                            "detail": f"{w}"})
        if w["end"] < w["start"]:
            issues.append({"severity": "HARD", "seg": f"word[{i}]", "kind": "inverted_word",
                            "detail": f"start={w['start']} end={w['end']}"})
        if i > 0 and w["start"] < words[i - 1]["start"] - 1e-6:
            issues.append({"severity": "HARD", "seg": f"word[{i}]", "kind": "non_monotonic_words",
                            "detail": f"word[{i}].start={w['start']} < word[{i-1}].start={words[i-1]['start']}"})
    return issues


def asr_coverage(words: list, lower_bound_end_sec):
    """Ratio of last-word-end to a duration reference. Returns (ratio_or_None, note)."""
    if not words:
        return None, "no words"
    last_end = max(w["end"] for w in words)
    if lower_bound_end_sec is None or lower_bound_end_sec <= 0:
        return None, "no duration reference available"
    return last_end / lower_bound_end_sec, "partial: video not mounted, compared against max segment end_sec (lower bound)"


def _parse_rate(rate_str):
    """ffprobe r_frame_rate/avg_frame_rate comes back as 'num/den'."""
    try:
        num, den = rate_str.split("/")
        den = float(den)
        return float(num) / den if den else None
    except (ValueError, AttributeError):
        return None


def ffprobe_reconcile(video_path: Path, tolerance=0.01):
    """Self-contained container sanity check: does the container's OWN nb_frames /
    its OWN frame rate agree with its OWN reported duration? This is independent of
    anything in our segments.json (no assumption about proc_fps or every) — it only
    catches VFR-mislabeled or otherwise internally-inconsistent source video, which
    is one of the accepted risks in DATA_QUALITY.md sec 6 ("VFR video fps trust").
    Returns a dict, or None if ffprobe isn't on PATH, or {"skipped": ...} if the
    container doesn't expose nb_frames without a full decode (not attempted here —
    too slow to run over the corpus; see docstring)."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(video_path)],
            capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout)
        stream = (data.get("streams") or [{}])[0]
        duration = float(data["format"]["duration"])
        nb_frames_raw = stream.get("nb_frames")
        fps = _parse_rate(stream.get("r_frame_rate", ""))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    if not nb_frames_raw or not str(nb_frames_raw).isdigit() or not fps:
        return {"skipped": "nb_frames/r_frame_rate not in container metadata without a full decode"}
    nb_frames = int(nb_frames_raw)
    expected = nb_frames / fps
    ok = abs(duration - expected) <= tolerance * max(duration, expected)
    return {"container_duration_s": duration, "nb_frames": nb_frames, "container_fps": fps,
            "expected_s": round(expected, 3), "ok": ok}


def check_review_timing(row: dict):
    """sign_start/sign_end sanity for one completed-review row. Returns issue list."""
    issues = []
    ss, se = row.get("sign_start", ""), row.get("sign_end", "")
    if ss in ("", None) and se in ("", None):
        return issues
    try:
        ss_f, se_f = float(ss), float(se)
    except (TypeError, ValueError):
        issues.append({"severity": "FLAG", "seg": row.get("file", "?"), "kind": "unparseable_timing",
                        "detail": f"sign_start={ss!r} sign_end={se!r}"})
        return issues
    if ss_f < 0 or se_f < 0:
        issues.append({"severity": "HARD", "seg": row.get("file", "?"), "kind": "negative_review_time",
                        "detail": f"sign_start={ss_f} sign_end={se_f}"})
    if se_f <= ss_f:
        issues.append({"severity": "HARD", "seg": row.get("file", "?"), "kind": "inverted_review_timing",
                        "detail": f"sign_start={ss_f} >= sign_end={se_f}"})
    elif se_f - ss_f >= REVIEW_SANITY_MAX_S:
        issues.append({"severity": "FLAG", "seg": row.get("file", "?"), "kind": "implausible_review_duration",
                        "detail": f"sign_end-sign_start={se_f - ss_f:.1f}s >= {REVIEW_SANITY_MAX_S}s"})
    return issues


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def _iter_segments_json(root: Path):
    """Yield (video_id, path, meta) for every segments.json under root (segments_all/ and arrays_*/)."""
    seen = set()
    for p in sorted((root / "segments_all").glob("*.segments.json")) if (root / "segments_all").is_dir() else []:
        vid = p.name[: -len(".segments.json")]
        if vid in seen:
            continue
        seen.add(vid)
        yield vid, p
    for d in sorted(root.glob("arrays_*")):
        p = d / "segments.json"
        if p.exists():
            vid = d.name[len("arrays_"):]
            if vid in seen:
                continue
            seen.add(vid)
            yield vid, p


def run(root: Path, out_dir: Path) -> dict:
    params = {"root": str(root), "review_sanity_max_s": REVIEW_SANITY_MAX_S}
    if not root.exists():
        return {"gate": "temporal", "status": "SKIPPED", "reason": f"root not found: {root}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    issues = []
    n_checked = 0
    partials = []

    for vid, p in _iter_segments_json(root):
        try:
            meta = json.load(open(p))
        except Exception as e:  # noqa: BLE001
            issues.append({"severity": "HARD", "video": vid, "seg": "", "kind": "unreadable_segments_json",
                            "detail": str(e)})
            continue
        fps, every = meta.get("fps"), meta.get("every", 1)
        proc_fps = (fps / every) if fps else None
        segs = meta.get("segments", [])
        n_checked += len(segs)
        for it in check_segment_boundaries(segs, proc_fps):
            it["video"] = vid
            issues.append(it)

        video_path = Path(meta.get("video", ""))
        if video_path.exists():
            res = ffprobe_reconcile(video_path)
            n_checked += 1
            if res is None:
                issues.append({"severity": "INFO", "video": vid, "seg": "", "kind": "ffprobe_unavailable",
                                "detail": "ffprobe not found on PATH"})
            elif "skipped" in res:
                issues.append({"severity": "INFO", "video": vid, "seg": "", "kind": "ffprobe_skipped",
                                "detail": res["skipped"]})
            elif "error" in res:
                issues.append({"severity": "INFO", "video": vid, "seg": "", "kind": "ffprobe_error",
                                "detail": res["error"]})
            elif not res.get("ok", False):
                issues.append({"severity": "FLAG", "video": vid, "seg": "", "kind": "fps_reconciliation_mismatch",
                                "detail": str(res)})
        else:
            issues.append({"severity": "INFO", "video": vid, "seg": "", "kind": "ffprobe_skipped",
                            "detail": f"video not mounted: {video_path}"})

    asr_dir = root / "asr"
    if asr_dir.is_dir():
        seg_by_vid = {}
        for vid, p in _iter_segments_json(root):
            try:
                seg_by_vid[vid] = json.load(open(p)).get("segments", [])
            except Exception:  # noqa: BLE001
                pass
        for p in sorted(asr_dir.glob("*.asr.json")):
            vid = p.name[: -len(".asr.json")]
            try:
                words = json.load(open(p))["words"]
            except Exception as e:  # noqa: BLE001
                issues.append({"severity": "HARD", "video": vid, "seg": "", "kind": "unreadable_asr_json",
                                "detail": str(e)})
                continue
            n_checked += len(words)
            for it in check_asr_words(words):
                it["video"] = vid
                issues.append(it)
            lower_bound = max((s["end_sec"] for s in seg_by_vid.get(vid, [])), default=None)
            ratio, note = asr_coverage(words, lower_bound)
            partials.append({"video": vid, "coverage_ratio": ratio, "note": note})

    review_paths = sorted(root.rglob("review*.csv"))
    for p in review_paths:
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception as e:  # noqa: BLE001
            issues.append({"severity": "FLAG", "video": p.stem, "seg": "", "kind": "unreadable_review_csv",
                            "detail": str(e)})
            continue
        if not rows or not REVIEW_HEADER.issubset(rows[0].keys()):
            continue  # candidate CSV (different schema), not this gate's concern
        n_checked += len(rows)
        for r in rows:
            for it in check_review_timing(r):
                it["video"] = str(p.relative_to(root))
                issues.append(it)

    details_path = out_dir / "temporal_issues.csv"
    with open(details_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["severity", "video", "seg", "kind", "detail"])
        w.writeheader()
        w.writerows(issues)

    partials_path = out_dir / "temporal_asr_coverage.csv"
    with open(partials_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "coverage_ratio", "note"])
        w.writeheader()
        w.writerows(partials)

    n_hard = sum(1 for i in issues if i["severity"] == "HARD")
    n_flag = sum(1 for i in issues if i["severity"] == "FLAG")
    status = "FAIL" if n_hard else ("FLAG" if n_flag else "PASS")

    return {"gate": "temporal", "status": status, "n_checked": n_checked, "n_failed": n_hard,
            "n_flagged": n_flag, "details_path": str(details_path), "params": params,
            "asr_coverage_path": str(partials_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.root, out_dir)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
