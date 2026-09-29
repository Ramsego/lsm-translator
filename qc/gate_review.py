"""
QC gate 5/5: REVIEW VALIDITY (FLAG-only) — are human review CSVs internally
well-formed, and where two versions of the same review exist, is the disagreement
between them surfaced rather than silently resolved?

Two CSV families in this corpus:

  - completed review CSVs (header: file,word,video_start,verdict,signer,sign_start,
    sign_end) — 57TvyH9902U/review.csv, review.recovered.csv, and any probe/review*.csv.
    verdict must be in {y, n, neg, ''}; if verdict is non-empty, signer should be too
    (warn, not hard); sign_start/sign_end must be both-present-or-both-absent and
    numeric.
  - candidate CSVs (header: file,word,video_start,context,verdict,dict_youtube_id,
    ref_clip) under review_csvs/ — lighter check: header matches, video_start is a
    non-negative number.

KNOWN ISSUE THIS GATE IS DESIGNED TO SURFACE, NOT FIX: 57TvyH9902U/review.csv and
review.recovered.csv are two different exports of the same 165-row review batch.
Verdict counts disagree (see DATA_QUALITY.md sec 4): this gate reports the
disagreement per file and per row, and explicitly does NOT pick a winner —
authoritativeness is UNRESOLVED. Treat this as a FLAG, not a bug in the gate.

NEVER hard-fails. Run standalone:
    python qc/gate_review.py --root phase3/local_drive_mirror
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_VERDICTS = {"y", "n", "neg", ""}
COMPLETED_HEADER = {"file", "word", "video_start", "verdict", "signer", "sign_start", "sign_end"}
CANDIDATE_HEADER = {"file", "word", "video_start", "context", "verdict", "dict_youtube_id", "ref_clip"}


# --------------------------------------------------------------------------- #
# core, testable functions
# --------------------------------------------------------------------------- #

def validate_review_row(row: dict):
    """Returns a list of {severity, kind, detail} issues for one completed-review row."""
    issues = []
    verdict = row.get("verdict", "")
    if verdict not in VALID_VERDICTS:
        issues.append({"severity": "FLAG", "kind": "invalid_verdict", "detail": f"verdict={verdict!r}"})
    if verdict and not row.get("signer", "").strip():
        issues.append({"severity": "FLAG", "kind": "verdict_without_signer",
                        "detail": f"verdict={verdict!r} but signer is blank"})

    ss, se = row.get("sign_start", ""), row.get("sign_end", "")
    ss_present, se_present = bool(str(ss).strip()), bool(str(se).strip())
    if ss_present != se_present:
        issues.append({"severity": "FLAG", "kind": "incomplete_timing_pair",
                        "detail": f"sign_start={ss!r} sign_end={se!r}"})
    elif ss_present and se_present:
        try:
            float(ss)
            float(se)
        except ValueError:
            issues.append({"severity": "FLAG", "kind": "unparseable_timing",
                            "detail": f"sign_start={ss!r} sign_end={se!r}"})
    return issues


def validate_candidate_row(row: dict):
    """Lighter check for review_csvs/*.review.csv candidate lists."""
    issues = []
    vs = row.get("video_start", "")
    try:
        if float(vs) < 0:
            issues.append({"severity": "FLAG", "kind": "negative_video_start", "detail": f"video_start={vs!r}"})
    except (TypeError, ValueError):
        issues.append({"severity": "FLAG", "kind": "unparseable_video_start", "detail": f"video_start={vs!r}"})
    return issues


def classify_review_csv(header) -> str:
    h = set(header)
    if COMPLETED_HEADER.issubset(h):
        return "completed"
    if CANDIDATE_HEADER.issubset(h):
        return "candidate"
    return "unknown"


def diff_review_files(path_a: Path, path_b: Path):
    """Per-file verdict counts + row-level disagreement, keyed on `file`. Never picks a winner."""
    rows_a = {r["file"]: r for r in csv.DictReader(open(path_a))}
    rows_b = {r["file"]: r for r in csv.DictReader(open(path_b))}

    def counts(rows):
        from collections import Counter
        return dict(Counter(r.get("verdict", "") for r in rows.values()))

    common = set(rows_a) & set(rows_b)
    disagree = sum(1 for k in common if rows_a[k].get("verdict") != rows_b[k].get("verdict"))
    return {
        "file_a": str(path_a), "file_b": str(path_b),
        "n_rows_a": len(rows_a), "n_rows_b": len(rows_b),
        "verdict_counts_a": counts(rows_a), "verdict_counts_b": counts(rows_b),
        "n_common_rows": len(common), "n_disagreeing_rows": disagree,
        "only_in_a": sorted(set(rows_a) - set(rows_b)), "only_in_b": sorted(set(rows_b) - set(rows_a)),
        "authoritativeness": "UNRESOLVED",
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run(root: Path, out_dir: Path, extra_dirs=None) -> dict:
    """extra_dirs: additional directories to scan for review*.csv, beyond root.
    Defaults to [REPO_ROOT/phase3/probe] (called out explicitly in the spec as a
    fixed extra location, not relative to --root). Pass extra_dirs=[] to scope the
    scan to root only — used by qc/test_gates.py so its fixtures stay hermetic."""
    params = {"root": str(root), "valid_verdicts": sorted(VALID_VERDICTS)}
    if not root.exists():
        return {"gate": "review", "status": "SKIPPED", "reason": f"root not found: {root}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": params}

    issues = []
    n_checked = 0

    if extra_dirs is None:
        probe_dir = REPO_ROOT / "phase3" / "probe"
        extra_dirs = [probe_dir] if probe_dir.is_dir() else []
    search_dirs = [root] + list(extra_dirs)
    seen_paths = set()
    completed_paths = []
    for d in search_dirs:
        for p in sorted(d.rglob("review*.csv")):
            if p.resolve() in seen_paths:
                continue
            seen_paths.add(p.resolve())
            completed_paths.append(p)

    for p in completed_paths:
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception as e:  # noqa: BLE001
            issues.append({"severity": "FLAG", "source": str(p), "row": "", "kind": "unreadable_csv",
                            "detail": str(e)})
            continue
        if not rows:
            continue
        kind = classify_review_csv(rows[0].keys())
        if kind != "completed":
            continue
        n_checked += len(rows)
        for i, r in enumerate(rows):
            for it in validate_review_row(r):
                it["source"] = str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
                it["row"] = i
                issues.append(it)

    # review.csv vs review.recovered.csv (or any *.csv / *.recovered.csv pair sharing a stem)
    diffs = []
    by_dir = {}
    for p in completed_paths:
        by_dir.setdefault(p.parent, []).append(p)
    for d, paths in by_dir.items():
        names = {p.name: p for p in paths}
        if "review.csv" in names and "review.recovered.csv" in names:
            diff = diff_review_files(names["review.csv"], names["review.recovered.csv"])
            diffs.append(diff)
            issues.append({"severity": "FLAG", "source": str(d.relative_to(REPO_ROOT)) if d.is_relative_to(REPO_ROOT) else str(d),
                            "row": "", "kind": "review_files_disagree",
                            "detail": f"{diff['n_disagreeing_rows']}/{diff['n_common_rows']} rows disagree; "
                                      f"a={diff['verdict_counts_a']} b={diff['verdict_counts_b']}; "
                                      f"authoritativeness UNRESOLVED"})

    candidate_dir = root / "review_csvs"
    n_candidates = 0
    if candidate_dir.is_dir():
        for p in sorted(candidate_dir.glob("*.review.csv")):
            try:
                rows = list(csv.DictReader(open(p)))
            except Exception as e:  # noqa: BLE001
                issues.append({"severity": "FLAG", "source": str(p), "row": "", "kind": "unreadable_csv",
                                "detail": str(e)})
                continue
            if not rows:
                continue
            kind = classify_review_csv(rows[0].keys())
            if kind != "candidate":
                issues.append({"severity": "FLAG", "source": str(p), "row": "", "kind": "unexpected_header",
                                "detail": f"header={list(rows[0].keys())}"})
                continue
            n_candidates += len(rows)
            for i, r in enumerate(rows):
                for it in validate_candidate_row(r):
                    it["source"] = str(p)
                    it["row"] = i
                    issues.append(it)
    n_checked += n_candidates

    details_path = out_dir / "review_issues.csv"
    with open(details_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["severity", "source", "row", "kind", "detail"])
        w.writeheader()
        w.writerows(issues)

    diffs_path = out_dir / "review_file_diffs.json"
    with open(diffs_path, "w") as f:
        json.dump(diffs, f, indent=2)

    n_flag = sum(1 for i in issues if i["severity"] == "FLAG")
    status = "FLAG" if n_flag else ("PASS" if n_checked else "SKIPPED")

    return {"gate": "review", "status": status, "n_checked": n_checked, "n_failed": 0,
            "n_flagged": n_flag, "details_path": str(details_path), "params": params,
            "diffs_path": str(diffs_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.root, out_dir)
    print(json.dumps(result, indent=2))
    sys.exit(0)  # review gate never fails the run


if __name__ == "__main__":
    main()
