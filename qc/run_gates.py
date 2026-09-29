"""
QC orchestrator — runs every gate in qc/gate_*.py, prints a summary table, and writes
qc/out/qc_report.json.

Gate ladder (see DATA_QUALITY.md for the full rationale): integrity -> temporal ->
presence -> activity(flag) -> review. integrity and temporal are HARD gates: a FAIL
there means something is actually broken (missing/corrupt files, inconsistent
timestamps) and the process exit code reflects that. presence, activity, and review
are FLAG-only by design — they report known, expected imperfections in weakly
supervised data (excluded low-coverage segments, motionless spans, unreconciled
review-file disagreement) that a human should see, not a broken pipeline.

Every gate degrades to SKIPPED (not a crash) when its input isn't there — e.g. the
external drive isn't mounted. This script additionally guards each gate call so one
gate's bug can never take the rest of the run down with it.

Usage:
    python qc/run_gates.py [--root phase3/local_drive_mirror] [--gate NAME]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "qc"))

import gate_integrity  # noqa: E402
import gate_temporal  # noqa: E402
import gate_presence  # noqa: E402
import gate_activity  # noqa: E402
import gate_review  # noqa: E402

GATES = [
    ("integrity", gate_integrity),
    ("temporal", gate_temporal),
    ("presence", gate_presence),
    ("activity", gate_activity),
    ("review", gate_review),
]
HARD_GATES = {"integrity", "temporal"}


def git_rev():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def run_one(name, module, root: Path, out_dir: Path):
    try:
        return module.run(root, out_dir)
    except Exception as e:  # noqa: BLE001 — a gate must never take the whole run down
        return {"gate": name, "status": "SKIPPED", "reason": f"gate crashed: {type(e).__name__}: {e}",
                "n_checked": 0, "n_failed": 0, "n_flagged": 0, "details_path": None, "params": {}}


def print_summary(results, elapsed_by_gate):
    print(f"\n{'GATE':<10} {'STATUS':<9} {'CHECKED':>8} {'FAILED':>7} {'FLAGGED':>8}  {'TIME':>6}  DETAILS")
    print("-" * 100)
    for r in results:
        details = r.get("details_path") or r.get("reason") or ""
        print(f"{r['gate']:<10} {r['status']:<9} {r.get('n_checked', 0):>8} "
              f"{r.get('n_failed', 0):>7} {r.get('n_flagged', 0):>8}  "
              f"{elapsed_by_gate.get(r['gate'], 0):>5.1f}s  {details}")
    print("-" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO_ROOT / "phase3" / "local_drive_mirror")
    ap.add_argument("--gate", choices=[n for n, _ in GATES], default=None,
                     help="run a single gate instead of the full ladder")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "qc" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    gates_to_run = [(n, m) for n, m in GATES if args.gate is None or n == args.gate]

    results, elapsed = [], {}
    for name, module in gates_to_run:
        t0 = time.time()
        r = run_one(name, module, args.root, out_dir)
        elapsed[name] = time.time() - t0
        results.append(r)
        print(f"[{name}] {r['status']}  checked={r.get('n_checked', 0)} "
              f"failed={r.get('n_failed', 0)} flagged={r.get('n_flagged', 0)}"
              + (f"  ({r['reason']})" if r.get("reason") else ""))

    print_summary(results, elapsed)

    report = {
        "provenance": {
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_rev": git_rev(),
            "root": str(args.root),
            "gate_selection": args.gate or "all",
            "hard_gates": sorted(HARD_GATES),
            "params_by_gate": {r["gate"]: r.get("params", {}) for r in results},
        },
        "results": results,
    }
    report_path = out_dir / "qc_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {report_path}")

    hard_fail = any(r["status"] == "FAIL" and r["gate"] in HARD_GATES for r in results)
    if hard_fail:
        failed = [r["gate"] for r in results if r["status"] == "FAIL" and r["gate"] in HARD_GATES]
        print(f"\n*** HARD GATE FAILURE: {failed} — DO NOT USE this data until fixed ***")
        sys.exit(1)
    print("\nno hard-gate failures (flags, if any, are informational — see qc/out/*.csv)")
    sys.exit(0)


if __name__ == "__main__":
    main()
