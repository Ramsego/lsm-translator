#!/usr/bin/env bash
# Runs every qc/test_*.py (dependency-free, no pytest), aggregates exit codes.
# Picks up qc/test_gates.py (this workstream) and qc/test_producers.py (the parallel
# producer-bugfix workstream) if present — tolerates its absence so this script works
# standalone.
set -u
cd "$(dirname "$0")/.."

overall=0
n_run=0
n_failed=0

for f in qc/test_*.py; do
  [ -e "$f" ] || continue
  n_run=$((n_run + 1))
  echo "=== $f ==="
  python3 "$f"
  rc=$?
  if [ $rc -ne 0 ]; then
    n_failed=$((n_failed + 1))
    overall=1
    echo "*** $f FAILED (exit $rc) ***"
  fi
  echo
done

echo "======================================"
if [ "$n_run" -eq 0 ]; then
  echo "no qc/test_*.py files found"
  exit 1
fi
echo "$((n_run - n_failed))/$n_run test files passed"
if [ $overall -ne 0 ]; then
  echo "RESULT: FAIL"
else
  echo "RESULT: PASS"
fi
exit $overall
