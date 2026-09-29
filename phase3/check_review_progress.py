#!/usr/bin/env python3
"""Read-only ground truth for review progress -- no server, no browser.

Reads review.csv (and review.recovered.csv as a fallback) directly off disk.
Run this whenever the browser's displayed clip number looks wrong, to check
what's actually saved independent of any tab, cache, or localStorage state.

Usage:  python3 check_review_progress.py <clip_dir>
"""

import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_server import load_state, rows_from_html


def main():
    clip_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
    rows = rows_from_html(clip_dir)
    state = load_state(clip_dir)
    reviewed = sum(1 for s in state.values() if s.get("verdict"))
    next_idx = next(
        (i for i, r in enumerate(rows, 1)
         if not state.get(r["file"], {}).get("verdict")),
        len(rows) + 1,
    )
    print("clip dir : %s" % clip_dir)
    print("on disk  : %d / %d reviewed" % (reviewed, len(rows)))
    if next_idx <= len(rows):
        print("next     : #%d  (%s)" % (next_idx, rows[next_idx - 1]["word"]))
    else:
        print("next     : none -- all clips reviewed")
    missing_signer = [
        r["file"] for r in rows
        if state.get(r["file"], {}).get("verdict") and not state[r["file"]].get("signer")
    ]
    if missing_signer:
        print("warning  : %d reviewed clip(s) have no signer set" % len(missing_signer))


if __name__ == "__main__":
    main()
