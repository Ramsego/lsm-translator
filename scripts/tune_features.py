"""
Feature-tuning sweep for the DTW spotter.

Evaluates several featurization variants on the meaningful query subset
(multi-example + cross-source clips) against the full 963-clip bank, and prints a
table ranked by cross-source top-5 — the metric that predicts cross-signer use.
Much faster than full leave-one-out (~1/6 the queries).

Usage:
    python scripts/tune_features.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
clf = __import__("05_classify")

# Curated variants (not full cartesian). Each is a partial cfg over DEFAULT_CFG.
VARIANTS = {
    "baseline (pos, per_frame, hands)":      {},
    "per_clip norm":                          {"norm": "per_clip"},
    "vel_only":                               {"vel_only": True},
    "pos + velocity":                         {"velocity": True},
    "vel_only + zscore":                      {"vel_only": True, "zscore": True},
    "pos + pose":                             {"include_pose": True},
    "vel_only + pose":                        {"vel_only": True, "include_pose": True},
    "vel_only + pose + per_clip":             {"vel_only": True, "include_pose": True, "norm": "per_clip"},
    "vel_only + zscore + pose":               {"vel_only": True, "zscore": True, "include_pose": True},
}


def main():
    # Determine the query subset once (labels with >1 example, or present in both sources).
    import sqlite3
    conn = sqlite3.connect(clf.DB_PATH)
    rows = conn.execute("SELECT youtube_id, label, source FROM videos").fetchall()
    conn.close()
    label_counts = defaultdict(int)
    src_by_label = defaultdict(set)
    for yid, label, source in rows:
        label_counts[label] += 1
        src_by_label[label].add(source)
    subset_labels = {l for l in label_counts
                     if label_counts[l] > 1 or len(src_by_label[l]) > 1}

    def is_query(ref):
        return ref["label"] in subset_labels

    print(f"Query subset: {sum(1 for _,l,_ in rows if l in subset_labels)} clips "
          f"(labels with >1 example or in both sources); refs = full bank.\n")
    print(f"{'variant':40s} {'cross t1':>9} {'cross t5':>9} {'multi t5':>9}  {'sec':>5}")
    print("-" * 80)

    results = []
    for name, partial in VARIANTS.items():
        cfg = {**clf.DEFAULT_CFG, **partial}
        t0 = time.time()
        bank = clf.load_bank(cfg)
        r = clf.evaluate(bank, query_filter=is_query)
        dt = time.time() - t0
        ct1, ct5 = clf._pct(r["cross"], "top1"), clf._pct(r["cross"], "top5")
        mt5 = clf._pct(r["multi"], "top5")
        results.append((ct5, ct1, mt5, name))
        print(f"{name:40s} {ct1:8.1f}% {ct5:8.1f}% {mt5:8.1f}%  {dt:5.0f}")

    results.sort(reverse=True)
    print("\nRanked by cross-source top-5:")
    for ct5, ct1, mt5, name in results:
        print(f"  {ct5:6.1f}%  {name}")
    print(f"\nBest: {results[0][3]}")


if __name__ == "__main__":
    main()
