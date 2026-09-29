"""
Dependency-free tests for qc/gate_*.py. Style copied from scripts/test_trim.py:
a check(name, cond) helper, PASS/FAIL prints, nonzero exit on any failure.

Builds one synthetic "dirty" corpus root (tempdir) that embeds one instance of every
case the gate ladder must catch, runs each gate's run() against it, and asserts the
right thing got flagged/failed. Then builds one minimal "clean" root and asserts all
gates come back PASS/SKIPPED with zero failures on it. Also unit-tests a handful of
core functions directly (not via a fixture) where that's the more precise assertion.

Run:
    python qc/test_gates.py
"""
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import gate_integrity  # noqa: E402
import gate_temporal  # noqa: E402
import gate_presence  # noqa: E402
import gate_activity  # noqa: E402
import gate_review  # noqa: E402

N_LANDMARKS = 124


def check(name: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #

def make_landmarks(n_frames, hand_present_frac=0.9, motion=True, seed=0):
    """[n_frames, 124, 3] float32 array with hand rows (0:42) present for a fraction
    of frames. When motion=True present-frame values vary by frame; when False they
    are constant (zero motion energy)."""
    rng = np.random.RandomState(seed)
    arr = np.full((n_frames, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    n_present = int(round(hand_present_frac * n_frames))
    present_frames = sorted(rng.choice(n_frames, size=n_present, replace=False)) if n_present else []
    for i, f in enumerate(present_frames):
        if motion:
            val = 0.1 + 0.01 * i
        else:
            val = 0.5
        arr[f, 0:42, :] = val
    # pose + face always present, doesn't matter for these tests
    arr[:, 42:75, :] = 0.3
    arr[:, 75:124, :] = 0.4
    return arr


def write_npz(path: Path, arr: np.ndarray):
    np.savez(path, landmarks=arr)


def build_dirty_root(tmp: Path):
    """One video ('DIRTY') whose segments.json embeds cases (a),(b),(c),(d),(e),(f),
    plus one clean control segment, plus a review.csv with case (g)."""
    root = tmp / "root"
    arr_dir = root / "arrays_DIRTY"
    arr_dir.mkdir(parents=True)

    fps, every = 10.0, 1
    proc_fps = fps / every

    segments = [
        # (a) overlap pair
        {"file": "seg_a1.npz", "start_frame": 0, "end_frame": 20, "start_sec": 0.0, "end_sec": 2.0,
         "n_frames": 20, "hand_coverage": 0.9},
        {"file": "seg_a2.npz", "start_frame": 15, "end_frame": 35, "start_sec": 1.5, "end_sec": 3.5,
         "n_frames": 20, "hand_coverage": 0.9},
        # (b) frame<->time mismatch: start_frame=40 -> should be 4.0s, recorded as 50.0
        {"file": "seg_b.npz", "start_frame": 40, "end_frame": 60, "start_sec": 50.0, "end_sec": 52.0,
         "n_frames": 20, "hand_coverage": 0.9},
        # (e) low coverage, long enough to otherwise be eligible (dur=20s)
        {"file": "seg_e.npz", "start_frame": 600, "end_frame": 800, "start_sec": 60.0, "end_sec": 80.0,
         "n_frames": 200, "hand_coverage": 0.3},
        # (f) motionless-but-present, long + high coverage, zero motion energy
        {"file": "seg_f.npz", "start_frame": 900, "end_frame": 1100, "start_sec": 90.0, "end_sec": 110.0,
         "n_frames": 200, "hand_coverage": 0.9},
        # clean control: long, high coverage, real motion, correct frame/time math
        {"file": "seg_good.npz", "start_frame": 1200, "end_frame": 1400, "start_sec": 120.0, "end_sec": 140.0,
         "n_frames": 200, "hand_coverage": 0.9},
        # (d) references a file that will NOT be written to disk
        {"file": "seg_missing.npz", "start_frame": 1500, "end_frame": 1520, "start_sec": 150.0, "end_sec": 152.0,
         "n_frames": 20, "hand_coverage": 0.9},
    ]
    meta = {"video": "/nonexistent/DIRTY.mp4", "schema_rows": 124,
            "crop_changes": [{"at_sec": 0.0, "crop": [0, 0, 10, 10]}],
            "scale": 1.0, "every": every, "fps": fps, "segments": segments}
    json.dump(meta, open(arr_dir / "segments.json", "w"))

    write_npz(arr_dir / "seg_a1.npz", make_landmarks(20, 0.9, motion=True, seed=1))
    write_npz(arr_dir / "seg_a2.npz", make_landmarks(20, 0.9, motion=True, seed=2))
    write_npz(arr_dir / "seg_b.npz", make_landmarks(20, 0.9, motion=True, seed=3))
    write_npz(arr_dir / "seg_e.npz", make_landmarks(200, 0.3, motion=True, seed=4))
    write_npz(arr_dir / "seg_f.npz", make_landmarks(200, 0.9, motion=False, seed=5))
    write_npz(arr_dir / "seg_good.npz", make_landmarks(200, 0.9, motion=True, seed=6))
    # (d): seg_missing.npz deliberately NOT written
    # (c): an orphan npz on disk, unreferenced by segments.json
    write_npz(arr_dir / "seg_orphan.npz", make_landmarks(20, 0.9, motion=True, seed=7))

    # asr — well-formed, doesn't need to be part of the dirty story
    asr_dir = root / "asr"
    asr_dir.mkdir()
    words = [{"start": 0.0, "end": 0.5, "word": "hola"}, {"start": 0.6, "end": 1.0, "word": "mundo"}]
    json.dump({"video": "x", "model": "test", "words": words}, open(asr_dir / "DIRTY.asr.json", "w"))

    # (g) review.csv: invalid verdict row + inverted sign_start>sign_end row + one clean row
    review_dir = root / "DIRTY"
    review_dir.mkdir()
    with open(review_dir / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "word", "video_start", "verdict", "signer",
                                          "sign_start", "sign_end"])
        w.writeheader()
        w.writerow({"file": "clean/clean_1.mp4", "word": "hola", "video_start": "1.0", "verdict": "y",
                    "signer": "Tester", "sign_start": "1.0", "sign_end": "2.0"})
        w.writerow({"file": "bad/bad_verdict.mp4", "word": "hola", "video_start": "2.0", "verdict": "maybe",
                    "signer": "Tester", "sign_start": "", "sign_end": ""})
        w.writerow({"file": "bad/bad_timing.mp4", "word": "hola", "video_start": "3.0", "verdict": "y",
                    "signer": "Tester", "sign_start": "5.0", "sign_end": "2.0"})

    return root


def build_clean_root(tmp: Path):
    """One video, one well-formed long/active/high-coverage segment, matching asr,
    a well-formed review.csv row, and a matching videos_manifest.csv row (written to
    the repo root candidates gate_integrity looks at is skipped here — we only assert
    the per-video checks, since the manifest lives outside the fixture)."""
    root = tmp / "clean_root"
    arr_dir = root / "arrays_CLEAN"
    arr_dir.mkdir(parents=True)
    fps, every = 10.0, 1

    segments = [
        {"file": "seg0.npz", "start_frame": 0, "end_frame": 200, "start_sec": 0.0, "end_sec": 20.0,
         "n_frames": 200, "hand_coverage": 0.9},
    ]
    meta = {"video": "/nonexistent/CLEAN.mp4", "schema_rows": 124,
            "crop_changes": [{"at_sec": 0.0, "crop": [0, 0, 10, 10]}],
            "scale": 1.0, "every": every, "fps": fps, "segments": segments}
    json.dump(meta, open(arr_dir / "segments.json", "w"))
    write_npz(arr_dir / "seg0.npz", make_landmarks(200, 0.9, motion=True, seed=42))

    asr_dir = root / "asr"
    asr_dir.mkdir()
    words = [{"start": 0.0, "end": 0.5, "word": "hola"}, {"start": 0.6, "end": 1.0, "word": "mundo"}]
    json.dump({"video": "x", "model": "test", "words": words}, open(asr_dir / "CLEAN.asr.json", "w"))

    review_dir = root / "CLEAN"
    review_dir.mkdir()
    with open(review_dir / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "word", "video_start", "verdict", "signer",
                                          "sign_start", "sign_end"])
        w.writeheader()
        w.writerow({"file": "clean/clean_1.mp4", "word": "hola", "video_start": "1.0", "verdict": "y",
                    "signer": "Tester", "sign_start": "1.0", "sign_end": "2.0"})

    return root


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_integrity_dirty(root, out_dir):
    r = gate_integrity.run(root, out_dir)
    issues = list(csv.DictReader(open(r["details_path"])))
    kinds = {i["kind"] for i in issues}
    ok = True
    ok &= check("integrity: dirty fixture status is FAIL", r["status"] == "FAIL")
    ok &= check("integrity: (c) orphan npz detected", "orphan_npz" in kinds)
    ok &= check("integrity: (d) missing referenced file detected", "missing_referenced_file" in kinds)
    return ok


def test_integrity_clean(root, out_dir):
    r = gate_integrity.run(root, out_dir)
    return check("integrity: clean fixture has zero hard failures", r["n_failed"] == 0)


def test_temporal_dirty(root, out_dir):
    r = gate_temporal.run(root, out_dir)
    issues = list(csv.DictReader(open(r["details_path"])))
    kinds = {i["kind"] for i in issues}
    ok = True
    ok &= check("temporal: dirty fixture status is FAIL", r["status"] == "FAIL")
    ok &= check("temporal: (a) overlapping segments detected", "overlapping_segments" in kinds)
    ok &= check("temporal: (b) frame/time mismatch detected", "frame_time_mismatch" in kinds)
    ok &= check("temporal: (g) inverted review timing detected", "inverted_review_timing" in kinds)
    return ok


def test_temporal_clean(root, out_dir):
    r = gate_temporal.run(root, out_dir)
    return check("temporal: clean fixture status is PASS", r["status"] == "PASS" and r["n_failed"] == 0)


def test_presence_dirty(root, out_dir):
    r = gate_presence.run(root, out_dir)
    rows = list(csv.DictReader(open(r["details_path"])))
    excluded = {row["seg"] for row in rows if "excluded" in row["flags"]}
    ok = True
    ok &= check("presence: dirty fixture status is FLAG", r["status"] == "FLAG")
    ok &= check("presence: (e) low-coverage segment excluded", "seg_e.npz" in excluded)
    ok &= check("presence: clean control segment NOT excluded", "seg_good.npz" not in excluded)
    return ok


def test_presence_clean(root, out_dir):
    r = gate_presence.run(root, out_dir)
    return check("presence: clean fixture status is PASS", r["status"] == "PASS")


def test_activity_dirty(root, out_dir):
    r = gate_activity.run(root, out_dir)
    rows = list(csv.DictReader(open(r["details_path"])))
    flagged = {row["seg"] for row in rows if row["flag"]}
    ok = True
    ok &= check("activity: dirty fixture status is FLAG", r["status"] == "FLAG")
    ok &= check("activity: (f) motionless segment flagged", "seg_f.npz" in flagged)
    ok &= check("activity: clean control segment NOT flagged", "seg_good.npz" not in flagged)
    return ok


def test_activity_clean(root, out_dir):
    r = gate_activity.run(root, out_dir)
    return check("activity: clean fixture status is PASS", r["status"] == "PASS")


def test_review_dirty(root, out_dir):
    r = gate_review.run(root, out_dir, extra_dirs=[])
    issues = list(csv.DictReader(open(r["details_path"])))
    kinds = {i["kind"] for i in issues}
    ok = True
    ok &= check("review: dirty fixture status is FLAG", r["status"] == "FLAG")
    ok &= check("review: (g) invalid verdict detected", "invalid_verdict" in kinds)
    return ok


def test_review_clean(root, out_dir):
    r = gate_review.run(root, out_dir, extra_dirs=[])
    return check("review: clean fixture status is PASS", r["status"] == "PASS")


# --------------------------------------------------------------------------- #
# direct core-function unit tests
# --------------------------------------------------------------------------- #

def test_check_segment_boundaries_unit():
    good = [{"file": "a", "start_frame": 0, "end_frame": 10, "start_sec": 0.0, "end_sec": 1.0, "n_frames": 10}]
    bad_overlap = good + [{"file": "b", "start_frame": 5, "end_frame": 20, "start_sec": 0.5, "end_sec": 2.0,
                            "n_frames": 15}]
    ok = True
    ok &= check("check_segment_boundaries: clean segment has no issues",
                len(gate_temporal.check_segment_boundaries(good, proc_fps=10.0)) == 0)
    ok &= check("check_segment_boundaries: overlap flagged",
                any(i["kind"] == "overlapping_segments"
                    for i in gate_temporal.check_segment_boundaries(bad_overlap, proc_fps=10.0)))
    return ok


def test_check_asr_words_unit():
    good = [{"start": 0.0, "end": 0.5}, {"start": 0.6, "end": 1.0}]
    bad = [{"start": 0.0, "end": 0.5}, {"start": -0.1, "end": 0.2}, {"start": 0.1, "end": 0.05}]
    ok = True
    ok &= check("check_asr_words: clean words have no issues", len(gate_temporal.check_asr_words(good)) == 0)
    issues = gate_temporal.check_asr_words(bad)
    kinds = {i["kind"] for i in issues}
    ok &= check("check_asr_words: negative time flagged", "negative_time" in kinds)
    ok &= check("check_asr_words: inverted word flagged", "inverted_word" in kinds)
    return ok


def test_activity_metrics_unit():
    bins_active = np.array([0.01] * 10)
    bins_still = np.array([0.0001] * 10)
    frac_a, still_a = gate_activity.activity_metrics(bins_active)
    frac_s, still_s = gate_activity.activity_metrics(bins_still)
    ok = True
    ok &= check("activity_metrics: fully active bins -> fraction_active==1", frac_a == 1.0)
    ok &= check("activity_metrics: fully still bins flagged", gate_activity.is_flagged(frac_s, still_s))
    ok &= check("activity_metrics: fully active bins not flagged", not gate_activity.is_flagged(frac_a, still_a))
    return ok


def test_diff_review_files_unit(tmp: Path):
    d = tmp / "diffcheck"
    d.mkdir()
    header = ["file", "word", "video_start", "verdict", "signer", "sign_start", "sign_end"]
    with open(d / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow({"file": "a.mp4", "word": "x", "video_start": "1", "verdict": "y",
                    "signer": "S", "sign_start": "1", "sign_end": "2"})
        w.writerow({"file": "b.mp4", "word": "x", "video_start": "2", "verdict": "n",
                    "signer": "S", "sign_start": "", "sign_end": ""})
    with open(d / "review.recovered.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow({"file": "a.mp4", "word": "x", "video_start": "1", "verdict": "n",
                    "signer": "S", "sign_start": "", "sign_end": ""})
        w.writerow({"file": "b.mp4", "word": "x", "video_start": "2", "verdict": "n",
                    "signer": "S", "sign_start": "", "sign_end": ""})
    diff = gate_review.diff_review_files(d / "review.csv", d / "review.recovered.csv")
    ok = True
    ok &= check("diff_review_files: detects exactly one disagreeing row", diff["n_disagreeing_rows"] == 1)
    ok &= check("diff_review_files: authoritativeness stays UNRESOLVED", diff["authoritativeness"] == "UNRESOLVED")
    return ok


def main():
    tmp = Path(tempfile.mkdtemp(prefix="qc_test_"))
    out_dir = tmp / "out"
    out_dir.mkdir()
    try:
        dirty_root = build_dirty_root(tmp)
        clean_root = build_clean_root(tmp)

        out_subdirs = {}
        for name in ["dirty_integrity", "clean_integrity", "dirty_temporal", "clean_temporal",
                     "dirty_presence", "clean_presence", "dirty_activity", "clean_activity",
                     "dirty_review", "clean_review"]:
            p = out_dir / name
            p.mkdir(exist_ok=True)
            out_subdirs[name] = p

        results = [
            test_integrity_dirty(dirty_root, out_subdirs["dirty_integrity"]),
            test_integrity_clean(clean_root, out_subdirs["clean_integrity"]),
            test_temporal_dirty(dirty_root, out_subdirs["dirty_temporal"]),
            test_temporal_clean(clean_root, out_subdirs["clean_temporal"]),
            test_presence_dirty(dirty_root, out_subdirs["dirty_presence"]),
            test_presence_clean(clean_root, out_subdirs["clean_presence"]),
            test_activity_dirty(dirty_root, out_subdirs["dirty_activity"]),
            test_activity_clean(clean_root, out_subdirs["clean_activity"]),
            test_review_dirty(dirty_root, out_subdirs["dirty_review"]),
            test_review_clean(clean_root, out_subdirs["clean_review"]),
            test_check_segment_boundaries_unit(),
            test_check_asr_words_unit(),
            test_activity_metrics_unit(),
            test_diff_review_files_unit(tmp),
        ]

        total = len(results)
        passed = sum(bool(r) for r in results)
        print(f"\n{passed}/{total} tests passed")
        if passed < total:
            sys.exit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
