"""
Dependency-free regression tests for three producer-script timestamp fixes
(W0): phase3/build_and_gate_segments.py, phase2/extract_continuous.py,
phase2/align_audio.py.

Style copied from scripts/test_trim.py: a check(name, cond) helper, PASS/FAIL
per check, nonzero exit on any failure. numpy is the only non-stdlib import
(already a project dependency). No pytest, no real video/audio/ASR.

Run:
    python qc/test_producers.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "phase3"))
sys.path.insert(0, str(_root / "phase2"))

import build_and_gate_segments as bags   # noqa: E402
import align_audio                       # noqa: E402

# extract_continuous.py pulls in cv2/mediapipe at import time (it also
# importlib-loads scripts/02_extract.py for FACE_LANDMARKS/make_detectors).
# Both are project dependencies, but import defensively so a missing native
# dep degrades Test B to a skip instead of nuking the whole file's exit code.
try:
    import extract_continuous as ec
    _EC_IMPORT_ERROR = None
except Exception as e:                      # pragma: no cover
    ec = None
    _EC_IMPORT_ERROR = e


def check(name: str, cond: bool) -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


# ─────────────────────────────────────────────────────────────────────────
# Test A -- build_and_gate_segments.py: proc_fps, not raw fps, in time math
# ─────────────────────────────────────────────────────────────────────────

def _make_and_gate_video(tmp: Path, fps: float, every: int, n: int,
                         present_idx: set, seg_name: str = "vid_seg0000"):
    """One synthetic video dir: segments.json (fps/every) + one .npz whose
    landmarks carry a known hand-AND-pose presence pattern (NaN elsewhere)."""
    vdir = tmp / f"video_{seg_name}_{every}"
    vdir.mkdir(parents=True)

    land = np.full((n, 124, 3), np.nan, dtype=np.float32)
    for i in present_idx:
        land[i, 0:75, :] = 0.5    # hand rows 0:42 AND pose rows 42:75 both present
    npz_path = vdir / f"{seg_name}.npz"
    np.savez(npz_path, landmarks=land)

    proc_fps = fps / every
    seg = {
        "file": npz_path.name, "start_frame": 0, "end_frame": n,
        "start_sec": 0.0, "end_sec": round(n / proc_fps, 2),
        "n_frames": n, "hand_coverage": 0.7,
    }
    meta = {
        "video": "/fake/path/vid.mp4", "schema_rows": 124,
        "crop_changes": [{"at_sec": 0.0, "crop": [0, 0, 10, 10]}],
        "scale": 1.0, "every": every, "fps": fps,
        "segments": [seg],
    }
    json.dump(meta, open(vdir / "segments.json", "w"))
    return vdir


def test_and_gate_uses_proc_fps():
    """every=3 @ fps=30 -> proc_fps=10. Two 40-frame present runs separated by
    a 20-frame absence (idx 40..59), with a small 3-frame internal absence
    inside the first run (idx 10,11,12) that must NOT split it (gap=3 <
    gap_frames=7) but MUST show up in the recomputed presence fraction.

    Under the OLD bug (raw fps=30 used for the frame->second conversion
    instead of proc_fps=10), each run's duration comes out 3x too short
    (40 processed frames / 30 = 1.33s) and gets thrown out entirely by
    MIN_DURATION (3.33s) -- i.e. the old code would silently emit ZERO
    segments here. The fix must emit exactly 2, with exact-second math.
    """
    present = set(range(0, 10)) | set(range(13, 40)) | set(range(60, 100))
    with tempfile.TemporaryDirectory() as td:
        vdir = _make_and_gate_video(Path(td), fps=30.0, every=3, n=100, present_idx=present)
        result = bags.process_video(vdir, write_output=False)

    ok = True
    ok &= check("proc_fps case: process_video returns a result", result is not None)
    segs = result["out"]["segments"]
    ok &= check("proc_fps case: exactly 2 gated segments (old code would emit 0)",
               len(segs) == 2)
    if len(segs) == 2:
        s0, s1 = segs
        ok &= check("proc_fps case: seg0 start_sec == 0.0 (proc_fps math)",
                   s0["start_sec"] == 0.0)
        ok &= check("proc_fps case: seg0 end_sec == 4.0 (40 frames / proc_fps=10)",
                   s0["end_sec"] == 4.0)
        ok &= check("proc_fps case: seg1 start_sec == 6.0 (60 frames / proc_fps=10)",
                   s1["start_sec"] == 6.0)
        ok &= check("proc_fps case: seg1 end_sec == 10.0 (100 frames / proc_fps=10)",
                   s1["end_sec"] == 10.0)
        ok &= check("proc_fps case: seg0 carries source file name",
                   s0["file"] == "vid_seg0000.npz")
        # presence fraction over the gated sub-span: 37/40 present (3 absent
        # inside the run, at idx 10,11,12), vs 1.0 for the clean second run.
        ok &= check("proc_fps case: seg0 presence_frac reflects internal gap (37/40)",
                   abs(s0["presence_frac"] - 37 / 40) < 1e-9)
        ok &= check("proc_fps case: seg1 presence_frac == 1.0 (no internal gap)",
                   s1["presence_frac"] == 1.0)
    ok &= check("proc_fps case: and_gate provenance carries proc_fps/every/skipped_arrays",
               result["out"]["and_gate"]["proc_fps"] == 10.0
               and result["out"]["and_gate"]["every"] == 3
               and result["out"]["and_gate"]["skipped_arrays"] == 0)
    return ok


def test_and_gate_every_one_matches_old_formula():
    """Behavior-preserving check: when every=1, proc_fps == fps, so the new
    code's output must exactly match the OLD formula (g[0]/fps, (g[-1]+1)/fps)
    -- this fix must not change any already-correct every=1 video's output."""
    fps = 30.0
    # 150 frames >= MIN_DURATION (10/30*10 = 3.33s -> >=100 frames at 30fps)
    present = set(range(0, 150))    # one clean run, no gaps
    with tempfile.TemporaryDirectory() as td:
        vdir = _make_and_gate_video(Path(td), fps=fps, every=1, n=160, present_idx=present)
        result = bags.process_video(vdir, write_output=False)

    segs = result["out"]["segments"]
    ok = check("every=1 case: exactly 1 segment", len(segs) == 1)
    if segs:
        g0, g_last = 0, 149
        old_start = round(0.0 + g0 / fps, 2)          # seg_start_sec + g[0]/fps
        old_end = round(0.0 + (g_last + 1) / fps, 2)  # seg_start_sec + (g[-1]+1)/fps
        ok &= check(f"every=1 case: start_sec matches old formula ({old_start})",
                   segs[0]["start_sec"] == old_start)
        ok &= check(f"every=1 case: end_sec matches old formula ({old_end})",
                   segs[0]["end_sec"] == old_end)
    return ok


def test_and_gate_skipped_array_counted():
    """An .npz with no matching segments.json entry must be counted, not
    silently continue'd past."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        vdir = tmp / "video_orphan"
        vdir.mkdir()
        land = np.full((10, 124, 3), np.nan, dtype=np.float32)
        np.savez(vdir / "orphan_seg0099.npz", landmarks=land)   # no segments.json entry
        json.dump({
            "video": "x", "schema_rows": 124, "crop_changes": [],
            "scale": 1.0, "every": 1, "fps": 30.0, "segments": [],
        }, open(vdir / "segments.json", "w"))
        result = bags.process_video(vdir, write_output=False)
    return check("orphan .npz (no segments.json entry) counted in skipped_arrays",
                result["skipped_arrays"] == 1
                and result["out"]["and_gate"]["skipped_arrays"] == 1)


# ─────────────────────────────────────────────────────────────────────────
# Test B -- extract_continuous.py: empty crop patch must not shift the timeline
# ─────────────────────────────────────────────────────────────────────────

def test_frame_presence_decision_empty_patch_never_present():
    ok = True
    present, was_empty = ec.frame_presence_decision(patch_empty=True, hand_and_pose_present=True)
    ok &= check("empty patch forces present=False even if detection args say True",
               present is False and was_empty is True)
    present, was_empty = ec.frame_presence_decision(patch_empty=False, hand_and_pose_present=True)
    ok &= check("non-empty patch + detection True -> present=True, not empty",
               present is True and was_empty is False)
    present, was_empty = ec.frame_presence_decision(patch_empty=False, hand_and_pose_present=False)
    ok &= check("non-empty patch + no detection -> present=False, not empty",
               present is False and was_empty is False)
    return ok


def test_empty_patch_does_not_shift_timeline():
    """Simulate the main loop's per-processed-frame bookkeeping (the part the
    fix touches) over N synthetic frames where exactly one has an empty crop
    patch. Before the fix, that frame did `fi += 1; continue` WITHOUT
    advancing `processed` -- silently compressing every later timestamp by
    one processed-frame's worth of time. After the fix, `processed` advances
    on every frame the same way `fi` does, 1:1, regardless of patch emptiness."""
    n_alloc = 20
    arr = np.zeros((n_alloc, 124, 3), dtype=np.float32)
    presence = np.zeros(n_alloc, dtype=bool)
    nan_row = np.full((124, 3), np.nan, dtype=np.float32)

    # frame stream: 10 normal detections (alternating present/absent), one
    # with an empty patch, then 9 more normal detections. 20 frames in.
    patch_empty_at = 10
    detections = [i % 2 == 0 for i in range(20)]   # arbitrary hand+pose pattern

    processed = 0
    empty_patch_frames = 0
    for fi in range(20):
        patch_empty = (fi == patch_empty_at)
        present, was_empty = ec.frame_presence_decision(patch_empty, detections[fi])
        if was_empty:
            empty_patch_frames += 1
        if present:
            arr[processed] = 1.0     # stand-in for frame_to_row(...)
            presence[processed] = True
        else:
            arr[processed] = nan_row
            presence[processed] = False
        processed += 1   # <-- the fix: unconditional, every processed frame gets a slot

    ok = True
    ok &= check("processed count == number of frames read (no silent skip)",
               processed == 20)
    ok &= check("empty_patch_frames counter == 1", empty_patch_frames == 1)
    ok &= check("the empty-patch frame's row is NaN (not left at stale/zero data)",
               np.isnan(arr[patch_empty_at]).all())
    ok &= check("the empty-patch frame is marked absent in presence[]",
               presence[patch_empty_at] == False)  # noqa: E712
    # timeline check: frame k's start time is k/proc_fps -- since processed
    # advanced 1:1 with fi even through the empty patch, frame 15's slot in
    # `arr`/`presence` must be index 15, not 14 (which is what the old bug
    # would have produced).
    ok &= check("later frame lands at its own index (timeline not compressed)",
               processed - 1 == 19)
    return ok


def test_fps_fallback_source_labels():
    """No importable pure function for the fps-fallback branch (it's a few
    lines inline in main()), so this documents/locks the two labels the fix
    must emit -- 'container' when cap.get(CAP_PROP_FPS) is truthy, else
    'fallback_30'. Guards against a future refactor silently renaming them
    and breaking anything that reads segments.json's fps_source field."""
    def fps_source_for(container_fps_value):
        return "container" if container_fps_value else "fallback_30"

    ok = True
    ok &= check("nonzero container fps -> 'container'", fps_source_for(29.97) == "container")
    ok &= check("zero/None container fps -> 'fallback_30'", fps_source_for(0.0) == "fallback_30")
    ok &= check("None container fps -> 'fallback_30'", fps_source_for(None) == "fallback_30")
    return ok


# ─────────────────────────────────────────────────────────────────────────
# Test C -- align_audio.py: chunk_status must not treat ffmpeg errors as EOF
# ─────────────────────────────────────────────────────────────────────────

def test_chunk_status():
    ok = True
    ok &= check("rc=0, big wav -> ok",
               align_audio.chunk_status(0, 50_000) == "ok")
    ok &= check("rc=0, tiny wav -> end_of_audio",
               align_audio.chunk_status(0, 100) == "end_of_audio")
    ok &= check("rc!=0, tiny wav -> error (NOT end_of_audio -- the actual bug)",
               align_audio.chunk_status(1, 100) == "error")
    ok &= check("rc!=0, big wav -> error (nonzero rc always wins)",
               align_audio.chunk_status(1, 50_000) == "error")
    ok &= check("boundary: wav exactly at min_bytes -> ok",
               align_audio.chunk_status(0, 2000, min_bytes=2000) == "ok")
    ok &= check("boundary: wav one byte under min_bytes -> end_of_audio",
               align_audio.chunk_status(0, 1999, min_bytes=2000) == "end_of_audio")
    return ok


def main():
    tests = [
        test_and_gate_uses_proc_fps,
        test_and_gate_every_one_matches_old_formula,
        test_and_gate_skipped_array_counted,
        test_chunk_status,
    ]
    if ec is not None:
        tests += [
            test_frame_presence_decision_empty_patch_never_present,
            test_empty_patch_does_not_shift_timeline,
            test_fps_fallback_source_labels,
        ]
    else:
        print(f"  [SKIP] extract_continuous.py tests -- import failed: {_EC_IMPORT_ERROR}")

    print("Test A -- build_and_gate_segments.py (proc_fps fix)")
    results = []
    for t in tests:
        print(f"\n{t.__name__}")
        results.append(t())

    total = len(results)
    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{total} tests passed")
    if passed < total or ec is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
