"""Unit tests for trim_window() in trim_arrays.py."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trim_arrays import trim_window, active_segment

HAND_ROWS = slice(0, 42)
POSE_ROWS = slice(42, 75)
FACE_ROWS = slice(75, 116)
N_LM = 116


def make_array(n_frames: int) -> np.ndarray:
    """All-NaN array of shape [n_frames, 116, 3]."""
    return np.full((n_frames, N_LM, 3), np.nan, dtype=np.float32)


def put_hand(arr: np.ndarray, frame: int):
    """Mark a single hand landmark as detected in the given frame."""
    arr[frame, 0, :] = [0.5, 0.5, 0.0]


def check(name: str, condition: bool):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def test_basic_trim():
    arr = make_array(100)
    # Sign from frames 30–60; rest is silent padding
    for f in range(30, 61):
        put_hand(arr, f)
    result = trim_window(arr, buffer=0)
    assert result is not None
    return check("basic trim: output starts at first hand frame",
                 result.shape[0] == 31)  # frames 30–60 inclusive


def test_buffer_kept():
    arr = make_array(100)
    for f in range(30, 61):
        put_hand(arr, f)
    result = trim_window(arr, buffer=5)
    assert result is not None
    return check("buffer: 5 extra frames kept on each side",
                 result.shape[0] == 31 + 10)  # 30-5=25 to 60+5=65 → 41 frames


def test_buffer_clamp():
    arr = make_array(20)
    for f in range(0, 20):
        put_hand(arr, f)
    # buffer=5 would go to -5 and 24, both must clamp
    result = trim_window(arr, buffer=5)
    assert result is not None
    return check("buffer clamps at array boundaries",
                 result.shape[0] == 20)


def test_no_hands_skipped():
    arr = make_array(50)
    # no hand data at all
    result = trim_window(arr, buffer=5)
    return check("no hands → returns None", result is None)


def test_already_trimmed():
    arr = make_array(30)
    for f in range(0, 30):
        put_hand(arr, f)
    result = trim_window(arr, buffer=0)
    assert result is not None
    return check("already tight array → same length", result.shape[0] == 30)


def test_face_pose_preserved():
    arr = make_array(50)
    for f in range(10, 40):
        put_hand(arr, f)
    # Put distinct values in face/pose rows
    for f in range(10, 40):
        arr[f, 80, :] = [f * 0.01, f * 0.01, 0.0]   # face
        arr[f, 50, :] = [f * 0.02, f * 0.02, 0.0]   # pose

    result = trim_window(arr, buffer=0)
    assert result is not None
    # Verify face/pose values in the trimmed array match original at frame 10
    original_face = arr[10, 80, :]
    trimmed_face = result[0, 80, :]
    return check("face/pose data preserved after trim",
                 np.allclose(original_face, trimmed_face))


def test_nan_fraction_drops():
    arr = make_array(100)
    for f in range(40, 60):
        put_hand(arr, f)
    original_nan = np.isnan(arr).mean()
    result = trim_window(arr, buffer=0)
    assert result is not None
    trimmed_nan = np.isnan(result).mean()
    return check("NaN fraction drops after trim",
                 trimmed_nan < original_nan)


def test_segment_drops_outro_card():
    # Real demo frames 30–70, then a 40-frame dead gap (outro card),
    # then a short stray detection at 115–120 (editing tail).
    arr = make_array(130)
    for f in range(30, 71):
        put_hand(arr, f)
    for f in range(115, 121):
        put_hand(arr, f)
    result = active_segment(arr, buffer=0, gap_threshold=20)
    assert result is not None
    # Should keep only the first (largest) segment: frames 30–70 = 41 frames
    return check("segment: drops outro card / editing tail",
                 result.shape[0] == 41)


def test_segment_keeps_largest():
    # Short take first (10 frames), big gap, longer take second (30 frames)
    arr = make_array(120)
    for f in range(5, 15):
        put_hand(arr, f)
    for f in range(60, 90):
        put_hand(arr, f)
    result = active_segment(arr, buffer=0, gap_threshold=20)
    assert result is not None
    return check("segment: keeps the largest take, not the first",
                 result.shape[0] == 30)


def test_segment_short_gaps_kept():
    # A brief within-sign pause (gap of 5) must NOT split the sign.
    arr = make_array(60)
    for f in range(10, 25):
        put_hand(arr, f)
    for f in range(31, 45):  # gap of 6 frames (< threshold)
        put_hand(arr, f)
    result = active_segment(arr, buffer=0, gap_threshold=20)
    assert result is not None
    # Whole span 10–44 kept = 35 frames
    return check("segment: short within-sign pause not split",
                 result.shape[0] == 35)


def test_segment_no_hands():
    arr = make_array(40)
    return check("segment: no hands → None", active_segment(arr) is None)


def main():
    tests = [
        test_basic_trim,
        test_buffer_kept,
        test_buffer_clamp,
        test_no_hands_skipped,
        test_already_trimmed,
        test_face_pose_preserved,
        test_nan_fraction_drops,
        test_segment_drops_outro_card,
        test_segment_keeps_largest,
        test_segment_short_gaps_kept,
        test_segment_no_hands,
    ]
    results = [t() for t in tests]
    total = len(results)
    passed = sum(results)
    print(f"\n{passed}/{total} tests passed")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
