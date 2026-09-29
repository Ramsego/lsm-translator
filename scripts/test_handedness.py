"""Unit tests for scripts/handedness.py."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from handedness import (
    flip_x_landmarks, wrist_path_length, motion_dominant,
    mirror_array, LEFT_HAND, RIGHT_HAND,
)

N_LM = 116


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    return condition


def make_array(n=10):
    a = np.full((n, N_LM, 3), np.nan, dtype=np.float32)
    return a


def test_mirror_is_inverse():
    a = make_array(5)
    a[:, 5, :] = [0.3, 0.4, 0.1]      # a left-hand point
    a[:, 30, :] = [0.7, 0.2, 0.0]     # a right-hand point
    a[:, 50, :] = [0.6, 0.6, 0.0]     # a pose point
    back = mirror_array(mirror_array(a))
    return check("mirror_array is its own inverse",
                 np.allclose(back, a, equal_nan=True))


def test_mirror_swaps_hands():
    a = make_array(1)
    a[0, 5, :] = [0.3, 0.4, 0.1]      # left-hand row 5
    m = mirror_array(a)
    # That point should now be in the right-hand block (row 21+5=26), x-flipped
    moved = m[0, 26, :]
    gone = np.isnan(m[0, 5, 0])
    return check("mirror swaps left->right hand block, x-flipped",
                 gone and np.allclose(moved, [0.7, 0.4, 0.1]))


def test_mirror_pose_flipped_not_reordered():
    a = make_array(1)
    a[0, 50, :] = [0.6, 0.6, 0.0]     # pose row stays a pose row
    m = mirror_array(a)
    return check("pose x-flipped but not reordered",
                 np.allclose(m[0, 50, :], [0.4, 0.6, 0.0]))


def test_motion_dominant_right():
    # right hand moves a lot, left barely
    lms = []
    for f in range(10):
        lms.append({"frame": f, "source": "right_hand", "landmark_index": 0,
                    "x": 0.1 + f * 0.05, "y": 0.5, "z": 0.0})
        lms.append({"frame": f, "source": "left_hand", "landmark_index": 0,
                    "x": 0.2, "y": 0.5, "z": 0.0})
    return check("motion_dominant detects right", motion_dominant(lms) == "right")


def test_motion_dominant_left():
    lms = []
    for f in range(10):
        lms.append({"frame": f, "source": "left_hand", "landmark_index": 0,
                    "x": 0.1 + f * 0.05, "y": 0.5, "z": 0.0})
        lms.append({"frame": f, "source": "right_hand", "landmark_index": 0,
                    "x": 0.2, "y": 0.5, "z": 0.0})
    return check("motion_dominant detects left", motion_dominant(lms) == "left")


def test_motion_dominant_tie_defaults_right():
    return check("motion_dominant defaults right with no data",
                 motion_dominant([]) == "right")


def test_flip_x_is_inverse():
    lms = [{"frame": 0, "source": "left_hand", "landmark_index": 0,
            "x": 0.3, "y": 0.4, "z": 0.1}]
    flip_x_landmarks(lms)
    once = lms[0]["x"]
    flip_x_landmarks(lms)
    return check("flip_x_landmarks round-trips (0.3 -> 0.7 -> 0.3)",
                 abs(once - 0.7) < 1e-9 and abs(lms[0]["x"] - 0.3) < 1e-9)


def main():
    tests = [
        test_mirror_is_inverse,
        test_mirror_swaps_hands,
        test_mirror_pose_flipped_not_reordered,
        test_motion_dominant_right,
        test_motion_dominant_left,
        test_motion_dominant_tie_defaults_right,
        test_flip_x_is_inverse,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    if sum(results) < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
