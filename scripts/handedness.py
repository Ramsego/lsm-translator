"""
Handedness utilities shared by extraction, the un-mirror pass, and the classifier.

Design decision: landmark arrays are stored in their RAW, as-detected orientation.
Handedness invariance is handled at DTW query time by comparing a query against both
itself and its mirror (see mirror_array). Nothing here transforms stored coordinates
based on a guessed dominant hand — `motion_dominant` is informational metadata only.
"""

import numpy as np

# Row layout of the [frames, 116, 3] array.
LEFT_HAND = slice(0, 21)
RIGHT_HAND = slice(21, 42)
# rows 42-74 pose, 75-115 face (x-flipped by a mirror but not reordered)


# ---- landmark-list helpers (work on the list-of-dicts in landmarks.json) ----

def flip_x_landmarks(landmarks: list) -> list:
    """Apply x -> 1-x to every record, in place. This is its own inverse and
    exactly undoes the old normalize_handedness mirror (which flipped x only,
    without swapping hand labels)."""
    for lm in landmarks:
        lm["x"] = 1.0 - lm["x"]
    return landmarks


def wrist_path_length(landmarks: list, source: str) -> float:
    """Cumulative frame-to-frame displacement of a hand's wrist (landmark 0).

    Reflection-invariant in magnitude, so it can be computed regardless of any
    prior x-flip. Used only to label the dominant hand as metadata.
    """
    pts = [(lm["frame"], lm["x"], lm["y"])
           for lm in landmarks
           if lm["source"] == source and lm["landmark_index"] == 0]
    if len(pts) < 2:
        return 0.0
    pts.sort()
    xy = np.array([(x, y) for _, x, y in pts], dtype=np.float64)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def motion_dominant(landmarks: list) -> str:
    """Return 'left' or 'right' — whichever hand moves more over the clip.
    Informational only; defaults to 'right' on a tie or no motion."""
    left = wrist_path_length(landmarks, "left_hand")
    right = wrist_path_length(landmarks, "right_hand")
    return "left" if left > right else "right"


# ---- array helper (for the DTW classifier) ----

def mirror_array(arr: np.ndarray) -> np.ndarray:
    """Return a true horizontal reflection of a [frames, 116, 3] array.

    A real mirror flips x (x -> 1-x) for ALL landmarks AND swaps the left/right
    hand blocks (rows 0-20 <-> 21-41), since a reflected left hand becomes a
    right hand. Pose and face rows are x-flipped but keep their order.

    Used at query time: min(dtw(q, ref), dtw(mirror_array(q), ref)) makes
    matching handedness-invariant without canonicalizing stored data.
    """
    out = arr.copy()
    out[:, :, 0] = 1.0 - out[:, :, 0]  # x -> 1-x everywhere (NaN stays NaN)
    left = out[:, LEFT_HAND, :].copy()
    out[:, LEFT_HAND, :] = out[:, RIGHT_HAND, :]
    out[:, RIGHT_HAND, :] = left
    return out
