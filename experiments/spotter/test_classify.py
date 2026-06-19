"""Unit tests for the DTW classifier core (experiments/spotter/classify.py)."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import classify as clf
from handedness import mirror_array

N_LM = 116


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def make_clip(T=15, seed=0):
    """A synthetic clip: right hand traces a path, shoulders present, rest NaN."""
    rng = np.random.default_rng(seed)
    a = np.full((T, N_LM, 3), np.nan, dtype=np.float32)
    # shoulders (rows 53,54) steady
    a[:, 53, :2] = [0.45, 0.30]
    a[:, 54, :2] = [0.55, 0.30]
    # right hand (rows 21..41) moves along a smooth diagonal
    for t in range(T):
        base = 0.50 + 0.01 * t
        for j in range(21):
            a[t, 21 + j, :2] = [base + 0.002 * j, 0.50 + 0.005 * t]
    return a


def test_identical_zero():
    a = make_clip()
    fa = clf.featurize(a)
    return check("identical sequences -> distance ~0", clf._dist(fa, fa) < 1e-6)


def test_translate_scale_invariant():
    a = make_clip()
    b = a.copy()
    # translate +0.1 in x and scale around origin by 1.3 (shoulders move too)
    b[:, :, 0] = b[:, :, 0] * 1.3 + 0.1
    b[:, :, 1] = b[:, :, 1] * 1.3 + 0.05
    norm_cfg = {"norm": "per_frame"}
    raw_cfg = {"norm": "none"}
    d_norm = clf._dist(clf.featurize(a, norm_cfg), clf.featurize(b, norm_cfg))
    d_raw = clf._dist(clf.featurize(a, raw_cfg), clf.featurize(b, raw_cfg))
    return check("normalization makes translate+scale copy ~match (norm << raw)",
                 d_norm < 1e-3 and d_norm < d_raw)


def test_mirror_matched_at_query():
    a = make_clip()
    am = mirror_array(a)
    # mirrored clip should match original via the mirror-at-query min
    bank = [{"youtube_id": "ref", "label": "X", "source": "s",
             "feat": clf.featurize(a), "feat_mirror": clf.featurize(mirror_array(a))}]
    qf = clf.featurize(am)
    qfm = clf.featurize(mirror_array(am))
    top = clf.classify(qf, qfm, bank, k=1)
    return check("mirrored query matches original (handedness-invariant)",
                 top[0][1] < 1e-6)


def test_different_is_farther():
    a = make_clip(seed=1)
    b = make_clip(seed=1)
    # c: hand moves the opposite direction
    c = a.copy()
    for t in range(c.shape[0]):
        for j in range(21):
            c[t, 21 + j, :2] = [0.50 - 0.01 * t, 0.50 - 0.005 * t]
    d_same = clf._dist(clf.featurize(a), clf.featurize(b))
    d_diff = clf._dist(clf.featurize(a), clf.featurize(c))
    return check("different trajectory is farther than matching one", d_diff > d_same)


def test_nan_frames_dropped():
    a = make_clip(T=12)
    a[0:3, 21:42, :] = np.nan       # first 3 frames have no hand
    f = clf.featurize(a)
    return check("all-NaN-hand frames dropped, no crash, finite features",
                 f.shape[0] == 9 and np.isfinite(f).all())


def test_velocity_position_invariant():
    # vel_only should make a pure translation match exactly (motion is identical).
    a = make_clip()
    b = a.copy()
    b[:, :, 0] += 0.2          # shift right; velocity unchanged
    cfg = {"norm": "none", "vel_only": True}
    d = clf._dist(clf.featurize(a, cfg), clf.featurize(b, cfg))
    return check("vel_only is position-invariant (translation -> dist ~0)", d < 1e-6)


def test_pose_changes_dim():
    a = make_clip()
    a[:, 53:59, :2] = 0.4      # give arm rows some value
    d_hands = clf.featurize(a, {"norm": "none", "include_pose": False}).shape[1]
    d_pose = clf.featurize(a, {"norm": "none", "include_pose": True}).shape[1]
    return check("include_pose widens the feature vector", d_pose > d_hands)


def main():
    tests = [
        test_identical_zero,
        test_translate_scale_invariant,
        test_mirror_matched_at_query,
        test_different_is_farther,
        test_nan_frames_dropped,
        test_velocity_position_invariant,
        test_pose_changes_dim,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} tests passed")
    if sum(results) < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
