"""Route 1 Step 2: activity signal + burst detection. Shared by clips and spans."""
import numpy as np

SMOOTH_W = 5
Z_ON = 1.0
MIN_BURST = 6
MAX_GAP = 5
MERGE_GAP = 10
OCC_DIST = 0.9

def compute_occluded(hand_dist, face_ok):
    return (hand_dist < OCC_DIST) | (~face_ok)

def moving_average_ignore_nan(d, w, min_valid=3):
    n = len(d)
    out = np.full(n, np.nan)
    half = w // 2
    for t in range(n):
        lo, hi = max(0, t-half), min(n, t+half+1)
        window = d[lo:hi]
        valid = window[~np.isnan(window)]
        if len(valid) >= min_valid:
            out[t] = np.mean(valid)
    return out

def robust_z(s):
    valid = s[~np.isnan(s)]
    if len(valid) < 3:
        return np.full_like(s, np.nan)
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        return np.full_like(s, 0.0)
    z = (s - med) / scale
    return z

def detect_bursts(aperture, hand_dist, face_ok,
                   smooth_w=SMOOTH_W, z_on=Z_ON, min_burst=MIN_BURST,
                   max_gap=MAX_GAP, merge_gap=MERGE_GAP, occ_dist=OCC_DIST):
    n = len(aperture)
    occluded = (hand_dist < occ_dist) | (~face_ok)
    d = np.full(n, np.nan)
    for t in range(1, n):
        if occluded[t] or occluded[t-1]:
            continue
        d[t] = abs(aperture[t] - aperture[t-1])
    s = moving_average_ignore_nan(d, smooth_w)
    z = robust_z(s)

    active = z >= z_on
    active = np.where(np.isnan(z), False, active)

    # find maximal runs tolerating gaps <= max_gap
    bursts = []
    t = 0
    while t < n:
        if not active[t]:
            t += 1; continue
        start = t
        end = t
        gap = 0
        t2 = t + 1
        while t2 < n:
            if active[t2]:
                end = t2; gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            t2 += 1
        bursts.append([start, end])
        t = end + gap + 1

    bursts = [b for b in bursts if (b[1]-b[0]+1) >= min_burst]

    merged = []
    for b in bursts:
        if merged and b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))

    return dict(occluded=occluded, d=d, s=s, z=z, bursts=[tuple(b) for b in merged])
