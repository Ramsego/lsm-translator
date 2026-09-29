import sys, csv, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from pathlib import Path
from burst_detect import detect_bursts

HERE = Path(__file__).parent
SEARCH_PRE, SEARCH_POST, PAD = 2.5, 1.0, 0.15

rows = [r for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"] == "y" and r["sign_start"]]

out_rows = []
n_none = 0
for r in rows:
    rel = r["file"].replace("../57TvyH9902U/", "")
    name = rel.replace("/", "__").replace(".mp4", ".npz")
    d = np.load(HERE / "sidecar" / name)
    aperture, hand_dist, face_ok, fps = d["aperture"], d["hand_dist"], d["face_ok"], float(d["fps"])
    res = detect_bursts(aperture, hand_dist, face_ok)

    video_start = float(r["video_start"])
    a_s = float(r["sign_start"]) - video_start
    b_s = float(r["sign_end"]) - video_start
    a_f, b_f = a_s*fps, b_s*fps
    n = len(aperture)
    search_lo = max(0, a_f - SEARCH_PRE*fps)
    search_hi = min(n-1, b_f + SEARCH_POST*fps)

    best_burst, best_overlap = None, -1
    for (bs, be) in res["bursts"]:
        overlap = max(0, min(be, search_hi) - max(bs, search_lo))
        if overlap > best_overlap or (overlap == best_overlap and best_burst is not None and abs(bs-a_f) < abs(best_burst[0]-a_f)):
            best_overlap, best_burst = overlap, (bs, be)

    if best_burst is None or best_overlap <= 0:
        n_none += 1
        out_rows.append(dict(file=rel, word=r["word"], sign_start=r["sign_start"], sign_end=r["sign_end"],
                              burst_on_s="", burst_off_s="", lead_s="", n_occluded_frames="", mouthing_detected=False))
        continue

    bs, be = best_burst
    pad_f = PAD*fps
    t0, t1 = max(0, int(bs - pad_f)), min(n, int(be + pad_f) + 1)
    n_occ_in_window = int(np.sum(res["occluded"][t0:t1]))
    lead_s = a_s - bs/fps
    out_rows.append(dict(file=rel, word=r["word"], sign_start=r["sign_start"], sign_end=r["sign_end"],
                          burst_on_s=f"{bs/fps:.3f}", burst_off_s=f"{be/fps:.3f}", lead_s=f"{lead_s:.3f}",
                          n_occluded_frames=n_occ_in_window, mouthing_detected=True))
    print(f"{r['word']:12s} {rel:35s} sign=[{a_s:.2f},{b_s:.2f}] burst=[{bs/fps:.2f},{be/fps:.2f}] lead={lead_s:+.2f}s occ_in_win={n_occ_in_window}")

with open(HERE / "snapped_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file","word","sign_start","sign_end","burst_on_s","burst_off_s","lead_s","n_occluded_frames","mouthing_detected"])
    w.writeheader(); w.writerows(out_rows)

print(f"\n{n_none}/{len(rows)} clips: mouthing_detected=False")
print("GATE: >10/26 none? ", "FAIL - STOP" if n_none > 10 else "PASS")
