import sys, csv
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from burst_detect import detect_bursts

HERE = Path(__file__).parent
rows = {r["file"].replace("../57TvyH9902U/",""): r
        for r in csv.DictReader(open("/Users/sebastiangonzalezramirez/Desktop/LSM_Translator/phase3/probe/review_2.csv"))
        if r["verdict"]=="y" and r["sign_start"]}

targets = [
    ("gracias/gracias_20m52s.mp4", "gracias_20m52s (expect: burst BEFORE/overlapping marked window)"),
    ("gracias/gracias_20m40s.mp4", "gracias_20m40s (expect: weak/no burst near marked window)"),
    ("joven/joven_25m49s.mp4", "joven_25m49s (expect: occlusion flagged where hand covers mouth ~7.2-7.5s)"),
    ("actividad/actividad_20m20s.mp4", "actividad_20m20s (expect: clear burst near marked window)"),
]
# futuro is not itself in review_2.csv under a 'sign' we're checking against joven/familia,
# but it IS one of the 26 verified clips (its own word). include it.
targets.append(("futuro/futuro_45m33s.mp4", "futuro_45m33s (own marked window + activity shape)"))

fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3*len(targets)))
for ax, (rel, title) in zip(axes, targets):
    name = rel.replace("/", "__").replace(".mp4", ".npz")
    d = np.load(HERE / "sidecar" / name)
    aperture, hand_dist, face_ok = d["aperture"], d["hand_dist"], d["face_ok"]
    r = detect_bursts(aperture, hand_dist, face_ok)
    fps = float(d["fps"])
    t = np.arange(len(aperture)) / fps

    ax.plot(t, aperture, color="tab:blue", lw=1, label="aperture")
    ax2 = ax.twinx()
    ax2.plot(t, r["z"], color="tab:orange", lw=0.8, alpha=0.7, label="z-score")
    ax2.axhline(1.0, color="tab:orange", ls=":", lw=0.8)

    for (a,b) in r["bursts"]:
        ax.axvspan(a/fps, b/fps, color="green", alpha=0.25)

    occ = r["occluded"]
    in_occ = False
    for i in range(len(occ)):
        if occ[i] and not in_occ:
            occ_start = i; in_occ = True
        if (not occ[i] or i==len(occ)-1) and in_occ:
            occ_end = i
            ax.axvspan(occ_start/fps, occ_end/fps, color="red", alpha=0.15, hatch="//")
            in_occ = False

    row = rows.get(rel)
    if row:
        vs = float(row["video_start"])
        ss, se = float(row["sign_start"])-vs, float(row["sign_end"])-vs
        ax.axvline(ss, color="black", ls="--", lw=1.2)
        ax.axvline(se, color="black", ls="--", lw=1.2)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("seconds"); ax.set_ylabel("aperture")
    ax2.set_ylabel("z")

plt.tight_layout()
out = HERE / "sanity_check.png"
plt.savefig(out, dpi=110)
print("wrote", out)
