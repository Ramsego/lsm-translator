"""
QC check flagged by both independent reviewers: dictionary reference videos are
clean/frontal, real footage has motion blur/occlusion/framing -- differential
landmark quality between the query (real footage) and candidate (dictionary) sides
could dominate the retrieval signal rather than sign content. Report hand-landmark
detection rate per side so this is measured, not assumed.

Usage: /opt/miniconda3/envs/lsm-probe/bin/python phase3/probe/fc_test/qc_poses.py
"""
import statistics as stats
from pathlib import Path

from pose_format import Pose

HERE = Path(__file__).parent


def hand_detection_rate(pose_path):
    with open(pose_path, "rb") as f:
        p = Pose.read(f.read())
    conf = p.body.confidence  # (frames, people, points)
    if conf.shape[0] == 0:
        return 0.0
    # MediaPipe Holistic point layout: 33 pose + 468 face + 21 left hand + 21 right hand.
    # Hands are the last 42 points -- what actually carries sign content.
    hand_conf = conf[:, 0, -42:]
    frame_has_hand = (hand_conf > 0).any(axis=1)
    return float(frame_has_hand.mean())


def summarize(label, pose_dir):
    rates = [hand_detection_rate(p) for p in sorted(Path(pose_dir).glob("*.pose"))]
    print(f"{label:12s} n={len(rates):3d}  mean={stats.fmean(rates):.1%}  "
          f"median={stats.median(rates):.1%}  min={min(rates):.1%}  max={max(rates):.1%}")
    low = [(p.name, r) for p, r in zip(sorted(Path(pose_dir).glob("*.pose")), rates) if r < 0.5]
    if low:
        print(f"  clips with <50% hand-detected frames ({len(low)}):")
        for name, r in low:
            print(f"    {name:30s} {r:.1%}")


if __name__ == "__main__":
    summarize("query", HERE / "query_poses")
    summarize("candidate", HERE / "candidate_poses")
