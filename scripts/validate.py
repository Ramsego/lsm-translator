import json
import statistics
import os
from pathlib import Path
from collections import defaultdict

VIDEOS_DIR = Path(os.environ.get("LSM_DATA_ROOT", "/Volumes/Crucial X8/LSM_Translator")) / "videos" / "isolated"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def load_video_data() -> list:
    videos = []
    for folder in sorted(VIDEOS_DIR.iterdir()):
        landmark_file = folder / "landmarks.json"
        info_files = list(folder.glob("*.info.json"))

        if not landmark_file.exists() or not info_files:
            continue

        with open(landmark_file) as f:
            lm_data = json.load(f)

        with open(info_files[0]) as f:
            info = json.load(f)

        videos.append({
            "youtube_id": folder.name,
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "channel": info.get("channel", ""),
            "upload_date": info.get("upload_date", ""),
            "dominant_hand": lm_data.get("dominant_hand", "unknown"),
            "landmarks": lm_data.get("landmarks", []),
        })

    return videos

def compute_frame_coverage(landmarks: list) -> dict:
    if not landmarks:
        return {"total_frames": 0, "hand_frames": 0, "coverage_pct": 0.0}

    total_frames = max(lm["frame"] for lm in landmarks) + 1
    hand_frames = len({lm["frame"] for lm in landmarks if lm["source"] in ("left_hand", "right_hand")})
    coverage_pct = round(hand_frames / total_frames * 100, 1) if total_frames > 0 else 0.0

    return {"total_frames": total_frames, "hand_frames": hand_frames, "coverage_pct": coverage_pct}

def check_labels(videos: list) -> dict:
    label_to_ids = defaultdict(list)
    for v in videos:
        label = v["title"].strip().lower()
        label_to_ids[label].append(v["youtube_id"])

    duplicates = {label: ids for label, ids in label_to_ids.items() if len(ids) > 1}
    suspicious = [label for label in label_to_ids if len(label) < 2 or len(label) > 60]

    return {
        "unique_labels": len(label_to_ids),
        "duplicates": duplicates,
        "suspicious_titles": suspicious,
    }

def run_validation():
    print("Loading video data...")
    videos = load_video_data()
    print(f"Found {len(videos)} processed videos.\n")

    # --- Coverage ---
    coverages = []
    zero_detection = []
    low_coverage = []

    for v in videos:
        cov = compute_frame_coverage(v["landmarks"])
        v["coverage"] = cov
        coverages.append(cov["coverage_pct"])
        if cov["total_frames"] == 0 or cov["hand_frames"] == 0:
            zero_detection.append(v["youtube_id"])
        elif cov["coverage_pct"] < 30:
            low_coverage.append((v["youtube_id"], cov["coverage_pct"]))

    # --- Dominant hand ---
    hand_counts = defaultdict(int)
    for v in videos:
        hand_counts[v["dominant_hand"]] += 1

    # --- Labels ---
    label_stats = check_labels(videos)

    # --- Durations ---
    durations = [v["duration"] for v in videos if v["duration"]]

    # --- Class balance ---
    label_counts = defaultdict(int)
    for v in videos:
        label_counts[v["title"].strip().lower()] += 1
    videos_per_label = list(label_counts.values())

    # --- Signers ---
    channels = {v["channel"] for v in videos if v["channel"]}

    # --- Build stats dict ---
    stats = {
        "total_videos": len(videos),
        "unique_signers": len(channels),
        "signer_channels": list(channels),
        "unique_labels": label_stats["unique_labels"],
        "duplicate_labels": label_stats["duplicates"],
        "suspicious_titles": label_stats["suspicious_titles"],
        "dominant_hand": dict(hand_counts),
        "coverage": {
            "mean_pct": round(statistics.mean(coverages), 1),
            "median_pct": round(statistics.median(coverages), 1),
            "min_pct": round(min(coverages), 1),
            "max_pct": round(max(coverages), 1),
            "zero_detection_count": len(zero_detection),
            "zero_detection_ids": zero_detection,
            "low_coverage_count": len(low_coverage),
            "low_coverage_ids": low_coverage,
        },
        "duration_seconds": {
            "mean": round(statistics.mean(durations), 1),
            "median": round(statistics.median(durations), 1),
            "min": round(min(durations), 1),
            "max": round(max(durations), 1),
            "std": round(statistics.stdev(durations), 1),
        },
        "class_balance": {
            "mean_videos_per_label": round(statistics.mean(videos_per_label), 2),
            "min_videos_per_label": min(videos_per_label),
            "max_videos_per_label": max(videos_per_label),
            "labels_with_1_video": sum(1 for c in videos_per_label if c == 1),
        },
    }

    # --- Print report ---
    print("=" * 50)
    print("DATASET VALIDATION REPORT")
    print("=" * 50)

    print(f"\nOVERVIEW")
    print(f"  Total videos:      {stats['total_videos']}")
    print(f"  Unique labels:     {stats['unique_labels']}")
    print(f"  Unique signers:    {stats['unique_signers']} — {', '.join(stats['signer_channels'])}")

    print(f"\nDOMINANT HAND")
    for hand, count in stats["dominant_hand"].items():
        print(f"  {hand}: {count} videos")

    print(f"\nMEDIAPIPE COVERAGE (% of frames with hand detections)")
    print(f"  Mean:   {stats['coverage']['mean_pct']}%")
    print(f"  Median: {stats['coverage']['median_pct']}%")
    print(f"  Min:    {stats['coverage']['min_pct']}%")
    print(f"  Max:    {stats['coverage']['max_pct']}%")
    print(f"  Zero detection:  {stats['coverage']['zero_detection_count']} videos")
    print(f"  Low coverage (<30%): {stats['coverage']['low_coverage_count']} videos")

    print(f"\nSIGN DURATION (seconds)")
    print(f"  Mean:   {stats['duration_seconds']['mean']}s")
    print(f"  Median: {stats['duration_seconds']['median']}s")
    print(f"  Std:    {stats['duration_seconds']['std']}s")
    print(f"  Min:    {stats['duration_seconds']['min']}s")
    print(f"  Max:    {stats['duration_seconds']['max']}s")

    print(f"\nCLASS BALANCE")
    print(f"  Mean videos per label: {stats['class_balance']['mean_videos_per_label']}")
    print(f"  Min videos per label:  {stats['class_balance']['min_videos_per_label']}")
    print(f"  Max videos per label:  {stats['class_balance']['max_videos_per_label']}")
    print(f"  Labels with only 1 video: {stats['class_balance']['labels_with_1_video']}")

    print(f"\nLABEL QUALITY")
    if stats["duplicate_labels"]:
        print(f"  Duplicates found:")
        for label, ids in stats["duplicate_labels"].items():
            print(f"    '{label}' → {ids}")
    else:
        print(f"  No duplicate labels.")
    if stats["suspicious_titles"]:
        print(f"  Suspicious titles: {stats['suspicious_titles']}")
    else:
        print(f"  No suspicious titles.")

    print("\n" + "=" * 50)

    # --- Save stats ---
    stats_path = DATA_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Stats saved to {stats_path}")

if __name__ == "__main__":
    run_validation()