"""
Gold-clip dataset: load human-verified (sign-clip → word) pairs for training.

This is the Stage-A data pipeline and is MODEL-AGNOSTIC: it yields featurized,
shoulder-normalized landmark sequences + word labels, collated into padded batches
(`[B, Tmax, D]` + lengths). A classifier can mean/last-pool over lengths now; a CTC
head later consumes the same padded sequences + input lengths. So the model choice
(CTC vs decoder-only vs SHuBERT-finetune) does not change this file.

Each gold label is one isolated sign in VIDEO-time `[start_sec, end_sec]`. We map that
window into the right continuous-extraction segment, slice the frames, and reuse the
spotter's `featurize()` so train-time normalization is IDENTICAL to what the spotter
validated (shoulder-relative, schema-agnostic over the 124-row layout).

Reuse:
  - experiments/spotter/classify.py : featurize, DEFAULT_CFG  (normalization)
  - scripts/handedness.py           : mirror_array            (flip augmentation)

Usage:
    # runnable today with NO real data — synthesizes a tiny gold + fake npz:
    python phase3/dataset.py --self-test

    # once annotation has produced gold_labels.json:
    python phase3/dataset.py --gold phase2/gold_labels.json \\
        --arrays-root "/Volumes/Crucial X8/LSM_Translator/arrays/mananera"
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ── reuse the spotter's normalization + the mirror augmentation ───────────────
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "experiments" / "spotter"))
sys.path.insert(0, str(_root / "scripts"))
from classify import featurize, DEFAULT_CFG          # noqa: E402
from handedness import mirror_array                  # noqa: E402

DEFAULT_GOLD = _root / "phase2" / "gold_labels.json"


# ── time → frame mapping ──────────────────────────────────────────────────────

def _load_segments(arrays_root: Path, video_id: str, _cache: dict):
    """Load + cache a video's segments.json. Returns (meta, proc_fps, seg_dir)."""
    if video_id in _cache:
        return _cache[video_id]
    seg_dir = arrays_root / video_id
    meta = json.load(open(seg_dir / "segments.json"))
    proc_fps = meta["fps"] / meta.get("every", 1)
    _cache[video_id] = (meta, proc_fps, seg_dir)
    return _cache[video_id]


def _locate(meta, proc_fps, start_sec, end_sec):
    """Return (seg_file, frame_lo, frame_hi) for a video-time window, or None if the
    window's midpoint falls in no segment (the sign landed in an extraction gap)."""
    mid = 0.5 * (start_sec + end_sec)
    for s in meta["segments"]:
        if s["start_sec"] <= mid <= s["end_sec"]:
            lo = round(start_sec * proc_fps) - s["start_frame"]
            hi = round(end_sec * proc_fps) - s["start_frame"]
            lo = max(0, min(lo, s["n_frames"]))
            hi = max(0, min(hi, s["n_frames"]))
            if hi <= lo:
                return None
            return s["file"], lo, hi
    return None


# ── vocab ─────────────────────────────────────────────────────────────────────

def build_vocab(labels, min_count=1):
    """word -> contiguous index, keeping words with >= min_count occurrences."""
    from collections import Counter
    counts = Counter(lbl["word"] for lbl in labels)
    words = sorted(w for w, c in counts.items() if c >= min_count)
    return {w: i for i, w in enumerate(words)}


# ── dataset ───────────────────────────────────────────────────────────────────

class GoldClipDataset(Dataset):
    def __init__(self, gold_json, arrays_root, vocab=None, min_count=1,
                 signers=None, exclude_signers=None, cfg=None, augment=False,
                 mirror_p=0.5, verbose=False):
        self.arrays_root = Path(arrays_root)
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.augment = augment
        self.mirror_p = mirror_p

        data = json.load(open(gold_json))
        labels = data["labels"] if isinstance(data, dict) else data
        self.vocab = vocab if vocab is not None else build_vocab(labels, min_count)
        self.itos = {i: w for w, i in self.vocab.items()}

        seg_cache = {}
        self.index = []          # (video_id, npz_path, lo, hi, word_idx, signer)
        skipped_gap = skipped_oov = 0
        for lbl in labels:
            word = lbl["word"]
            if word not in self.vocab:
                skipped_oov += 1
                continue
            sig = lbl.get("signer", "")
            if signers is not None and sig not in signers:
                continue
            if exclude_signers is not None and sig in exclude_signers:
                continue
            try:
                meta, proc_fps, seg_dir = _load_segments(
                    self.arrays_root, lbl["video_id"], seg_cache)
            except FileNotFoundError:
                skipped_gap += 1
                continue
            loc = _locate(meta, proc_fps, lbl["start_sec"], lbl["end_sec"])
            if loc is None:
                skipped_gap += 1
                continue
            seg_file, lo, hi = loc
            self.index.append((lbl["video_id"], seg_dir / seg_file, lo, hi,
                               self.vocab[word], sig))

        self.skipped_gap = skipped_gap
        self.skipped_oov = skipped_oov
        if verbose:
            print(f"  resolved {len(self.index)} clips  "
                  f"(skipped {skipped_gap} in-gap, {skipped_oov} below-min-count)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        _, npz_path, lo, hi, word_idx, signer = self.index[i]
        arr = np.load(npz_path)["landmarks"][lo:hi]      # [T, 124, 3]
        if self.augment and random.random() < self.mirror_p:
            arr = mirror_array(arr)
        feat = featurize(arr, self.cfg)                  # [T', D]
        return torch.from_numpy(feat).float(), word_idx, signer


# ── collate: pad variable-length sequences ────────────────────────────────────

def collate_pad(batch):
    """[(feat[T,D], label, signer), ...] -> (padded[B,Tmax,D], lengths[B], labels[B])."""
    feats, labels, _ = zip(*batch)
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


# ── leave-one-signer-out ──────────────────────────────────────────────────────

def loso_splits(gold_json, arrays_root, min_count=1, cfg=None, augment_train=True):
    """Yield (held_out_signer, train_ds, test_ds) per signer, sharing one vocab."""
    data = json.load(open(gold_json))
    labels = data["labels"] if isinstance(data, dict) else data
    vocab = build_vocab(labels, min_count)
    signers = sorted({lbl.get("signer", "") for lbl in labels})
    for held in signers:
        train = GoldClipDataset(gold_json, arrays_root, vocab=vocab,
                                exclude_signers={held}, cfg=cfg, augment=augment_train)
        test = GoldClipDataset(gold_json, arrays_root, vocab=vocab,
                               signers={held}, cfg=cfg, augment=False)
        yield held, train, test


# ── self-test (runnable with zero real data) ──────────────────────────────────

def _self_test():
    import tempfile
    from torch.utils.data import DataLoader

    tmp = Path(tempfile.mkdtemp(prefix="dataset_selftest_"))
    vid = "FAKEVIDEO01"
    seg_dir = tmp / vid
    seg_dir.mkdir(parents=True)
    fps, every = 30.0, 1
    proc_fps = fps / every

    # one long synthetic segment with plausible shoulder rows (so featurize's
    # shoulder-width normalization doesn't divide by ~0)
    n = 3000
    rng = np.random.default_rng(0)
    land = rng.random((n, 124, 3)).astype(np.float32) * 0.1 + 0.5
    land[:, 53, :2] = [0.45, 0.5]   # L shoulder (row 42+11)
    land[:, 54, :2] = [0.55, 0.5]   # R shoulder (row 42+12)
    np.savez_compressed(seg_dir / f"{vid}_seg0000.npz", landmarks=land)
    json.dump({
        "fps": fps, "every": every, "schema_rows": 124,
        "segments": [{"file": f"{vid}_seg0000.npz", "start_frame": 0,
                      "end_frame": n, "start_sec": 0.0,
                      "end_sec": round(n / proc_fps, 2), "n_frames": n,
                      "hand_coverage": 1.0}],
    }, open(seg_dir / "segments.json", "w"))

    words = ["gobierno", "salud", "pueblo", "tener", "tener:neg"]
    labels = []
    for k, w in enumerate(words):
        for j in range(4):                       # 4 clips per word, 2 signers
            t0 = 5.0 + (k * 4 + j) * 6.0
            labels.append({"word": w, "video_id": vid,
                           "signer": f"{vid}:{1 + (j % 2)}",
                           "start_sec": round(t0, 1), "end_sec": round(t0 + 1.5, 1),
                           "localized": True, "tier": "gold", "source": "human"})
    gold = tmp / "gold_labels.json"
    json.dump({"labels": labels, "implicit_signs": []}, open(gold, "w"))

    ds = GoldClipDataset(gold, tmp, min_count=1, augment=True, verbose=True)
    print(f"vocab ({len(ds.vocab)}): {ds.vocab}")
    print(f"examples: {len(ds)}  skipped_gap={ds.skipped_gap} skipped_oov={ds.skipped_oov}")

    feat0, y0, sig0 = ds[0]
    D = feat0.shape[1]
    expected_D = 2 * (42 + (6 if ds.cfg["include_pose"] else 0))   # hands+arms, x/y
    assert D == expected_D, f"feature dim {D} != expected {expected_D} (cfg drift?)"
    print(f"sample: feat={tuple(feat0.shape)}  label={ds.itos[y0]}  signer={sig0}  D ok ({D})")

    loader = DataLoader(ds, batch_size=6, shuffle=True, collate_fn=collate_pad)
    padded, lengths, ys = next(iter(loader))
    print(f"batch: padded={tuple(padded.shape)}  lengths={lengths.tolist()}  labels={ys.tolist()}")
    assert padded.shape[0] == len(lengths) == len(ys)
    assert padded.shape[2] == D

    print("\nLOSO split:")
    for held, tr, te in loso_splits(gold, tmp):
        print(f"  held-out {held}: train={len(tr)} test={len(te)}")

    print("\n✅ self-test passed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="Run on synthetic data (no real gold/arrays needed).")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--arrays-root", type=Path, default=None)
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.arrays_root is None:
        ap.error("--arrays-root is required (or use --self-test)")

    ds = GoldClipDataset(args.gold, args.arrays_root, min_count=args.min_count,
                         verbose=True)
    print(f"vocab size: {len(ds.vocab)}")
    print(f"examples:   {len(ds)}  (skipped {ds.skipped_gap} in-gap, "
          f"{ds.skipped_oov} below-min-count)")
    from collections import Counter
    per_signer = Counter(sig for *_, sig in ds.index)
    print(f"per-signer: {dict(per_signer)}")
    if len(ds):
        feat, y, sig = ds[0]
        print(f"sample: feat={tuple(feat.shape)}  label={ds.itos[y]}  signer={sig}")


if __name__ == "__main__":
    main()
