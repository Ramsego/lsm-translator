# Re-run: which language tag did §3 use? — RESOLVED 2026-08-01

> **✅ RUN COMPLETE. Answer: the tag makes no difference; §3 stands as published.**
>
> | tag | words | INTERP top-1 | INTERP MRR | DICT top-1 | DICT MRR |
> |---|---|---|---|---|---|
> | `<en> <mfs>` | EN | 3.8% | 0.187 | 12.5% | 0.324 |
> | `<es> <mfs>` | EN | 3.8% | 0.192 | 12.5% | 0.322 |
> | `<en> <ase>` | EN | 11.5% | 0.249 | 6.2% | 0.272 |
> | `<es> <mfs>` | ES | 11.5% | 0.274 | 0.0% | 0.163 |
>
> Chance top-1 6.25%, MRR 0.211. English words held fixed, `<en>` vs `<es>` differ by 0.005
> and 0.002 MRR — nothing. The published §3 numbers reproduce (3.8%/0.182 → 3.8%/0.187;
> 12.5%/0.316 → 12.5%/0.324). Full reading in `RESULTS.md` §3. Instructions kept below for
> reproducibility.

# Original instructions

**The one question this answers:** `RESULTS.md` §3 reports text-to-pose numbers (interpreter
3.8% top-1, dictionary 12.5%) but the Colab cells that produced them were never saved, so the
**language tag** is unrecorded. SignCLIP's convention is `<spoken_lang> <sign_lang>` and its
text encoder is English-only, so an `<es>` tag is out of distribution even when the words are
English. This run settles it and re-measures §3 cleanly.

**Scope:** only §3's absolute numbers are at stake. §1 and §2 are pose-to-pose and never touch
the text encoder. §4 (the p=0.038 domain gap) holds the text fixed and varies only the pose
source, so a wrong tag cancels on both sides.

Time: ~20 minutes, most of it the checkpoint download. No GPU needed.

---

## Step 0 — files to upload (from this Mac)

Two zips, both self-contained (each includes its own manifest):

```
phase3/probe/fc_test/query_poses.zip            (9.2 MB, 26 interpreter poses)
phase3/probe/fc_test/candidate_poses_min.zip    (9.7 MB, 16 reference poses)   <-- use this
```

**Use `candidate_poses_min.zip`, not `candidate_poses.zip`.** The full 91 MB archive holds all
144 candidates because the *pose-to-pose forced choice* (Cells 4–5) needs the 128 distractors.
This ablation only ever loads the 16 `role == "true"` reference signs, so the minimal zip is
byte-for-byte sufficient — verified: 16/16 vocab resolvable, 26/26 query clips retained,
chance 6.25%, identical to the full archive. It cuts the upload from ~100 MB to ~19 MB.

Rebuild it any time from `candidate_poses/` + `candidate_manifest.csv` if it goes missing.

**If uploads are still slow, use Google Drive instead of the `files.upload()` widget** — that
widget streams through the notebook channel and is slow and flaky above a few MB. Put both zips
in Drive once via drive.google.com, then in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -o "/content/drive/MyDrive/query_poses.zip"         -d /content/query_data
!unzip -o "/content/drive/MyDrive/candidate_poses_min.zip" -d /content/candidate_data
```

This also survives session disconnects, so a re-run costs nothing but the checkpoint download.

## Step 1 — new Colab notebook

Go to https://colab.research.google.com → **New notebook**. Runtime type doesn't matter (CPU
is fine). Run everything in one sitting — an idle disconnect costs you the 2.6 GB download.

## Step 2 — install (paste as cell 1, run)

```python
%cd /content
!pip install "pip<24.1" -q
!rm -rf fairseq
!git clone --depth 1 https://github.com/J22Melody/fairseq.git
%cd /content/fairseq
!pip install -e . -q
%cd /content/fairseq/examples/MMPT
!pip install -e . -q
!pip install -q transformers==4.51.2 hydra-core==1.0.7 pose-format gdown
print("Install OK")
```

Red dependency-conflict warnings are normal (fairseq pins old omegaconf/hydra). Only the final
`Install OK` matters.

## Step 3 — checkpoint, ~2.6 GB (paste as cell 2, run, wait)

```python
%cd /content/fairseq/examples/MMPT
import os
os.makedirs("runs/retri_v1_1/baseline_temporal", exist_ok=True)
!gdown --folder https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc -O /content/weights
!find /content/weights -name '*.pt'
!cp /content/weights/baseline_temporal_checkpoint_best.pt runs/retri_v1_1/baseline_temporal/checkpoint_best.pt
!ls -la runs/retri_v1_1/baseline_temporal/
```

If `find` prints a different filename, edit the `cp` line to match before running it.

## Step 4 — upload the data (paste as cell 3, run)

```python
from google.colab import files
files.upload()   # choose query_poses.zip
!unzip -o query_poses.zip -d /content/query_data
files.upload()   # choose candidate_poses_min.zip
!unzip -o candidate_poses_min.zip -d /content/candidate_data
!ls /content/query_data/*.pose | wc -l      # expect 26
!ls /content/candidate_data/*.pose | wc -l  # expect 16
```

If the upload widget stalls, drag the zips into the file panel on the left instead, then run
just the `unzip`/`ls` lines — or use the Drive route in Step 0, which is faster and persists.

## Step 5 — THE RUN (paste as cell 4, run)

Self-contained: embeds both pose sets and sweeps the tag. Nothing else needed.

```python
%cd /content/fairseq/examples/MMPT
import csv, os
import numpy as np
from pose_format import Pose
from demo_sign import embed_pose, embed_text

def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def load_pose(path):
    with open(path, "rb") as f:
        p = Pose.read(f.read())
    if p.body.data.shape[0] > 256:          # same 256-frame cap as the original run
        p.body.data = p.body.data[:256]
        p.body.confidence = p.body.confidence[:256]
    return p

ES2EN = {
    "gracias": "thank you", "joven": "young", "actividad": "activity",
    "atender": "attend", "equipo": "team", "decidir": "decide",
    "derecho": "right", "experiencia": "experience", "ganar": "win",
    "familia": "family", "encontrar": "find", "edad": "age",
    "director": "director", "aprovechar": "take advantage",
    "cambiar": "change", "futuro": "future",
}

# dictionary side: the 16 reference signs (role == "true")
crows = list(csv.DictReader(open("/content/candidate_data/candidate_manifest.csv")))
true_yid = {r["word"]: r["youtube_id"] for r in crows if r["role"] == "true"}

qrows_all = list(csv.DictReader(open("/content/query_data/query_manifest.csv")))
vocab = [w for w in sorted({r["word"] for r in qrows_all})
         if w in true_yid and os.path.exists(f"/content/candidate_data/{true_yid[w]}.pose")]
qrows = [r for r in qrows_all if r["word"] in vocab]          # keep sides consistent
q_words = [r["word"] for r in qrows]

Q = unit(embed_pose([load_pose(f"/content/query_data/{r['clip_id']}.pose") for r in qrows]))
D = unit(embed_pose([load_pose(f"/content/candidate_data/{true_yid[w]}.pose") for w in vocab]))
print(f"interpreter poses {len(Q)} | dictionary poses {len(D)} | vocab {len(vocab)}")
missing = [w for w in ES2EN if w not in vocab]
if missing: print("  (dropped, no reference pose:", missing, ")")

def score(T, poses, gold):
    """Each pose ranks all |vocab| texts; report top-1 and MRR."""
    sim = poses @ T.T
    t1 = rr = 0.0
    for i, g in enumerate(gold):
        rank = int(np.where(np.argsort(-sim[i]) == g)[0][0]) + 1
        t1 += (rank == 1)
        rr += 1.0 / rank
    return t1 / len(gold), rr / len(gold)

q_gold = [vocab.index(w) for w in q_words]
d_gold = list(range(len(vocab)))
chance = 1 / len(vocab)
chance_mrr = float(np.mean([1 / r for r in range(1, len(vocab) + 1)]))
print(f"chance: top-1 {chance:.1%}   MRR {chance_mrr:.3f}\n")
print(f"{'tag':<12}{'words':<7}{'INTERP top1':>12}{'INTERP MRR':>12}{'DICT top1':>11}{'DICT MRR':>10}")
print("-" * 64)
for tag, use_en in [("<en> <mfs>", True),    # correct configuration
                    ("<es> <mfs>", True),    # tag OOD, words EN -- the suspected §3 run
                    ("<en> <ase>", True),    # wrong sign language: is the tag doing work?
                    ("<es> <mfs>", False)]:  # fully Spanish -- June's original 3.3% failure
    wl = [ES2EN[w] for w in vocab] if use_en else vocab
    T = unit(embed_text([f"{tag} {w}" for w in wl]))
    it1, imrr = score(T, Q, q_gold)
    dt1, dmrr = score(T, D, d_gold)
    print(f"{tag:<12}{'EN' if use_en else 'ES':<7}{it1:>11.1%}{imrr:>12.3f}{dt1:>10.1%}{dmrr:>10.3f}")
```

## Step 6 — send me the output

Paste back the whole printed block (the two header lines plus the four result rows). That is
all I need. I will interpret it and update `RESULTS.md` §3, `PIPELINE_MAP.md`, and the reply
to Zifan.

---

## What each outcome means (written before seeing the numbers)

| observation | conclusion |
|---|---|
| `<en> <mfs>` DICT clearly beats `<es> <mfs>` DICT | §3 was run under a broken prompt; its absolute numbers are superseded by the `<en>` row |
| the two are close | the tag was not the problem; §3's numbers stand as published |
| `<en> <ase>` ≈ `<en> <mfs>` again | replicates the 23%/20% tie on a second, harder word set — supports the reading that language conditioning lives in in-domain signer/studio cues, and that what transfers is the iconicity-driven shared space (SignCLIP paper §2, §6.1, §6.2) |
| `<en> <ase>` clearly below `<en> <mfs>` | the tag *does* carry LSM-specific information even out-of-domain — the earlier tie was particular to those 30 concrete words, and LSM-specific retrieval is real but weak |
| INTERP stays flat while DICT rises under any tag | the domain gap is the binding constraint, independent of prompting — strengthens the §4 conclusion and the fine-tuning prerequisite |

Note the interpreter column is expected to be weak in every row; §1/§2 already established that
pose-to-pose retrieval on this footage is at chance. The informative comparison is **within a
column, across tags** — not between the two columns.
