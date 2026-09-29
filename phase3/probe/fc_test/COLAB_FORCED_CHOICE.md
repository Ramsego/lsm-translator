# Forced-choice SignCLIP test — run on Google Colab

Answers the load-bearing question flagged in review: can SignCLIP's pretrained
pose-to-pose embedding tell the true sign apart from a small set of distractors, on
this project's own footage? If not, fine-tuning is a prerequisite for the whole
candidate-generation pipeline, not an optional nice-to-have.

**Design (reviewed by two independent ML-expert passes before running):**
- 26 query trials: human-verified (word, tight video clip) instances from
  `review_2.csv`, one interpreter, one 30-min slice. Selected for a different purpose
  (checking DP alignment quality) before this test was conceived — not cherry-picked
  for SignCLIP-friendliness.
- Two paired distractor conditions per trial, k=5 (true + 4 distractors) each:
  **random** (tests basic recognition) and **semantic-near** (tests fine-grained
  disambiguation — the harder, more realistic condition, since that's what the
  eventual LLM candidate-generation stage would actually produce as confusable
  options). Distractors hand-picked/seeded from the 886-label Phase-1 inventory, not
  SignCLIP's own training data — see provenance note below.
- **Pose-to-pose only, no text encoder used anywhere in this test.** Earlier probing
  on this project found SignCLIP's text encoder is English-only; Spanish-language text
  queries measured near-chance for a reason unrelated to sign recognition. Avoided
  entirely here.
- **Provenance check:** this project's 886-label inventory comes from `wikisigns.org`
  and a YouTube channel source ("chnt") — confirmed distinct from Spreadthesign, the
  corpus SignCLIP was pretrained on. So a positive result here reflects genuine
  cross-corpus transfer, not memorization of the reference clips.
- **Pre-registered decision thresholds (fixed before seeing results):**
  top-1 accuracy ≥45% (exact-binomial p<0.01 vs. chance=20%) → clear go, build the
  candidate-generation pipeline. <25% → clear no-go, fine-tune SignCLIP on this
  project's own footage first. 25–45% → inconclusive at this n; informative but not
  decisive alone.
- Reported: Wilson 95% CIs (not just a point estimate — n=26 is a real constraint,
  see review), MRR and top-3 as secondary metrics (more statistically efficient per
  trial than top-1), per-word breakdown, and k=3 in addition to k=5 for sensitivity.

**What this result does NOT license, regardless of outcome:** any claim beyond this
one signer, this one 30-minute slice, this 16-word subset of the 886-label inventory.
It's a go/no-go gate for whether to invest in the next pipeline stage, not a general
capability estimate for LSM.

**QC finding worth reading results alongside (measured, not assumed):** both reviewers
predicted the confound would run dictionary-clean vs. footage-messy. Measured hand-landmark
detection rate says the opposite. Query clips (tightly cut to the human-marked sign
moment): 99.9% of frames have detected hand landmarks. Candidate/dictionary clips: 65.8%
mean, 45% median, with 80/144 below 50% -- these reference videos are not tightly
trimmed to the sign itself (some carry lead-in/lead-out where the signer's hands are
out of frame or at rest), so a meaningful fraction of each candidate's pose sequence
is dead frames, not a cleaner version of the query side. If results come back weak,
this is a live alternative explanation to "SignCLIP can't discriminate" -- the
candidate embeddings may be diluted by padding, independent of the model's real
capability. Not corrected for here (would need per-clip trimming to the high-confidence
window, deferred as its own step rather than risk introducing new bias mid-pipeline);
disclosed so it's part of how any result gets read, not silently absorbed into the verdict.

---

### Cell 1 — install SignCLIP (MMPT / fairseq)
Same as the original probe. Skip if running in the same notebook session as an earlier
Colab run.
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
print("Install OK — proceed to Cell 2.")
```

### Cell 2 — download the multilingual checkpoint (skip if already have it)
```python
%cd /content/fairseq/examples/MMPT
import os
os.makedirs("runs/retri_v1_1/baseline_temporal", exist_ok=True)
!gdown --folder https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc -O /content/weights
!find /content/weights -name '*.pt'
!cp /content/weights/baseline_temporal_checkpoint_best.pt runs/retri_v1_1/baseline_temporal/checkpoint_best.pt
```

### Cell 3 — upload the data
Two zips, built locally: `query_poses.zip` (26 files + manifest) and
`candidate_poses.zip` (144 files + manifest).
```python
from google.colab import files
files.upload()   # query_poses.zip
!unzip -o query_poses.zip -d /content/query_data
files.upload()   # candidate_poses.zip
!unzip -o candidate_poses.zip -d /content/candidate_data
!ls /content/query_data/*.pose | wc -l      # expect 26
!ls /content/candidate_data/*.pose | wc -l  # expect 144 (some fewer if extraction dropped any)
```

### Cell 4 — embed everything (pose-to-pose, no text encoder)
```python
%cd /content/fairseq/examples/MMPT
import csv, numpy as np
from pose_format import Pose
from demo_sign import embed_pose

def norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

def load_pose(path):
    with open(path, "rb") as f:
        p = Pose.read(f.read())
    if p.body.data.shape[0] > 256:
        p.body.data = p.body.data[:256]
        p.body.confidence = p.body.confidence[:256]
    return p

# query side
qrows = list(csv.DictReader(open("/content/query_data/query_manifest.csv")))
q_ids = [r["clip_id"] for r in qrows]
q_words = [r["word"] for r in qrows]
q_poses = [load_pose(f"/content/query_data/{cid}.pose") for cid in q_ids]
Q = norm(embed_pose(q_poses))   # (26, D)

# candidate side -- one embedding per distinct youtube_id, looked up per trial below
crows = list(csv.DictReader(open("/content/candidate_data/candidate_manifest.csv")))
cand_ids = sorted({r["youtube_id"] for r in crows})
cand_poses, kept_ids = [], []
for yid in cand_ids:
    path = f"/content/candidate_data/{yid}.pose"
    import os
    if os.path.exists(path):
        cand_poses.append(load_pose(path)); kept_ids.append(yid)
    else:
        print(f"  (missing pose, dropped: {yid})")
C = norm(embed_pose(cand_poses))            # (n_candidates, D)
id_to_row = {yid: i for i, yid in enumerate(kept_ids)}

sim_all = Q @ C.T   # (26, n_candidates) -- full similarity matrix, sliced per trial below
print(f"Embedded {len(q_ids)} query clips, {len(kept_ids)} distinct candidate signs.")
```

### Cell 5 — pre-registered forced-choice evaluation
```python
import math
from collections import defaultdict

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def run_condition(role, k_size):
    """role: 'random_distractor' or 'semantic_distractor'. k_size: 3 or 5 (true + k_size-1 distractors)."""
    by_word = defaultdict(lambda: defaultdict(list))
    for r in crows:
        by_word[r["word"]][r["role"]].append(r["youtube_id"])

    top1 = top3 = 0
    rr_sum = 0.0
    n_valid = 0
    per_word = defaultdict(lambda: [0, 0])  # word -> [hits, total]
    for i, (qid, word) in enumerate(zip(q_ids, q_words)):
        true_yid = by_word[word]["true"][0]
        distractor_yids = by_word[word][role][:k_size - 1]
        cand_yids = [true_yid] + distractor_yids
        cand_yids = [y for y in cand_yids if y in id_to_row]
        if true_yid not in cand_yids or len(cand_yids) < 2:
            continue  # true or all distractors missing pose -- skip, don't silently miscount
        n_valid += 1
        rows_idx = [id_to_row[y] for y in cand_yids]
        sims = sim_all[i, rows_idx]
        order = np.argsort(-sims)
        true_pos = list(order).index(0)  # true is always cand_yids[0]
        top1 += (true_pos == 0)
        top3 += (true_pos < min(3, len(cand_yids)))
        rr_sum += 1.0 / (true_pos + 1)
        per_word[word][0] += (true_pos == 0)
        per_word[word][1] += 1

    chance = 1 / k_size
    lo, hi = wilson_ci(top1, n_valid, z=1.96)
    print(f"\n── {role}, k={k_size} (chance={chance:.1%}) ──")
    print(f"  n valid trials : {n_valid}/26")
    print(f"  top-1 accuracy : {top1}/{n_valid} = {top1/n_valid:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  top-3 accuracy : {top3/n_valid:.1%}")
    print(f"  MRR            : {rr_sum/n_valid:.3f}")
    print(f"  per-word: " + ", ".join(f"{w}={h}/{t}" for w, (h, t) in sorted(per_word.items())))
    return top1 / n_valid, lo, hi

results = {}
for role in ["random_distractor", "semantic_distractor"]:
    for k in [5, 3]:
        results[(role, k)] = run_condition(role, k)
```

### Cell 6 — pre-registered verdict
```python
acc, lo, hi = results[("random_distractor", 5)]
print("\n──────────── VERDICT (pre-registered, random condition, k=5) ────────────")
if lo > 0.20 and acc >= 0.45:
    print(f"GO — top-1={acc:.1%}, CI excludes chance and clears the 45% bar.")
    print("Candidate-generation pipeline is worth building as designed.")
elif hi < 0.25:
    print(f"NO-GO — top-1={acc:.1%}, CI stays near/below chance.")
    print("Fine-tune SignCLIP on this project's own footage before building further.")
else:
    print(f"INCONCLUSIVE — top-1={acc:.1%}, CI [{lo:.1%},{hi:.1%}] straddles the pre-registered bands.")
    print("Informative but not decisive at n=26; treat as a pilot, consider growing n before deciding.")
print("\nCompare against semantic_distractor condition above: a much larger drop there than in")
print("random_distractor would indicate the recognition itself is fine but fine-grained sense")
print("disambiguation is the harder problem — relevant to how much weight to put on the LLM stage.")
```

---
**Notes for the operator (same environment caveats as the original probe):**
- Cells 1–2 can be skipped if resuming an existing Colab session with the checkpoint loaded.
- If `demo_sign.embed_pose` isn't the right entry point in the current fairseq/MMPT
  checkout, check `demo_sign.py` in `examples/MMPT` for the current pose-embedding
  function signature — this project's earlier probe already validated this exact
  import path works for pose-to-pose embedding (no text encoder involved), so it
  should not need re-deriving from scratch.
- Report back the six numbers from Cell 5 (top-1/top-3/MRR × 2 conditions) plus the
  Cell 6 verdict line.

---

### Cell 7 — text-to-pose WITH THE LANGUAGE TAG VARIED (added 2026-08-01)

**Why this cell exists.** The §3 numbers in `RESULTS.md` (text-to-pose, English prompts)
were produced by cells written ad hoc in Colab and never pasted back here, so the exact
prompt string — specifically the **language tag** — is unrecorded. That matters: SignCLIP's
tag convention is `<spoken_language> <sign_language>`, its text encoder was trained on
English only, and an `<es>` tag is therefore out of distribution regardless of whether the
words themselves are English. Prompting `<es> <mfs> agua` is exactly what produced this
project's original 3.3% (chance) result in June 2026.

It also settles a live question from the earlier 30-word probe: `<en> <mfs>` scored 23%
but the **wrong-sign-language control `<en> <ase>` scored 20%**, essentially tied. By this
project's own pre-registered reading rule (`COLAB_PROBE.md`, Cell 6), a tied control means
the sign-language tag is doing no work and LSM-specificity is unconfirmed. Re-testing on the
fc_test data with a proper per-tag breakdown resolves both at once.

Run after Cell 4 (needs `Q`, `q_words`, `C`, `kept_ids`, `id_to_row`, `crows` in memory).
**Paste the printed table back into `RESULTS.md` §3 when done.**

```python
import numpy as np
from collections import defaultdict
from demo_sign import embed_text

ES2EN = {
    "gracias": "thank you", "joven": "young", "actividad": "activity",
    "atender": "attend", "equipo": "team", "decidir": "decide",
    "derecho": "right", "experiencia": "experience", "ganar": "win",
    "familia": "family", "encontrar": "find", "edad": "age",
    "director": "director", "aprovechar": "take advantage", "cambiar": "change",
    "futuro": "future",
}
vocab = sorted(set(q_words))                       # the 16 gold word types
assert all(w in ES2EN for w in vocab), [w for w in vocab if w not in ES2EN]

# dictionary pose for each word = its 'true' candidate row
true_yid = {}
for r in crows:
    if r["role"] == "true":
        true_yid[r["word"]] = r["youtube_id"]
dict_rows = [id_to_row[true_yid[w]] for w in vocab if true_yid.get(w) in id_to_row]
dict_words = [w for w in vocab if true_yid.get(w) in id_to_row]
D_pose = C[dict_rows]                              # (16, D) dictionary poses

def score(T, poses, gold_idx):
    """For each pose, rank the |vocab| texts; return top-1 and MRR."""
    sim = poses @ T.T                              # (n_poses, n_vocab)
    top1 = rr = 0.0
    for i, g in enumerate(gold_idx):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == g)[0][0]) + 1
        top1 += (rank == 1); rr += 1.0 / rank
    n = len(gold_idx)
    return top1 / n, rr / n

q_gold = [vocab.index(w) for w in q_words]
d_gold = [vocab.index(w) for w in dict_words]
chance = 1 / len(vocab)
chance_mrr = float(np.mean([1/r for r in range(1, len(vocab)+1)]))

print(f"vocab={len(vocab)}  chance top-1={chance:.1%}  chance MRR={chance_mrr:.3f}")
print(f"{'tag':<14}{'words':<8}{'interp top1':>12}{'interp MRR':>12}{'dict top1':>11}{'dict MRR':>10}")
for tag, use_en in [("<en> <mfs>", True),    # correct configuration
                    ("<es> <mfs>", True),    # tag OOD, words English -- the suspected §3 run
                    ("<en> <ase>", True),    # wrong sign language: is the tag doing work?
                    ("<es> <mfs>", False)]:  # fully Spanish -- June's original failure
    wl = [ES2EN[w] for w in vocab] if use_en else vocab
    T = embed_text([f"{tag} {w}" for w in wl])
    T = np.asarray(T, dtype=np.float32)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    it1, imrr = score(T, Q, q_gold)
    dt1, dmrr = score(T, D_pose, d_gold)
    print(f"{tag:<14}{'EN' if use_en else 'ES':<8}{it1:>11.1%}{imrr:>12.3f}{dt1:>10.1%}{dmrr:>10.3f}")
```

**How to read it.**

- If `<en> <mfs>` beats `<es> <mfs>` on the dictionary column, §3 was run under a broken
  prompt and its absolute numbers are superseded. Report the corrected row.
- If `<en> <ase>` ties `<en> <mfs>` again, then whatever above-chance signal exists is **not
  LSM-specific** — it is the English text encoder plus general visual-semantic alignment. That
  is a more interesting finding than a prompting bug and is worth telling the SignCLIP authors.
- **§4 (the paired dictionary-vs-interpreter domain gap, p=0.038) is unaffected either way.**
  It holds the text embedding fixed and varies only the pose source, so a wrong tag cancels
  on both sides of the comparison. §1 and §2 are pose-to-pose and never touch the text
  encoder at all. Only §3's absolute numbers are at risk here.
