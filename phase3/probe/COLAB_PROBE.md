# SignCLIP-LSM probe — run on Google Colab

Goal: does SignCLIP already understand Mexican LSM? Six cells, paste into a **new**
notebook, run top to bottom in one sitting (Colab disconnects idle runtimes, which
would cost you the 2.6 GB checkpoint download again).

Open https://colab.research.google.com → New notebook.
(Runtime → GPU is nice but not required; 30 clips run fine on CPU.)

---

## Background — why Cell 6 exists

June's first probe (30 dictionary signs, prompted as `<es> <mfs> agua`) came back at
3.3% top-1 — chance. That was reported as "SignCLIP doesn't know LSM."

Zifan Jiang (SignCLIP/SEA first author, contacted 2026-07 after we shared the result)
said the text encoder was trained on **English only** (resource limit), and that LSM
**is** in the Spreadthesign training data. Our manifest labels are Spanish, so
`<es> <mfs> agua` likely put the text side out of distribution — the 3.3% may reflect
a prompting mistake, not a real gap in the model.

Cell 6 re-runs the same poses with English words to isolate that. Cell 4 reproduces
the original (confounded) Spanish result for comparison; Cell 5 is a sanity check that
the install/checkpoint/poses are all working, independent of the LSM question.

---

### Cell 1 — install SignCLIP (MMPT / fairseq)
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
Expect a lot of red pip dependency-conflict warnings (fairseq pins ancient
omegaconf/hydra) — that's normal. What matters is the final print.

### Cell 2 — download the multilingual checkpoint (~2.6 GB, the slow step)
```python
%cd /content/fairseq/examples/MMPT
import os
os.makedirs("runs/retri_v1_1/baseline_temporal", exist_ok=True)
!gdown --folder https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc -O /content/weights
!find /content/weights -name '*.pt'
!cp /content/weights/baseline_temporal_checkpoint_best.pt runs/retri_v1_1/baseline_temporal/checkpoint_best.pt
!ls -la runs/retri_v1_1/baseline_temporal/   # expect a ~2.6 GB checkpoint_best.pt
```
If the filename printed by `find` differs from `baseline_temporal_checkpoint_best.pt`,
edit the `cp` line to match before running it.

### Cell 3 — upload the data
```python
from google.colab import files
files.upload()   # pick signclip_probe_data.zip from your Mac
!unzip -o signclip_probe_data.zip -d /content/probe_data
!ls /content/probe_data/*.pose | wc -l
!head -3 /content/probe_data/manifest.csv
```
Expect `30` and the manifest starting with `agua`,`mujer`. If the upload widget
misbehaves, drag the zip into the file panel on the left instead, then just run the
`unzip`/`ls`/`head` lines.

### Cell 4 — build pose embeddings + reproduce June's baseline
```python
%cd /content/fairseq/examples/MMPT
import csv, numpy as np
from pose_format import Pose
from demo_sign import embed_pose, embed_text

def norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

rows = list(csv.DictReader(open("/content/probe_data/manifest.csv")))
words, poses = [], []
for r in rows:
    with open(f"/content/probe_data/{r['youtube_id']}.pose", "rb") as f:
        p = Pose.read(f.read())
    if p.body.data.shape[0] > 256:
        p.body.data = p.body.data[:256]
        p.body.confidence = p.body.confidence[:256]
    poses.append(p); words.append(r["label"])

P = norm(embed_pose(poses))
T_es = norm(embed_text([f"<es> <mfs> {w}" for w in words]))
sim = T_es @ P.T
n = len(words)
t1 = float(np.mean([i in np.argsort(-sim[i])[:1] for i in range(n)]))
print(f"Built {n} pose embeddings.")
print(f"Baseline (Spanish, June's original run): top-1 = {t1:.1%}  (chance = {1/n:.1%})")
```
Should print `top-1 = 3.3%` — matching June, confirming the session is behaving the
same way. `P` and `words` stay in memory for the next two cells.

### Cell 5 — sanity check (`house.pose`)
```python
with open("house.pose", "rb") as f:
    house = Pose.read(f.read())
cands = ["house", "dog", "water", "police", "school", "car", "sun", "tree"]
hp = norm(embed_pose([house]))
tt = norm(embed_text([f"<en> <ase> {w}" for w in cands]))
sims = (tt @ hp.T).ravel()
for w, s in sorted(zip(cands, sims), key=lambda x: -x[1]):
    print(f"  {w:8s} {s:.3f}")
```
Expect `house` on top around `0.668`. This confirms install + checkpoint + pose
extraction are all correct, independent of the LSM question — it prompts in English
(`<en> <ase> house`), which is exactly why it passed in June while the Spanish LSM run
failed: it never touched the path that turned out to be broken. If `house` is NOT on
top here, stop — something is wrong upstream and the numbers below won't mean anything.

### Cell 6 — the actual test (English words, isolates the real question)
```python
ES2EN = {
    "agua": "water", "mujer": "woman", "hombre": "man", "beber": "drink",
    "dormir": "sleep", "caminar": "walk", "correr": "run", "hablar": "talk",
    "escribir": "write", "rojo": "red", "azul": "blue", "verde": "green",
    "amarillo": "yellow", "negro": "black", "blanco": "white", "fuego": "fire",
    "flor": "flower", "libro": "book", "mesa": "table", "silla": "chair",
    "puerta": "door", "ventana": "window", "dinero": "money", "comida": "food",
    "leche": "milk", "pan": "bread", "café": "coffee", "gracias": "thank you",
    "hola": "hello", "amigo": "friend",
}
en_words = [ES2EN[w] for w in words]          # `words` and `P` carry over from Cell 4

# Vary ONLY the text side; P (poses) is fixed. The <en>/<es> pair isolates tag from language.
for tag, wl in [("<en> <mfs>", en_words), ("<es> <mfs>", en_words),
                ("<en> <ase>", en_words),                      # wrong-language control
                ("<es> <mfs>", words)]:                        # the old, confounded run
    T = norm(embed_text([f"{tag} {w}" for w in wl]))
    sim = T @ P.T
    n = len(wl)
    t1 = float(np.mean([i in np.argsort(-sim[i])[:1] for i in range(n)]))
    lang = "EN" if wl is en_words else "ES"
    print(f"{tag:12s} words={lang}   top-1 = {t1:5.1%}")
print(f"(chance = {1/len(en_words):.1%})")
```

**Reading the result.** `<en> <mfs>` + English clearly above chance → SignCLIP does see
LSM, and June's "doesn't transfer" conclusion was an artifact of the Spanish prompt —
the fine-tuning plan gets cheaper. Still at chance while `<en> <ase>` is also at chance
→ the pose side genuinely carries nothing for our signers, and June's conclusion
survives on better evidence. If `<en> <ase>` scores as well as `<en> <mfs>`, the
sign-language tag isn't doing any work and something upstream needs a closer look.

> **⚠️ CORRECTION 2026-08-01 — the last sentence of that rule is wrong, and this run hit
> exactly that case.** Observed: `<en> <mfs>` 23%, `<en> <ase>` 20%, i.e. tied. The rule
> read that as a defect. The SignCLIP paper (Jiang et al., EMNLP 2024) says otherwise:
> the tag *is* highly informative in-domain (§6.2 gets 0.99 recall@1 identifying the sign
> language from the tag alone), but the authors caveat that the model "can learn to
> identify **signers** for particular sign languages" — signer/studio cues that do not
> exist in our broadcast footage. Meanwhile cross-lingual sharing is deliberate: §2 cites
> "lexical similarity due to iconicity" as making transfer between sign languages easier
> than between spoken languages, and §6.1 credits "the hypothesized multilingual transfer
> effect thanks to sign language iconicity."
>
> **Correct reading: tag-insensitivity out-of-domain is expected, not a bug.** The
> above-chance signal is consistent with SignCLIP's intended shared, iconicity-driven
> semantic space. It does mean the 23% should not be cited as evidence of *LSM-specific*
> retrieval — but it is not evidence of a broken measurement either. See
> `fc_test/RESULTS.md` §"Vocabulary difficulty" for the full write-up.

---

## Part 2 — does it work cross-signer, on real interpreter footage?

Everything above used 30 clean, single-signer dictionary poses — best case for any
sign-language model. The question that actually decides whether hand-labelling can be
skipped is different: **does the same word, signed by a different person on live
broadcast footage, land near the matching dictionary sign?** That's what Part 2 tests.

Data: `phase3/probe/extract_interpreter_poses.sh` pulls already-cut mañanera clips
(from `phase3/gold/cut_reviews.sh`, June) for the 8 words that overlap the dictionary
set (agua, amigo, gracias, hablar, hombre, mesa, mujer, verde), up to 4 different
signers each — 32 candidate clips.

**These are weak labels (transcript-timing windows, never verified) and two things can
make one wrong (flagged 2026-07-29):** (1) the interpreter may have signed a different/
related word for the same concept instead (e.g. dinero → RECURSOS — LSM signs concepts,
not literal words), so the target sign may not be in the clip at all; (2) the lag
estimate is a soft ±3.5s prior, so even when the sign is there, a blind first-8.5s
truncation of the loose 14s window can miss or clip it. Both would show up as a false
"miss" in the SignCLIP test — indistinguishable from a real model failure unless caught
first. So before embedding anything:

1. `python phase3/probe/make_interp_review_csv.py` — builds `review.csv` for the 32
   clips (dictionary reference clip shown side by side, same review format used for
   the full gold set).
2. `python phase2/build_review_form.py --dir phase3/probe` — writes `review.html`.
3. Open `phase3/probe/review.html`, watch all 32 (a couple of minutes), mark
   → yes / ↓ doesn't appear / g negated, and press **s**/**e** at the sign's start/end
   in each clip you mark yes. Download `review.csv`, overwrite the one in `phase3/probe/`.
4. `conda activate lsm-probe && python phase3/probe/cut_verified_interp_clips.py` —
   drops anything not verified 'y', cuts a tight clip from the marked start/end
   (±0.3s buffer) for the rest, re-extracts `.pose`, rebuilds
   `signclip_probe_data_interpreter.zip`. It also prints which clips got dropped and
   why — that breakdown (how many were substituted/absent out of 32) is itself a real
   number worth keeping: it's your empirical weak-label noise rate.

Upload the resulting zip below. Because this is pose-to-pose, no text prompt is
involved — the earlier `<mfs>`-tag ambiguity doesn't come up here.

This is a **pose-to-pose** comparison — no text prompt, so the earlier `<mfs>`-tag
question doesn't even come up here. It reuses `P` and `words` already in memory from
Cell 4, so run this in the *same* notebook session, after Cell 4 (Cells 5–7 optional).

### Cell 8 — upload interpreter poses and test cross-signer retrieval
```python
from google.colab import files
files.upload()   # pick signclip_probe_data_interpreter.zip
!unzip -o signclip_probe_data_interpreter.zip -d /content/interp_data
```
```python
import csv
irows = list(csv.DictReader(open("/content/interp_data/manifest.csv")))
iwords, ipose_objs, isigners = [], [], []
for r in irows:
    path = f"/content/interp_data/{r['clip_id']}.pose"
    with open(path, "rb") as f:
        p = Pose.read(f.read())
    if p.body.data.shape[0] > 256:
        p.body.data = p.body.data[:256]
        p.body.confidence = p.body.confidence[:256]
    ipose_objs.append(p); iwords.append(r["word"]); isigners.append(r["signer"])

IP = norm(embed_pose(ipose_objs))     # interpreter-clip embeddings, same space as P
sim = IP @ P.T                         # (n_interp, 30) — each row: similarity to all 30 dict signs
dict_words = words                     # from Cell 4, the 30 dictionary labels, same order as P

correct1 = correct5 = 0
for i, (w, signer) in enumerate(zip(iwords, isigners)):
    order = np.argsort(-sim[i])
    ranked = [dict_words[j] for j in order]
    hit1 = ranked[0] == w
    hit5 = w in ranked[:5]
    correct1 += hit1; correct5 += hit5
    print(f"{w:10s} (signer {signer:14s}) → top match: {ranked[0]:10s}"
          f"  [{'✓' if hit1 else ' '}]   top-5 has it: {'✓' if hit5 else '✗'}")

n = len(iwords)
print(f"\ncross-signer, dict→interpreter top-1 = {correct1/n:.1%}   top-5 = {correct5/n:.1%}"
      f"   (chance top-1 = {1/30:.1%}, n={n})")
```

**Reading the result.** Clearly above chance (roughly >13%, i.e. >4x) → real cross-signer
signal exists on your actual target domain, and Jiang's suggested rung 2 (frozen
embeddings, zero training) is worth building out properly — hand-labelling for
TRAINING may indeed be skippable, only a modest eval set needed. At or near chance →
the dictionary-to-broadcast domain gap is too large for the frozen model alone;
fine-tuning (or full hand-labelling) is still the path, but now for a documented reason
rather than a guess.

---
**Likely snags (Colab is the right place to fix them, not locally — the authors'
inference environment is Linux/CUDA-only and doesn't build on arm64 Mac):**
- Cell 1: fairseq pins old `omegaconf`/`hydra` — dependency-conflict warnings are expected.
- Cell 2: the Drive filename for the checkpoint may differ from
  `baseline_temporal_checkpoint_best.pt` — match the `cp` line to the `find` output.
- Report the four printed top-1 numbers back and I'll tell you what they mean + next step.
