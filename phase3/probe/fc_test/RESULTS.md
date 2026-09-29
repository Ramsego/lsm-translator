# Forced-choice SignCLIP test — results (2026-07-31)

**Question:** can off-the-shelf SignCLIP serve as the ranking stage of the proposed
candidate-generation pipeline (SEA → LLM-constrained selection over the 886-label
inventory → SignCLIP picks the right candidate from video)? If it can't distinguish a
true sign from a handful of distractors on *this* footage, no amount of upstream
candidate quality rescues the pipeline, and fine-tuning becomes a prerequisite rather
than an option.

**Design** was reviewed by two independent ML-expert passes before anything was run.
Both flagged the same things: n=26 is underpowered for a modest effect; run random *and*
semantic distractors as separate conditions; report CIs not bare significance; account
for word-level clustering (26 trials span only 16 words); check for pretraining overlap;
QC pose quality on both sides. All incorporated. Thresholds pre-registered before seeing
any number.

## Data

- **Query side:** 26 human-verified instances from `review_2.csv` — a person watched real
  mañanera interpreter footage and hand-marked exact sign start/end for a given Spanish
  word. One interpreter, one 30-min slice of one video. Selected months earlier for a
  different purpose (DP alignment checking), not curated for this test.
- **Candidate side:** each word's own Phase-1 reference sign + 4 random + 4 semantic-near
  distractors drawn from the 886-label inventory (144 distinct videos).
- **Provenance:** the 886-label inventory comes from wikisigns.org and a YouTube channel
  source — **not** Spreadthesign, SignCLIP's pretraining corpus. No memorization path.

## Results

### 1. Forced choice, pose-to-pose (pre-registered primary)

| condition | k | top-1 | 95% CI | top-3 | MRR |
|---|---|---|---|---|---|
| random distractors | 5 | 38.5% (10/26) | [22.4%, 57.5%] | 76.9% | 0.599 |
| random distractors | 3 | 46.2% (12/26) | [28.8%, 64.5%] | — | 0.686 |
| semantic-near | 5 | 34.6% (9/26) | [19.4%, 53.8%] | 73.1% | 0.568 |
| semantic-near | 3 | 53.8% (14/26) | [35.5%, 71.2%] | — | 0.731 |

Chance = 20% (k=5), 33.3% (k=3). **Pre-registered verdict: INCONCLUSIVE** — CI neither
excludes chance nor clears the 45% GO bar. (Note: top-3 at k=3 is tautologically 100%
and was a design error in the notebook — meaningless, ignore it. k=3 is also the same
trial with two distractors dropped, not an independent test.)

### 2. Full-inventory retrieval, pose-to-pose (exploratory, same data)

Ranking each query against all 144 candidates rather than a 5-way choice — a much
sharper null, and it uses information the forced choice discards.

- observed MRR **0.0695** vs null mean 0.0435 (null 95th pct 0.0790)
- **permutation p = 0.102**, 20,000 word-level shuffles (shuffling at word level so
  `decidir`'s 4 repeats can't inflate significance)

Above chance, still not significant. Working backwards: observed total reciprocal rank
1.81 vs null 1.13 — that gap is roughly "one or two queries landed near rank 1, the rest
scattered," not a broad modest signal. **Not pre-registered**; more sensitive read on the
same data, exploratory status.

### 3. Text-to-pose with English prompts

SignCLIP is a text↔pose contrastive model — pose-to-pose is off-label use. Its text
encoder is English-only (confirmed with a SignCLIP author; Spanish prompts measured
near-chance for that reason, not a capability gap). So this tests the configuration the
model was actually trained for, using English translations of the 16 words.

| pose source | top-1 | chance | MRR | chance MRR | median rank |
|---|---|---|---|---|---|
| **interpreter footage** (26 clips) | 3.8% (1/26) | 6.25% | 0.182 | 0.211 | 9.0 / 16 |
| **dictionary refs** (16 clips) | 12.5% (2/16) | 6.25% | 0.316 | 0.211 | — |

Interpreter footage is **flat at chance** (marginally below is noise at this n).
Dictionary references are **above chance but weakly** (~1.5× on MRR, roughly p≈0.04).

### 4. Paired comparison — the cleanest measurement we have

Same 16 words, same text embedding vectors, only the pose source differs:

**Dictionary beats interpreter on 12 of 16 words, sign test one-sided p = 0.038.**

This is the domain gap, measured directly and controlling for vocabulary.

## Two hypotheses tested and killed

**Padding does not explain the weak control.** QC found candidate clips average only
65.8% hand-detected frames (median 45%; 9 of the 16 true signs sit at 31–42%, i.e. mostly
dead frames from untrimmed lead-in/lead-out). Natural hypothesis: degraded candidate
embeddings are suppressing the signal. Tested by splitting the 16 control words:

| subset | n | mean rank | MRR |
|---|---|---|---|
| clean (100% hand-detected) | 7 | 6.86 | 0.335 |
| padded (31–42%) | 9 | 6.44 | 0.302 |

Essentially identical, both above chance. **Padding is not the problem** — re-trimming all
144 clips would not have rescued this. Useful negative: saves the work.

**The language tag is ruled out — now measured, not assumed (ablation run 2026-08-01).**
The concern was that §3 might have used an `<es>` tag, which SignCLIP's English-only text
encoder never saw in training. Re-ran the full §3 protocol sweeping the tag with the poses and
words held fixed (`RERUN_TAG_ABLATION.md`):

| tag | words | INTERP top-1 | INTERP MRR | DICT top-1 | DICT MRR |
|---|---|---|---|---|---|
| `<en> <mfs>` | EN | 3.8% | 0.187 | 12.5% | 0.324 |
| `<es> <mfs>` | EN | 3.8% | 0.192 | 12.5% | 0.322 |
| `<en> <ase>` | EN | 11.5% | 0.249 | 6.2% | 0.272 |
| `<es> <mfs>` | ES | 11.5% | 0.274 | 0.0% | 0.163 |

Chance: top-1 6.25%, MRR 0.211.

**Result: the tag changes nothing.** With English words held fixed, `<en>` and `<es>` give
*identical* top-1 in both columns (3.8%/3.8% and 12.5%/12.5%) and MRR deltas of 0.005 and
0.002. §3 was not corrupted by prompting, whichever tag it used, and **the originally reported
§3 numbers reproduce closely** (published 3.8%/MRR 0.182 interpreter, 12.5%/MRR 0.316
dictionary; re-run 3.8%/0.187 and 12.5%/0.324). §3 stands as published.

**What the last two rows are, and are not.** The `<en> <ase>` and Spanish-words rows show a
*higher* interpreter score (11.5% = 3/26 vs 1/26). That is not ASL beating LSM. Both are
indistinguishable from chance (binomial p=0.22, 95% CI [2.4%, 30.2%], expected 1.6 hits), and
the interpreter column rises exactly when the text side is *degraded* — the signature of a
ranking collapsing toward random, not of signal. The systematic behaviour is all in the
dictionary column: English words put it above chance (MRR 0.324 vs 0.211) and Spanish words
push it *below* chance (0.163, top-1 0/16), which is the mechanism behind the original June
3.3% failure, reproduced here.

**One suggestive but underpowered observation:** on dictionary clips `<en> <mfs>` beats
`<en> <ase>` on MRR (0.324 vs 0.272), i.e. naming the right sign language helps on citation
form. At n=16 this is not significant and should not be cited as a finding — but it is the
direction the SignCLIP paper's §6.2 predicts, and it contrasts with the interpreter column
where the tag does nothing. Consistent with language conditioning being tied to in-domain
recording conditions.

## What remains as explanation

Three compounding factors, not cleanly separable at this n:

1. **Register/domain gap** — broadcast interpreting is fast and co-articulated; SignCLIP's
   training data is slow citation form. Directly supported by §4 (p=0.038).
2. **Clip duration asymmetry** — interpreter clips average 56 frames (45–78, i.e. ~1.5–2.6s
   at 30fps, tightly cut to the marked sign); dictionary clips average 170. A 3× difference
   in sequence length going into the same encoder.
3. **Vocabulary difficulty** — these 16 words are abstract political/administrative
   vocabulary (*derecho, actividad, experiencia, decidir, director, futuro*). An earlier
   probe on 30 *concrete* words (agua, perro, casa, rojo, mesa) hit 23% top-1 against 3.3%
   chance — ~7× chance, far stronger than the ~1.5× seen here on dictionary clips. Part of
   this result is that our vocabulary is harder, not only that our footage is harder.

   **Compare the two probes by ratio, not raw percent** (added 2026-08-01): 23% against a
   1/30 chance floor is ~7× chance; 12.5% against a 1/16 floor is 2× chance. The candidate
   pools differ, so the raw 23%→12.5% understates the drop — in ratio terms it is 7×→2×.

   **Three candidate mechanisms, and one that can be ruled out.** *Not* vocabulary coverage:
   all of `derecho, actividad, futuro, experiencia` return entries on Spreadthesign's Mexican
   edition (checked 2026-08-01; crude markup probe, and presence today is not proof of
   presence in SignCLIP's filtered 2023 crawl of 18,423 concepts — but nothing marks the
   abstract words as absent). What remains:
   - **Iconicity.** The SignCLIP paper names iconicity as the mechanism behind its
     cross-lingual transfer (§2, §6.1) and ranks signs by cross-language embedding variance,
     with the universal-handshape SCORPION at the top. Concrete words have signs that
     resemble their referent; *derecho, experiencia, aprovechar, futuro* do not. Abstract
     vocabulary removes exactly the property the model leans on.
   - **Semantic clustering.** The 30-word set spans colours, food, furniture and motion —
     mutually distant signs. These 16 are one register of abstract administrative verbs and
     nouns, plausibly sharing neutral signing space and movement profiles, so they are more
     confusable with each other regardless of model quality.
   - **Lossy English translation** — a weakness of §3's own design. `agua`→`water` is exact;
     `atender`→`attend` is not (closer to "see to / care for"), and `aprovechar`→`take
     advantage` is a phrase, not a word. The text side is noisier for this word set than for
     the concrete one, independent of anything on the pose side.

   **Practical consequence for the project:** mañanera vocabulary *is* abstract
   administrative language. The harder half of the distribution is the half we need.

   **On that 23%, and the tied control — reading corrected 2026-08-01 after reading the
   SignCLIP paper (Jiang et al., EMNLP 2024).** In the same ablation the wrong-sign-language
   control `<en> <ase>` scored **20%**, statistically tied with `<en> <mfs>`'s 23%. The
   reading rule pre-registered in `COLAB_PROBE.md` Cell 6 called that "the sign-language tag
   isn't doing any work → something upstream needs a closer look." **That rule was written
   without knowledge of the paper's §6.2 and it over-reads the tie.** What the paper actually
   establishes:

   - **The tag is not inert.** §6.2: sign language identification "can be achieved by simply
     ranking text prompts of different sign languages without actual content, e.g.
     `<en><ase>` for ASL," reaching **0.99 recall@1** in-domain. The model encodes language
     identity strongly.
   - **But the authors caveat that result themselves:** "the identification task is ill-defined
     in that the model can learn to identify **signers** for particular sign languages." The
     tag may be keying on signer/studio identity, not linguistic structure — and those cues do
     not exist in our broadcast footage.
   - **Cross-lingual sharing is a design goal, not an accident.** §2: "lexical similarity due
     to iconicity ... makes transferring between different sign languages easier than the text
     of different spoken languages." §6.1 attributes its own results partly to "the
     hypothesized multilingual transfer effect thanks to sign language iconicity," and the
     paper ranks 302 signs by pose-embedding variance *across 20+ sign languages* to ask which
     signs are most iconic crosslingually.

   **Corrected interpretation:** a tag-insensitive result on out-of-domain footage is the
   *expected* behaviour, not a broken test. The residual above-chance signal is consistent
   with the shared, iconicity-driven semantic space SignCLIP was built to have. Our tie is
   therefore evidence about *where* the language conditioning lives (in-domain signer/studio
   cues), not evidence that our measurement was faulty.

## Conclusion

**Off-the-shelf SignCLIP is not usable as the ranking stage on this footage.** Three
independent measurements converge: forced choice inconclusive, full-inventory retrieval
non-significant, text-to-pose flat at chance on interpreter clips. Fine-tuning on this
project's own footage is a **prerequisite**, not an optional improvement — now
evidence-backed rather than assumed.

**The SignCLIP paper reaches the same conclusion about its own model** (added 2026-08-01),
which converts this from "our result contradicts expectations" to "our result reproduces a
documented limitation on a new language":

> "zero-shot performance on out-of-domain data is deficient. We posit that to reach
> noticeable performance on out-of-domain data, few-shot learning or fine-tuning is
> essential given the current scale of pretraining." (§6.1)

> "SignCLIP demonstrates excellent in-domain performance but falls short of immediate
> zero-shot prediction on downstream ISLR tasks. ... As a middle ground between full
> zero-shot prediction and full supervision, few-shot learning or fine-tuning is essential
> to tackle domain shift." (Conclusion)

The paper's Limitations section also states that evaluation "focuses ... on ASL, one of the
highest-resource sign languages," leaving other languages to future work — so LSM was in
pretraining but never evaluated. Our measurements are, as far as we know, the first
published-grade evidence on how the model behaves for LSM, and specifically for LSM in a
broadcast-interpreting register rather than dictionary citation form.

The pipeline *design* survives intact. What fails is the specific choice of an
off-the-shelf scorer at the final stage.

## What this does NOT license

- Any claim beyond **one interpreter, one 30-minute slice, 16 words**. Not a general
  statement about LSM, about SignCLIP, or about broadcast interpreting.
- A clean attribution to domain gap alone. The positive control is itself weak (~1.5×
  chance), so "these words are hard for SignCLIP" and "this footage is hard for SignCLIP"
  cannot be fully separated here. Both are likely true.
- A negative claim about SignCLIP's quality generally — §3 confirms it works on citation
  form, and the earlier concrete-word probe worked well. This is a transfer result, not a
  model-quality result.

## Reproduce

- `cut_query_clips.py` → 26 tight query clips from marked timestamps
- `select_distractors.py` → candidate manifest (seeded, hand-audited semantic picks)
- `copy_candidate_videos.py` → pull the 144 reference videos
- `qc_poses.py` → landmark-completeness QC on both sides
- `COLAB_FORCED_CHOICE.md` → Colab cells 1–11 (embedding + all analyses)

## Audit addendum (2026-08-01) — two clarifications from methodology review

**1. LSM is very likely IN SignCLIP's pretraining — which sharpens, not weakens, the
conclusion.** Spreadthesign operates a Mexican edition (`spreadthesign.com/es.mx/`,
verified live 2026-08-01 as its own locale, not a redirect), and SignCLIP pretrained on
a Spreadthesign scrape spanning 41 sign languages (Jiang et al., arXiv 2407.01264 —
the paper does not enumerate the list, but the es.mx edition and our earlier
concrete-word probe result, 23% top-1 vs 3.3% chance on LSM dictionary clips, both
point to LSM being present). This means the failure documented here canNOT be read as
"SignCLIP never saw LSM." It saw LSM *citation form* — and §3 confirms it retains
(weak) signal on our citation-form clips. What it fails on is **broadcast interpreting
register**: fast, co-articulated, tightly-cut clips of abstract vocabulary. The
domain-gap interpretation (§4, p=0.038) is therefore the primary explanation, with less
weight than previously implied on "unfamiliar language."

**2. One additional alternative explanation, previously implicit: lexical-variant
mismatch.** Query clips are human-verified (the word IS signed in them), but the
answer key assumes the interpreter's sign matches the wikisigns/ChNt dictionary
variant. LSM has regional/lexical variants; where the interpreter uses a different
variant of the same concept, the trial is unwinnable regardless of model quality, and
this deflates all pose-to-pose numbers in an unmeasured way. At n=26 this cannot be
separated from the register gap. It does not change the practical verdict (fine-tuning
on our own footage fixes both), but it belongs in any writeup of these numbers.
