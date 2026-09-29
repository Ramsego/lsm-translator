# Review batch 2 — how the words and windows were chosen

Batch: `<drive>/review/batch2` — **203 clips × 18 s, 29 words, 13 videos**, generated 2026-08-01 by
`phase2/make_review_clips_v2.py`. Targets in `phase2/target_words.json`.

Purpose: grow the verified instrument from 26 → ~100+ clips, and — more importantly —
break its single-signer limitation. Every alignment number in this project is currently
measured on 26 clips from one interpreter in one 30-minute slice.

## The three defects this batch fixes

### 1. Clips with no interpreter on screen

`make_review_clips.py` opens `segments.json` but reads only `crop_changes` from it. The
`segments` list — the record of when the interpreter is actually present — is never
consulted, so a clip was cut for every transcript hit regardless. Combined with the old
hand-based presence gate (which fired on any hands in the inset region: football players'
arms, a trouser leg, a violin, a flower vase), review batches could contain clips with no
interpreter at all.

v2 requires the **entire** clip window to fall inside one segment from
`segments_revalidated.json` (pose-gated, calibrated at 100% vs 0% on known good/bad cases).
706 of 4,283 candidate windows were rejected on this basis.

**Verified after generation, not assumed:** all 203 clips were re-probed with MediaPipe
Holistic at 15%/50%/85% of their duration — **203/203 have pose detected in all three
probes, and hands visible in at least one.** Audit script and raw results in the scratchpad;
the criterion is identical to the one `revalidate_segments.py` uses.

### 2. The clip is now defined by LENGTH, not by a predicted lag

**Revised 2026-08-01 after review.** The first version of this batch still positioned each
clip around `word_end + lag`. That inherits every weakness of the lag estimate: it varies
**4.52–7.25 s across the corpus**, carries ~3 s of internal spread inside a single video,
and is missing altogether for some videos (a corpus default gets substituted). When the lag
is wrong the clip moves off the sign and the reviewer records "doesn't appear" — the error
is invisible and lands in the gold set as a false negative.

The fix is to stop predicting. **The clip starts 1 s before the word is spoken and runs 18 s.**
No lag term anywhere. Validation against the 26 verified signs, measured relative to the
spoken word:

| | |
|---|---|
| sign starts | +2.33 s to +13.63 s (median +3.43 s) |
| latest sign end | +14.33 s |
| corpus worst case (slowest interpreter) | ≈ +15.2 s |
| **window −1 s / +17 s** | **contains 26/26, 2.7 s of margin** |

`--anchor lag` still restores the old behaviour for reproducibility.

### 2b. Why the old window was mis-centred (the evidence that prompted this)

Measured on the 26 verified signs in `review_2.csv`, relative to the old anchor
(`word_end + lag`):

| | |
|---|---|
| median offset of the true sign | **−2.90 s** (the interpreter signs *earlier* than predicted) |
| standard deviation | 3.43 s |
| signs within 0.35 s of clip start | **7 of 26** |
| signs in the last 2 s of the clip | **0 of 26** |

The pile-up at the clip's leading edge with an empty trailing edge is the signature of a
**censored distribution**: the old window (−4 s / +10 s) started too late, so an unknown
number of signs fell off the front entirely. Those would have been marked "doesn't appear"
by the reviewer — recorded as *the interpreter omitted the sign* when in fact the clip
simply started after it. That is a false-negative class in the existing gold data, and it
is worth re-checking the current `n` verdicts against it.

This is what the length-based window in §2 replaces: anchoring at the spoken word makes
the leading edge structurally safe, because nothing can be signed before the word is heard.

Note part of the negative offset is **not error**: LSM reorders ~42% of pairs relative to
Spanish and fronts time markers 87% of the time (measured on the gloss corpus), so a word's
sign legitimately precedes its Spanish position within a clause. This is the same
phenomenon Workstream E models, observed from the other side.

### 3. Word sense was assumed, not checked — the `mesa` problem

`mesa` produced unusable clips because the mañanera says *mesa de trabajo* (a committee),
never *mesa* (a table). The dictionary sign is for the furniture; the answer key was wrong
by construction.

v2 filters on measured collocations. Right-neighbour profiles were computed on the real
corpus before choosing, which produced two exclusions and six block lists:

| word | measurement | decision |
|---|---|---|
| **estado** | 135/498 right-neighbours are `unidos` (Estados Unidos — a proper noun, fingerspelled); much of the rest is the participle of *estar* ("hemos estado instalando") | **excluded** |
| **equipo** | three live senses: sports team (*equipos profesionales*), equipment (*equipos tecnológicos*), staff (*excelente equipo. El gabinete*) | **excluded** |
| **día** | dominated by *el día de hoy*, *todos los días*, *buenos días*, *Día del Amor* | 14 block phrases |
| **mundo** | *todo el mundo* = "everybody"; *Copa del Mundo*; *mundo maya* | 4 block phrases |
| **médico** | adjective uses (*centro médico*, *equipo médico*) would not be signed DOCTOR | 10 block phrases |
| **casa** | *casa por casa* = door-to-door (23×) | 3 block phrases |
| **trabajo** | *mesa/grupo de trabajo* — the original failure, generalized | 5 block phrases |
| **cama** | consistently literal hospital beds (*camas con ventilador*, *número de camas*) | kept, no filter needed |

Only 6 occurrences were actually dropped by block matching, because the exclusions did the
heavy lifting — the filters mostly serve as guards for future batches.

## Word list

29 families, chosen for: gated frequency, spread across videos, plausibility of a
conventional LSM lexical sign, and sense stability in this register.

- **Continuity with the existing 26-clip set** (so old and new measurements stay comparable):
  `gracias, actividad, atender, derecho, director, pregunta, hablar, familia, joven`
- **Concrete, high-frequency, health/COVID register:**
  `salud, hospital, cama, enfermedad, vacuna, medico`
- **Concrete civic/everyday:**
  `persona, gobierno, pais, ano, semana, maestro, ciudad, pueblo, nino, escuela, agua,
  mujer, trabajo, gente, seguridad`

Deliberately **not** included, and why: pronouns (`nosotros, ellos, ustedes`) are indexical
pointing rather than lexical signs; copulas and prepositions (`estar, contra`) are among the
most-deleted tokens in the gloss corpus; proper nouns (`México, López, IMSS, Oaxaca`) are
fingerspelled or name signs; discourse markers (`bueno, entonces, ahora`) are rarely lexical.

Selection is round-robin across videos within each word, so no word is single-signer.
Coverage: 13 videos (5–26 clips each). The remaining 3 corpus videos have no ASR on the
drive — `Z-pG_c9HoT8` is the transcript flagged as possibly incomplete last session;
`R1QhGx6JdDk` and `imsUjdIQtSA` are also missing one.

## Where LLMs are used here, and where they are deliberately not

Used, and load-bearing:

1. **Morphological expansion.** spaCy is not installed in either environment, so there is no
   lemmatizer; the old script silently fell back to identity matching, which misses every
   inflected form. The surface-form families in `target_words.json` are LLM-written and
   exhaustive per word. This is the same mechanism as the `FORMS` map in
   `check_alignment_v2.py`, which the 2026-07-31 audit showed strictly beats stem-prefix
   matching (that heuristic had both false negatives — `encontrar`→`encuentra` — and
   false positives it introduced itself — `director`→`directamente`).
2. **Sense and signability triage.** Deciding that `equipo` has three senses, that `estado`
   is mostly *Estados Unidos*, that pronouns are indexical and proper nouns fingerspelled.
   Every such judgement here was **checked against measured collocation counts** before being
   applied, not trusted on its own.

Deliberately not used:

3. **Not as an answer key.** The LLM chooses what to put in front of the reviewer; it never
   decides whether a sign is present. That keeps the failure mode safe — a bad guess costs
   review minutes, and can never write a wrong label into the gold set. This is the same
   boundary that made the earlier "LLM constrained to the 886" experiment harmful: it forced
   substitutes and manufactured wrong answer keys.
4. **Not open-vocabulary candidate proposal — yet.** That belongs to the mining stage
   (see `PIPELINE_MAP.md` Stage 2/3 and the `concept-cloud-open-vocab` design note), where
   the consumer is a fine-tuned encoder scoring recurrence, not a human answering "is this
   word signed here?" A review clip needs one specific target word to ask about.

**The natural next increment** is per-occurrence sense vetting: pass each candidate's real
sentence to an LLM and ask whether the target reading is the intended one, instead of
filtering by collocation lists. That subsumes the block lists and would let `equipo` and
`estado` back in with per-clip sense tags. It is worth doing if this batch's yield shows
sense confusion is still costing clips — which the reviewer's verdicts will reveal.

## Reviewing this batch

```bash
open "/Volumes/Crucial X8/LSM_Translator/review/batch2/review.html"
```

Keys: `→` yes · `↓` doesn't appear · `g` negated form · `s`/`e` mark sign start/end ·
`Backspace` back. Progress auto-saves to localStorage, so it can be done across sessions.
Fill the interpreter description field whenever the signer changes — **per-clip signer tags
are what make leave-one-signer-out evaluation possible**, and this batch's whole point is
multi-signer coverage.

When finished: Download review.csv, overwrite the one in the folder, then `collect_gold.py`.

Twelve of the 29 words have a side-by-side dictionary reference clip (`persona, gracias,
pregunta, hablar, atender, actividad, derecho, agua, mujer, joven, familia, director`); the
rest have no single-word entry in the 886 inventory and will show the interpreter pane only.
