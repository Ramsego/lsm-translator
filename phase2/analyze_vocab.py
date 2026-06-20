"""
Frequency analysis on an aligned transcript to select target words for review clips.

Lemmatizes Spanish words (spaCy es_core_news_sm) so conjugations group under their root
sign: estamos/están/estaba → estar. Flags temporal markers and acronyms.

Usage:
    python phase2/analyze_vocab.py --aligned <aligned.json> [--top 80]

Output: ranked table to stdout. Pick words for the master --words list, then pass them
to make_review_clips.py.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import spacy

STOPWORDS = {w for w in (
    "el la los las un una unos unas lo al del de a en y o u que como cuando donde quien "
    "cual cuanto porque por para con sin sobre entre yo tu te me mi nos nosotros ellos "
    "ella el ellas su sus se si no ni es son ser estar haber este esta esto ese esa eso "
    "aquel aqui ahi alla muy mas menos ya pero tambien cosa cosas entonces gracias "
    "pues bueno ahora asi etcetera usted ustedes").split()}

TEMPORALS = {
    "hoy", "ayer", "mañana", "manana", "semana", "mes", "año", "anio",
    "antes", "despues", "después", "siempre", "nunca", "ahora",
    "tarde", "noche", "dia", "día", "dias", "días", "semanas",
    "meses", "años", "anios",
}


def bare(w: str) -> str:
    """Lowercase, strip accents — for stopword/temporal lookup only."""
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_acronym(raw: str) -> bool:
    return bool(re.match(r'^[A-ZÁÉÍÓÚÑ]{2,}$', raw.strip(".,;:¿?¡!()")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", type=Path, required=True)
    ap.add_argument("--top", type=int, default=80)
    args = ap.parse_args()

    print("Loading spaCy es_core_news_sm...")
    nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])

    data = json.load(open(args.aligned))
    words = data["words"]

    # collect raw tokens, preserving original form
    tokens = [w["word"].strip().strip(".,;:¿?¡!()") for w in words]

    # split into acronyms and regular words
    acronym_counter: Counter = Counter()
    regular_tokens = []
    for tok in tokens:
        if is_acronym(tok):
            acronym_counter[tok] += 1
        else:
            regular_tokens.append(tok)

    # lemmatize in one batch (fast)
    text = " ".join(regular_tokens)
    doc = nlp(text)

    lemma_counter: Counter = Counter()
    lemma_to_example: dict = {}   # lemma → a clean original form for display
    for token in doc:
        lemma = token.lemma_.lower().strip()
        b = bare(lemma)
        if len(b) < 3:
            continue
        if b in STOPWORDS:
            continue
        lemma_counter[lemma] += 1
        # keep the most-common original surface form for display
        lemma_to_example.setdefault(lemma, token.text)

    print(f"\n=== TOP {args.top} CONTENT WORDS (lemmatized) ===")
    print(f"{'lemma':<20} {'example':<18} {'n':>4}  flags")
    print("-" * 55)
    for lemma, count in lemma_counter.most_common(args.top):
        flags = []
        if bare(lemma) in TEMPORALS:
            flags.append("TEMPORAL")
        example = lemma_to_example.get(lemma, lemma)
        print(f"{lemma:<20} {example:<18} {count:>4}  {'  '.join(flags)}")

    # filter acronym false positives: drop all-caps words that are >6 chars
    # (likely speaker name tags leaking from transcript headers)
    real_acronyms = {k: v for k, v in acronym_counter.items()
                     if len(k) <= 6 and v >= 2}
    if real_acronyms:
        print(f"\n=== ACRONYMS (≤6 chars, ≥2 occurrences) ===")
        for a, c in sorted(real_acronyms.items(), key=lambda x: -x[1]):
            print(f"  {a}: {c}")
    else:
        print("\n=== ACRONYMS: none meeting threshold ===")

    print(f"\nTotal unique lemmas: {len(lemma_counter)}")
    print(f"Source: {args.aligned}")


if __name__ == "__main__":
    main()
