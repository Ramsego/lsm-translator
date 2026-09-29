"""
Forced-choice SignCLIP test — candidate-side label selection.

For each of the 16 gold word types (from review_2.csv's 26 usable instances), pick:
  - its own Phase-1 reference label (exact match against data/metadata.csv)
  - 4 RANDOM distractor labels (seeded, reproducible)
  - 4 SEMANTIC-NEAR distractor labels (hand-picked by reading the full 886-label
    vocabulary; documented below so the choice is auditable rather than a black box)

Two conditions per Fable's review: random-only tests basic recognition, semantic-near
tests fine-grained disambiguation. Distractor pools are disjoint from each other and
from the set of all 16 true labels, so no trial's distractor is secretly another
trial's correct answer.

Usage: python3 phase3/probe/fc_test/select_distractors.py
"""
import csv
import random
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
METADATA = REPO / "data/metadata.csv"
OUT = Path(__file__).parent / "candidate_manifest.csv"

TRUE_WORDS = [
    "gracias", "joven", "actividad", "atender", "equipo", "decidir", "derecho",
    "experiencia", "ganar", "familia", "encontrar", "edad", "director",
    "aprovechar", "cambiar", "futuro",
]

# Hand-picked from the full 886-label list (phase3/probe/fc_test/select_distractors.py
# was written after reading the whole vocabulary, not generated) -- same broad semantic
# domain as the true word, using only labels that actually exist in data/metadata.csv:
SEMANTIC_NEAR = {
    "gracias": ["de nada", "por favor", "con permiso", "buenos días"],
    "joven": ["adolescente", "adulto", "viejo", "bebé (a)"],  # bare "bebé" has no video on drive
    "actividad": ["compromiso (a)", "conducta", "costumbres", "ejercer"],
    "atender": ["ayudar", "enseñar", "enfrentar", "guiar"],
    "equipo": ["amigo", "compas", "amistad", "fila"],  # "compañero" has no video on drive
    "decidir": ["elegir", "escoger", "acordar", "confirmar"],
    "derecho": ["constitución", "discriminación", "corrupción", "delito"],
    "experiencia": ["aprender", "comprender", "educar", "conocer a detalle"],
    "ganar": ["conseguir", "conquistar", "fracasar", "aprobado"],
    "familia": ["primo", "tío", "nieto", "cuñado"],
    "encontrar": ["buscar", "esconder", "descubrir", "aparecer"],
    "edad": ["fecha", "apellido (a)", "calificación", "estado (a)"],
    "director": ["coronel", "capitán", "dueño", "arquitecto"],
    "aprovechar": ["ahorrar", "gastar", "desear", "consentir"],
    "cambiar": ["corregir", "aumentar", "doblar", "construir"],  # "arreglar" has no video on drive
    "futuro": ["continuar", "eterna", "diariamente (a)", "en punto"],
}


def _norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower().strip())
                   if not unicodedata.combining(c))


def main():
    rows = list(csv.DictReader(open(METADATA)))
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"].strip().lower(), r)

    true_rows = {w: by_label[w] for w in TRUE_WORDS}
    for w, r in true_rows.items():
        print(f"  true  {w:14s} -> {r['youtube_id']} ({r['source']})")

    # Verify every semantic pick actually resolves
    for w, picks in SEMANTIC_NEAR.items():
        for p in picks:
            assert p.strip().lower() in by_label, f"semantic distractor {p!r} for {w!r} not in metadata"

    # 20/963 metadata rows have no source video on the mounted drive (checked
    # 2026-07-31) -- restrict the random pool to confirmed-available labels so sampling
    # doesn't need a re-run/patch cycle after the fact
    available = set(open(Path(__file__).parent / "available_labels.txt", encoding="utf-8")
                     .read().splitlines())
    excluded = (set(TRUE_WORDS)
                | {p.strip().lower() for picks in SEMANTIC_NEAR.values() for p in picks})
    pool = [l for l in by_label if l not in excluded and l in available]

    rng = random.Random(42)
    used_random = set()
    random_near = {}
    for w in TRUE_WORDS:
        # exclude anything sharing a >=5-char stem with the true word (crude guard
        # against picking a morphological variant as a "random" distractor)
        stem = _norm(w)[:5]
        candidates = [l for l in pool if l not in used_random and not _norm(l).startswith(stem)]
        picks = rng.sample(candidates, 4)
        used_random.update(picks)
        random_near[w] = picks

    out_rows = []
    for w in TRUE_WORDS:
        out_rows.append(dict(word=w, role="true", label=w,
                              youtube_id=true_rows[w]["youtube_id"], source=true_rows[w]["source"]))
        for p in SEMANTIC_NEAR[w]:
            r = by_label[p.strip().lower()]
            out_rows.append(dict(word=w, role="semantic_distractor", label=p,
                                  youtube_id=r["youtube_id"], source=r["source"]))
        for p in random_near[w]:
            r = by_label[p]
            out_rows.append(dict(word=w, role="random_distractor", label=p,
                                  youtube_id=r["youtube_id"], source=r["source"]))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=["word", "role", "label", "youtube_id", "source"])
        wcsv.writeheader()
        wcsv.writerows(out_rows)

    n_videos = len({r["youtube_id"] for r in out_rows})
    print(f"\n{len(out_rows)} rows ({n_videos} distinct source videos needed) -> {OUT}")


if __name__ == "__main__":
    main()
