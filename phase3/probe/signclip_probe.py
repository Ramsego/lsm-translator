"""
SignCLIP-LSM probe — does SignCLIP already understand Mexican Sign Language?

Two sub-commands:
  select : choose 30 dictionary signs that have a source video, copy videos to clips/
  eval   : embed the 30 poses + their Spanish words with SignCLIP, report retrieval accuracy

Plain-English verdict from `eval`:
  top-1 well above 1/30 (~3%)  → SignCLIP SEES LSM. Green light: recognizer ~for free.
  near chance                  → it doesn't. We train our own pose encoder instead.

⚠️  THE ONE PART TO VERIFY LIVE: the two functions `embed_poses()` and `embed_texts()`
    call SignCLIP. I cannot guarantee SignCLIP's exact Python API from memory — the
    operator (run this with Opus, see README note) must open SignCLIP's repo
    (https://github.com/J22Melody/fairseq, signclip branch) read its embedding example,
    and make these two functions return an (N, D) L2-normalized numpy array.
    EVERYTHING ELSE in this file is correct and final.
"""

import argparse
import csv
import shutil
import sys
import unicodedata
from pathlib import Path

import numpy as np

LANG_TAG = "<es> <mfs>"   # Spanish text → Mexican Sign Language, SignCLIP's tag convention

# Common, concrete, single-word signs likely present in Spreadthesign's Mexican vocabulary.
# A FAIR test set — unlike the first-30-alphabetical sample (proper nouns, phrases, dup variants).
COMMON = [
    "casa", "agua", "perro", "gato", "madre", "padre", "hermano", "hermana", "niño", "niña",
    "mujer", "hombre", "comer", "beber", "dormir", "trabajar", "caminar", "correr", "hablar",
    "escribir", "leer", "rojo", "azul", "verde", "amarillo", "negro", "blanco", "sol", "luna",
    "fuego", "árbol", "flor", "libro", "mesa", "silla", "puerta", "ventana", "coche", "escuela",
    "ciudad", "méxico", "dinero", "comida", "leche", "pan", "café", "gracias", "hola", "amigo",
    "amor", "feliz", "triste", "grande", "pequeño", "frío", "caliente", "día", "noche", "año",
    "hora", "nombre", "número", "bueno", "malo", "blanco", "lunes", "rojo", "familia", "tiempo",
]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower().strip())
    return "".join(c for c in s if not unicodedata.combining(c))


# ────────────────────────────────────────────────────────────────────────────
# select : pick 30 signs whose source video exists on the drive
# ────────────────────────────────────────────────────────────────────────────
def find_video(drive: Path, youtube_id: str) -> Path | None:
    for p in (drive / "videos").rglob(f"{youtube_id}.*"):
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
            return p
    return None


def cmd_select(args):
    repo = Path(__file__).resolve().parents[2]
    rows = list(csv.DictReader(open(repo / "data" / "metadata.csv")))
    drive = Path(args.drive)

    # Keep only clean, distinct labels: single word, no parentheses (drops variant markers
    # like "aceite (a)", states like "acapulco (gro)", and phrases like "a el le importa").
    clean = {}
    for r in rows:
        label = r["label"].strip()
        if not label or "(" in label or " " in label:
            continue
        clean.setdefault(_norm(label), r)

    picked, used = [], set()

    def try_pick(key):
        if key in used or key not in clean:
            return False
        r = clean[key]
        if not find_video(drive, r["youtube_id"]):
            return False
        shutil.copy(find_video(drive, r["youtube_id"]), Path(args.clips) / f"{r['youtube_id']}.mp4")
        picked.append({"youtube_id": r["youtube_id"], "label": r["label"].strip(),
                       "clip": f"{r['youtube_id']}.mp4"})
        used.add(key)
        return True

    # 1) prefer the common concrete words
    for w in COMMON:
        if len(picked) >= args.n:
            break
        try_pick(_norm(w))
    # 2) top up with other clean words, spread across the alphabet (not all "a")
    if len(picked) < args.n:
        keys = sorted(k for k in clean if k not in used)
        step = max(1, len(keys) // (args.n * 3))
        for k in keys[::step]:
            if len(picked) >= args.n:
                break
            try_pick(k)

    man = Path(args.clips) / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["youtube_id", "label", "clip"])
        w.writeheader(); w.writerows(picked)
    print(f"  selected {len(picked)} signs → {man}")
    if len(picked) < args.n:
        print(f"  ⚠️ only found {len(picked)} (wanted {args.n}); check drive is mounted")


# ────────────────────────────────────────────────────────────────────────────
# SignCLIP bridge — ⚠️ FILL THESE TWO BY READING THE SIGNCLIP REPO
# ────────────────────────────────────────────────────────────────────────────
def _load_signclip():
    """Load the SignCLIP model + weights. Return whatever the embed fns need."""
    raise NotImplementedError(
        "Load SignCLIP here. See https://github.com/J22Melody/fairseq (signclip). "
        "Download the pretrained weights, then return the model object.")


def embed_poses(pose_paths: list[Path], model) -> np.ndarray:
    """(N .pose files) → (N, D) L2-normalized numpy array of SignCLIP sign embeddings."""
    raise NotImplementedError("Embed each .pose with SignCLIP's video/pose encoder.")


def embed_texts(words: list[str], model) -> np.ndarray:
    """(N Spanish words) → (N, D) L2-normalized array. Prepend LANG_TAG to each word."""
    raise NotImplementedError("Embed [f'{LANG_TAG} {w}' for w in words] with SignCLIP's text encoder.")


# ────────────────────────────────────────────────────────────────────────────
# eval : retrieval accuracy (this logic is final/correct)
# ────────────────────────────────────────────────────────────────────────────
def cmd_eval(args):
    rows = list(csv.DictReader(open(Path(args.clips) / "manifest.csv")))
    words = [r["label"] for r in rows]
    pose_paths, kept_words = [], []
    for r, w in zip(rows, words):
        p = Path(args.poses) / f"{r['youtube_id']}.pose"
        if p.exists():
            pose_paths.append(p); kept_words.append(w)
        else:
            print(f"  (skip {w}: no .pose)")

    n = len(kept_words)
    if n < 5:
        sys.exit("Too few poses to evaluate — re-run extract_poses.sh")

    model = _load_signclip()
    P = embed_poses(pose_paths, model)     # (n, D)
    T = embed_texts(kept_words, model)     # (n, D)

    sim = T @ P.T                          # (n, n): text i vs pose j; row i's truth is column i
    def topk(k):
        hits = sum(i in np.argsort(-sim[i])[:k] for i in range(n))
        return hits / n

    chance = 1 / n
    print("\n──────────── SignCLIP-LSM probe result ────────────")
    print(f"  signs tested : {n}   (chance top-1 = {chance:.1%})")
    print(f"  text→sign top-1 : {topk(1):.1%}")
    print(f"  text→sign top-5 : {topk(5):.1%}")
    verdict = ("GREEN — SignCLIP sees LSM. Recognizer is ~free; proceed."
               if topk(1) > 4 * chance else
               "RED — near chance. SignCLIP doesn't transfer; train our own encoder.")
    print(f"  VERDICT: {verdict}")
    print("────────────────────────────────────────────────────")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--drive", required=True)
    s.add_argument("--clips", required=True); s.add_argument("--n", type=int, default=30)
    s.set_defaults(func=cmd_select)
    e = sub.add_parser("eval"); e.add_argument("--clips", required=True)
    e.add_argument("--poses", required=True); e.set_defaults(func=cmd_eval)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
