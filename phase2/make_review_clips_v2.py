"""
Corpus-wide review-clip generation with interpreter-presence gating.

Supersedes make_review_clips.py (kept as the baseline). Three differences, each of
which fixes a measured defect:

1. INTERPRETER PRESENCE IS ENFORCED.
   make_review_clips.py opens segments.json but reads only `crop_changes` from it --
   the `segments` list is never consulted, so a clip is cut for every transcript hit
   whether or not the interpreter is on screen. This script requires the ENTIRE clip
   window to sit inside one pose-revalidated segment (segments_revalidated.json,
   which re-gates on pose rather than hands and so rejects B-roll: measured 2026-07-31
   on football players' arms, a trouser leg, a violin, a flower vase).

2. THE CLIP IS DEFINED BY LENGTH, NOT BY A PREDICTED LAG (--anchor word, the default).
   The old cutter placed a short window around `word_end + lag`. That makes the clip's
   position depend on an estimate that (a) varies 4.52-7.25s across this corpus, (b) has
   ~3s of internal spread within a single video, and (c) is missing entirely for some
   videos, where a corpus default gets substituted. A wrong lag silently moves the clip
   off the sign, and the reviewer records "sign doesn't appear".

   Evidence it was already doing damage: on the 26 verified signs in review_2.csv, the
   true sign sat a median 2.90s EARLIER than the old anchor predicted, with 7 of 26
   pinned within 0.35s of the clip's start and 0 of 26 in the final 2s -- a pile-up at
   the leading edge, i.e. a censored distribution. Signs that fell off the front were
   invisible and would have been scored as omissions.

   The fix is to stop predicting. The clip starts when the word is SPOKEN and simply
   runs long enough to outlast any plausible interpreter delay. Measured relative to the
   spoken word, the 26 verified signs start at +2.33s to +13.63s and end by +14.33s;
   allowing for the fastest and slowest interpreters in the corpus the worst case is
   ~15.2s. Default window is therefore -1s / +17s (18s), which contains 26/26 with 2.7s
   of margin and needs no lag term at all. `--anchor lag` restores the old behaviour.

   Note the interpreter genuinely can sign a word earlier than its Spanish position:
   LSM fronts time markers 87% of the time and reorders ~42% of pairs (measured on the
   gloss corpus). Anchoring at the spoken word rather than at a predicted delay handles
   that for free.

3. WORD SENSE IS FILTERED, NOT ASSUMED.
   Targets come from phase2/target_words.json as explicit surface-form families with
   per-word `block` collocations. An occurrence whose context matches a block phrase is
   skipped: "mesa de trabajo" is a committee, not a table -- the failure mode that made
   `mesa` unreviewable in an earlier batch.

Selection is balanced across words and across videos, because the existing 26-clip
instrument is one interpreter in one video and its single biggest weakness is that
every downstream number is measured on that one signer.

Usage:
    python3 phase2/make_review_clips_v2.py \
        --drive "/Volumes/Crucial X8/LSM_Translator" \
        --out   "/Volumes/Crucial X8/LSM_Translator/review/batch2" \
        --words salud hospital cama ... --per-word 7
"""

import argparse
import csv
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}
SENT_END = (".", "!", "?", "…")


def norm(w: str) -> str:
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", w.lower())


def norm_phrase(s: str) -> str:
    return " ".join(norm(t) for t in s.split() if norm(t))


def crop_at(crop_changes, t):
    prior = [c for c in crop_changes if c["at_sec"] <= t]
    return (prior[-1] if prior else crop_changes[0])["crop"]


def get_context(words, idx, max_words=30, fallback=12):
    """Sentence around idx, target wrapped in **...**. Walks to real punctuation."""
    n = len(words)
    half = max_words // 2
    left = idx
    while left > 0 and (idx - left) < half:
        prev = words[left - 1]["word"].strip()
        if prev and prev[-1] in SENT_END:
            break
        left -= 1
    right = idx
    while right < n - 1 and (right - idx) < half:
        cur = words[right]["word"].strip()
        if cur and cur[-1] in SENT_END:
            break
        right += 1
    if right - left < 3:
        left, right = max(0, idx - fallback), min(n - 1, idx + fallback)
    return " ".join(f"**{words[i]['word'].strip()}**" if i == idx
                    else words[i]["word"].strip() for i in range(left, right + 1))


def find_source_video(videos_root: Path, youtube_id: str):
    for p in videos_root.rglob(f"{youtube_id}.*"):
        if p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"):
            return p
    return None


def collect_candidates(drive: Path, families, win_pre, win_post, min_sep, anchor):
    """One pass over every video: every occurrence that survives presence + sense gating."""
    form2fam = {norm(f): fam for fam, v in families.items() for f in v["forms"]}
    blocks = {fam: [norm_phrase(b) for b in v.get("block", [])] for fam, v in families.items()}

    arrays_root = drive / "arrays" / "mananera"
    cands = []
    stats = defaultdict(int)
    for vdir in sorted(arrays_root.iterdir()):
        vid = vdir.name
        seg_f = vdir / "segments_revalidated.json"
        asr_f = drive / "videos" / "mananera" / vid / f"{vid}.asr.json"
        src = drive / "videos" / "mananera" / vid / f"{vid}.mp4"
        if not (seg_f.exists() and asr_f.exists() and src.exists()):
            stats["video_skipped_missing_inputs"] += 1
            continue
        meta = json.load(open(seg_f))
        segs = sorted((s["start_sec"], s["end_sec"]) for s in meta["segments"])
        crops = sorted(meta["crop_changes"], key=lambda c: c["at_sec"])
        lag_f = vdir / "lag_estimate.json"
        if lag_f.exists():
            lag = json.load(open(lag_f)).get("lag_sec", 6.33)
            lag_src = "measured"
        else:
            lag, lag_src = 6.33, "corpus_default"
        words = json.load(open(asr_f))["words"]

        for i, w in enumerate(words):
            fam = form2fam.get(norm(w["word"].strip(".,;:¿?¡!()»«\"'")))
            if not fam:
                continue
            stats["hits_raw"] += 1
            ctx = get_context(words, i)
            ctx_n = norm_phrase(ctx.replace("**", ""))
            if any(b and b in ctx_n for b in blocks[fam]):
                stats["dropped_blocked_sense"] += 1
                continue
            if anchor == "word":
                # Length-based: the clip starts when the word is spoken and simply runs
                # long enough to outlast any plausible interpreter delay. No lag term.
                t0 = w["end"] - win_pre
                t1 = w["end"] + win_post
            else:
                t0 = w["end"] + lag - win_pre
                t1 = w["end"] + lag + win_post
            if t0 < 0:
                stats["dropped_before_video_start"] += 1
                continue
            seg = next(((a, b) for a, b in segs if a <= t0 and t1 <= b), None)
            if seg is None:
                stats["dropped_no_interpreter"] += 1
                continue
            stats["kept"] += 1
            cands.append(dict(video=vid, word=fam, surface=w["word"].strip(),
                              t0=t0, dur=win_pre + win_post,
                              word_start=w["start"], word_end=w["end"],
                              lag=lag, lag_src=lag_src, seg_start=seg[0], seg_end=seg[1],
                              context=ctx, crop=crop_at(crops, t0), src=str(src)))
    # drop near-duplicates: same word+video within min_sep seconds
    cands.sort(key=lambda c: (c["word"], c["video"], c["t0"]))
    deduped, last = [], {}
    for c in cands:
        k = (c["word"], c["video"])
        if k in last and c["t0"] - last[k] < min_sep:
            stats["dropped_overlapping"] += 1
            continue
        last[k] = c["t0"]
        deduped.append(c)
    return deduped, stats


def balanced_pick(cands, per_word, words_wanted):
    """Round-robin across videos within each word, so no word is one-signer-only."""
    by_word = defaultdict(lambda: defaultdict(list))
    for c in cands:
        by_word[c["word"]][c["video"]].append(c)
    picked = []
    for word in words_wanted:
        vids = by_word.get(word)
        if not vids:
            continue
        # spread each video's candidates evenly in time, then interleave videos
        queues = []
        for vid, lst in sorted(vids.items(), key=lambda kv: -len(kv[1])):
            lst = sorted(lst, key=lambda c: c["t0"])
            step = max(1, len(lst) // max(1, per_word))
            queues.append(lst[::step] or lst)
        out, qi = [], 0
        while len(out) < per_word and any(queues):
            q = queues[qi % len(queues)]
            if q:
                out.append(q.pop(0))
            else:
                queues = [x for x in queues if x]
                if not queues:
                    break
                qi -= 1
            qi += 1
        picked.extend(out[:per_word])
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--targets", type=Path, default=Path("phase2/target_words.json"))
    ap.add_argument("--words", nargs="*", default=None, help="Subset of families to use.")
    ap.add_argument("--per-word", type=int, default=7)
    ap.add_argument("--anchor", choices=["word", "lag"], default="word",
                    help="word = clip spans a fixed LENGTH from when the word is spoken "
                         "(no lag term, default); lag = centre on word_end + lag estimate.")
    ap.add_argument("--win", type=float, default=1.0, help="Seconds BEFORE the anchor.")
    ap.add_argument("--win-post", type=float, default=17.0, help="Seconds AFTER the anchor.")
    ap.add_argument("--min-sep", type=float, default=20.0,
                    help="Min seconds between two clips of the same word in one video.")
    ap.add_argument("--pad", type=float, default=0.25, help="Crop padding fraction.")
    ap.add_argument("--ref-seconds", type=float, default=8.0)
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    ap.add_argument("--dry-run", action="store_true", help="Report the plan, cut nothing.")
    args = ap.parse_args()

    tw = json.load(open(args.targets))
    families = {k: v for k, v in tw.items()
                if not k.startswith("_") and "exclude" not in v}
    excluded = {k: v["exclude"] for k, v in tw.items()
                if not k.startswith("_") and "exclude" in v}
    if args.words:
        missing = [w for w in args.words if w not in families]
        if missing:
            raise SystemExit(f"unknown/excluded families: {missing}")
        families = {k: families[k] for k in args.words}
    words_wanted = list(families)

    print(f"targets: {len(words_wanted)} families, per-word quota {args.per_word}")
    if excluded:
        print(f"excluded by sense audit: {', '.join(excluded)}")
    anchor_name = "the spoken word" if args.anchor == "word" else "the lag point"
    print(f"window: -{args.win}s / +{args.win_post}s about {anchor_name} "
          f"({args.win + args.win_post:.0f}s clips)\n")

    cands, stats = collect_candidates(args.drive, families, args.win, args.win_post,
                                      args.min_sep, args.anchor)
    print("gating:")
    for k in ("hits_raw", "dropped_blocked_sense", "dropped_no_interpreter",
              "dropped_before_video_start", "dropped_overlapping", "kept"):
        if stats.get(k):
            print(f"  {k:<28} {stats[k]:>6,}")
    print(f"  usable after dedupe          {len(cands):>6,}\n")

    picked = balanced_pick(cands, args.per_word, words_wanted)
    per_w = defaultdict(int)
    per_v = defaultdict(int)
    for c in picked:
        per_w[c["word"]] += 1
        per_v[c["video"]] += 1
    print(f"selected {len(picked)} clips across {len(per_w)} words and {len(per_v)} videos")
    short = {w: per_w.get(w, 0) for w in words_wanted if per_w.get(w, 0) < args.per_word}
    if short:
        print(f"  under quota: {short}")
    print(f"  per video: {dict(sorted(per_v.items(), key=lambda kv: -kv[1]))}\n")
    if args.dry_run:
        return

    # dictionary reference clip per family (side-by-side in the review form)
    dict_ref = {}
    if args.metadata.exists():
        for r in csv.DictReader(open(args.metadata)):
            lab = r["label"].strip()
            if lab and "(" not in lab and " " not in lab:
                dict_ref.setdefault(norm(lab), r["youtube_id"])

    args.out.mkdir(parents=True, exist_ok=True)
    ref_dir = args.out / "_ref"
    ref_dir.mkdir(exist_ok=True)
    videos_root = args.drive / "videos"

    rows = []
    for n, c in enumerate(sorted(picked, key=lambda c: (c["word"], c["video"], c["t0"])), 1):
        wdir = args.out / c["word"]
        wdir.mkdir(exist_ok=True)

        dict_yt, ref_rel = dict_ref.get(c["word"], ""), ""
        if dict_yt:
            ref_path = ref_dir / f"{c['word']}.mp4"
            if not ref_path.exists():
                src = find_source_video(videos_root, dict_yt)
                if src:
                    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-t", f"{args.ref_seconds:.1f}",
                                    "-vf", "scale=-2:360", "-an", "-loglevel", "error",
                                    str(ref_path)], check=False)
            if ref_path.exists():
                ref_rel = str(ref_path.relative_to(args.out))

        x, y, cw, ch = c["crop"]
        px, py = int(cw * args.pad), int(ch * args.pad)
        cx, cy = max(0, x - px), max(0, y - py)
        cw2, ch2 = cw + 2 * px, ch + 2 * py
        t0 = c["t0"]
        out = wdir / f"{c['word']}_{c['video']}_{int(t0)//60:02d}m{int(t0)%60:02d}s.mp4"
        subprocess.run(["ffmpeg", "-y", "-ss", f"{t0:.2f}", "-i", c["src"],
                        "-t", f"{c['dur']:.2f}",
                        "-filter:v", f"crop={cw2}:{ch2}:{cx}:{cy},scale=2*iw:2*ih",
                        "-an", "-loglevel", "error", str(out)], check=False)
        if n % 25 == 0:
            print(f"  cut {n}/{len(picked)}")
        # word_start/word_end are the WHOLE POINT of a review batch: the deliverable is
        # the pair (when the word was spoken, when it was signed).  With --anchor word the
        # clip begins at `word_end - win_pre`, so word_end is recoverable as
        # `video_start + win_pre` -- but only if you know that convention, and word_START
        # is not recoverable at all because word durations vary (0.95-1.05s spread on
        # word_end vs -0.48..+1.04s on word_start, measured on batch2).  Writing both here
        # removes the need to re-derive anything.  Note review.html drops every column
        # except file/word/video_start/verdict/signer/sign_start/sign_end when the reviewer
        # exports, so join back on `file` (see phase2/make_word_times.py).
        rows.append({"file": str(out.relative_to(args.out)), "word": c["word"],
                     "video_start": round(t0, 1), "context": c["context"],
                     "verdict": "", "signer": "", "sign_start": "", "sign_end": "",
                     "dict_youtube_id": dict_yt, "ref_clip": ref_rel,
                     "video": c["video"], "surface": c["surface"],
                     "word_start": round(c["word_start"], 3),
                     "word_end": round(c["word_end"], 3),
                     "lag": round(c["lag"], 2), "lag_src": c["lag_src"]})

    csv_path = args.out / "review.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    json.dump({"window_pre": args.win, "window_post": args.win_post,
               "min_sep": args.min_sep, "per_word": args.per_word,
               "n_clips": len(rows), "words": sorted(per_w), "videos": sorted(per_v),
               "gating": dict(stats), "excluded_families": excluded},
              open(args.out / "batch_manifest.json", "w"), indent=1)

    print(f"\n{len(rows)} clips -> {args.out}")
    print(f"review sheet -> {csv_path}")
    print(f"next: python3 phase2/build_review_form.py --dir {args.out}")


if __name__ == "__main__":
    main()
