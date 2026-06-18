"""
Patch wikisigns_map.csv: replace slug-derived word labels with the actual h1
from each entry page. Only fetches pages where the word looks truncated or
wrong (i.e. contains a URL-encoded character or doesn't match the h1).

Writes corrected CSV to data/wikisigns_map.csv in place.
"""
import csv
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

MAP = Path("data/wikisigns_map.csv")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
H1_RE = re.compile(r'<h1[^>]*>([^<]+)</h1>', re.I)
_SKIP = {"sign language dictionaries of the world", "lengua de señas mexicana"}

def get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")

def extract_h1(html: str, fallback: str) -> str:
    h1s = [h.strip() for h in H1_RE.findall(html) if h.strip().lower() not in _SKIP]
    return h1s[0] if h1s else fallback

def main():
    with open(MAP, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Dedupe by slug — one fetch per unique entry page
    slugs_seen = {}
    for row in rows:
        slug = row["slug"]
        if slug not in slugs_seen:
            slugs_seen[slug] = {"entry_url": row["entry_url"], "word": row["word"]}

    print(f"Unique slugs to check: {len(slugs_seen)}")

    corrected = {}
    for i, (slug, info) in enumerate(slugs_seen.items()):
        try:
            html = get(info["entry_url"])
            word = extract_h1(html, urllib.parse.unquote(slug))
        except Exception as e:
            print(f"  ERR {slug}: {e}")
            word = info["word"]
        corrected[slug] = word
        if word != info["word"]:
            print(f"  FIXED: {info['word']!r} -> {word!r}")
        if i % 50 == 0:
            print(f"  [{i}/{len(slugs_seen)}]")
        time.sleep(1)

    # Rewrite CSV with corrected words
    with open(MAP, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "slug", "youtube_id", "entry_url"])
        writer.writeheader()
        for row in rows:
            row["word"] = corrected.get(row["slug"], row["word"])
            writer.writerow(row)

    print(f"\nDone. {sum(1 for s, info in slugs_seen.items() if corrected.get(s) != info['word'])} labels corrected.")

if __name__ == "__main__":
    main()
