import csv
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

BASE = "https://www.wikisigns.org"
INDEX = BASE + "/es/lsm/diccionario"
OUT = Path("data/wikisigns_map.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

YT_RE = re.compile(
    r"(?:youtube(?:-nocookie)?\.com/embed/|youtu\.be/|youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})"
)
H1_RE = re.compile(r'<h1[^>]*>([^<]+)</h1>', re.I)
_SKIP = {"sign language dictionaries of the world", "lengua de señas mexicana"}

def get(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")

def find_last_page(index_html: str) -> int:
    pages = [int(n) for n in re.findall(r"[?&]page=(\d+)", index_html)]
    return max(pages) if pages else 0

def collect_entry_slugs() -> list:
    first = get(INDEX)
    last = find_last_page(first)
    print(f"Index has pages 0..{last}")
    slugs = set()
    for page in range(0, last + 1):
        html = first if page == 0 else get(f"{INDEX}?page={page}")
        found = re.findall(r"/es/lsm/([a-z0-9%\-]+)", html, re.I)
        for s in found:
            if s.lower() == "diccionario":
                continue
            slugs.add(s)
        print(f"  page {page}: {len(found)} links, {len(slugs)} unique so far")
        time.sleep(1)
    return sorted(slugs)

def load_done() -> set:
    if not OUT.exists():
        return set()
    with open(OUT, encoding="utf-8") as f:
        return {row["slug"] for row in csv.DictReader(f)}

def main():
    slugs = collect_entry_slugs()
    print(f"\nTotal entries: {len(slugs)}")

    done = load_done()
    print(f"Already mapped: {len(done)} (will skip)")

    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["word", "slug", "youtube_id", "entry_url"])

        for i, slug in enumerate(slugs):
            if slug in done:
                continue
            url = f"{BASE}/es/lsm/{slug}"
            try:
                html = get(url)
            except Exception as e:
                print(f"  ERR {slug}: {e}")
                continue

            h1s = [h.strip() for h in H1_RE.findall(html) if h.strip().lower() not in _SKIP]
            word = h1s[0] if h1s else urllib.parse.unquote(slug)

            yt_ids = sorted(set(YT_RE.findall(html)))
            if not yt_ids:
                print(f"  no video: {word}")
            for yid in yt_ids:
                writer.writerow([word, slug, yid, url])

            f.flush()
            if i % 25 == 0:
                print(f"  [{i}/{len(slugs)}] {word} -> {yt_ids}")
            time.sleep(1)

    print(f"\nDone. Mapping written to {OUT}")

if __name__ == "__main__":
    main()
