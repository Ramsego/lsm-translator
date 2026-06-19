"""
Fetch a mañanera versión estenográfica (Spanish transcript) from gob.mx by date.

The transcript has no timestamps; it's the text we later align to video via audio ASR.

Usage:
    python phase2/fetch_estenografica.py 2026-06-18
"""

import argparse
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

OUT_DIR = Path("phase2/transcripts")
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
UA = {"User-Agent": "Mozilla/5.0 (research; LSM dataset)"}


def url_for(d: date) -> str:
    m = MONTHS_ES[d.month - 1]
    return ("https://www.gob.mx/presidencia/articulos/"
            f"version-estenografica-conferencia-de-prensa-de-la-presidenta-"
            f"claudia-sheinbaum-pardo-del-{d.day}-de-{m}-de-{d.year}")


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?</style>", " ", html)
    # Keep paragraph breaks
    html = re.sub(r"(?i)</p\s*>", "\n\n", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = (text.replace("&aacute;", "á").replace("&eacute;", "é")
            .replace("&iacute;", "í").replace("&oacute;", "ó")
            .replace("&uacute;", "ú").replace("&ntilde;", "ñ")
            .replace("&Aacute;", "Á").replace("&iexcl;", "¡")
            .replace("&iquest;", "¿").replace("&nbsp;", " ").replace("&amp;", "&"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="Conference date YYYY-MM-DD.")
    args = ap.parse_args()
    d = date.fromisoformat(args.date)
    url = url_for(d)
    print(f"Fetching {url}")

    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    text = strip_html(html)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.date}.txt"
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"Saved {out}  ({words} words)")
    print("\n--- first 400 chars ---")
    print(text[:400])


if __name__ == "__main__":
    main()
