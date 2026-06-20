"""
Fetch a mañanera versión estenográfica (Spanish transcript) from gob.mx by date.

gob.mx serves a JS bot-challenge; we use Playwright (headless Chromium) with
playwright-stealth if available, and WAIT for the challenge to clear before reading.
The conference URL slug differs by president, so we try both the Sheinbaum and the
AMLO (COVID-era) patterns and keep whichever returns real content.

Usage:
    python phase2/fetch_estenografica.py 2026-06-18
    python phase2/fetch_estenografica.py 2020-06-30 --out-dir phase2/transcripts/
"""

import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

OUT_DIR = Path("phase2/transcripts")
MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
             "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]

# URL slug templates differ by president/era; we try each until one returns content.
# {d}=day {m}=month-name {y}=year
SLUG_TEMPLATES = [
    # Sheinbaum (2024–)
    "version-estenografica-conferencia-de-prensa-de-la-presidenta-"
    "claudia-sheinbaum-pardo-del-{d}-de-{m}-de-{y}",
    # AMLO mañanera (2019–2024)
    "version-estenografica-de-la-conferencia-de-prensa-matutina-del-presidente-"
    "andres-manuel-lopez-obrador-del-{d}-de-{m}-de-{y}",
    "version-estenografica-de-la-conferencia-de-prensa-matutina-del-presidente-"
    "andres-manuel-lopez-obrador-{d}-de-{m}-de-{y}",
    "version-estenografica-conferencia-de-prensa-del-presidente-"
    "andres-manuel-lopez-obrador-del-{d}-de-{m}-de-{y}",
]


def candidate_urls(d: date):
    m = MONTHS_ES[d.month - 1]
    base = "https://www.gob.mx/presidencia/articulos/"
    for tmpl in SLUG_TEMPLATES:
        yield base + tmpl.format(d=d.day, m=m, y=d.year)


def strip_html(html: str) -> str:
    html = re.sub(r"(?is)<script.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?</style>", " ", html)
    html = re.sub(r"(?i)</p\s*>", "\n\n", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = text.replace("&nbsp;", " ").replace("&aacute;", "á").replace("&eacute;", "é") \
               .replace("&iacute;", "í").replace("&oacute;", "ó").replace("&uacute;", "ú") \
               .replace("&ntilde;", "ñ").replace("&iexcl;", "¡").replace("&iquest;", "¿") \
               .replace("&amp;", "&")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# the estenográfica body starts at this dateline and ends before the page footer
_BODY_START = re.compile(r"Conferencia encabezada por", re.I)
_BODY_END = re.compile(
    r"(Última modificación|Contesta nuestra encuesta|Protección de Datos Personales"
    r"|Síguenos en|Enlaces|Datos Abiertos)", re.I)


def extract_body(text: str) -> str:
    """Trim gob.mx nav/related-links/footer boilerplate down to the conference body."""
    m = _BODY_START.search(text)
    if m:
        text = text[m.start():]
    e = _BODY_END.search(text)
    if e:
        text = text[:e.start()]
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_page(p, headless=True):
    browser = p.chromium.launch(
        headless=headless,
        args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
    )
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        locale="es-MX",
        timezone_id="America/Mexico_City",
        viewport={"width": 1280, "height": 800},
    )
    page = ctx.new_page()
    # best-effort stealth patches (works with or without playwright-stealth)
    page.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
        "Object.defineProperty(navigator,'languages',{get:()=>['es-MX','es']});"
        "Object.defineProperty(navigator,'plugins',{get:()=>[1,2,3,4,5]});"
    )
    try:
        from playwright_stealth import stealth_sync
        stealth_sync(page)
    except Exception:
        pass  # stealth optional; init-script patches still apply
    return browser, page


def fetch_url(url: str, challenge_wait=30, headless=True) -> str:
    """Load url, wait for the JS bot-challenge to clear (page text grows), return HTML."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = _make_page(p, headless=headless)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # poll until the challenge resolves (body text becomes substantial) or 404
            deadline = time.time() + challenge_wait
            best = ""
            while time.time() < deadline:
                txt = page.inner_text("body")
                if len(txt) > len(best):
                    best = txt
                if "Challenge Validation" not in txt and len(txt.split()) > 500:
                    break
                if "no encontrada" in txt.lower() or "404" in page.title():
                    break
                page.wait_for_timeout(1000)
            html = page.content()
        finally:
            browser.close()
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="Conference date YYYY-MM-DD (also used to name the output file).")
    ap.add_argument("--url", default=None,
                    help="Fetch this exact article URL instead of date-constructed candidates "
                         "(e.g. a gob.mx/salud COVID briefing).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--headed", action="store_true",
                    help="Run a visible browser (clears the gob.mx JS challenge).")
    args = ap.parse_args()

    d = date.fromisoformat(args.date)
    url_list = [args.url] if args.url else list(candidate_urls(d))

    best_text, best_url = "", None
    for url in url_list:
        print(f"Trying: {url}")
        try:
            html = fetch_url(url, headless=not args.headed)
        except Exception as e:
            print(f"  error: {e}")
            continue
        text = extract_body(strip_html(html))
        words = len(text.split())
        print(f"  → {words} words (body)")
        if words > len(best_text.split()):
            best_text, best_url = text, url
        if words > 500 and "Challenge Validation" not in text:
            break

    words = len(best_text.split())
    if words < 500:
        print(f"\nFAILED: best attempt had only {words} words "
              f"(bot-challenge or wrong date/slug).")
        print(best_text[:500])
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.date}.txt"
    out.write_text(best_text, encoding="utf-8")
    print(f"\nSaved {out}  ({words} words)  from {best_url}")
    print("\n--- first 400 chars ---")
    print(best_text[:400])


if __name__ == "__main__":
    main()
