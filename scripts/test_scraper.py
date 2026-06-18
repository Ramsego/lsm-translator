import re
import urllib.parse

H1_RE = re.compile(r'<h1[^>]*>([^<]+)</h1>', re.I)
_SKIP = {"sign language dictionaries of the world", "lengua de señas mexicana"}

def extract_word(html: str, slug: str) -> str:
    h1s = [h.strip() for h in H1_RE.findall(html) if h.strip().lower() not in _SKIP]
    return h1s[0] if h1s else urllib.parse.unquote(slug)

def check(name: str, condition: bool) -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return condition

def make_page(*h1s):
    tags = "".join(f"<h1>{h}</h1>" for h in h1s)
    return f"<html><body>{tags}</body></html>"

SITE_H1S = ["Sign Language Dictionaries of the World", "Lengua de Señas Mexicana"]

def run():
    results = []

    results.append(check(
        "Normal entry",
        extract_word(make_page(*SITE_H1S, "gato"), "gato") == "gato"
    ))

    results.append(check(
        "Multi-label entry (naranja/Fanta)",
        extract_word(
            make_page(*SITE_H1S, "naranja, Fanta (Color, fruta, Refresco)"),
            "naranja"
        ) == "naranja, Fanta (Color, fruta, Refresco)"
    ))

    results.append(check(
        "Multi-word entry (Abandonar, Dejar)",
        extract_word(make_page(*SITE_H1S, "Abandonar, Dejar"), "abandonar") == "Abandonar, Dejar"
    ))

    results.append(check(
        "Skip site-wide h1s, keep entry h1",
        extract_word(make_page(*SITE_H1S, "perro"), "perro") == "perro"
    ))

    results.append(check(
        "Fallback to slug when no entry h1",
        extract_word(make_page(*SITE_H1S), "mi%20palabra") == "mi palabra"
    ))

    results.append(check(
        "Slug with parens falls back (malformed page like Fanta%20%28Color)",
        extract_word(make_page(*SITE_H1S), "Fanta%20%28Color") == "Fanta (Color"
    ))

    results.append(check(
        "Accented characters preserved",
        extract_word(make_page(*SITE_H1S, "Águila"), "aguila") == "Águila"
    ))

    results.append(check(
        "Entry h1 with parentheses and commas",
        extract_word(
            make_page(*SITE_H1S, "Correr (A), Huir"),
            "correr"
        ) == "Correr (A), Huir"
    ))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    print("Testing scraper h1 extraction logic...\n")
    ok = run()
    raise SystemExit(0 if ok else 1)
