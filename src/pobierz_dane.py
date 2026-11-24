"""Pobiera teksty źródłowe do data/raw/.

Książki: Sherlock Holmes (Project Gutenberg, mirror pglaf) i „Lalka” t. 1 (Wolne Lektury).
Poezja: liryka Mickiewicza i Norwida z API Wolnych Lektur (domena publiczna).
Teksty Skolima nie są pobierane - plik data/poezja/skolim_raw.txt dostarczamy ręcznie.
"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
POEZJA_RAW = ROOT / "data" / "poezja" / "raw"

BOOKS = {
    "sherlock.txt": [
        "https://www.gutenberg.org/files/1661/1661-0.txt",
        "https://gutenberg.pglaf.org/1/6/6/1661/1661-0.txt",
    ],
    "lalka.txt": ["https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt"],
}

# Autorzy poezji z Wolnych Lektur (slug autora w API).
POETS = {"mickiewicz": "adam-mickiewicz", "norwid": "cyprian-kamil-norwid"}

# Przekłady i utwory obcojęzyczne, które API też zwraca w „liryce”.
FOREIGN_PREFIXES = ("mickevicius", "mickievicius", "mickiewicius", "krymo-sonetai")
FOREIGN_SLUGS = {
    "mickiewicz-the-tempest", "mickiewicz-to-my-cicerone",
    "mickiewicz-to-upon-the-alps-in-splugen-1829", "mickiewicz-uncertainty",
    "norwid-in-verona", "norwid-my-little-song-ii",
}

UA = {"User-Agent": "Mozilla/5.0 (lab5 ngram)"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download_books() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, urls in BOOKS.items():
        target = RAW / name
        if target.exists() and target.stat().st_size > 100_000:
            print(f"jest już {name}")
            continue
        for url in urls:
            try:
                data = fetch(url)
            except Exception as e:  # serwer niedostępny - próbujemy mirrora
                print(f"  {url}: {e}")
                continue
            if len(data) > 100_000:
                target.write_bytes(data)
                print(f"pobrano {name} z {url} ({len(data)} B)")
                break
        else:
            raise SystemExit(f"Nie udało się pobrać {name}")


def leaf_slugs(slugs: list[str]) -> list[str]:
    """Zbiory (np. 'sonety-krymskie') zawierają utwory-dzieci; bierzemy tylko liście,
    żeby żaden wiersz nie trafił do korpusu dwa razy."""
    s = set(slugs)
    return [x for x in slugs if not any(o != x and o.startswith(x + "-") for o in s)]


def download_poems() -> None:
    for poet, api_slug in POETS.items():
        out_dir = POEZJA_RAW / poet
        out_dir.mkdir(parents=True, exist_ok=True)
        books = json.loads(fetch(f"https://wolnelektury.pl/api/authors/{api_slug}/kinds/liryka/books/"))
        slugs = [b["slug"] for b in books]
        slugs = [x for x in slugs if not x.startswith(FOREIGN_PREFIXES) and x not in FOREIGN_SLUGS]
        slugs = leaf_slugs(slugs)
        print(f"{poet}: {len(slugs)} utworów")
        for slug in slugs:
            target = out_dir / f"{slug}.txt"
            if target.exists():
                continue
            target.write_bytes(fetch(f"https://wolnelektury.pl/media/book/txt/{slug}.txt"))
            time.sleep(0.2)


if __name__ == "__main__":
    download_books()
    download_poems()
