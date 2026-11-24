"""Czyszczenie tekstów, podział na zdania/wersy i tokenizacja.

Zasady z zadania: lowercase, interpunkcja jako osobne tokeny, zdania (dla poezji wersy)
jako osobne sekwencje; znaczniki <s> i </s> dokłada model (ngram_lm.pad).
"""
import random
import re
from pathlib import Path

from nltk.tokenize import sent_tokenize

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
POEZJA = ROOT / "data" / "poezja"
CLEAN = ROOT / "data" / "clean"

# słowo (z apostrofem/łącznikiem w środku, np. don't, biało-czerwony) albo pojedynczy znak interpunkcji
TOKEN_RE = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]")
WORD_RE = re.compile(r"\w", re.UNICODE)


# ---------------------------------------------------------------- książki

def clean_sherlock(text: str) -> str:
    start = text.index("*** START OF")
    start = text.index("***", start + 5) + 3          # koniec linii START
    end = text.index("*** END OF")
    body = text[start:end]
    body = body[body.index("I. A SCANDAL IN BOHEMIA", body.index("Contents") + 10):]  # pomijamy spis treści
    body = body.replace("_", "")                       # kursywa w Gutenbergu: _the_
    # nagłówki opowiadań i rozdziałów (wiersze z samych wielkich liter / cyfr rzymskich)
    lines = [ln for ln in body.splitlines() if not re.fullmatch(r"\s*([IVXL]+\.)?[\sA-Z’'\-.,]*", ln) or ln.strip() == ""]
    return "\n".join(lines)


def clean_wolnelektury(text: str) -> str:
    """Usuwa stopkę (od '-----') i nagłówek (autor, tytuł, ISBN) Wolnych Lektur."""
    text = text.split("\n-----\n")[0]
    return text


def clean_lalka(text: str) -> str:
    text = clean_wolnelektury(text)
    lines = text.splitlines()
    # nagłówek: autor, tytuł, ISBN, „Tom I” - treść zaczyna się od pierwszego rozdziału
    first = next(i for i, ln in enumerate(lines) if ln.startswith("I. Jak wygląda firma"))
    out = []
    for ln in lines[first:]:
        if re.fullmatch(r"[IVXL]+\. .*", ln.strip()) and len(ln) < 120:   # tytuły rozdziałów
            continue
        out.append(ln)
    return "\n".join(out)


def paragraphs(text: str) -> list[str]:
    """Łączy złamane linie w akapity (akapity oddzielone pustą linią)."""
    paras = re.split(r"\n\s*\n", text)
    return [re.sub(r"\s+", " ", p).strip() for p in paras if p.strip()]


def tokenize(s: str) -> list[str]:
    # cudzysłowy otwierające (“ „ ‘) i zamykające (” ’) zostają osobnymi tokenami,
    # żeby przy sklejaniu wiedzieć, z której strony dokleić spację
    s = s.lower().replace("’", "'").replace('"', "”")
    return TOKEN_RE.findall(s)


def book_sentences(name: str) -> list[list[str]]:
    raw = (RAW / f"{name}.txt").read_text(encoding="utf-8-sig")
    if name == "sherlock":
        text, lang = clean_sherlock(raw), "english"
    else:
        text, lang = clean_lalka(raw), "polish"
    sents = []
    for p in paragraphs(text):
        for s in sent_tokenize(p, language=lang):
            toks = tokenize(s)
            if any(WORD_RE.search(t) for t in toks):
                sents.append(toks)
    return sents


def word_tokens(tokens) -> list[str]:
    """Same słowa (bez interpunkcji i znaczników) - do liczenia długości i n-gramów."""
    return [t for t in tokens if WORD_RE.search(t) and t not in ("<s>", "</s>", "<unk>")]


def detokenize(tokens: list[str]) -> str:
    """Skleja tokeny w czytelny tekst: </s> kończy zdanie, interpunkcja bez spacji przed."""
    out, sentence = [], []
    for t in tokens:
        if t == "<s>":
            continue
        if t == "</s>":
            if sentence:
                out.append(_join(sentence))
            sentence = []
            continue
        sentence.append(t)
    if sentence:
        out.append(_join(sentence))
    return " ".join(out)


def _join(toks: list[str]) -> str:
    s = ""
    glue_next = False                     # po cudzysłowie/nawiasie otwierającym bez spacji
    for t in toks:
        if t in ".,;:!?)…»”'" :
            s += t
        elif glue_next or not s:
            s += t
        else:
            s += " " + t
        glue_next = t in "(«“„‘"
    return s


# ---------------------------------------------------------------- poezja

# Rdzenie przekleństw (zakaz z polecenia: „bez przekleństw”). Wers z takim słowem jest
# usuwany z korpusu, więc model nie może go wygenerować.
PROFANITY = re.compile(
    r"\b\w*(jeb|kurw|chuj|huj|pierd[oa]l|pizd|kutas|skurw|cwel|dziwk|szmat)\w*\b", re.IGNORECASE)

FILLER = re.compile(r"^(la|laj|na|nanana|ha|aj|jaj|ej|uuu|o+|oj(oj)*|a)$", re.IGNORECASE)


def clean_skolim(raw: str) -> list[list[str]]:
    """Zwraca listę piosenek, każda to lista wersów (napisy z tekstowo.pl).

    - „Tekst pochodzi z …” = koniec piosenki;
    - usuwamy oznaczenia „Ref.:”, „Zw. I”, numery zwrotek „1.”, krotności „(2 x)”, „/ x2”;
    - refren powtórzony w piosence zostawiamy raz (powtórzenia sztucznie zawyżają liczności);
    - usuwamy wersy złożone tylko z wypełniaczy („la la la”, „Ooooo”, „na na na”);
    - usuwamy wersy z przekleństwami.
    """
    songs, stanzas, stanza = [], [], []

    def close_stanza():
        nonlocal stanza
        if stanza:
            stanzas.append(stanza)
        stanza = []

    for line in raw.splitlines():
        ln = line.strip()
        if ln.startswith("Tekst pochodzi z"):
            close_stanza()
            seen, verses = set(), []
            for st in stanzas:
                key = tuple(v.lower() for v in st)
                if key in seen:
                    continue
                seen.add(key)
                verses.extend(st)
            songs.append(verses)
            stanzas = []
            continue
        if not ln:
            close_stanza()
            continue
        ln = re.sub(r"^(Ref\.?:?|Ref x ?\d|Zw\. ?[IVX]+)\s*", "", ln, flags=re.IGNORECASE)
        ln = re.sub(r"^\d\.\s*", "", ln)
        ln = re.sub(r"\(\s*\d\s*x\s*\)|/\s*x\s*\d|\((\?|…|\.\.\.)\)", "", ln)
        ln = ln.strip()
        if not ln:
            continue
        words = re.findall(r"\w+", ln)
        if not words or all(FILLER.match(w) or w.lower() == "ja" for w in words):
            continue
        if PROFANITY.search(ln):
            continue
        stanza.append(ln)
    close_stanza()
    return [s for s in songs if s]


def clean_wl_poem(raw: str) -> list[str]:
    """Wiersz z Wolnych Lektur: pomija nagłówek (autor + tytuł, oddzielone 3 pustymi liniami)."""
    text = clean_wolnelektury(raw)
    parts = re.split(r"\n\s*\n\s*\n\s*\n", text, maxsplit=1)   # nagłówek kończą >=3 puste linie
    body = parts[1] if len(parts) == 2 else text
    verses = []
    for ln in body.splitlines():
        ln = ln.replace("*", "").strip()               # *…* = kursywa w Wolnych Lekturach
        if not ln or ln.startswith("ISBN") or set(ln) <= set(". "):
            continue
        if re.fullmatch(r"[IVXLCivxlc]+\.?|\d+\.?", ln):  # numery części wiersza
            continue
        if PROFANITY.search(ln) or len(ln) > 150:   # długie linie to przypisy/proza
            continue
        verses.append(ln)
    return verses


# Utwory prozą, które API Wolnych Lektur zalicza do liryki - nie są wierszami.
PROSE = {"norwid-czarne-kwiaty", "norwid-biale-kwiaty"}


def load_poets(balance_seed: int = 7) -> dict[str, list[str]]:
    """Wersy każdego autora. Mickiewicza i Norwida przycinamy (losując całe wiersze, seed
    stały) do podobnej liczby słów co Skolim, żeby żaden autor nie zdominował modelu."""
    skolim = [v for song in clean_skolim((POEZJA / "skolim_raw.txt").read_text(encoding="utf-8")) for v in song]
    budget = sum(len(word_tokens(tokenize(v))) for v in skolim)
    authors = {"skolim": skolim}
    for poet in ("mickiewicz", "norwid"):
        poems = [clean_wl_poem(p.read_text(encoding="utf-8")) for p in sorted((POEZJA / "raw" / poet).glob("*.txt"))
                 if p.stem not in PROSE]
        poems = [p for p in poems if p]
        random.Random(balance_seed).shuffle(poems)
        chosen, n = [], 0
        for p in poems:
            if n >= budget:
                break
            chosen.extend(p)
            n += sum(len(word_tokens(tokenize(v))) for v in p)
        authors[poet] = chosen
    return authors


def save_clean() -> None:
    """Zapisuje oczyszczone teksty do data/clean/ (do wglądu i do markovify)."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    for name in ("sherlock", "lalka"):
        sents = book_sentences(name)
        (CLEAN / f"{name}_zdania.txt").write_text("\n".join(" ".join(s) for s in sents), encoding="utf-8")
    for author, verses in load_poets().items():
        (CLEAN / f"poezja_{author}.txt").write_text("\n".join(verses), encoding="utf-8")


if __name__ == "__main__":
    save_clean()
    for name in ("sherlock", "lalka"):
        s = book_sentences(name)
        print(name, "zdań:", len(s), "tokenów:", sum(map(len, s)), "słów:", sum(len(word_tokens(x)) for x in s))
    for a, v in load_poets().items():
        print(a, "wersów:", len(v), "słów:", sum(len(word_tokens(tokenize(x))) for x in v))
