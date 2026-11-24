"""Wspólne ustawienia eksperymentów: języki, seedy, zapis wyników, odsetek skopiowanych n-gramów."""
import json
import textwrap
from pathlib import Path

from preprocess import WORD_RE, book_sentences, detokenize, word_tokens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "wyniki"

RANDOM_SEED = 42
NUM_WORDS = 100

# Seed musi występować w KAŻDYM zbiorze uczącym, także w pierwszych 10% książki,
# więc bierzemy dwa słowa z początku (ostatnie słowa książki nie występują w 10%).
LANGS = {
    "en": {"book": "sherlock", "seed": ["sherlock", "holmes"]},
    "pl": {"book": "lalka", "seed": ["pan", "wokulski"]},
}
FRACTIONS = [10, 50, 80, 100]


def is_word(tok: str) -> bool:
    return bool(WORD_RE.search(tok)) and tok not in ("<s>", "</s>", "<unk>")


_cache = {}


def sentences(lang: str):
    if lang not in _cache:
        _cache[lang] = book_sentences(LANGS[lang]["book"])
    return _cache[lang]


def head(sents, pct):
    """Pierwsze pct% zdań książki (od początku)."""
    return sents[: round(len(sents) * pct / 100)]


def ngram_set(token_lists, k=5):
    """Zbiór k-gramów słów (bez interpunkcji) z ciągłego strumienia tekstu."""
    words = [w for s in token_lists for w in word_tokens(s)]
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def copied_ratio(generated_tokens, source_ngrams, k=5):
    """Jaka część k-gramów słów wygenerowanego tekstu występuje dosłownie w źródle."""
    w = word_tokens(generated_tokens)
    grams = [tuple(w[i:i + k]) for i in range(len(w) - k + 1)]
    return sum(g in source_ngrams for g in grams) / len(grams) if grams else 0.0


def save_text(path: Path, meta: dict, tokens: list[str], seed_tokens=()):
    """Zapisuje tekst z nagłówkiem parametrów (wymóg powtarzalności z zadania)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = detokenize(list(seed_tokens) + tokens)
    head_lines = [f"{k}: {v}" for k, v in meta.items()]
    body = "\n".join(textwrap.wrap(text, 100))
    path.write_text("\n".join(head_lines) + "\n" + "-" * 60 + "\n" + body + "\n", encoding="utf-8")
    return text


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
