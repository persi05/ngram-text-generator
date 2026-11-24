"""Zadanie 5.0: poezja (Skolim, Mickiewicz, Norwid), jednostką jest wers.

- 5 wierszy po 6 wersów z modelu słownego (MLE i Kneser-Ney, n = 2, 3) oraz z markovify
  (state_size 1, 2, 3; z domyślnym testem oryginalności i bez niego);
- odsetek skopiowanych 5- i 6-gramów słów: wiersze vs korpus poezji, teksty prozą (3.0, 4.0)
  vs książka - tabele do sprawozdania.
"""
import random

import markovify

from common import OUT, copied_ratio, ngram_set, save_json
from ngram_lm import EOS, NGramLM
from preprocess import load_poets, tokenize

N_POEMS, N_VERSES = 5, 6
POEM_SEEDS = [1, 2, 3, 4, 5]          # random_seed dla kolejnych wierszy

authors = load_poets()
verses = [v for a in authors.values() for v in a]
tok_verses = [tokenize(v) for v in verses]
src = {k: ngram_set(tok_verses, k) for k in (5, 6)}


def lines_from_tokens(toks):
    from preprocess import detokenize
    lines, cur = [], []
    for t in toks:
        if t == EOS:
            lines.append(detokenize(cur))
            cur = []
        else:
            cur.append(t)
    return lines


def score(poems_tokens):
    """Średni odsetek skopiowanych k-gramów (liczony na sklejonym tekście wszystkich wierszy)."""
    allt = [t for p in poems_tokens for t in p]
    return {f"copied{k}": copied_ratio(allt, src[k], k) for k in (5, 6)}


configs = []

# --- model słowny
for method, n, T in [("mle", 2, 1.0), ("mle", 3, 1.0), ("kn", 2, 1.0), ("kn", 3, 1.0), ("kn", 3, 0.7)]:
    lm = NGramLM(n, method).fit(tok_verses)
    poems, poems_tok = [], []
    for rs in POEM_SEEDS:
        toks = lm.generate(seed_tokens=(), random_seed=rs, temperature=T, num_sentences=N_VERSES)
        poems_tok.append(toks)
        poems.append(lines_from_tokens(toks))
    name = f"slowny_{method}_n{n}" + (f"_T{T}" if T != 1.0 else "")
    configs.append({"name": name, "method": {"mle": "MLE", "kn": "Kneser-Ney"}[method], "n": n,
                    "T": T, "random_seeds": POEM_SEEDS, "poems": poems, **score(poems_tok)})


# --- markovify (Markov na słowach z oryginalną wielkością liter i interpunkcją przyklejoną do słów)
def markovify_poems(state_size, test_output):
    model = markovify.NewlineText("\n".join(verses), state_size=state_size)
    poems = []
    for rs in POEM_SEEDS:
        random.seed(rs)                       # markovify losuje modułem random
        lines = []
        while len(lines) < N_VERSES:
            s = model.make_sentence(tries=200, test_output=test_output)
            if s:
                lines.append(s)
        poems.append(lines)
    return poems


for state in (1, 2, 3):
    for test in (True, False):
        poems = markovify_poems(state, test)
        poems_tok = [[t for ln in p for t in tokenize(ln) + [EOS]] for p in poems]
        configs.append({"name": f"markovify_s{state}" + ("" if test else "_bez_testu"),
                        "method": "markovify" + ("" if test else " (bez testu)"), "n": state + 1,
                        "T": None, "random_seeds": POEM_SEEDS, "poems": poems, **score(poems_tok)})

for c in configs:
    path = OUT / "zad5" / f"wiersze_{c['name']}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (f"korpus: Skolim + Mickiewicz + Norwid ({len(verses)} wersów), jednostka: wers\n"
            f"metoda: {c['method']} | n: {c['n']}" + (f" | temperatura: {c['T']}" if c["T"] else "")
            + f" | random_seed kolejnych wierszy: {c['random_seeds']}\n"
            f"odsetek skopiowanych 5-gramów: {c['copied5']:.1%}, 6-gramów: {c['copied6']:.1%}\n" + "-" * 60)
    body = "\n\n".join(f"Wiersz {i + 1} (random_seed={rs})\n" + "\n".join(p)
                       for i, (rs, p) in enumerate(zip(c["random_seeds"], c["poems"])))
    path.write_text(head + "\n" + body + "\n", encoding="utf-8")
    print(f"{c['name']:28s} 5g={c['copied5']:6.1%} 6g={c['copied6']:6.1%} | {' / '.join(c['poems'][0][:2])}")

# --- skopiowane n-gramy dla tekstów prozą (źródło = cała książka). Jedna próbka 100 słów
# daje tylko ~96 5-gramów, więc wynik jest mocno losowy - uśredniamy po 20 próbkach
# (random_seed 1..20, po 100 słów, ten sam seed startowy co w 3.0/4.0).
from common import LANGS, NUM_WORDS, is_word, sentences  # noqa: E402

PROSE_SEEDS = list(range(1, 21))
PROSE_CONFIGS = [("mle", 1, 1.0), ("mle", 2, 1.0), ("mle", 3, 1.0), ("mle", 4, 1.0),
                 ("laplace", 2, 1.0), ("laplace", 3, 1.0), ("kn", 2, 1.0), ("kn", 3, 1.0),
                 ("kn", 3, 0.5), ("kn", 3, 1.5)]
NAMES = {"mle": "MLE", "laplace": "Laplace", "kn": "Kneser-Ney"}
prose = []
for lang, cfg in LANGS.items():
    sents = sentences(lang)
    book = {k: ngram_set(sents, k) for k in (5, 6)}
    for method, n, T in PROSE_CONFIGS:
        lm = NGramLM(n, method).fit(sents)
        r = {5: [], 6: []}
        for rs in PROSE_SEEDS:
            toks = lm.generate(NUM_WORDS, cfg["seed"], rs, temperature=T, is_word=is_word)
            for k in (5, 6):
                r[k].append(copied_ratio(toks, book[k], k))
        row = {"lang": lang, "method": NAMES[method], "n": n, "T": T,
               "copied5": sum(r[5]) / len(r[5]), "copied6": sum(r[6]) / len(r[6]),
               "copied5_min": min(r[5]), "copied5_max": max(r[5]), "samples": len(PROSE_SEEDS)}
        prose.append(row)
        print(f"proza {lang} {method:8s} n={n} T={T}: 5g={row['copied5']:.1%} "
              f"({row['copied5_min']:.0%}-{row['copied5_max']:.0%}) 6g={row['copied6']:.1%}")

stats = {a: {"verses": len(v), "words": sum(len([t for t in tokenize(x) if t.isalnum() or t[0].isalpha()]) for x in v)}
         for a, v in authors.items()}
save_json(OUT / "zad5" / "podsumowanie.json", {"poetry": configs, "prose": prose, "authors": stats})
