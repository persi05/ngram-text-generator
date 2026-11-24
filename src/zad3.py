"""Zadanie 3.0: MLE dla n = 1, 2, 3, oba języki, uczenie na 10/50/80/100% tekstu,
po 100 słów z każdego modelu. Wyniki: wyniki/zad3/<język>_mle_n<n>_<część>proc.txt."""
from common import (FRACTIONS, LANGS, NUM_WORDS, OUT, RANDOM_SEED, copied_ratio, head,
                    is_word, ngram_set, save_json, save_text, sentences)
from ngram_lm import NGramLM
from preprocess import word_tokens

rows = []
for lang, cfg in LANGS.items():
    sents = sentences(lang)
    for pct in FRACTIONS:
        train = head(sents, pct)
        src5 = ngram_set(train)
        n_words = sum(len(word_tokens(s)) for s in train)
        for n in (1, 2, 3):
            lm = NGramLM(n, "mle").fit(train)
            seed = cfg["seed"]
            toks = lm.generate(NUM_WORDS, seed, RANDOM_SEED, is_word=is_word)
            ratio = copied_ratio(toks, src5)
            meta = {
                "język": lang, "metoda": "MLE", "n": n,
                "część tekstu": f"{pct}% od początku ({len(train)} zdań, {n_words} słów)",
                "seed": " ".join(seed), "random_seed": RANDOM_SEED,
                "liczba słów": NUM_WORDS,
                "odsetek 5-gramów skopiowanych z tekstu uczącego": f"{ratio:.1%}",
            }
            text = save_text(OUT / "zad3" / f"{lang}_mle_n{n}_{pct}proc.txt", meta, toks, seed)
            rows.append({"lang": lang, "n": n, "pct": pct, "copied5": ratio, "vocab": lm.V,
                         "train_words": n_words, "text": text})
            print(f"{lang} n={n} {pct:>3}%  kopia5={ratio:6.1%}  {text[:90]}")

save_json(OUT / "zad3" / "podsumowanie.json", rows)

# Jedna próbka 100 słów to za mało, żeby ocenić przepisywanie - dla n = 2, 3 uśredniamy
# odsetek skopiowanych 5-gramów po 20 próbkach (random_seed 1..20) dla każdej części tekstu.
# Dodatkowo: średnia liczba różnych następników kontekstu (im mniejsza, tym model bardziej
# „idzie po szynach” tekstu źródłowego).
avg = []
for lang, cfg in LANGS.items():
    sents = sentences(lang)
    for pct in FRACTIONS:
        train = head(sents, pct)
        src5 = ngram_set(train)
        for n in (2, 3):
            lm = NGramLM(n, "mle").fit(train)
            vals = [copied_ratio(lm.generate(NUM_WORDS, cfg["seed"], rs, is_word=is_word), src5)
                    for rs in range(1, 21)]
            branching = sum(e[2] for e in lm.tables[n].values()) / len(lm.tables[n])
            avg.append({"lang": lang, "n": n, "pct": pct, "copied5_avg": sum(vals) / len(vals),
                        "successors_avg": branching})
            print(f"średnio {lang} n={n} {pct:>3}%: kopia5={sum(vals) / len(vals):.1%} "
                  f"następników/kontekst={branching:.2f}")
save_json(OUT / "zad3" / "srednie_kopiowanie.json", avg)
