"""Zadanie 4.0:
1) generowanie z modeli Laplace i Kneser-Ney (n = 2, 3, 100% tekstu),
2) perplexity: uczenie na pierwszych 90% zdań, test na ostatnich 10%, wspólny słownik
   (słowa z < 2 wystąpieniami w części uczącej -> <unk>),
3) temperatura 0.5 / 1.0 / 1.5 dla najlepszego modelu (najniższa perplexity).
"""
import math

from common import (LANGS, NUM_WORDS, OUT, RANDOM_SEED, copied_ratio, is_word, ngram_set,
                    save_json, save_text, sentences)
from ngram_lm import NGramLM, apply_unk, build_vocab
from preprocess import word_tokens

METHODS = {"mle": "MLE", "laplace": "Laplace", "kn": "Kneser-Ney"}
TEMPS = [0.5, 1.0, 1.5]


def text_stats(toks, src5):
    w = word_tokens(toks)
    n_sent = max(toks.count("</s>"), 1)
    return {"copied5": copied_ratio(toks, src5), "distinct": len(set(w)) / len(w),
            "words_per_sentence": len(w) / n_sent}


results = {"generation": [], "perplexity": [], "temperature": []}

for lang, cfg in LANGS.items():
    sents = sentences(lang)
    src5 = ngram_set(sents)

    # --- 1) generowanie z modeli wygładzanych
    for method in ("laplace", "kn"):
        for n in (2, 3):
            lm = NGramLM(n, method).fit(sents)
            toks = lm.generate(NUM_WORDS, cfg["seed"], RANDOM_SEED, is_word=is_word)
            st = text_stats(toks, src5)
            meta = {"język": lang, "metoda": METHODS[method], "n": n,
                    "część tekstu": "100%", "seed": " ".join(cfg["seed"]),
                    "random_seed": RANDOM_SEED, "temperatura": 1.0,
                    **({"dyskont D": lm.d} if method == "kn" else {}),
                    "odsetek 5-gramów skopiowanych z tekstu uczącego": f"{st['copied5']:.1%}"}
            text = save_text(OUT / "zad4" / f"{lang}_{method}_n{n}_100proc.txt", meta, toks, cfg["seed"])
            results["generation"].append({"lang": lang, "method": method, "n": n, **st, "text": text})
            print(f"{lang} {method} n={n} kopia5={st['copied5']:.1%} {text[:80]}")

    # --- 2) perplexity
    cut = round(len(sents) * 0.9)
    vocab = build_vocab(sents[:cut], min_count=2)
    train, test = apply_unk(sents[:cut], vocab), apply_unk(sents[cut:], vocab)
    unk_rate = sum(w == "<unk>" for s in test for w in s) / sum(map(len, test))
    for method in METHODS:
        for n in (1, 2, 3):
            if method == "kn" and n == 1:
                continue
            lm = NGramLM(n, method).fit(train, vocab=vocab)
            r = lm.perplexity(test)
            results["perplexity"].append({"lang": lang, "method": method, "n": n, "V": lm.V,
                                          "unk_rate": unk_rate, **r})
            print(f"{lang} PP {method:8s} n={n}: {r['ppl']:10.1f}  zera={r['zero']}/{r['tokens']}"
                  f"  PP(bez zer)={r['ppl_seen']:.1f}")

    # --- 3) temperatura dla najlepszego modelu
    finite = [r for r in results["perplexity"] if r["lang"] == lang and math.isfinite(r["ppl"])]
    best = min(finite, key=lambda r: r["ppl"])
    lm = NGramLM(best["n"], best["method"]).fit(sents)
    for T in TEMPS:
        toks = lm.generate(NUM_WORDS, cfg["seed"], RANDOM_SEED, temperature=T, is_word=is_word)
        st = text_stats(toks, src5)
        meta = {"język": lang, "metoda": METHODS[best["method"]], "n": best["n"],
                "część tekstu": "100%", "seed": " ".join(cfg["seed"]),
                "random_seed": RANDOM_SEED, "temperatura": T,
                "odsetek 5-gramów skopiowanych z tekstu uczącego": f"{st['copied5']:.1%}"}
        text = save_text(OUT / "zad4" / f"{lang}_{best['method']}_n{best['n']}_T{T}.txt", meta, toks, cfg["seed"])
        results["temperature"].append({"lang": lang, "method": best["method"], "n": best["n"],
                                       "T": T, **st, "text": text})
        print(f"{lang} T={T} kopia5={st['copied5']:.1%} różne={st['distinct']:.2f} {text[:80]}")

save_json(OUT / "zad4" / "podsumowanie.json", results)
