"""Test poprawności własnej implementacji (ngram_lm.py) na 500 zdaniach Sherlocka.

1. MLE: P(w|h) musi być identyczne z nltk.lm.MLE.
2. Laplace: identyczne z nltk.lm.Laplace po uwzględnieniu, że nltk liczy w V także <s> i <UNK>.
3. Każdy rozkład P(·|h) (MLE, Laplace, KN) sumuje się do 1.
"""
import random

import numpy as np
from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

from ngram_lm import NGramLM
from preprocess import book_sentences

sents = book_sentences("sherlock")[:500]
rng = random.Random(0)
ok = True

for n in (1, 2, 3):
    ours = {m: NGramLM(n, m).fit(sents) for m in ("mle", "laplace") + (("kn",) if n > 1 else ())}
    ref_mle, ref_lap = MLE(n), Laplace(n)
    # nltk dla n = 1 nie dokleja </s>; nasz model tak (generator musi kończyć zdania),
    # więc dla porównania dodajemy </s> jawnie.
    data = [s + ["</s>"] for s in sents] if n == 1 else sents
    train, vocab = padded_everygram_pipeline(n, data)
    ref_mle.fit(train, vocab)
    train, vocab = padded_everygram_pipeline(n, data)
    ref_lap.fit(train, vocab)
    V_nltk = len(ref_lap.vocab)

    # losowe n-gramy z tekstu + losowe (często niewidziane) pary
    checks = []
    for _ in range(300):
        s = ours["mle"].pad(rng.choice(sents))
        i = rng.randrange(n - 1, len(s))
        checks.append((s[i], tuple(s[i - n + 1:i])))
        checks.append((rng.choice(ours["mle"].vocab), tuple(s[i - n + 1:i])))

    max_mle = max_lap = 0.0
    for w, h in checks:
        hi = tuple(ours["mle"].idx[t] for t in h)
        wi = ours["mle"].idx[w]
        max_mle = max(max_mle, abs(ours["mle"].prob(wi, hi) - ref_mle.score(w, list(h))))
        d, tot, _ = ours["mle"].tables[n].get(hi, ({}, 0, 0))
        lap_with_nltk_V = (d.get(wi, 0) + 1) / (tot + V_nltk)
        max_lap = max(max_lap, abs(lap_with_nltk_V - ref_lap.score(w, list(h))))

    sums = {m: max(abs(lm.dist(tuple(lm.idx[t] for t in h)).sum() - 1) for _, h in checks)
            for m, lm in ours.items()}
    print(f"n={n}: max|MLE - nltk| = {max_mle:.2e}, max|Laplace - nltk| = {max_lap:.2e}, "
          f"max|Σp - 1| = " + ", ".join(f"{m}: {v:.1e}" for m, v in sums.items()))
    ok &= max_mle < 1e-12 and max_lap < 1e-12 and all(v < 1e-9 for v in sums.values())

print("OK - implementacja zgodna" if ok else "BŁĄD")
