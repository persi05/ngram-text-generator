"""Model n-gramowy słów: MLE, Laplace i interpolowany Kneser-Ney.

Własna implementacja (zamiast nltk.lm), bo nltk.KneserNeyInterpolated liczy rozkład
po całym słowniku bardzo wolno. Zgodność z nltk sprawdza sprawdz_nltk.py.

Oznaczenia: h - kontekst (n-1 poprzednich tokenów), w - słowo, c(.) - liczba wystąpień,
V - rozmiar słownika (bez <s>, którego model nigdy nie przewiduje).
"""
from collections import Counter, defaultdict
import math

import numpy as np

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class NGramLM:
    def __init__(self, n: int, method: str = "mle", discount: float = 0.75):
        assert method in ("mle", "laplace", "kn")
        if method == "kn":
            assert n >= 2, "Kneser-Ney ma sens od n = 2"
        self.n, self.method, self.d = n, method, discount

    # ------------------------------------------------------------ uczenie
    def pad(self, sent):
        return [BOS] * (self.n - 1) + list(sent) + [EOS]

    def fit(self, sentences, vocab=None):
        """sentences: lista list tokenów (bez <s>/</s>). vocab: opcjonalny stały słownik
        (do perplexity - wtedy zdania muszą już mieć <unk> w miejsce rzadkich słów)."""
        words = set(vocab) if vocab is not None else {w for s in sentences for w in s}
        words.discard(BOS)
        words.add(EOS)
        self.vocab = sorted(words)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx[BOS] = -1                       # <s> występuje tylko w kontekście
        self.V = len(self.vocab)

        # liczności n-gramów wszystkich rzędów: ngrams[k][(h, w)] dla |h| = k-1
        ngrams = {k: Counter() for k in range(1, self.n + 1)}
        for s in sentences:
            ids = [self.idx[t] for t in self.pad(s)]
            for i in range(self.n - 1, len(ids)):
                for k in range(1, self.n + 1):
                    ngrams[k][(tuple(ids[i - k + 1:i]), ids[i])] += 1

        # tabele[k]: h -> {w: wartość}; dla MLE/Laplace tylko najwyższy rząd z c(h,w),
        # dla KN najwyższy rząd z c(h,w), niższe z liczbami kontynuacji N1+(• h w).
        self.tables = {}
        self.tables[self.n] = self._group(ngrams[self.n])
        if self.method == "kn":
            for k in range(1, self.n):
                cont = Counter()
                for (h, w) in ngrams[k + 1]:          # różne (k+1)-gramy v h w
                    cont[(h[1:], w)] += 1             # → +1 dla k-gramu (h w)
                self.tables[k] = self._group(cont)
        self._cache = {}
        return self

    @staticmethod
    def _group(counter):
        table = defaultdict(dict)
        for (h, w), c in counter.items():
            table[h][w] = c
        # dla każdego kontekstu: (słownik w->c, suma c(h), liczba typów N1+(h •))
        return {h: (d, sum(d.values()), len(d)) for h, d in table.items()}

    # ------------------------------------------------------------ prawdopodobieństwa
    def prob(self, w: int, h: tuple) -> float:
        """P(w | h) dla indeksów słów; h ma długość n-1."""
        if self.method == "mle":
            e = self.tables[self.n].get(h)
            return e[0].get(w, 0) / e[1] if e else 0.0
        if self.method == "laplace":
            d, tot, _ = self.tables[self.n].get(h, ({}, 0, 0))
            return (d.get(w, 0) + 1) / (tot + self.V)
        return self._kn(w, h, self.n)

    def _kn(self, w, h, k):
        if k == 0:
            return 1.0 / self.V                       # rozkład jednostajny na dole rekurencji
        e = self.tables[k].get(h)
        lower = self._kn(w, h[1:], k - 1)
        if e is None:                                 # niewidziany kontekst: cała masa niżej
            return lower
        d, tot, types = e
        return max(d.get(w, 0) - self.d, 0) / tot + self.d * types / tot * lower

    def dist(self, h: tuple) -> np.ndarray:
        """Cały rozkład P(· | h) jako wektor długości V (do losowania)."""
        if h in self._cache:
            return self._cache[h]
        if self.method == "mle":
            p = np.zeros(self.V)
            e = self.tables[self.n].get(h)
            if e:
                ids, c = self._arrays(self.n, h)
                p[ids] = c / e[1]
        elif self.method == "laplace":
            d, tot, _ = self.tables[self.n].get(h, ({}, 0, 0))
            p = np.full(self.V, 1.0 / (tot + self.V))
            if d:
                ids, c = self._arrays(self.n, h)
                p[ids] = (c + 1) / (tot + self.V)
        else:
            p = self._kn_vec(h, self.n)
        if len(self._cache) < 50_000:
            self._cache[h] = p
        return p

    def _arrays(self, k, h):
        d = self.tables[k][h][0]
        return np.fromiter(d.keys(), int, len(d)), np.fromiter(d.values(), float, len(d))

    def _kn_vec(self, h, k):
        if k == 0:
            return np.full(self.V, 1.0 / self.V)
        lower = self._kn_vec(h[1:], k - 1)
        e = self.tables[k].get(h)
        if e is None:
            return lower
        _, tot, types = e
        ids, c = self._arrays(k, h)
        p = self.d * types / tot * lower
        p[ids] += np.maximum(c - self.d, 0) / tot
        return p

    # ------------------------------------------------------------ generowanie
    def start_context(self, seed_tokens=()):
        ctx = [BOS] * (self.n - 1) + list(seed_tokens)
        return tuple(self.idx[t] for t in ctx[len(ctx) - (self.n - 1):]) if self.n > 1 else ()

    def generate(self, num_words=None, seed_tokens=(), random_seed=42, temperature=1.0, is_word=None,
                 num_sentences=None):
        """Losuje kolejne tokeny, aż powstanie num_words słów (interpunkcja się nie liczy)
        albo num_sentences zdań/wersów (do poezji).
        Po </s> kontekst wraca do <s> … <s> - nowe zdanie, a nie losowanie z kontekstu </s>."""
        rng = np.random.default_rng(random_seed)
        h = self.start_context(seed_tokens)
        if self.method == "mle" and self.n > 1 and h not in self.tables[self.n]:
            raise ValueError(f"Seed {seed_tokens} nie występuje w danych uczących")
        out, words = [], 0
        while (words < num_words) if num_sentences is None else (out.count(EOS) < num_sentences):
            p = self.dist(h)
            if temperature != 1.0:
                p = np.where(p > 0, p, 0) ** (1.0 / temperature)
            p = p / p.sum()
            w = self.vocab[rng.choice(self.V, p=p)]
            out.append(w)
            if w == EOS:
                h = self.start_context()
                continue
            if is_word is None or is_word(w):
                words += 1
            if self.n > 1:
                h = h[1:] + (self.idx[w],)
        return out

    # ------------------------------------------------------------ ocena
    def perplexity(self, sentences):
        """PP = exp(-1/N Σ log P(w_i | h_i)) po wszystkich tokenach testu łącznie z </s>.
        Jeśli choć jedno P = 0 (MLE, niewidziany n-gram) - perplexity jest nieskończona."""
        log_sum, n_tok, n_zero = 0.0, 0, 0
        for s in sentences:
            ids = [self.idx.get(t, self.idx.get(UNK)) for t in self.pad(s)]
            for i in range(self.n - 1, len(ids)):
                p = self.prob(ids[i], tuple(ids[i - self.n + 1:i]))
                n_tok += 1
                if p <= 0:
                    n_zero += 1
                else:
                    log_sum += math.log(p)
        ppl = math.inf if n_zero else math.exp(-log_sum / n_tok)
        return {"ppl": ppl, "tokens": n_tok, "zero": n_zero,
                "ppl_seen": math.exp(-log_sum / (n_tok - n_zero)) if n_tok > n_zero else math.inf}


def build_vocab(sentences, min_count=2):
    """Słownik do perplexity: słowa z co najmniej min_count wystąpieniami + <unk>."""
    c = Counter(w for s in sentences for w in s)
    return {w for w, k in c.items() if k >= min_count} | {UNK}


def apply_unk(sentences, vocab):
    return [[w if w in vocab else UNK for w in s] for s in sentences]
