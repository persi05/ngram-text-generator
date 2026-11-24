# N-gram Text Generator

A from-scratch n-gram language model for generating Polish and English text

Authors: Grzegorz Rzepiela, Patryk Persak.

## Features

- **Three smoothing methods** implemented by hand in `ngram_lm.py`:
  - Maximum Likelihood Estimation (MLE).
  - Laplace (add-one).
  - Interpolated Kneser-Ney (discount D = 0.75).
- **Text generation** from a seed phrase, with optional temperature sampling.
- **Perplexity** evaluation on a held-out test set.
- **Poetry generation** from several authors, using both our word model and `markovify`.
- **Plagiarism check**: measures how much of the generated text is copied verbatim from the training data
  (the share of word 5-grams and 6-grams that also appear in the source).
- **Verified against `nltk.lm`**: our MLE and Laplace probabilities match nltk exactly, and every
  distribution sums to 1.

## Data

| Language | Work | Source | Sentences | Words |
|---|---|---|---|---|
| EN | A. Conan Doyle, *The Adventures of Sherlock Holmes* | Project Gutenberg (1661) | 6,863 | 104,493 |
| PL | B. Prus, *Lalka*, vol. I | Wolne Lektury | 9,188 | 116,784 |
| PL, poetry | Skolim, A. Mickiewicz (lyric poems), C. K. Norwid (lyric poems) | Skolim lyrics pasted manually, the rest from Wolne Lektury | 2,534 verses | ~4.5k per author |

**Preprocessing:**
- Site headers and footers, tables of contents and chapter titles are removed.
- Prose is split into sentences with `nltk.sent_tokenize`, separately for Polish and English.
  Poetry is split into verses.
- Text is lowercased and punctuation becomes separate tokens.
- Each sentence or verse is wrapped in `<s>` … `</s>`.
- Lines containing profanity are dropped from the poetry corpus.

## Project structure

```
src/
  pobierz_dane.py   downloads the books and the Mickiewicz/Norwid poems into data/
  preprocess.py     cleaning, sentence/verse splitting, tokenization, profanity filter
  ngram_lm.py       the n-gram model: MLE, Laplace, interpolated Kneser-Ney
  sprawdz_nltk.py   test: our MLE/Laplace equal nltk.lm, every distribution sums to 1
  common.py         seeds, saving outputs with their parameters, copied n-gram ratio
  zad3.py           level 3.0: MLE n=1,2,3 × PL/EN × 10/50/80/100% of the text
  zad4.py           level 4.0: Laplace and Kneser-Ney, perplexity (90/10 split), temperature
  zad5.py           level 5.0: poems (word model + markovify), copied n-gram tables
wyniki/zad3, zad4, zad5   generated .txt files plus .json summaries
```

**Output files:**
- Prose files are named `<lang>_<method>_n<n>_<part>proc.txt`, e.g. `pl_mle_n3_50proc.txt`.
- Temperature runs are named `<lang>_kn_n3_T<temperature>.txt`.
- Every file starts with a header listing its parameters: n, method, seed, random_seed and temperature.

## Usage

```
pip install -r requirements.txt
cd src
python pobierz_dane.py        # download data (once)
python sprawdz_nltk.py        # optional sanity check against nltk
python zad3.py && python zad4.py && python zad5.py
```

**Reproducibility:** all sampling uses a fixed `random_seed`, so results are reproducible.
Prose uses `random_seed = 42` and poems use `1..5`.

**Skolim lyrics:** they are copyrighted and therefore not included in the repository. `zad5.py`
expects them in `data/poezja/skolim_raw.txt`, in the format copied from tekstowo.pl, where a line
starting with "Tekst pochodzi z …" ends each song.

## Results

### Level 3.0: MLE

**Seed phrases:** `sherlock holmes` (EN) and `pan wokulski` (PL). The task suggests the last words of
the book, but those do not appear in the first 10% of the text, and the seed must be present in every
training subset. So we use words from the beginning of the book.

**From which n does the text sound correct?**
- **n = 1:** a bag of words with no syntax, e.g. *"the himself turned saw. that the holmes for with
  opening…"*. The model only knows word frequencies, so sentence length and comma placement are random.
- **n = 2:** neighbouring word pairs fit, e.g. *"pan wokulski wrócił do tego rodzaju budowle"*, but the
  sentence as a whole makes no sense. Long-range agreement is missing, which is especially visible in
  Polish.
- **n = 3:** text sounds correct over spans of several words, e.g. *"pan wokulski — poprawił się
  rzecki — pisze mi, panie klejn, przybijcie z łaski swojej literki"*. For both languages correctness
  starts at n = 3, but the price is copying.

**Does the model copy the book?** Yes, from n = 3 onwards. The table shows the share of word 5-grams in
the generated text that appear verbatim in the training text. Each value is the average of 20 samples
of 100 words.

| | n = 2 | n = 3 | n = 4 (extra) |
|---|---|---|---|
| EN, 100% of text | 0.1% | 19.1% | 59.3% |
| PL, 100% of text | 0.2% | 25.7% | 60.2% |

A trigram context has only 1.5–1.9 distinct successors on average. The model therefore often follows the
original "on rails" for a whole sentence. At n = 4 more than half of the output is quotation.

**How does the amount of training text matter?** Copied 5-grams for MLE, n = 3:

| part of text | 10% | 50% | 80% | 100% |
|---|---|---|---|---|
| EN | 34.0% | 23.6% | 19.5% | 19.1% |
| PL | 42.0% | 29.5% | 27.4% | 25.7% |

- **Fewer continuations:** with less data, each context has fewer distinct continuations. A Polish
  context has 1.24 successors on average at 10% of the text, compared with 1.49 at 100%.
- **Near-copy at 10%:** the 10% model reproduces the book almost word for word.
- **Bigger vocabulary:** the Polish vocabulary grows from 4.3k to 22k words, so the larger model has more
  options and produces new word combinations.
- **Correctness vs. originality:** local correctness at n = 3 stays about the same, while originality
  increases.
- **Polish vs. English:** Polish copies more than English. Rich inflection makes each word form rarer,
  so contexts are sparser.

### Level 4.0: smoothing, perplexity, temperature

**Perplexity** (train on the first 90%, test on the last 10%):

| Model | EN | PL |
|---|---|---|
| Kneser-Ney, n = 3 | 67.7 | 106.2 |
| Laplace, n = 3 | 1,456 | 2,908 |
| MLE, n ≥ 2 | ∞ | ∞ |

- **Laplace** gets worse as n grows, because add-one gives too much probability mass to unseen events.
- **MLE** has infinite perplexity for n ≥ 2: 19% (EN) and 25% (PL) of the test bigrams never occur in
  the training part.

**Temperature** (Kneser-Ney, n = 3):
- **T = 0.5:** safe, repetitive phrases and more copying (9–13% of 5-grams, vs. ~1% at T = 1.0).
- **T = 1.5:** a salad of rare words with no sentence endings.

### Level 5.0: poetry

Five poems of six verses each are generated from a mixed corpus of Skolim, Mickiewicz and Norwid,
using our word model (MLE / Kneser-Ney) and `markovify` (`state_size` 1–3). The results mix disco polo
with Romanticism, e.g. (`wiersze_slowny_mle_n3.txt`):

> bądź moją kardashian, róbmy to co mam dla ciebie<br>
> niechaj rozkosznie usypiam na wieki,

Copied 5-grams: 14% for the word model (MLE, n = 3) and 21% for markovify (`state_size` = 2).
