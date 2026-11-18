---
title: Tokenisation and BPE
level: advanced
minutes: 40
summary: How text becomes the integers a language model reads. Characters versus words versus subwords, the byte-pair encoding algorithm step by step, and the surprising ways tokenisation shapes what models can do.
---

## Why tokenise at all

A model needs a finite vocabulary of symbols, each mapped to an [embedding](#/lesson/word-embeddings). The choice of symbols is a trade-off:

| Unit | Vocabulary | Sequence length | Problem |
|---|---|---|---|
| Characters or bytes | tiny (256 bytes) | very long | the model must learn spelling, and attention cost grows with length |
| Words | huge (millions with names, typos, inflections) | short | unknown words, no sharing between "run" and "running" |
| **Subwords** | 30k–200k | moderate | the standard compromise |

Subword tokenisers keep frequent words whole ("the", "machine") and split rare ones into pieces ("tokenisation" → "token", "isation"), so any string can be encoded and common patterns are cheap.

## Byte-pair encoding

BPE builds the vocabulary bottom-up from a training corpus:

1. Start with each word split into characters (or bytes), with word frequencies counted.
2. Count every adjacent pair of symbols, weighted by word frequency.
3. **Merge** the most frequent pair into a new symbol, everywhere it occurs.
4. Repeat until the vocabulary reaches the target size.

The ordered list of merges *is* the tokeniser. To encode new text, split it into characters and apply the learned merges in the order they were learned. Byte-level BPE (GPT-2 onwards) starts from the 256 byte values, so every possible string, in any language or with emoji, is representable with no "unknown" token. WordPiece (BERT) and the Unigram method (SentencePiece) are close relatives that choose merges or pieces by likelihood instead of raw counts.

## The number of tokens matters

- **Cost and context**: models are billed and limited per token. A language underrepresented in the tokeniser's training data needs several times more tokens for the same sentence, making it slower, more expensive and giving it less effective context.
- **Arithmetic and spelling**: "1234567" may be split into chunks like "123", "45", "67", which hides the digit structure; asking a model to count the letters in a word is hard when it never sees the letters.
- **Leading spaces**: " cat" and "cat" are usually different tokens, which is why prompts are sensitive to trailing spaces.
- **Compression**: a good tokeniser averages about 3–4 characters of English per token.

## Special tokens

Vocabularies also reserve tokens with no text meaning: beginning and end of sequence, padding, separators, and in chat models markers for system, user and assistant turns. They must never be produced by encoding ordinary user text, or a user could forge a turn boundary.

<!-- exercise: pair-counts | Count pairs and merge -->
Words are given as a dict mapping a tuple of symbols to its frequency, e.g. `{("l","o","w"): 5}`. Write `pair_counts(vocab)` returning a dict from adjacent symbol pairs to their total frequency, and `merge(vocab, pair)` returning a new dict where every occurrence of `pair` inside a word is replaced by the concatenated symbol (scan left to right, non-overlapping).

~~~python starter
from collections import Counter

def pair_counts(vocab):
    raise NotImplementedError

def merge(vocab, pair):
    raise NotImplementedError

vocab = {("l", "o", "w"): 5, ("l", "o", "w", "e", "r"): 2, ("n", "e", "w"): 6}
print(pair_counts(vocab))
print(merge(vocab, ("l", "o")))
~~~

~~~python solution
from collections import Counter

def pair_counts(vocab):
    counts = Counter()
    for word, freq in vocab.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return dict(counts)

def merge(vocab, pair):
    out = {}
    for word, freq in vocab.items():
        new, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                new.append(word[i] + word[i + 1])
                i += 2
            else:
                new.append(word[i])
                i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out

vocab = {("l", "o", "w"): 5, ("l", "o", "w", "e", "r"): 2, ("n", "e", "w"): 6}
print(pair_counts(vocab))
print(merge(vocab, ("l", "o")))
~~~

~~~python tests
vocab = {("l", "o", "w"): 5, ("l", "o", "w", "e", "r"): 2, ("n", "e", "w"): 6}

def test_counts():
    """pairs are weighted by word frequency"""
    c = pair_counts(vocab)
    assert c[("l", "o")] == 7 and c[("o", "w")] == 7 and c[("e", "w")] == 6 and c[("w", "e")] == 2

def test_merge():
    """merging ('l', 'o') replaces it in every word"""
    m = merge(vocab, ("l", "o"))
    assert m == {("lo", "w"): 5, ("lo", "w", "e", "r"): 2, ("n", "e", "w"): 6}

def test_non_overlapping():
    """('a', 'a') in 'aaa' merges once from the left"""
    assert merge({("a", "a", "a"): 1}, ("a", "a")) == {("aa", "a"): 1}
~~~

<!-- /exercise -->

<!-- exercise: bpe | Train BPE and encode -->
Write `train_bpe(word_freqs, n_merges)` where `word_freqs` maps words (strings) to counts. Split each word into characters, then repeat `n_merges` times: find the most frequent pair (ties broken by the **lexicographically smallest** pair), merge it, and record it. Return the list of merges. Then `encode(word, merges)` splits a word into characters and applies the merges in order. Helpers `pair_counts` and `merge` are provided. Stop early if no pairs remain.

~~~python starter
from collections import Counter

def pair_counts(vocab):
    c = Counter()
    for w, f in vocab.items():
        for a, b in zip(w, w[1:]):
            c[(a, b)] += f
    return c

def merge(vocab, pair):
    out = {}
    for w, f in vocab.items():
        new, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                new.append(w[i] + w[i + 1]); i += 2
            else:
                new.append(w[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + f
    return out

def train_bpe(word_freqs, n_merges):
    raise NotImplementedError

def encode(word, merges):
    raise NotImplementedError
~~~

~~~python solution
from collections import Counter

def pair_counts(vocab):
    c = Counter()
    for w, f in vocab.items():
        for a, b in zip(w, w[1:]):
            c[(a, b)] += f
    return c

def merge(vocab, pair):
    out = {}
    for w, f in vocab.items():
        new, i = [], 0
        while i < len(w):
            if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                new.append(w[i] + w[i + 1]); i += 2
            else:
                new.append(w[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + f
    return out

def train_bpe(word_freqs, n_merges):
    vocab = {tuple(w): f for w, f in word_freqs.items()}
    merges = []
    for _ in range(n_merges):
        counts = pair_counts(vocab)
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        vocab = merge(vocab, best)
        merges.append(best)
    return merges

def encode(word, merges):
    tokens = {tuple(word): 1}
    for pair in merges:
        tokens = merge(tokens, pair)
    return list(next(iter(tokens)))
~~~

~~~python tests
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}

def test_first_merges():
    """the first merges follow the counts: (e, s) and (es, t) are most frequent"""
    m = train_bpe(corpus, 2)
    assert m == [("e", "s"), ("es", "t")]

def test_tie_break():
    """ties are broken by the smallest pair"""
    assert train_bpe({"ab": 1, "cd": 1}, 1) == [("a", "b")]

def test_encode_known_and_new():
    """encoding applies merges in order and handles unseen words"""
    m = train_bpe(corpus, 6)
    assert "".join(encode("lowest", m)) == "lowest"
    assert len(encode("newest", m)) < len("newest")

def test_stops_when_done():
    """no more pairs means no more merges"""
    assert len(train_bpe({"ab": 3}, 10)) == 1
~~~

<!-- hint: `min(counts, key=lambda p: (-counts[p], p))` picks the most frequent pair and, on ties, the smallest one. -->
<!-- /exercise -->

<!-- quiz -->
1. Why do modern language models use subword tokens rather than whole words?
   - [x] Any string can be encoded with a moderate vocabulary, and frequent words stay as single tokens
   - [ ] Subwords make every sequence shorter than words would
   - [ ] Words cannot be embedded
   > Word vocabularies explode and still miss new words; characters make sequences very long.
2. What is the learned output of BPE training?
   - [ ] A neural network
   - [x] An ordered list of merges (and the resulting vocabulary)
   - [ ] A probability for each word
   > Encoding replays the merges in the order they were learned.
3. Why can the same sentence cost more tokens in Indonesian than in English for some tokenisers?
   - [ ] Indonesian words are always longer
   - [x] If the tokeniser was trained mostly on English, fewer Indonesian patterns became merged tokens
   - [ ] Tokenisers only support English
   > Merges reflect the training corpus; underrepresented languages get split into more pieces.
4. Why do byte-level BPE tokenisers never need an "unknown" token?
   - [x] They start from all 256 byte values, so any text can be represented as bytes
   - [ ] They delete unknown characters
   - [ ] They have a vocabulary of every possible word
   > Rare characters fall back to their individual bytes.
<!-- /quiz -->
