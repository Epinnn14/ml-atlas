---
title: Embeddings and word2vec
level: intermediate
minutes: 40
summary: Represent words, products or users as dense vectors where similar things are close. The distributional hypothesis, skip-gram with negative sampling, vector arithmetic analogies, and embeddings as lookup tables in every modern model.
---

## From one-hot to dense vectors

A vocabulary of 50 000 words as [one-hot vectors](#/lesson/data-preprocessing) gives 50 000 sparse dimensions in which every pair of words is equally far apart: "cat" is as different from "dog" as from "carburettor". An **embedding** maps each word to a dense vector of, say, 300 numbers, learned so that words used in similar ways get similar vectors.

## The distributional hypothesis

"You shall know a word by the company it keeps." Words that appear in similar contexts tend to mean similar things: *coffee* and *tea* both appear near *cup*, *hot* and *drink*. So we can learn word vectors from raw text by training them to predict context, with no labels at all.

## Skip-gram

For each word in a corpus (the **centre** word), take the words within a **window** of $\pm m$ positions as its **context**. Skip-gram learns two vectors per word, $\mathbf v_w$ as centre and $\mathbf u_c$ as context, so that the probability of a context word given a centre word is high:

<!-- formula: Skip-gram probability -->
$$P(c\mid w) = \frac{\exp(\mathbf u_c^\top\mathbf v_w)}{\sum_{c'\in V}\exp(\mathbf u_{c'}^\top\mathbf v_w)}$$

The softmax over the whole vocabulary is expensive. **Negative sampling** replaces it with a binary task: tell the real pair $(w, c)$ apart from $k$ random "negative" words $n_1..n_k$:

<!-- formula: Skip-gram with negative sampling loss -->
$$\mathcal L = -\log\sigma(\mathbf u_c^\top\mathbf v_w) - \sum_{j=1}^{k}\log\sigma(-\mathbf u_{n_j}^\top\mathbf v_w)$$

Negatives are drawn from the word frequency distribution raised to the power 0.75, which samples rare words a little more often. Frequent words like "the" are also randomly dropped from the training text (subsampling).

## Geometry of meaning

Trained embeddings show striking structure. Similar words have high [cosine similarity](#/lesson/linear-algebra), and some relations become directions:

<!-- formula: Analogy by vector arithmetic -->
$$\mathbf v_{\text{king}} - \mathbf v_{\text{man}} + \mathbf v_{\text{woman}} \approx \mathbf v_{\text{queen}}$$

Solve an analogy "$a$ is to $b$ as $c$ is to ?" by finding the word whose vector is most cosine-similar to $\mathbf v_b - \mathbf v_a + \mathbf v_c$, excluding $a$, $b$ and $c$ themselves. These regularities are approximate and also encode the biases of the training text (gender and ethnic stereotypes), which is a real concern when embeddings feed decisions.

## Embeddings everywhere

An embedding layer is just a learnable lookup table: a matrix $E$ of shape (vocabulary size, dimension), where token $i$'s vector is row $i$. Multiplying a one-hot vector by $E$ gives the same row, but the lookup is far cheaper. Every [Transformer](#/lesson/transformer) begins with one. The same idea embeds users and items in [recommender systems](#/lesson/recommender-systems), categories with thousands of values in tabular models, and whole sentences or documents for semantic search, where nearest-neighbour lookup over embeddings powers retrieval for [language models](#/lesson/language-models).

Static word vectors give "bank" one vector for both river and money. Contextual embeddings from Transformers give each occurrence its own vector depending on the sentence.

<!-- exercise: skipgram-pairs | Training pairs for skip-gram -->
Write `skipgram_pairs(tokens, window)` that returns the list of `(centre, context)` pairs: for each position `i`, every token at positions `i - window .. i + window` except `i` itself, staying inside the sequence. Keep the order: by centre position, then by context position.

~~~python starter
def skipgram_pairs(tokens, window):
    raise NotImplementedError

print(skipgram_pairs(["the", "cat", "sat", "down"], 1))
~~~

~~~python solution
def skipgram_pairs(tokens, window):
    pairs = []
    for i, centre in enumerate(tokens):
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if j != i:
                pairs.append((centre, tokens[j]))
    return pairs

print(skipgram_pairs(["the", "cat", "sat", "down"], 1))
~~~

~~~python tests
def test_window_one():
    """window 1 on four tokens"""
    assert skipgram_pairs(["the", "cat", "sat", "down"], 1) == [
        ("the", "cat"), ("cat", "the"), ("cat", "sat"), ("sat", "cat"), ("sat", "down"), ("down", "sat")]

def test_count():
    """window 2 on six tokens gives 2 + 3 + 4 + 4 + 3 + 2 = 18 pairs"""
    t = list("abcdef")
    assert len(skipgram_pairs(t, 2)) == 2 + 3 + 4 + 4 + 3 + 2

def test_no_self_pairs():
    """a word is not its own context position"""
    t = ["a", "b", "a"]
    assert ("a", "a") in skipgram_pairs(t, 2) and len(skipgram_pairs(t, 2)) == 6
~~~

<!-- /exercise -->

<!-- exercise: analogy | Solve analogies with cosine similarity -->
Write `analogy(E, vocab, a, b, c)` where `E` is an `(V, d)` embedding matrix and `vocab` a list of words in row order. Return the word whose vector has the highest cosine similarity to `E[b] - E[a] + E[c]`, excluding `a`, `b` and `c`.

~~~python starter
import numpy as np

def analogy(E, vocab, a, b, c):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def analogy(E, vocab, a, b, c):
    idx = {w: i for i, w in enumerate(vocab)}
    target = E[idx[b]] - E[idx[a]] + E[idx[c]]
    sims = (E @ target) / (np.linalg.norm(E, axis=1) * np.linalg.norm(target) + 1e-12)
    for w in (a, b, c):
        sims[idx[w]] = -np.inf
    return vocab[int(np.argmax(sims))]
~~~

~~~python tests
import numpy as np

# toy embeddings built from two hidden directions: "royalty" and "gender", plus noise words
rng = np.random.default_rng(0)
royal, female = rng.normal(size=8), rng.normal(size=8)
base = {w: rng.normal(size=8) * 0.1 for w in ["person", "fruit", "car", "city"]}
vocab = ["man", "woman", "king", "queen", "prince", "princess", "apple", "banana", "car", "city"]
E = np.array([
    base["person"], base["person"] + female, base["person"] + royal, base["person"] + royal + female,
    base["person"] + 0.6 * royal, base["person"] + 0.6 * royal + female,
    base["fruit"], base["fruit"] + 0.1, base["car"], base["city"]])

def test_king_queen():
    """man : woman :: king : queen"""
    assert analogy(E, vocab, "man", "woman", "king") == "queen"

def test_prince():
    """man : woman :: prince : princess"""
    assert analogy(E, vocab, "man", "woman", "prince") == "princess"

def test_excludes_inputs():
    """never returns one of the three query words"""
    assert analogy(E, vocab, "apple", "apple", "banana") not in {"apple", "banana"}
~~~

<!-- hint: Normalise: cosine similarity of every row with the target is `E @ t / (norms_of_rows * norm_of_t)`. Set the scores of the three query words to minus infinity before taking the argmax. -->
<!-- /exercise -->

<!-- quiz -->
1. What signal does word2vec learn from?
   - [ ] Human-labelled word similarities
   - [x] Which words appear near each other in raw text
   - [ ] Dictionary definitions
   > It is self-supervised: predicting context from a word creates the training signal for free.
2. Why does negative sampling make training faster?
   - [x] It replaces a softmax over the whole vocabulary with a few binary comparisons per pair
   - [ ] It uses fewer training sentences
   - [ ] It removes the need for gradients
   > Each update touches the true context word and $k$ sampled words instead of all 50 000.
3. Multiplying a one-hot vector for word 7 by an embedding matrix $E$ gives...
   - [ ] the sum of all embeddings
   - [x] row 7 of $E$
   - [ ] a scalar similarity
   > That is why an embedding layer is implemented as a fast table lookup.
4. A static embedding gives "bank" a single vector. What problem does that cause?
   - [x] It mixes the meanings "river bank" and "money bank"; contextual models give each occurrence its own vector
   - [ ] It cannot be compared with other words
   - [ ] It has no cosine similarity
   > One vector per word type cannot separate senses; Transformers produce context-dependent vectors.
<!-- /quiz -->
