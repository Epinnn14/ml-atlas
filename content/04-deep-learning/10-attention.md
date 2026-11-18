---
title: Attention
level: advanced
minutes: 50
summary: Let every position look at every other position and decide what is relevant. Queries, keys and values, scaled dot-product attention, masks, and multi-head attention, the core operation of Transformers.
---

## A soft lookup

A Python dictionary returns the value whose key equals the query exactly. **Attention** is a soft, differentiable version: compare a query with every key, turn the similarities into weights with a [softmax](#/lesson/softmax-regression), and return the weighted average of the values. Nothing is looked up exactly; everything contributes in proportion to how relevant it is.

For a sequence, each position produces three vectors from its embedding $\mathbf x_i$ with learned matrices:

<!-- formula: Queries, keys and values -->
$$\mathbf q_i = W_Q\mathbf x_i, \qquad \mathbf k_i = W_K\mathbf x_i, \qquad \mathbf v_i = W_V\mathbf x_i$$

The query says what this position is looking for, the key what it offers to others, the value what it passes on when chosen.

## Scaled dot-product attention

Stacking all positions as rows, $Q$ and $K$ have shape $(T, d_k)$ and $V$ has shape $(T, d_v)$:

<!-- formula: Scaled dot-product attention -->
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

$QK^\top$ is a $T\times T$ matrix of query–key scores. Each row goes through a softmax, giving the **attention weights** of one position over all positions, and multiplies $V$. The output at each position is a context-dependent mix of every position's value. In "the animal didn't cross the street because **it** was too tired", the row for *it* can put most weight on *animal*.

**Why divide by $\sqrt{d_k}$?** If query and key entries have unit variance, their dot product has variance $d_k$. Large scores push the softmax into a near one-hot regime where gradients vanish. Dividing by $\sqrt{d_k}$ keeps the scores at unit variance.

## Masks

- A **causal mask** stops position $i$ from attending to later positions $j > i$, needed when predicting the next token: set those scores to $-\infty$ before the softmax, so their weights become exactly zero.
- A **padding mask** hides the padding tokens added to make sequences in a batch the same length.

## Multi-head attention

One softmax produces one weighting per position, but a word may need to attend to its subject for one purpose and to a nearby adjective for another. **Multi-head attention** runs $h$ attentions in parallel, each on its own lower-dimensional projection ($d_k = d/h$), then concatenates the results and mixes them with an output matrix $W_O$:

<!-- formula: Multi-head attention -->
$$\operatorname{MHA}(X) = \operatorname{concat}(\text{head}_1, \dots, \text{head}_h)\,W_O, \quad \text{head}_j = \operatorname{Attention}(XW_Q^{j}, XW_K^{j}, XW_V^{j})$$

The cost is about the same as one full-width head. Different heads learn different patterns: previous-token heads, heads that track syntax, heads that copy.

## Self-attention versus cross-attention

When $Q$, $K$ and $V$ all come from the same sequence it is **self-attention**. In **cross-attention** the queries come from one sequence and the keys and values from another, for example a translation decoder attending to the encoded source sentence, or an image model attending to a text prompt.

## Cost

Attention compares every pair of positions: time and memory grow as $O(T^2)$ in the sequence length. That is the main obstacle for long documents, and the reason for memory-efficient kernels (FlashAttention), sliding-window and sparse attention, and caching keys and values during generation ([language models](#/lesson/language-models)).

<!-- exercise: sdpa | Scaled dot-product attention -->
Write `attention(Q, K, V, mask=None)` returning `(output, weights)`. `mask` is an optional boolean array of shape `(T_q, T_k)` that is `True` where attention is **allowed**; disallowed scores become `-inf` before a numerically stable softmax over the last axis. Also write `causal_mask(T)` returning the `(T, T)` boolean lower-triangular mask.

~~~python starter
import numpy as np

def causal_mask(T):
    raise NotImplementedError

def attention(Q, K, V, mask=None):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def causal_mask(T):
    return np.tril(np.ones((T, T), dtype=bool))

def attention(Q, K, V, mask=None):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    W = np.exp(scores)
    W = W / W.sum(axis=-1, keepdims=True)
    return W @ V, W
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
T, dk, dv = 5, 4, 3
Q, K, V = rng.normal(size=(T, dk)), rng.normal(size=(T, dk)), rng.normal(size=(T, dv))

def test_shapes_and_rows():
    """output is (T, dv) and every row of weights sums to one"""
    out, W = attention(Q, K, V)
    assert out.shape == (T, dv) and W.shape == (T, T) and approx(W.sum(1), np.ones(T))

def test_formula():
    """matches softmax(Q K^T / sqrt(dk)) V"""
    S = Q @ K.T / 2.0
    W = np.exp(S) / np.exp(S).sum(1, keepdims=True)
    out, _ = attention(Q, K, V)
    assert approx(out, W @ V)

def test_causal():
    """with a causal mask, position i ignores later positions"""
    m = causal_mask(T)
    assert m.dtype == bool and m[0, 1] == False and m[3, 2] == True
    out, W = attention(Q, K, V, m)
    assert approx(np.triu(W, 1), np.zeros((T, T)))
    V2 = V.copy(); V2[-1] += 100
    assert approx(attention(Q, K, V2, m)[0][:-1], out[:-1])

def test_first_position_copies_itself():
    """under a causal mask, the first position can only attend to itself"""
    out, W = attention(Q, K, V, causal_mask(T))
    assert approx(W[0, 0], 1.0) and approx(out[0], V[0])

def test_lookup():
    """a query matching one key strongly retrieves that key's value"""
    Kc = np.eye(4) * 10; Vc = np.arange(8.0).reshape(4, 2)
    out, _ = attention(np.array([[0, 0, 10.0, 0]]), Kc, Vc)
    assert approx(out, [[4.0, 5.0]], tol=1e-6)
~~~

<!-- hint: `np.where(mask, scores, -np.inf)` blocks the disallowed positions; subtracting the row maximum keeps the softmax stable. -->
<!-- /exercise -->

<!-- exercise: mha | Multi-head self-attention -->
Write `multi_head(X, Wq, Wk, Wv, Wo, n_heads, mask=None)`. Project `X` (shape `(T, d)`) with the `(d, d)` matrices `Wq, Wk, Wv`, split the last dimension into `n_heads` heads of size `d // n_heads` (head `j` uses columns `j*dh:(j+1)*dh`), run the provided `attention` on each head, concatenate the head outputs in order, and multiply by `Wo`.

~~~python starter
import numpy as np

def attention(Q, K, V, mask=None):
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    W = np.exp(S - S.max(-1, keepdims=True)); W /= W.sum(-1, keepdims=True)
    return W @ V

def multi_head(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def attention(Q, K, V, mask=None):
    S = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        S = np.where(mask, S, -np.inf)
    W = np.exp(S - S.max(-1, keepdims=True)); W /= W.sum(-1, keepdims=True)
    return W @ V

def multi_head(X, Wq, Wk, Wv, Wo, n_heads, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    dh = Q.shape[1] // n_heads
    heads = [attention(Q[:, j*dh:(j+1)*dh], K[:, j*dh:(j+1)*dh], V[:, j*dh:(j+1)*dh], mask)
             for j in range(n_heads)]
    return np.concatenate(heads, axis=1) @ Wo
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
T, d = 6, 8
X = rng.normal(size=(T, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4))

def test_shape():
    """output has the shape of the input"""
    assert multi_head(X, Wq, Wk, Wv, Wo, 2).shape == (T, d)

def test_one_head_is_plain_attention():
    """with one head it is ordinary attention followed by Wo"""
    assert approx(multi_head(X, Wq, Wk, Wv, Wo, 1), attention(X @ Wq, X @ Wk, X @ Wv) @ Wo)

def test_heads_are_separate():
    """head 2 only sees its own slice of Q, K, V"""
    out = multi_head(X, Wq, Wk, Wv, np.eye(d), 2)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    assert approx(out[:, 4:], attention(Q[:, 4:], K[:, 4:], V[:, 4:]))

def test_causal_mask_passed():
    """the mask reaches every head"""
    m = np.tril(np.ones((T, T), bool))
    X2 = X.copy(); X2[-1] += 5
    a, b = multi_head(X, Wq, Wk, Wv, Wo, 4, m), multi_head(X2, Wq, Wk, Wv, Wo, 4, m)
    assert approx(a[:-1], b[:-1])
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why are attention scores divided by $\sqrt{d_k}$?
   - [ ] To make the output smaller
   - [x] Dot products of $d_k$-dimensional vectors grow in variance with $d_k$; scaling keeps the softmax from saturating
   - [ ] To normalise the values $V$
   > Saturated softmaxes are nearly one-hot, and their gradients nearly vanish.
2. What does a causal mask do?
   - [x] Prevents each position from attending to later positions, so a model cannot peek at the tokens it must predict
   - [ ] Removes padding tokens
   - [ ] Makes attention linear in sequence length
   > Masked scores become $-\infty$ and receive zero weight after the softmax.
3. A model doubles its context length from 4 000 to 8 000 tokens. Roughly how does the attention score matrix grow?
   - [ ] Twice as large
   - [x] Four times as large
   - [ ] It stays the same size
   > The score matrix is $T\times T$, so its size grows with the square of $T$.
4. In a translation model's decoder, where do cross-attention's keys and values come from?
   - [ ] The decoder's own previous tokens
   - [x] The encoder's representation of the source sentence
   - [ ] A fixed lookup table
   > Queries come from the target side; keys and values from the source, letting each output word look at the relevant input words.
<!-- /quiz -->
