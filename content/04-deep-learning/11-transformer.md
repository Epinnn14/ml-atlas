---
title: The Transformer
level: advanced
minutes: 60
summary: The architecture behind modern language, vision and audio models. Token embeddings plus positions, a stack of blocks each made of attention and a feed-forward network wrapped in residuals and layer norm, and the decoder-only variant used by large language models.
---

## The big picture

A Transformer turns a sequence of tokens into a sequence of vectors, refining every position's vector layer by layer using information from the other positions. The original 2017 model had an encoder and a decoder for translation. Today the most common form is **decoder-only**: a stack of identical blocks with causal [attention](#/lesson/attention), trained to predict the next token ([language models](#/lesson/language-models)). Encoder-only models (BERT) are used for classification and embeddings.

## Input: tokens and positions

Tokens ([tokenisation](#/lesson/tokenization)) are mapped to vectors by an [embedding](#/lesson/word-embeddings) table. Attention itself is blind to order: shuffle the positions and each output is shuffled the same way. So position information is added. The original paper used fixed sinusoids of different frequencies:

<!-- formula: Sinusoidal positional encoding -->
$$PE_{(p, 2i)} = \sin\!\Big(\frac{p}{10000^{2i/d}}\Big), \qquad PE_{(p, 2i+1)} = \cos\!\Big(\frac{p}{10000^{2i/d}}\Big)$$

Each pair of dimensions is a clock running at its own speed, from fast (every few tokens) to very slow, so every position gets a unique pattern and relative offsets correspond to fixed rotations. Modern models usually use **rotary position embeddings** (RoPE), which rotate queries and keys by position so attention scores depend on relative distance, or learned position vectors.

## The block

Each block applies two sub-layers, each wrapped in a [residual connection](#/lesson/initialization-normalization) with [layer normalisation](#/lesson/initialization-normalization). In the now-standard **pre-norm** arrangement:

<!-- formula: Pre-norm Transformer block -->
$$\begin{aligned}
\mathbf h &= \mathbf x + \operatorname{MHA}\big(\operatorname{LN}_1(\mathbf x)\big)\\
\mathbf y &= \mathbf h + \operatorname{FFN}\big(\operatorname{LN}_2(\mathbf h)\big)
\end{aligned}$$

- **Multi-head attention** mixes information **across positions**.
- The **feed-forward network** processes **each position separately** with the same two-layer MLP, usually 4 times wider than the model:

<!-- formula: Position-wise feed-forward network -->
$$\operatorname{FFN}(\mathbf x) = W_2\,\operatorname{GELU}(W_1\mathbf x + \mathbf b_1) + \mathbf b_2$$

A useful mental model: the residual stream $\mathbf x$ is a shared working memory; attention moves information between positions, the MLP transforms it in place, and each block reads from and writes back to the stream. Most of the parameters live in the MLPs. Recent models use a gated variant (SwiGLU) and RMSNorm, but the structure is the same.

## Output

After the last block, a final layer norm and a linear layer (often sharing weights with the input embedding) produce logits over the vocabulary at every position; a [softmax](#/lesson/softmax-regression) gives the next-token distribution. Training uses cross-entropy at **every position at once**: with the causal mask, position $t$ predicts token $t+1$ from tokens $1..t$, so one sequence of length $T$ yields $T$ training examples in a single parallel pass.

## Size

Per block, attention has $4d^2$ weights ($W_Q, W_K, W_V, W_O$) and the MLP $8d^2$ (two $d\times 4d$ matrices), about $12d^2$ in total. A model with $L$ blocks has roughly

<!-- formula: Transformer parameter count -->
$$N \approx 12\,L\,d^2 + V d$$

parameters, where the second term is the embedding table for a vocabulary of size $V$. GPT-2 small ($L = 12$, $d = 768$, $V \approx 50\,000$) gives about 124 million.

<!-- exercise: posenc | Sinusoidal positional encoding -->
Write `positional_encoding(T, d)` returning the `(T, d)` matrix from the formula (for even `d`): even columns get sines, odd columns cosines, with frequency index `i` shared by columns `2i` and `2i+1`.

~~~python starter
import numpy as np

def positional_encoding(T, d):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def positional_encoding(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / 10000 ** (2 * i / d)
    PE = np.zeros((T, d))
    PE[:, 0::2] = np.sin(angle)
    PE[:, 1::2] = np.cos(angle)
    return PE
~~~

~~~python tests
import numpy as np

def test_shape_and_first_row():
    """position 0 is [0, 1, 0, 1, ...]"""
    PE = positional_encoding(10, 6)
    assert PE.shape == (10, 6) and approx(PE[0], [0, 1, 0, 1, 0, 1])

def test_values():
    """matches the formula for a few entries"""
    PE = positional_encoding(50, 8)
    assert approx(PE[7, 0], np.sin(7)) and approx(PE[7, 1], np.cos(7))
    assert approx(PE[7, 4], np.sin(7 / 10000 ** (4 / 8))) and approx(PE[7, 5], np.cos(7 / 10000 ** (4 / 8)))

def test_bounded_and_unique():
    """values lie in [-1, 1] and every position gets a different vector"""
    PE = positional_encoding(200, 16)
    assert np.all(np.abs(PE) <= 1) and len({tuple(np.round(r, 6)) for r in PE}) == 200

def test_relative_shift_is_rotation():
    """each sin/cos pair has the same length at every position"""
    PE = positional_encoding(30, 8)
    assert approx(PE[:, 0] ** 2 + PE[:, 1] ** 2, np.ones(30))
~~~

<!-- /exercise -->

<!-- exercise: block | A decoder block -->
Write `decoder_block(X, p)` implementing the pre-norm block with **causal** multi-head attention and a GELU feed-forward network. `p` is a dict with `ln1_g, ln1_b, ln2_g, ln2_b` (shape `(d,)`), `Wq, Wk, Wv, Wo` (shape `(d, d)`), `W1 (d, 4d)`, `b1 (4d,)`, `W2 (4d, d)`, `b2 (d,)`, and `n_heads`. Helper functions `layer_norm`, `gelu` and `mha` are provided; you only assemble the block.

~~~python starter
import numpy as np

def layer_norm(X, g, b, eps=1e-5):
    mu, var = X.mean(-1, keepdims=True), X.var(-1, keepdims=True)
    return g * (X - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    mask = np.tril(np.ones((T, T), bool))
    out = []
    for j in range(n_heads):
        s = slice(j * dh, (j + 1) * dh)
        S = Q[:, s] @ K[:, s].T / np.sqrt(dh)
        if causal:
            S = np.where(mask, S, -np.inf)
        W = np.exp(S - S.max(-1, keepdims=True)); W /= W.sum(-1, keepdims=True)
        out.append(W @ V[:, s])
    return np.concatenate(out, axis=1) @ Wo

def decoder_block(X, p):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def layer_norm(X, g, b, eps=1e-5):
    mu, var = X.mean(-1, keepdims=True), X.var(-1, keepdims=True)
    return g * (X - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def mha(X, Wq, Wk, Wv, Wo, n_heads, causal=True):
    T, d = X.shape
    dh = d // n_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    mask = np.tril(np.ones((T, T), bool))
    out = []
    for j in range(n_heads):
        s = slice(j * dh, (j + 1) * dh)
        S = Q[:, s] @ K[:, s].T / np.sqrt(dh)
        if causal:
            S = np.where(mask, S, -np.inf)
        W = np.exp(S - S.max(-1, keepdims=True)); W /= W.sum(-1, keepdims=True)
        out.append(W @ V[:, s])
    return np.concatenate(out, axis=1) @ Wo

def decoder_block(X, p):
    h = X + mha(layer_norm(X, p["ln1_g"], p["ln1_b"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], p["n_heads"])
    ff = gelu(layer_norm(h, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    return h + ff
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
T, d = 7, 8
def params(scale=1.0):
    r = lambda *s: rng.normal(size=s) * scale / np.sqrt(s[0])
    return {"ln1_g": np.ones(d), "ln1_b": np.zeros(d), "ln2_g": np.ones(d), "ln2_b": np.zeros(d),
            "Wq": r(d, d), "Wk": r(d, d), "Wv": r(d, d), "Wo": r(d, d),
            "W1": r(d, 4 * d), "b1": np.zeros(4 * d), "W2": r(4 * d, d), "b2": np.zeros(d), "n_heads": 2}
p = params()
X = rng.normal(size=(T, d))

def test_shape():
    """keeps the shape (T, d)"""
    assert decoder_block(X, p).shape == (T, d)

def test_zero_weights_is_identity():
    """with zero output projections both residual branches vanish"""
    q = dict(p, Wo=np.zeros((d, d)), W2=np.zeros((4 * d, d)))
    assert approx(decoder_block(X, q), X)

def test_causal():
    """changing the last token does not change earlier outputs"""
    X2 = X.copy(); X2[-1] += 3
    a, b = decoder_block(X, p), decoder_block(X2, p)
    assert approx(a[:-1], b[:-1]) and not approx(a[-1], b[-1])

def test_matches_reference():
    """matches the pre-norm equations"""
    ln = lambda Z, g, b: g * (Z - Z.mean(-1, keepdims=True)) / np.sqrt(Z.var(-1, keepdims=True) + 1e-5) + b
    h = X + mha(ln(X, p["ln1_g"], p["ln1_b"]), p["Wq"], p["Wk"], p["Wv"], p["Wo"], 2)
    y = h + gelu(ln(h, p["ln2_g"], p["ln2_b"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]
    assert approx(decoder_block(X, p), y)

def test_ffn_is_per_position():
    """with attention switched off, each position is processed independently"""
    q = dict(p, Wo=np.zeros((d, d)))
    full = decoder_block(X, q)
    single = decoder_block(X[3:4], q)
    assert approx(full[3:4], single)
~~~

<!-- hint: First sub-layer: `h = X + mha(layer_norm(X, ...), ...)`. Second: `h + FFN(layer_norm(h, ...))`. -->
<!-- /exercise -->

<!-- quiz -->
1. Why does a Transformer need positional information at all?
   - [x] Attention is permutation-equivariant: without positions, shuffling the tokens just shuffles the outputs
   - [ ] To make the embedding table smaller
   - [ ] Because layer norm removes the order
   > Order matters for language ("dog bites man"), so it must be injected explicitly.
2. In a Transformer block, which part mixes information between different positions?
   - [x] Multi-head attention
   - [ ] The feed-forward network
   - [ ] Layer normalisation
   > The FFN acts on each position independently; attention is the only cross-position operation.
3. Roughly how many parameters does one block with $d = 1024$ have?
   - [ ] About 1 million
   - [x] About 12.6 million ($12 d^2$)
   - [ ] About 1 billion
   > $4d^2$ for attention plus $8d^2$ for a 4× wide MLP gives $12 \times 1024^2 \approx 12.6$M.
4. How does a decoder-only model get $T$ training examples from one sequence of $T$ tokens in one forward pass?
   - [ ] It runs the model $T$ times
   - [x] The causal mask lets every position predict its next token using only earlier tokens, all computed in parallel
   - [ ] It duplicates the sequence $T$ times
   > Masking makes each position's prediction legitimate, so all $T$ losses are computed at once.
<!-- /quiz -->
