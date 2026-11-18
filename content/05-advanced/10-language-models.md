---
title: Large language models
level: senior
minutes: 60
summary: How LLMs are trained, measured, sampled and adapted. Next-token prediction and perplexity, temperature, top-k and nucleus sampling, the KV cache, scaling laws, fine-tuning with LoRA, and retrieval-augmented generation.
---

## Next-token prediction

A language model assigns a probability to text by predicting one token at a time, each conditioned on everything before it ([tokenisation](#/lesson/tokenization)):

<!-- formula: Autoregressive factorisation -->
$$p(x_1, \dots, x_T) = \prod_{t=1}^{T}p(x_t\mid x_1, \dots, x_{t-1})$$

A decoder-only [Transformer](#/lesson/transformer) computes all these conditionals in parallel during training, and is trained with cross-entropy on huge text corpora. Nothing else is needed for the model to pick up grammar, facts, reasoning patterns and code: predicting the next token well requires modelling whatever produced the text.

## Perplexity

The standard intrinsic measure is **perplexity**, the exponential of the average negative log-likelihood per token:

<!-- formula: Perplexity -->
$$\text{PPL} = \exp\Big(-\frac1T\sum_{t=1}^{T}\log p(x_t\mid x_{<t})\Big)$$

It is the effective number of tokens the model is choosing between at each step: a perplexity of 10 means as uncertain as a uniform choice among 10. Lower is better, but perplexities are only comparable with the same tokeniser and test text.

## Pretraining, scaling and post-training

**Scaling laws**: test loss falls as a smooth power law in the number of parameters $N$, the number of training tokens $D$ and the compute $C \approx 6ND$ floating-point operations. For a fixed compute budget there is an optimal balance; the Chinchilla result suggested roughly 20 tokens per parameter, though models meant to be served cheaply are trained on far more data than that.

A pretrained model continues text; it does not yet follow instructions. **Post-training** turns it into an assistant: supervised fine-tuning on example conversations, then preference optimisation (RLHF with [PPO](#/lesson/policy-gradients), or DPO) on human or AI judgements of which answers are better.

## Sampling

At generation time the model outputs logits for the next token. How you choose from them changes the text a lot:

- **Greedy**: always the most likely token. Deterministic, often repetitive.
- **Temperature**: divide logits by $T$ before the softmax. $T < 1$ sharpens, $T > 1$ flattens ([softmax](#/lesson/softmax-regression)).
- **Top-k**: keep only the $k$ most likely tokens, renormalise, sample.
- **Top-p (nucleus)**: keep the smallest set of most likely tokens whose total probability reaches $p$ (say 0.9). Adapts to the situation: few candidates when the model is confident, many when it is not.

<!-- formula: Nucleus (top-p) set -->
$$V_p = \text{smallest set of top-ranked tokens with } \sum_{x\in V_p}p(x) \ge p$$

## Inference: the KV cache

Generating token $t$ needs attention over all previous tokens. Recomputing their keys and values at every step would be quadratic waste, so they are stored in a **KV cache** and each new step only computes the new token's query, key and value. The cache grows linearly with context length and dominates memory for long contexts, which motivates grouped-query attention (several query heads share keys and values) and cache quantisation.

## Adapting models: LoRA

Fine-tuning all weights of a large model is expensive in memory. **Low-rank adaptation** freezes the pretrained matrix $W_0$ ($d\times k$) and learns a low-rank update:

<!-- formula: LoRA update -->
$$W = W_0 + \frac{\alpha}{r}BA, \qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r \ll \min(d, k)$$

Only $r(d + k)$ parameters are trained instead of $dk$: for $d = k = 4096$ and $r = 8$, 65 536 instead of 16.8 million, a 256× reduction per matrix. $B$ starts at zero so training begins from the original model, and the update can be merged into $W_0$ afterwards at no inference cost. Combined with a 4-bit quantised base model (QLoRA), it fine-tunes large models on a single GPU.

## Retrieval and tools

Models know only what was in their training data, up to a cut-off, and can state falsehoods fluently ("hallucination"). **Retrieval-augmented generation** (RAG) embeds documents, finds the ones nearest to the question by [cosine similarity](#/lesson/word-embeddings), and puts them into the prompt so the answer can quote current, private, checkable sources. Tool use extends this: the model emits structured calls to search engines, calculators or code interpreters and reads back the results.

<!-- exercise: perplexity | Perplexity from token probabilities -->
Write `perplexity(token_probs)` where `token_probs` is the probability the model gave to each actual token of a text, and `perplexity_from_logits(logits, targets)` computing it from a `(T, V)` logit array and the integer targets, via a stable log-softmax.

~~~python starter
import numpy as np

def perplexity(token_probs):
    raise NotImplementedError

def perplexity_from_logits(logits, targets):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def perplexity(token_probs):
    return float(np.exp(-np.mean(np.log(np.asarray(token_probs, float)))))

def perplexity_from_logits(logits, targets):
    Z = np.asarray(logits, float)
    m = Z.max(axis=1, keepdims=True)
    logp = Z - m - np.log(np.exp(Z - m).sum(axis=1, keepdims=True))
    return float(np.exp(-np.mean(logp[np.arange(len(targets)), targets])))
~~~

~~~python tests
import numpy as np

def test_uniform():
    """a uniform guess over 10 tokens has perplexity 10"""
    assert approx(perplexity([0.1] * 20), 10.0)

def test_perfect():
    """probability 1 for every token gives perplexity 1"""
    assert approx(perplexity([1.0, 1.0]), 1.0)

def test_geometric_mean():
    """perplexity is 1 / geometric mean of the probabilities"""
    p = np.array([0.5, 0.25, 0.125])
    assert approx(perplexity(p), 1 / np.prod(p) ** (1 / 3))

def test_from_logits():
    """matches the probability version and handles huge logits"""
    Z = np.random.default_rng(0).normal(size=(6, 5)); t = np.array([0, 3, 1, 4, 2, 2])
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    assert approx(perplexity_from_logits(Z, t), perplexity(P[np.arange(6), t]))
    assert np.isfinite(perplexity_from_logits(Z * 500, t))
~~~

<!-- /exercise -->

<!-- exercise: nucleus | Temperature and nucleus sampling -->
Write `nucleus_probs(logits, temperature=1.0, top_p=0.9)` returning the probability vector you would sample from: apply temperature, softmax, sort tokens by probability (descending, ties by lower index first, use `np.argsort(-p, kind="stable")`), keep the smallest prefix whose cumulative probability reaches `top_p`, zero the rest and renormalise. Then `lora_params(d, k, r)` returns the number of trainable LoRA parameters for one `d×k` matrix.

~~~python starter
import numpy as np

def nucleus_probs(logits, temperature=1.0, top_p=0.9):
    raise NotImplementedError

def lora_params(d, k, r):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def nucleus_probs(logits, temperature=1.0, top_p=0.9):
    z = np.asarray(logits, float) / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order])
    keep = order[: int(np.searchsorted(cum, top_p - 1e-12)) + 1]
    out = np.zeros_like(p)
    out[keep] = p[keep]
    return out / out.sum()

def lora_params(d, k, r):
    return r * (d + k)
~~~

~~~python tests
import numpy as np

logits = np.log(np.array([0.5, 0.2, 0.15, 0.1, 0.05]))

def test_keeps_smallest_prefix():
    """top_p = 0.8 keeps the tokens with 0.5, 0.2 and 0.15 (cumulative 0.85)"""
    p = nucleus_probs(logits, 1.0, 0.8)
    assert approx(p, np.array([0.5, 0.2, 0.15, 0, 0]) / 0.85)

def test_exact_boundary():
    """top_p = 0.7 stops exactly at 0.5 + 0.2"""
    p = nucleus_probs(logits, 1.0, 0.7)
    assert approx(p, [0.5 / 0.7, 0.2 / 0.7, 0, 0, 0])

def test_top_p_one_is_full_softmax():
    """top_p = 1 keeps everything"""
    assert approx(nucleus_probs(logits, 1.0, 1.0), [0.5, 0.2, 0.15, 0.1, 0.05])

def test_low_temperature_is_greedy():
    """a very low temperature leaves only the top token"""
    assert approx(nucleus_probs(logits, 0.01, 0.9), [1, 0, 0, 0, 0])

def test_lora():
    """rank 8 on a 4096 x 4096 matrix trains 65 536 parameters"""
    assert lora_params(4096, 4096, 8) == 65536
~~~

<!-- hint: `np.searchsorted(cum, top_p)` gives the index of the first cumulative value that reaches `top_p`; keep the tokens up to and including it. -->
<!-- /exercise -->

<!-- quiz -->
1. Model A has perplexity 12 and model B 18 on the same test text with the same tokeniser. What can you say?
   - [x] A assigns higher probability to the actual text, about as if choosing among 12 options per token instead of 18
   - [ ] A is 50% more accurate at answering questions
   - [ ] B has more parameters
   > Perplexity measures predictive fit to that text; it does not directly measure usefulness.
2. Why does nucleus (top-p) sampling adapt better than top-k?
   - [ ] It is deterministic
   - [x] The number of candidate tokens shrinks when the model is confident and grows when it is unsure
   - [ ] It ignores the model's probabilities
   > A fixed $k$ either cuts reasonable options or admits junk; a probability mass threshold adjusts automatically.
3. What does the KV cache save during generation?
   - [ ] The model's weights
   - [x] Recomputing keys and values of all earlier tokens at every new step
   - [ ] The tokeniser's merges
   > Each step only computes the new token's projections and attends to the cached ones.
4. Why is LoRA's $B$ matrix initialised to zero?
   - [x] So $BA = 0$ at the start and fine-tuning begins exactly from the pretrained model
   - [ ] To make the rank zero forever
   - [ ] To save memory
   > Training then moves away from the original weights only as far as the new data requires.
<!-- /quiz -->
