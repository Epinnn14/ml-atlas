---
title: Neural networks and backpropagation
level: intermediate
minutes: 60
summary: Stack linear layers with non-linear activations, compute the loss with a forward pass, and get every gradient with one backward pass. The equations of a two-layer network written out and implemented.
---

## Layers

A **multilayer perceptron** (MLP) alternates linear maps and element-wise non-linearities. For a batch $X$ of shape $(n, d)$, a two-layer network for $K$ classes is:

<!-- formula: Two-layer network forward pass -->
$$H = \phi(XW_1 + \mathbf b_1), \qquad Z = HW_2 + \mathbf b_2, \qquad P = \operatorname{softmax}(Z)$$

with $W_1$ of shape $(d, h)$, $W_2$ of shape $(h, K)$. $H$ is the **hidden layer** of $h$ units, $Z$ the logits, and the loss is the [cross-entropy](#/lesson/softmax-regression) of $P$. The hidden layer learns its own features; the output layer is just softmax regression on them.

## Why non-linear activations

Without $\phi$, two linear layers collapse into one: $X W_1W_2$ is still linear. The activation is what lets networks bend decision boundaries. Common choices:

| Activation | Formula | Derivative | Notes |
|---|---|---|---|
| sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ | saturates, gradients vanish |
| tanh | $\tanh z$ | $1 - \tanh^2 z$ | zero-centred, still saturates |
| ReLU | $\max(0, z)$ | $1$ if $z>0$ else $0$ | the default; cheap; units can "die" |
| GELU | $z\,\Phi(z)$ | smooth | used in Transformers |

With enough hidden units, a two-layer network can approximate any continuous function on a bounded region (the **universal approximation theorem**). Depth makes many functions far cheaper to represent than width does, which is why practice favours deep networks.

## Backpropagation

Training needs the gradient of the loss with respect to every weight. Backpropagation applies the [chain rule](#/lesson/calculus-gradients) from the loss backwards, reusing intermediate results. For our network with mean cross-entropy over $n$ examples and one-hot labels $Y$:

<!-- formula: Backpropagation through a two-layer network -->
$$\begin{aligned}
\delta_2 &= \tfrac1n(P - Y) &&\text{(gradient w.r.t. the logits)}\\
\nabla W_2 &= H^\top\delta_2, \quad \nabla\mathbf b_2 = \textstyle\sum_i \delta_{2,i}\\
\delta_1 &= (\delta_2 W_2^\top)\odot\phi'(A_1) &&\text{with } A_1 = XW_1 + \mathbf b_1\\
\nabla W_1 &= X^\top\delta_1, \quad \nabla\mathbf b_1 = \textstyle\sum_i \delta_{1,i}
\end{aligned}$$

Read it as a pattern that repeats for every layer: the gradient for a weight matrix is (layer input)ᵀ × (gradient at the layer output), and the gradient passes to the layer below by multiplying with the transposed weights and the activation's derivative. The backward pass costs about twice the forward pass, regardless of the number of parameters. Always check a new implementation against [numerical gradients](#/lesson/calculus-gradients); [autograd](#/lesson/autograd) automates the whole thing.

## Training loop

1. Initialise weights with small random values ([initialisation](#/lesson/initialization-normalization)); never all zeros, or every hidden unit stays identical.
2. For each mini-batch: forward pass, loss, backward pass, [optimiser](#/lesson/optimizers) step.
3. Watch training and validation loss; stop or regularise when they diverge ([regularisation](#/lesson/regularization-dl)).

<!-- exercise: mlp-forward | Forward pass and loss -->
Write `forward(X, params)` for the two-layer ReLU network above, returning `(P, cache)` where `P` are the softmax probabilities and `cache` is a dict holding at least `"A1"` (pre-activation) and `"H"` (hidden output). Write `loss(P, y)` returning the mean cross-entropy for integer labels `y`. `params` has keys `W1, b1, W2, b2`.

~~~python starter
import numpy as np

def forward(X, params):
    raise NotImplementedError

def loss(P, y):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def forward(X, params):
    A1 = X @ params["W1"] + params["b1"]
    H = np.maximum(0, A1)
    Z = H @ params["W2"] + params["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return P, {"X": X, "A1": A1, "H": H}

def loss(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
params = {"W1": rng.normal(size=(3, 5)), "b1": rng.normal(size=5),
          "W2": rng.normal(size=(5, 4)), "b2": rng.normal(size=4)}
X = rng.normal(size=(6, 3)); y = rng.integers(0, 4, 6)

def test_shapes():
    """P is (n, K) and the cache has A1 and H of shape (n, h)"""
    P, cache = forward(X, params)
    assert P.shape == (6, 4) and cache["A1"].shape == (6, 5) and cache["H"].shape == (6, 5)

def test_probabilities():
    """rows of P are distributions"""
    P, _ = forward(X, params)
    assert approx(P.sum(1), np.ones(6)) and np.all(P >= 0)

def test_relu():
    """H is ReLU of A1"""
    _, c = forward(X, params)
    assert approx(c["H"], np.maximum(0, c["A1"]))

def test_matches_reference():
    """matches a direct computation of the network"""
    H = np.maximum(0, X @ params["W1"] + params["b1"])
    Z = H @ params["W2"] + params["b2"]
    Pref = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    P, _ = forward(X, params)
    assert approx(P, Pref) and approx(loss(P, y), -np.mean(np.log(Pref[np.arange(6), y])))
~~~

<!-- /exercise -->

<!-- exercise: backprop | Backpropagation by hand -->
Write `backward(P, y, cache, params)` returning a dict of gradients `dW1, db1, dW2, db2` of the mean cross-entropy, using the equations above (ReLU derivative: 1 where `A1 > 0`). A working `forward` and `loss` are provided; the tests compare your gradients with numerical ones.

~~~python starter
import numpy as np

def forward(X, params):
    A1 = X @ params["W1"] + params["b1"]
    H = np.maximum(0, A1)
    Z = H @ params["W2"] + params["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return P, {"X": X, "A1": A1, "H": H}

def loss(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))

def backward(P, y, cache, params):
    n = len(y)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def forward(X, params):
    A1 = X @ params["W1"] + params["b1"]
    H = np.maximum(0, A1)
    Z = H @ params["W2"] + params["b2"]
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    return P, {"X": X, "A1": A1, "H": H}

def loss(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))

def backward(P, y, cache, params):
    n = len(y)
    d2 = P.copy()
    d2[np.arange(n), y] -= 1
    d2 /= n
    dW2 = cache["H"].T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ params["W2"].T) * (cache["A1"] > 0)
    dW1 = cache["X"].T @ d1
    db1 = d1.sum(axis=0)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
params = {"W1": rng.normal(size=(3, 6)), "b1": rng.normal(size=6) * 0.1,
          "W2": rng.normal(size=(6, 3)), "b2": rng.normal(size=3) * 0.1}
X = rng.normal(size=(8, 3)); y = rng.integers(0, 3, 8)

def numgrad(name, h=1e-6):
    g = np.zeros_like(params[name])
    for idx in np.ndindex(g.shape):
        old = params[name][idx]
        params[name][idx] = old + h; up = loss(forward(X, params)[0], y)
        params[name][idx] = old - h; down = loss(forward(X, params)[0], y)
        params[name][idx] = old
        g[idx] = (up - down) / (2 * h)
    return g

def grads():
    P, cache = forward(X, params)
    return backward(P, y, cache, params)

def test_output_layer():
    """dW2 and db2 match numerical gradients"""
    g = grads()
    assert approx(g["dW2"], numgrad("W2"), tol=1e-5) and approx(g["db2"], numgrad("b2"), tol=1e-5)

def test_hidden_layer():
    """dW1 and db1 match numerical gradients"""
    g = grads()
    assert approx(g["dW1"], numgrad("W1"), tol=1e-5) and approx(g["db1"], numgrad("b1"), tol=1e-5)

def test_training_works():
    """gradient steps lower the loss on a small spiral-like problem"""
    r = np.random.default_rng(2)
    p = {"W1": r.normal(size=(3, 6)) * 0.5, "b1": np.zeros(6), "W2": r.normal(size=(6, 3)) * 0.5, "b2": np.zeros(3)}
    start = loss(forward(X, p)[0], y)
    for _ in range(300):
        P, c = forward(X, p)
        g = backward(P, y, c, p)
        for k in p:
            p[k] -= 0.5 * g["d" + k]
    assert loss(forward(X, p)[0], y) < 0.5 * start
~~~

<!-- hint: Start from `d2 = P` with 1 subtracted at the true class of each row, divided by `n`. -->
<!-- hint: The gradient flows into the hidden layer as `d2 @ W2.T`, then is zeroed where the ReLU was inactive: multiply by `(A1 > 0)`. -->
<!-- /exercise -->

<!-- quiz -->
1. What happens if you remove every activation function from a 10-layer network?
   - [ ] It becomes more powerful
   - [x] It collapses to a single linear map, no more expressive than linear regression
   - [ ] It cannot be trained
   > A product of matrices is a matrix, so stacked linear layers represent only linear functions.
2. Why must the weights not all start at zero?
   - [ ] Zero weights make the loss infinite
   - [x] Every hidden unit would compute the same thing and receive the same gradient, so they would stay identical forever
   - [ ] Zero is not a valid floating-point value
   > Random initialisation breaks the symmetry between units.
3. In backpropagation, the gradient for a layer's weight matrix is...
   - [x] the layer's input, transposed, times the gradient at the layer's output
   - [ ] the layer's output times its input
   - [ ] the loss divided by the weights
   > $\nabla W = (\text{input})^\top\delta$, the same pattern for every linear layer.
4. Roughly how does the cost of the backward pass compare with the forward pass?
   - [ ] It grows with the square of the number of parameters
   - [x] It is a small constant multiple, about two times the forward pass
   - [ ] It is free
   > Reverse-mode differentiation computes all gradients in one sweep, which is why training huge networks is feasible.
<!-- /quiz -->
