---
title: Gradient descent
level: beginner
minutes: 35
summary: The optimiser behind almost all of machine learning, why the learning rate makes or breaks it, and how stochastic mini-batches make it scale to millions of examples.
---

## Walking downhill

To minimise a loss $L(\mathbf w)$ we start somewhere and repeatedly take a small step against the gradient:

<!-- formula: Gradient descent update -->
$$\mathbf w_{t+1} = \mathbf w_t - \eta\,\nabla L(\mathbf w_t)$$

$\eta > 0$ is the **learning rate** (step size). The gradient points uphill, so subtracting it goes down. Near a minimum the gradient shrinks, so the steps shrink by themselves.

## The learning rate

Take the simplest loss, $L(w) = \tfrac{a}{2}w^2$ with gradient $aw$. One step gives $w_{t+1} = (1 - \eta a)\,w_t$. So:

- $0 < \eta < 1/a$: $w$ shrinks smoothly towards 0.
- $1/a < \eta < 2/a$: $w$ overshoots and flips sign each step, but still converges.
- $\eta > 2/a$: $|1 - \eta a| > 1$ and $w$ **diverges**, growing every step.

For a loss with many directions, the steepest direction (largest curvature) sets the limit, while the flattest direction sets how many steps you need. When curvatures differ a lot (an **ill-conditioned** problem, for example features on very different scales) gradient descent zig-zags across the steep direction and crawls along the flat one. This is one reason to [standardise features](#/lesson/data-preprocessing), and why [momentum and Adam](#/lesson/optimizers) exist.

A practical recipe: try learning rates on a log scale (1, 0.1, 0.01, ...), plot the loss against the step, and keep the largest one that still decreases smoothly.

## Convexity

A function is **convex** if the line segment between any two points of its graph lies above the graph. For convex losses every local minimum is a global minimum, so gradient descent with a small enough step finds the best possible parameters. Linear and logistic regression have convex losses; neural networks do not, which is why their training depends on initialisation and luck in ways linear models do not.

## Stochastic and mini-batch gradient descent

The loss is an average over $n$ examples, so its gradient is an average of $n$ per-example gradients. With millions of examples one step costs a full pass over the data. **Stochastic gradient descent** (SGD) estimates the gradient from a small random **mini-batch** $B$ instead:

<!-- formula: Mini-batch gradient estimate -->
$$\nabla L(\mathbf w) \approx \frac{1}{|B|}\sum_{i\in B}\nabla \ell_i(\mathbf w)$$

The estimate is noisy but unbiased, and a step costs $|B|$ examples instead of $n$. In practice:

- Shuffle the data each **epoch** (one pass over all examples) and walk through it in batches of 32 to 512.
- The noise means SGD keeps bouncing near the minimum; decaying the learning rate over time lets it settle.
- The noise also helps neural networks escape poor regions, so it is a feature as much as a cost.

<!-- exercise: gd | Gradient descent on any function -->
Write `gradient_descent(grad, w0, lr, steps)` that applies the update rule `steps` times and returns the final `w` and the list of all iterates (including `w0`), as `(w, history)`.

~~~python starter
import numpy as np

def gradient_descent(grad, w0, lr, steps):
    w = np.array(w0, dtype=float)
    history = [w.copy()]
    raise NotImplementedError

# minimise (w0 - 3)^2 + 10 (w1 + 1)^2
grad = lambda w: np.array([2 * (w[0] - 3), 20 * (w[1] + 1)])
w, hist = gradient_descent(grad, [0.0, 0.0], lr=0.04, steps=200)
print(w)   # close to [3, -1]
~~~

~~~python solution
import numpy as np

def gradient_descent(grad, w0, lr, steps):
    w = np.array(w0, dtype=float)
    history = [w.copy()]
    for _ in range(steps):
        w = w - lr * grad(w)
        history.append(w.copy())
    return w, history

grad = lambda w: np.array([2 * (w[0] - 3), 20 * (w[1] + 1)])
w, hist = gradient_descent(grad, [0.0, 0.0], lr=0.04, steps=200)
print(w)
~~~

~~~python tests
import numpy as np

grad = lambda w: np.array([2 * (w[0] - 3), 20 * (w[1] + 1)])

def test_converges():
    """reaches the minimum (3, -1)"""
    w, _ = gradient_descent(grad, [0.0, 0.0], lr=0.04, steps=300)
    assert approx(w, [3.0, -1.0], tol=1e-5)

def test_history():
    """history has steps + 1 entries and starts at w0"""
    _, h = gradient_descent(grad, [5.0, 5.0], lr=0.01, steps=10)
    assert len(h) == 11 and approx(h[0], [5.0, 5.0])

def test_one_step():
    """a single step is w0 - lr * grad(w0)"""
    w, _ = gradient_descent(grad, [0.0, 0.0], lr=0.1, steps=1)
    assert approx(w, [0.6, -2.0])

def test_diverges_when_lr_too_big():
    """with lr above 2 / curvature (= 0.1 here) the steep direction blows up"""
    w, _ = gradient_descent(grad, [0.0, 0.0], lr=0.11, steps=60)
    assert abs(w[1]) > 100

def test_history_is_copied():
    """history entries are separate arrays, not the same object updated in place"""
    _, h = gradient_descent(grad, [0.0, 0.0], lr=0.04, steps=3)
    assert not approx(h[0], h[-1])
~~~

<!-- hint: Store `w.copy()` in the history; otherwise every entry may point to the same array. -->
<!-- /exercise -->

<!-- exercise: sgd | Mini-batch SGD for linear regression -->
Fit a linear model $\hat y = X\mathbf w$ (no bias) by mini-batch SGD on the mean squared error. Write `sgd_linear(X, y, lr, epochs, batch_size, seed)`:

1. `rng = np.random.default_rng(seed)`, `w = np.zeros(d)`.
2. Each epoch: `order = rng.permutation(n)`, then for each consecutive slice `idx` of length `batch_size` take `Xb, yb = X[idx], y[idx]` and update with the gradient of the batch MSE, $\frac{2}{|B|}X_B^\top(X_B\mathbf w - \mathbf y_B)$.
3. Return `w`.

~~~python starter
import numpy as np

def sgd_linear(X, y, lr=0.05, epochs=20, batch_size=32, seed=0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    raise NotImplementedError

rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.normal(size=1000)
print(sgd_linear(X, y))
~~~

~~~python solution
import numpy as np

def sgd_linear(X, y, lr=0.05, epochs=20, batch_size=32, seed=0):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            w -= lr * 2 / len(idx) * Xb.T @ (Xb @ w - yb)
    return w

rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 3))
y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * rng.normal(size=1000)
print(sgd_linear(X, y))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(1000, 3))
w_true = np.array([2.0, -1.0, 0.5])
y = X @ w_true + 0.1 * rng.normal(size=1000)

def test_recovers_weights():
    """finds weights close to the true ones"""
    assert approx(sgd_linear(X, y), w_true, tol=0.03)

def test_close_to_least_squares():
    """ends close to the exact least-squares solution"""
    w_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.max(np.abs(sgd_linear(X, y, lr=0.01, epochs=40) - w_ls)) < 0.02

def test_one_epoch_full_batch_is_gd():
    """with batch_size = n it is plain gradient descent"""
    w = sgd_linear(X, y, lr=0.1, epochs=1, batch_size=1000)
    expected = 0.1 * 2 / 1000 * X.T @ y   # one step from w = 0
    assert approx(w, expected, tol=1e-9)

def test_reproducible():
    """same seed, same result"""
    assert approx(sgd_linear(X, y, epochs=2, seed=4), sgd_linear(X, y, epochs=2, seed=4))
~~~

<!-- hint: `for start in range(0, n, batch_size): idx = order[start:start + batch_size]` walks through the shuffled order. -->
<!-- hint: Divide by `len(idx)`, not `batch_size`: the last batch can be shorter. -->
<!-- /exercise -->

<!-- quiz -->
1. The training loss goes up and down wildly and then becomes `nan`. What is the most likely cause?
   - [x] The learning rate is too large
   - [ ] The learning rate is too small
   - [ ] There is too little training data
   > A step size beyond the stability limit makes each update overshoot further than the last, until the numbers overflow.
2. Why is the mini-batch gradient useful even though it is noisy?
   - [ ] It is more accurate than the full gradient
   - [x] It is an unbiased estimate that costs a small fraction of a full pass, so you can take many more steps
   - [ ] It guarantees convergence to a global minimum for any loss
   > On average it points the same way as the full gradient, and many cheap noisy steps beat a few expensive exact ones.
3. One feature is measured in millimetres (values around 10 000) and another is a fraction between 0 and 1. What happens to plain gradient descent?
   - [ ] Nothing, gradient descent ignores feature scales
   - [x] The loss surface is badly stretched, forcing a tiny learning rate and very slow progress along the small feature
   - [ ] It converges faster because the large feature dominates
   > Very different scales create very different curvatures. Standardising the features makes the surface round and the problem well conditioned.
4. For which loss is any local minimum guaranteed to be a global minimum?
   - [ ] A deep neural network's loss
   - [x] A convex loss such as least squares
   - [ ] Any loss with a smooth gradient
   > Convexity rules out separate local valleys. Neural network losses are not convex.
<!-- /quiz -->
