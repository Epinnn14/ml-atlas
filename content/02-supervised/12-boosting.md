---
title: Boosting
level: advanced
minutes: 50
summary: Build a strong model from weak ones added in sequence, each fixing the mistakes of those before. AdaBoost's reweighting, gradient boosting as gradient descent on functions, and what XGBoost and LightGBM add.
---

## Sequential, not parallel

[Bagging](#/lesson/random-forests) trains independent models and averages them to cut variance. **Boosting** trains small, high-bias models (often stumps or depth-3 trees) one after another, each focusing on what the current ensemble still gets wrong. It mainly reduces bias, and with care it produces the most accurate models known for tabular data.

## AdaBoost

For labels $y_i \in \{-1, +1\}$, AdaBoost keeps a weight $w_i$ on every example, starting uniform. In round $t$:

1. Fit a weak classifier $h_t$ to the weighted data.
2. Compute its weighted error $\varepsilon_t = \sum_i w_i\,[h_t(\mathbf x_i) \ne y_i]$ (weights sum to 1).
3. Give it a vote proportional to how good it is:

<!-- formula: AdaBoost model weight -->
$$\alpha_t = \frac12\ln\frac{1 - \varepsilon_t}{\varepsilon_t}$$

4. Increase the weights of misclassified examples, decrease the others, and renormalise:

<!-- formula: AdaBoost weight update -->
$$w_i \leftarrow \frac{w_i\, e^{-\alpha_t y_i h_t(\mathbf x_i)}}{Z_t}$$

The final classifier is $\operatorname{sign}\big(\sum_t \alpha_t h_t(\mathbf x)\big)$. A classifier with error 0.5 gets $\alpha = 0$; one with small error gets a big vote. After the update, the previous classifier has exactly 50% weighted error, so the next one must find something new. AdaBoost turns out to be stagewise minimisation of the exponential loss $\sum_i e^{-y_i F(\mathbf x_i)}$.

## Gradient boosting

Gradient boosting generalises the idea to any differentiable loss. The model is a sum $F_M(\mathbf x) = F_0 + \sum_m \nu\, h_m(\mathbf x)$. At each stage, compute the **negative gradient** of the loss with respect to the current predictions, called pseudo-residuals, and fit the next tree to them:

<!-- formula: Pseudo-residuals -->
$$r_i = -\frac{\partial L\big(y_i, F(\mathbf x_i)\big)}{\partial F(\mathbf x_i)}$$

<!-- formula: Gradient boosting update -->
$$F_m(\mathbf x) = F_{m-1}(\mathbf x) + \nu\, h_m(\mathbf x)$$

For squared error $L = \tfrac12(y - F)^2$ the pseudo-residuals are just the residuals $y_i - F(\mathbf x_i)$: each tree fits what is still unexplained. This is [gradient descent](#/lesson/gradient-descent) where each step is a function (a tree) instead of a weight change. $F_0$ is the best constant (the mean for squared error).

The **learning rate** (shrinkage) $\nu$, around 0.05–0.1, makes each tree count only partly. Smaller $\nu$ needs more trees but generalises better. Unlike random forests, boosting **can** overfit with too many trees, so the number of rounds is chosen by validation with **early stopping**.

## What XGBoost, LightGBM and CatBoost add

- A second-order (Newton) step using the curvature of the loss, and explicit L1/L2 penalties on leaf values.
- Histogram-based split finding: bucket each feature into ~256 bins, making training fast on millions of rows.
- Row and column subsampling per tree (stochastic gradient boosting), borrowed from forests.
- Native handling of missing values and categorical features.

Typical knobs: learning rate, number of trees (with early stopping), tree depth or number of leaves, minimum examples per leaf, and subsampling rates.

<!-- exercise: adaboost-round | One round of AdaBoost -->
Write `adaboost_round(w, y, pred)` for current example weights `w` (summing to 1), labels `y` and weak-classifier predictions `pred` (both in $\{-1, +1\}$). Return `(alpha, new_w)` using the formulas above, with `new_w` normalised to sum to 1.

~~~python starter
import numpy as np

def adaboost_round(w, y, pred):
    w, y, pred = np.asarray(w, float), np.asarray(y), np.asarray(pred)
    raise NotImplementedError

w = np.full(5, 0.2)
y = np.array([1, 1, -1, -1, 1])
pred = np.array([1, 1, -1, 1, 1])        # one mistake
print(adaboost_round(w, y, pred))
~~~

~~~python solution
import numpy as np

def adaboost_round(w, y, pred):
    w, y, pred = np.asarray(w, float), np.asarray(y), np.asarray(pred)
    eps = np.sum(w[pred != y])
    alpha = 0.5 * np.log((1 - eps) / eps)
    new_w = w * np.exp(-alpha * y * pred)
    return float(alpha), new_w / new_w.sum()

w = np.full(5, 0.2)
y = np.array([1, 1, -1, -1, 1])
pred = np.array([1, 1, -1, 1, 1])
print(adaboost_round(w, y, pred))
~~~

~~~python tests
import numpy as np

w = np.full(5, 0.2)
y = np.array([1, 1, -1, -1, 1])
pred = np.array([1, 1, -1, 1, 1])

def test_alpha():
    """error 0.2 gives alpha = 0.5 ln 4"""
    alpha, _ = adaboost_round(w, y, pred)
    assert approx(alpha, 0.5 * np.log(4))

def test_weights_normalised():
    """new weights sum to one"""
    assert approx(adaboost_round(w, y, pred)[1].sum(), 1.0)

def test_mistake_gets_half():
    """after the update the misclassified examples hold exactly half the weight"""
    _, nw = adaboost_round(w, y, pred)
    assert approx(nw[pred != y].sum(), 0.5)

def test_coin_flip_has_no_vote():
    """a classifier with weighted error 0.5 gets alpha 0"""
    alpha, _ = adaboost_round(np.full(4, 0.25), np.array([1, 1, -1, -1]), np.array([1, -1, -1, 1]))
    assert approx(alpha, 0.0)
~~~

<!-- /exercise -->

<!-- exercise: gbm | Gradient boosting for regression -->
Write `fit_gbm(x, y, n_rounds=100, lr=0.1)` for a 1-D input `x`: start with `F0 = y.mean()`, then each round fit a regression stump (provided) to the residuals `y - F`, add `lr` times its predictions to `F`, and store it. Return `(F0, stumps)`. `predict_gbm(x, F0, stumps, lr=0.1)` returns `F0 + lr * sum of stump predictions`.

~~~python starter
import numpy as np

def fit_reg_stump(x, r):
    """Threshold on x minimising squared error of two leaf means."""
    order = np.argsort(x)
    xs, rs = x[order], r[order]
    best = (np.inf, None, 0.0, 0.0)
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            continue
        l, rr = rs[:i], rs[i:]
        sse = ((l - l.mean()) ** 2).sum() + ((rr - rr.mean()) ** 2).sum()
        if sse < best[0]:
            best = (sse, (xs[i] + xs[i - 1]) / 2, l.mean(), rr.mean())
    return best[1:]

def predict_reg_stump(stump, x):
    t, left, right = stump
    return np.where(x <= t, left, right)

def fit_gbm(x, y, n_rounds=100, lr=0.1):
    raise NotImplementedError

def predict_gbm(x, F0, stumps, lr=0.1):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def fit_reg_stump(x, r):
    """Threshold on x minimising squared error of two leaf means."""
    order = np.argsort(x)
    xs, rs = x[order], r[order]
    best = (np.inf, None, 0.0, 0.0)
    for i in range(1, len(xs)):
        if xs[i] == xs[i - 1]:
            continue
        l, rr = rs[:i], rs[i:]
        sse = ((l - l.mean()) ** 2).sum() + ((rr - rr.mean()) ** 2).sum()
        if sse < best[0]:
            best = (sse, (xs[i] + xs[i - 1]) / 2, l.mean(), rr.mean())
    return best[1:]

def predict_reg_stump(stump, x):
    t, left, right = stump
    return np.where(x <= t, left, right)

def fit_gbm(x, y, n_rounds=100, lr=0.1):
    F0 = float(np.mean(y))
    F = np.full(len(y), F0)
    stumps = []
    for _ in range(n_rounds):
        s = fit_reg_stump(x, y - F)
        F = F + lr * predict_reg_stump(s, x)
        stumps.append(s)
    return F0, stumps

def predict_gbm(x, F0, stumps, lr=0.1):
    return F0 + lr * sum(predict_reg_stump(s, x) for s in stumps)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 6, 120))
y = np.sin(x) + 0.1 * rng.normal(size=120)

def mse(F0, st):
    return np.mean((predict_gbm(x, F0, st) - y) ** 2)

def test_starts_at_mean():
    """with zero rounds the model predicts the mean"""
    F0, st = fit_gbm(x, y, n_rounds=0)
    assert approx(F0, y.mean()) and st == []

def test_error_falls():
    """training error falls as rounds are added"""
    errs = [mse(*fit_gbm(x, y, n_rounds=r)) for r in (1, 10, 50, 200)]
    assert all(a > b for a, b in zip(errs, errs[1:]))

def test_fits_the_curve():
    """200 rounds of stumps approximate sin(x) well"""
    assert mse(*fit_gbm(x, y, n_rounds=200)) < 0.03

def test_first_round():
    """the first stump is fitted to y - mean(y)"""
    F0, st = fit_gbm(x, y, n_rounds=1)
    assert st[0] == fit_reg_stump(x, y - y.mean())
~~~

<!-- hint: Keep the running prediction `F` on the training points; the residuals each round are `y - F`. -->
<!-- /exercise -->

<!-- quiz -->
1. What does each new tree in gradient boosting with squared-error loss try to predict?
   - [ ] The original targets $y$
   - [x] The residuals of the current ensemble, $y - F(\mathbf x)$
   - [ ] A bootstrap sample of the targets
   > For squared error, the negative gradient with respect to the prediction is the residual.
2. You lower the learning rate from 0.3 to 0.03. What else should change?
   - [x] Increase the number of trees, roughly tenfold, and keep early stopping
   - [ ] Decrease the number of trees
   - [ ] Nothing
   > Each tree now contributes a tenth as much, so more trees are needed to reach the same fit; the slower path usually generalises better.
3. Can boosting overfit as more trees are added?
   - [x] Yes, which is why the number of rounds is chosen with a validation set and early stopping
   - [ ] No, like random forests it only levels off
   - [ ] Only when using stumps
   > Each round reduces training loss further and can start fitting noise. Random forests average, boosting keeps adding.
4. In AdaBoost, what happens to an example that the current weak classifier gets wrong?
   - [ ] It is removed from the data
   - [x] Its weight increases, so the next weak classifier focuses on it
   - [ ] Its label is flipped
   > The update multiplies its weight by $e^{\alpha}$ while correct examples are multiplied by $e^{-\alpha}$.
<!-- /quiz -->
