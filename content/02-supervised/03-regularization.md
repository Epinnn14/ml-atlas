---
title: Regularisation: ridge and lasso
level: intermediate
minutes: 45
summary: Penalising large weights to fight overfitting. Ridge has a closed form and shrinks every weight; lasso, solved by coordinate descent, sets many weights exactly to zero.
---

## Penalise complexity directly

Instead of choosing between a simple and a flexible model, keep the flexible one and add a penalty that grows with the size of the weights:

<!-- formula: Regularised least squares -->
$$\min_{\mathbf w}\; \lVert X\mathbf w - \mathbf y\rVert^2 + \lambda\,\Omega(\mathbf w)$$

$\lambda \ge 0$ sets the strength. $\lambda = 0$ is ordinary least squares; a very large $\lambda$ forces the weights to zero. Like the polynomial degree, $\lambda$ is a **hyperparameter** chosen on validation data ([cross-validation](#/lesson/cross-validation)).

Two practical rules:

- **Standardise the features first.** The penalty treats all weights alike, so they must be on comparable scales.
- **Do not penalise the intercept.** Shrinking it towards zero would bias every prediction. The clean way: centre $X$ and $\mathbf y$, fit without intercept, then recover $b = \bar y - \bar{\mathbf x}^\top \mathbf w$.

## Ridge: the L2 penalty

With $\Omega(\mathbf w) = \lVert\mathbf w\rVert_2^2$ the gradient is $2X^\top(X\mathbf w - \mathbf y) + 2\lambda\mathbf w$, and setting it to zero gives a closed form:

<!-- formula: Ridge regression -->
$$\hat{\mathbf w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1}X^\top\mathbf y$$

Adding $\lambda I$ makes the matrix invertible even when features are collinear or $d > n$, so ridge is also a cure for unstable weights. It shrinks all weights smoothly towards zero but rarely makes any exactly zero. In Bayesian terms it is the [MAP estimate](#/lesson/statistics-estimation) under a Gaussian prior on the weights ([Bayesian regression](#/lesson/bayesian-regression)).

## Lasso: the L1 penalty

With $\Omega(\mathbf w) = \lVert\mathbf w\rVert_1 = \sum_j |w_j|$ (here usually written with a factor $\tfrac12$ on the squared error) the solution is **sparse**: many weights become exactly zero, so the lasso also selects features. The reason is geometric: the L1 ball has corners on the axes, and the error contours usually touch it at a corner.

The absolute value has no derivative at 0, so there is no closed form. **Coordinate descent** solves it by updating one weight at a time with the others fixed. Each one-dimensional problem has an exact answer given by **soft thresholding**:

<!-- formula: Soft thresholding -->
$$S(z, \gamma) = \operatorname{sign}(z)\,\max(|z| - \gamma,\; 0)$$

For the objective $\tfrac12\lVert X\mathbf w - \mathbf y\rVert^2 + \lambda\lVert\mathbf w\rVert_1$, the update of weight $j$ is

<!-- formula: Lasso coordinate update -->
$$w_j \leftarrow \frac{S\!\left(\mathbf x_j^\top \mathbf r_j,\; \lambda\right)}{\mathbf x_j^\top\mathbf x_j}, \qquad \mathbf r_j = \mathbf y - X\mathbf w + \mathbf x_j w_j$$

where $\mathbf x_j$ is column $j$ and $\mathbf r_j$ is the residual with feature $j$'s contribution added back. Sweep over all $j$ repeatedly until the weights stop changing.

## Elastic net and the regularisation path

**Elastic net** mixes both penalties, $\lambda_1\lVert\mathbf w\rVert_1 + \lambda_2\lVert\mathbf w\rVert_2^2$: sparse like the lasso, but stable when features are correlated (lasso tends to keep one of a correlated group at random). Plotting each weight against $\lambda$ gives the **regularisation path**: as $\lambda$ grows, ridge weights glide towards zero together, while lasso weights drop out one by one.

The same idea appears in neural networks as **weight decay** ([regularisation in deep learning](#/lesson/regularization-dl)).

<!-- exercise: ridge | Ridge regression with an unpenalised intercept -->
Write `ridge(X, y, lam)` returning `(w, b)`. Centre `X` (column means) and `y`, solve $(X_c^\top X_c + \lambda I)\mathbf w = X_c^\top \mathbf y_c$ with `np.linalg.solve`, then set $b = \bar y - \bar{\mathbf x}^\top\mathbf w$.

~~~python starter
import numpy as np

def ridge(X, y, lam):
    X, y = np.asarray(X, float), np.asarray(y, float)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ridge(X, y, lam):
    X, y = np.asarray(X, float), np.asarray(y, float)
    x_mean, y_mean = X.mean(axis=0), y.mean()
    Xc, yc = X - x_mean, y - y_mean
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    return w, float(y_mean - x_mean @ w)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 4))
y = X @ np.array([2.0, 0.0, -1.0, 3.0]) + 10 + 0.5 * rng.normal(size=60)

def test_zero_lambda_is_ols():
    """lambda = 0 gives ordinary least squares with intercept"""
    w, b = ridge(X, y, 0.0)
    coef = np.linalg.lstsq(np.hstack([X, np.ones((60, 1))]), y, rcond=None)[0]
    assert approx(w, coef[:4], tol=1e-8) and approx(b, coef[4], tol=1e-8)

def test_shrinks():
    """a larger lambda gives a smaller weight vector"""
    norms = [np.linalg.norm(ridge(X, y, lam)[0]) for lam in (0.0, 10.0, 100.0, 1000.0)]
    assert all(a > b for a, b in zip(norms, norms[1:]))

def test_intercept_not_shrunk():
    """with a huge lambda, weights go to 0 but the intercept goes to mean(y)"""
    w, b = ridge(X, y, 1e9)
    assert np.max(np.abs(w)) < 1e-4 and abs(b - y.mean()) < 1e-3

def test_collinear_features():
    """works when two columns are identical (OLS would be singular)"""
    Xd = np.hstack([X, X[:, :1]])
    w, b = ridge(Xd, y, 1.0)
    assert np.all(np.isfinite(w)) and approx(w[0], w[4], tol=1e-8)
~~~

<!-- hint: `np.eye(d)` is the identity; only the weights, not the intercept, get the penalty because the data are centred. -->
<!-- /exercise -->

<!-- exercise: lasso | Lasso by coordinate descent -->
Write `soft_threshold(z, gamma)` and `lasso(X, y, lam, sweeps=200)` that minimises $\tfrac12\lVert X\mathbf w - \mathbf y\rVert^2 + \lambda\lVert\mathbf w\rVert_1$ (no intercept: the tests use centred data) by coordinate descent, starting from zeros and updating $w_0, w_1, \dots$ in order on each sweep.

~~~python starter
import numpy as np

def soft_threshold(z, gamma):
    raise NotImplementedError

def lasso(X, y, lam, sweeps=200):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    w = np.zeros(d)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)

def lasso(X, y, lam, sweeps=200):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    w = np.zeros(d)
    col_sq = np.sum(X ** 2, axis=0)
    r = y - X @ w
    for _ in range(sweeps):
        for j in range(d):
            r += X[:, j] * w[j]                 # add feature j back
            w[j] = soft_threshold(X[:, j] @ r, lam) / col_sq[j]
            r -= X[:, j] * w[j]                 # remove it with the new weight
    return w
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(100, 8))
X -= X.mean(axis=0)
w_true = np.array([4.0, 0, 0, -3.0, 0, 0, 2.0, 0])
y = X @ w_true + 0.3 * rng.normal(size=100)
y -= y.mean()

def test_soft_threshold():
    """S(3, 1) = 2, S(-3, 1) = -2, S(0.5, 1) = 0"""
    assert approx(soft_threshold(np.array([3.0, -3.0, 0.5]), 1.0), [2.0, -2.0, 0.0])

def test_sparse():
    """a moderate lambda keeps the 3 real features and zeroes the rest"""
    w = lasso(X, y, 20.0)
    assert set(np.flatnonzero(np.abs(w) > 1e-8)) == {0, 3, 6}

def test_zero_lambda_is_least_squares():
    """lambda = 0 gives the least-squares solution"""
    assert approx(lasso(X, y, 0.0, sweeps=500), np.linalg.lstsq(X, y, rcond=None)[0], tol=1e-6)

def test_huge_lambda_all_zero():
    """a lambda larger than max |X^T y| zeroes every weight"""
    assert approx(lasso(X, y, np.max(np.abs(X.T @ y)) + 1), np.zeros(8))

def test_optimality():
    """at the solution, |x_j^T r| <= lambda for every zero weight"""
    lam = 20.0
    w = lasso(X, y, lam, sweeps=500)
    g = X.T @ (y - X @ w)
    zero = np.abs(w) < 1e-10
    assert np.all(np.abs(g[zero]) <= lam + 1e-6)
    assert approx(g[~zero], lam * np.sign(w[~zero]), tol=1e-5)
~~~

<!-- hint: Keep a running residual `r = y - X @ w`. Before updating `w[j]`, add `X[:, j] * w[j]` back to it; afterwards subtract the new contribution. -->
<!-- hint: The new weight is `soft_threshold(X[:, j] @ r, lam) / (X[:, j] @ X[:, j])`. -->
<!-- /exercise -->

<!-- quiz -->
1. You need a model that uses only a handful of 500 candidate features. Which penalty fits best?
   - [ ] Ridge (L2)
   - [x] Lasso (L1)
   - [ ] No penalty
   > The L1 penalty sets many weights exactly to zero, performing feature selection; ridge only shrinks them.
2. Why is the intercept usually left out of the penalty?
   - [ ] It is always zero after standardising
   - [x] Shrinking it would pull every prediction towards zero, a bias that has nothing to do with model complexity
   - [ ] It makes the closed form impossible
   > The intercept just sets the level of the predictions; penalising it means the model's answer depends on where the target's zero happens to be.
3. Your data has more features than examples ($d > n$). What does ridge offer that ordinary least squares does not?
   - [x] $X^\top X + \lambda I$ is invertible, so there is a unique, stable solution
   - [ ] It removes features automatically
   - [ ] It makes the problem convex
   > With $d > n$, $X^\top X$ is singular and least squares has infinitely many solutions. Adding $\lambda I$ fixes that.
4. What happens to ridge weights as $\lambda \to \infty$?
   - [ ] They become sparse, with most exactly zero
   - [x] All of them shrink towards zero together
   - [ ] They grow without bound
   > The penalty dominates and pulls every weight smoothly towards zero, but none reaches it exactly at finite $\lambda$.
<!-- /quiz -->
