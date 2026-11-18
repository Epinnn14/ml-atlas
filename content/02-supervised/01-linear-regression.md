---
title: Linear regression
level: beginner
minutes: 40
summary: The first real model. Fit a weighted sum of features by least squares, solve it exactly with the normal equation, read what the weights mean, and measure the fit with R².
---

## The model

Linear regression predicts a number as a weighted sum of the features plus an intercept:

<!-- formula: Linear regression -->
$$\hat y = \mathbf w^\top \mathbf x + b = w_1x_1 + \dots + w_dx_d + b$$

Each weight $w_j$ says how much the prediction changes when feature $j$ grows by one unit and the others stay the same. The intercept $b$ is the prediction when every feature is zero.

A common trick folds the intercept into the weights: add a constant feature $x_0 = 1$ to every example, so $b$ becomes just another weight. For a data matrix that means adding a column of ones.

## Least squares

We choose $\mathbf w$ to minimise the sum of squared residuals, which is maximum likelihood when the noise is Gaussian ([estimation](#/lesson/statistics-estimation)):

<!-- formula: Least-squares objective -->
$$\min_{\mathbf w}\; \lVert X\mathbf w - \mathbf y\rVert^2$$

The gradient is $2X^\top(X\mathbf w - \mathbf y)$ ([gradients](#/lesson/calculus-gradients)). Setting it to zero gives the **normal equation**:

<!-- formula: Normal equation -->
$$X^\top X\,\hat{\mathbf w} = X^\top\mathbf y \quad\Longrightarrow\quad \hat{\mathbf w} = (X^\top X)^{-1}X^\top \mathbf y$$

The loss is convex, so this is the global minimum. Solve it with `np.linalg.solve(X.T @ X, X.T @ y)`, or better `np.linalg.lstsq(X, y)`, which also copes when $X^\top X$ is singular.

## The geometry

The predictions $X\hat{\mathbf w}$ are the point in the space spanned by the feature columns that is closest to $\mathbf y$, the orthogonal [projection](#/lesson/linear-algebra). The residual $\mathbf y - X\hat{\mathbf w}$ is orthogonal to every column of $X$, which is just the normal equation read differently: $X^\top(\mathbf y - X\hat{\mathbf w}) = \mathbf 0$. With an intercept column, this means the residuals sum to zero.

## When to use gradient descent instead

Solving the normal equation costs about $O(nd^2 + d^3)$. That is instant for a few hundred features, but with hundreds of thousands of features, or data that does not fit in memory, [gradient descent](#/lesson/gradient-descent) on the same loss is the practical route.

## Measuring the fit: R²

The **coefficient of determination** compares the model's squared error with that of always predicting the mean:

<!-- formula: R squared -->
$$R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$

$R^2 = 1$ is a perfect fit, $0$ is no better than the mean, and on a test set it can be negative: worse than the mean.

## Reading the weights, carefully

- Weights are only comparable when features share a scale. Standardise first if you want to compare their sizes.
- Two strongly correlated features (**collinearity**) can share their effect in unstable ways: one weight large positive, the other large negative. Predictions are fine, interpretation is not. [Regularisation](#/lesson/regularization) stabilises this.
- A weight measures association within this model, not cause and effect ([causal inference](#/lesson/causal-inference)).
- "Linear" means linear in the weights. Features can be anything: $x^2$, $\log x$, $x_1x_2$. Adding such features lets a linear model fit curves, which leads straight to the [bias–variance trade-off](#/lesson/bias-variance).

<!-- exercise: ols | Fit linear regression with an intercept -->
Write `fit_ols(X, y)` that returns `(w, b)`: the weights (shape `(d,)`) and the intercept (a float). Add a column of ones, solve least squares with `np.linalg.lstsq`, and split the result. Then write `predict(X, w, b)`.

~~~python starter
import numpy as np

def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    raise NotImplementedError

def predict(X, w, b):
    raise NotImplementedError

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + 0.1 * rng.normal(size=100)
print(fit_ols(X, y))   # weights near [3, -2], intercept near 5
~~~

~~~python solution
import numpy as np

def fit_ols(X, y):
    X = np.asarray(X, dtype=float)
    A = np.hstack([X, np.ones((len(X), 1))])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return coef[:-1], float(coef[-1])

def predict(X, w, b):
    return np.asarray(X, dtype=float) @ w + b

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + 0.1 * rng.normal(size=100)
print(fit_ols(X, y))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([3.0, -2.0, 0.5]) + 5 + 0.05 * rng.normal(size=200)

def test_recovers_parameters():
    """recovers the weights and the intercept"""
    w, b = fit_ols(X, y)
    assert approx(w, [3.0, -2.0, 0.5], tol=0.02) and abs(b - 5) < 0.02

def test_types():
    """returns a (d,) array and a float"""
    w, b = fit_ols(X, y)
    assert np.shape(w) == (3,) and isinstance(b, float)

def test_residuals_orthogonal():
    """residuals sum to zero and are orthogonal to every feature"""
    w, b = fit_ols(X, y)
    r = y - predict(X, w, b)
    assert abs(r.sum()) < 1e-8 and approx(X.T @ r, np.zeros(3), tol=1e-8)

def test_exact_line():
    """two points define a line exactly"""
    w, b = fit_ols(np.array([[0.0], [2.0]]), np.array([1.0, 5.0]))
    assert approx(w, [2.0]) and approx(b, 1.0)
~~~

<!-- hint: `np.hstack([X, np.ones((n, 1))])` adds the column of ones; the last coefficient is then the intercept. -->
<!-- /exercise -->

<!-- exercise: r2 | The R² score -->
Write `r2_score(y, y_hat)` returning $R^2$ as a float.

~~~python starter
import numpy as np

def r2_score(y, y_hat):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def r2_score(y, y_hat):
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot)
~~~

~~~python tests
import numpy as np

y = np.array([1.0, 2.0, 4.0, 7.0])

def test_perfect():
    """a perfect prediction scores 1"""
    assert approx(r2_score(y, y), 1.0)

def test_mean_scores_zero():
    """predicting the mean scores 0"""
    assert approx(r2_score(y, np.full(4, y.mean())), 0.0)

def test_can_be_negative():
    """a prediction worse than the mean is negative"""
    assert r2_score(y, y[::-1]) < 0

def test_value():
    """matches the formula on a small example"""
    yh = np.array([1.5, 2.0, 3.5, 7.5])
    assert approx(r2_score(y, yh), 1 - 0.75 / 21.0)
~~~

<!-- /exercise -->

<!-- quiz -->
1. In $\hat y = 2.5x_1 - 0.8x_2 + 10$, what does the $-0.8$ mean?
   - [ ] $x_2$ is less important than $x_1$
   - [x] Raising $x_2$ by one unit, with $x_1$ fixed, lowers the prediction by 0.8
   - [ ] $x_2$ causes the target to decrease
   > A weight is a partial effect within the model. Importance comparisons need features on the same scale, and causal claims need more than a regression.
2. After fitting least squares with an intercept, what is always true of the training residuals?
   - [ ] They are all zero
   - [x] They sum to zero and are orthogonal to each feature column
   - [ ] They follow a normal distribution
   > The normal equation says $X^\top(\mathbf y - X\hat{\mathbf w}) = 0$; the row for the column of ones gives a zero sum.
3. A model scores $R^2 = -0.3$ on the test set. What does that mean?
   - [x] It predicts the test targets worse than simply predicting their mean
   - [ ] There is a bug, because $R^2$ cannot be negative
   - [ ] It explains 30% of the variance in reverse
   > On data the model was not fitted to, $R^2$ is unbounded below.
4. Can linear regression fit the curve $y = x^2$?
   - [ ] No, it can only fit straight lines
   - [x] Yes, if you add $x^2$ as a feature, because the model only needs to be linear in its weights
   - [ ] Only with gradient descent
   > With features $(x, x^2)$ the model $w_1x + w_2x^2 + b$ is still linear in $\mathbf w$, so least squares works unchanged.
<!-- /quiz -->
