---
title: Bayesian linear regression
level: advanced
minutes: 45
summary: Instead of one best set of weights, keep a distribution over all plausible weights. The Gaussian posterior in closed form, predictions with honest error bars, and how ridge regression appears as its most probable point.
---

## Uncertainty about the weights

[Linear regression](#/lesson/linear-regression) returns one weight vector. With little data, many weight vectors fit almost equally well, and a single answer hides that. The Bayesian approach treats the weights as uncertain: start from a **prior**, update it with the data through [Bayes' rule](#/lesson/probability), and keep the whole **posterior**. Predictions then come with uncertainty that shrinks where we have data and grows where we do not.

## Prior and likelihood

Use a zero-mean Gaussian prior on the weights with precision (inverse variance) $\alpha$, and Gaussian noise with precision $\beta$:

<!-- formula: Bayesian linear regression model -->
$$p(\mathbf w) = \mathcal N(\mathbf 0,\ \alpha^{-1}I), \qquad p(y\mid\mathbf x, \mathbf w) = \mathcal N(\mathbf w^\top\mathbf x,\ \beta^{-1})$$

## The posterior

Gaussian prior times Gaussian likelihood is again Gaussian (the prior is **conjugate**), so the posterior has a closed form:

<!-- formula: Posterior over the weights -->
$$p(\mathbf w\mid X, \mathbf y) = \mathcal N(\mathbf m_N, S_N), \qquad S_N = \big(\alpha I + \beta X^\top X\big)^{-1}, \qquad \mathbf m_N = \beta\,S_NX^\top\mathbf y$$

Compare with [ridge regression](#/lesson/regularization): the posterior mean $\mathbf m_N = (X^\top X + \tfrac{\alpha}{\beta}I)^{-1}X^\top\mathbf y$ is exactly the ridge solution with $\lambda = \alpha/\beta$. Ridge is the **MAP** estimate; the Bayesian view adds the covariance $S_N$, which says how uncertain each direction of the weights still is. More data makes $X^\top X$ larger and $S_N$ smaller.

## Predictions with error bars

For a new input $\mathbf x_*$, average the prediction over all plausible weights. The **predictive distribution** is Gaussian too:

<!-- formula: Predictive distribution -->
$$p(y_*\mid\mathbf x_*, X, \mathbf y) = \mathcal N\big(\mathbf m_N^\top\mathbf x_*,\ \ \sigma_*^2\big), \qquad \sigma_*^2 = \underbrace{\beta^{-1}}_{\text{noise}} + \underbrace{\mathbf x_*^\top S_N\mathbf x_*}_{\text{weight uncertainty}}$$

The two terms separate **aleatoric** uncertainty (noise in the data, irreducible) from **epistemic** uncertainty (we have not seen enough data, which shrinks with more data). Far from the training inputs the second term grows: the model knows when it is extrapolating. This is exactly the information you need for [calibrated uncertainty](#/lesson/calibration-uncertainty) and for exploration in [bandits](#/lesson/bandits).

## Choosing α and β

The hyperparameters can be set by maximising the **marginal likelihood** (evidence) $p(\mathbf y\mid X, \alpha, \beta)$, which integrates the weights out and automatically balances fit against complexity: "empirical Bayes". The same idea tunes kernels in [Gaussian processes](#/lesson/gaussian-processes), which are Bayesian linear regression with infinitely many features.

## Beyond Gaussians

For logistic regression or neural networks the posterior has no closed form. Options: the Laplace approximation (a Gaussian at the MAP), variational inference, Markov chain Monte Carlo, or cheap stand-ins such as deep ensembles and MC dropout.

<!-- exercise: posterior | The posterior over the weights -->
Write `posterior(X, y, alpha, beta)` returning `(m_N, S_N)` from the formulas above. Use `np.linalg.inv` for $S_N$ (the matrices here are small).

~~~python starter
import numpy as np

def posterior(X, y, alpha, beta):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def posterior(X, y, alpha, beta):
    d = X.shape[1]
    S_N = np.linalg.inv(alpha * np.eye(d) + beta * X.T @ X)
    m_N = beta * S_N @ X.T @ y
    return m_N, S_N
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = np.c_[np.ones(30), rng.uniform(-1, 1, 30)]
y = X @ np.array([0.5, -1.5]) + rng.normal(scale=0.2, size=30)

def test_shapes():
    """mean is (d,) and covariance (d, d)"""
    m, S = posterior(X, y, 2.0, 25.0)
    assert m.shape == (2,) and S.shape == (2, 2)

def test_mean_is_ridge():
    """the posterior mean equals ridge with lambda = alpha / beta"""
    m, _ = posterior(X, y, 2.0, 25.0)
    ridge = np.linalg.solve(X.T @ X + (2.0 / 25.0) * np.eye(2), X.T @ y)
    assert approx(m, ridge)

def test_no_data_is_prior():
    """with no data the posterior is the prior"""
    m, S = posterior(np.zeros((0, 2)), np.zeros(0), 4.0, 10.0)
    assert approx(m, [0.0, 0.0]) and approx(S, np.eye(2) / 4.0)

def test_shrinks_with_data():
    """more data gives a smaller posterior covariance"""
    _, S_small = posterior(X[:5], y[:5], 2.0, 25.0)
    _, S_big = posterior(X, y, 2.0, 25.0)
    assert np.trace(S_big) < np.trace(S_small)
~~~

<!-- /exercise -->

<!-- exercise: predictive | Predictions with uncertainty -->
Write `predictive(X_new, m_N, S_N, beta)` returning `(mean, var)` for every row of `X_new`, with the variance including both the noise and the weight-uncertainty term. Compute the quadratic form for all rows at once (no Python loop).

~~~python starter
import numpy as np

def predictive(X_new, m_N, S_N, beta):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def predictive(X_new, m_N, S_N, beta):
    mean = X_new @ m_N
    var = 1 / beta + np.sum((X_new @ S_N) * X_new, axis=1)
    return mean, var
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(-1, 1, 20)
X = np.c_[np.ones(20), x]
y = 0.5 - 1.5 * x + rng.normal(scale=0.2, size=20)
alpha, beta = 2.0, 25.0
S = np.linalg.inv(alpha * np.eye(2) + beta * X.T @ X)
m = beta * S @ X.T @ y

def test_values():
    """matches the formula row by row"""
    Xn = np.c_[np.ones(3), [-0.5, 0.0, 3.0]]
    mu, var = predictive(Xn, m, S, beta)
    assert approx(mu, Xn @ m) and approx(var, [1 / beta + r @ S @ r for r in Xn])

def test_at_least_noise():
    """the variance is never below the noise variance 1/beta"""
    _, var = predictive(np.c_[np.ones(50), np.linspace(-5, 5, 50)], m, S, beta)
    assert np.all(var >= 1 / beta)

def test_grows_when_extrapolating():
    """far outside the data the uncertainty is larger than inside it"""
    _, var = predictive(np.c_[np.ones(2), [0.0, 6.0]], m, S, beta)
    assert var[1] > 3 * var[0]
~~~

<!-- hint: For every row $\mathbf x$, $\mathbf x^\top S\mathbf x$ is `np.sum((X_new @ S_N) * X_new, axis=1)`. -->
<!-- /exercise -->

<!-- quiz -->
1. What is the relationship between Bayesian linear regression and ridge regression?
   - [ ] They are unrelated
   - [x] The posterior mean (and mode) equals the ridge solution with $\lambda = \alpha/\beta$
   - [ ] Ridge is the posterior variance
   > A Gaussian prior on the weights is an L2 penalty; Bayes adds the full posterior covariance on top.
2. Why does the predictive variance grow far from the training inputs?
   - [x] The weight-uncertainty term $\mathbf x_*^\top S_N\mathbf x_*$ grows along directions the data did not pin down
   - [ ] The noise variance increases there
   - [ ] The prior becomes wider
   > Uncertainty about the slope matters more the further you extrapolate.
3. Which part of the predictive variance does not shrink with more data?
   - [ ] The weight-uncertainty term
   - [x] The noise term $\beta^{-1}$ (aleatoric uncertainty)
   - [ ] Neither; both go to zero
   > Even the true weights cannot predict the random noise.
4. What does "conjugate prior" mean here?
   - [ ] The prior has the same variance as the noise
   - [x] The posterior has the same family (Gaussian) as the prior, so it has a closed form
   - [ ] The prior is chosen after seeing the data
   > Gaussian prior and Gaussian likelihood give a Gaussian posterior.
<!-- /quiz -->
