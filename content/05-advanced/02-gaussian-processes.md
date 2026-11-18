---
title: Gaussian processes
level: senior
minutes: 55
summary: A prior directly over functions, defined by a kernel. Posterior mean and variance in closed form, the Cholesky way to compute them, and tuning the kernel by the marginal likelihood.
---

## Distributions over functions

A **Gaussian process** (GP) is a distribution over functions such that the values at any finite set of inputs are jointly Gaussian. It is defined by a mean function (usually 0) and a **kernel** $k(\mathbf x, \mathbf x')$ that gives the covariance between the function's values at two inputs:

<!-- formula: Gaussian process prior -->
$$f \sim \mathcal{GP}\big(0,\ k(\mathbf x, \mathbf x')\big) \quad\Longleftrightarrow\quad \big(f(\mathbf x_1),\dots,f(\mathbf x_n)\big) \sim \mathcal N(\mathbf 0, K),\ \ K_{ij} = k(\mathbf x_i, \mathbf x_j)$$

The kernel encodes assumptions about the function. The squared-exponential (RBF) kernel, the one from [SVMs](#/lesson/svm), says nearby inputs have similar outputs and the function is very smooth:

<!-- formula: Squared-exponential kernel -->
$$k(\mathbf x, \mathbf x') = \sigma_f^2\exp\!\Big(-\frac{\lVert\mathbf x - \mathbf x'\rVert^2}{2\ell^2}\Big)$$

The **length-scale** $\ell$ says how far you move before the function changes substantially; $\sigma_f^2$ sets its typical amplitude. Other kernels encode roughness (Matérn), periodicity, or linear trends, and kernels can be added and multiplied to combine assumptions. A GP is [Bayesian linear regression](#/lesson/bayesian-regression) with (possibly infinitely many) features implied by the kernel.

## Posterior

Observe $\mathbf y = f(X) + \boldsymbol\varepsilon$ with noise variance $\sigma_n^2$. The training outputs and the function values at test points $X_*$ are jointly Gaussian, and conditioning gives the posterior in closed form:

<!-- formula: GP posterior mean and covariance -->
$$\boldsymbol\mu_* = K_*^\top(K + \sigma_n^2I)^{-1}\mathbf y, \qquad \Sigma_* = K_{**} - K_*^\top(K + \sigma_n^2I)^{-1}K_*$$

where $K = k(X, X)$, $K_* = k(X, X_*)$ and $K_{**} = k(X_*, X_*)$. The mean interpolates the data smoothly; the variance is small near observations and returns to the prior $\sigma_f^2$ far from them. Add $\sigma_n^2$ to predict noisy observations rather than the function itself.

## Computing it stably

Never invert the matrix. Factor $K + \sigma_n^2 I = LL^\top$ with a **Cholesky decomposition** and solve triangular systems:

1. $\boldsymbol\alpha = L^\top\backslash(L\backslash\mathbf y)$, so $\boldsymbol\mu_* = K_*^\top\boldsymbol\alpha$.
2. $V = L\backslash K_*$, so the predictive variances are $\operatorname{diag}(K_{**}) - \sum_{\text{rows}} V^2$.

A small "jitter" (say $10^{-8}$) on the diagonal keeps the factorisation from failing in floating point.

## Learning the kernel

The kernel's parameters are fitted by maximising the **log marginal likelihood**:

<!-- formula: GP log marginal likelihood -->
$$\log p(\mathbf y\mid X) = -\tfrac12\mathbf y^\top(K + \sigma_n^2I)^{-1}\mathbf y - \tfrac12\log\lvert K + \sigma_n^2I\rvert - \tfrac n2\log 2\pi$$

The first term rewards fitting the data, the second penalises complexity (a short length-scale makes the matrix better conditioned but "explains" any data). With the Cholesky factor, $\log\lvert K + \sigma_n^2 I\rvert = 2\sum_i\log L_{ii}$. This built-in Occam's razor is a major appeal of GPs: no validation set is needed to choose $\ell$.

## Strengths and limits

GPs give well-calibrated uncertainty from little data, which makes them the standard model inside **Bayesian optimisation**: tuning expensive things (hyperparameters, experiments) by choosing each next trial where the GP predicts a good outcome or is very uncertain. Their cost is $O(n^3)$ time and $O(n^2)$ memory, so exact GPs suit up to a few thousand points; sparse and inducing-point approximations go further.

<!-- exercise: gp-posterior | GP regression with Cholesky -->
Write `gp_posterior(X, y, X_star, kernel, noise)` returning `(mean, var)` of the latent function at `X_star` (variance of $f$, not including the noise), computed with `np.linalg.cholesky` and `np.linalg.solve` (no `inv`). `kernel(A, B)` returns a kernel matrix; add `noise` (a variance) to the training diagonal.

~~~python starter
import numpy as np

def rbf(A, B, ell=1.0, sf2=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return sf2 * np.exp(-d2 / (2 * ell ** 2))

def gp_posterior(X, y, X_star, kernel, noise):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def rbf(A, B, ell=1.0, sf2=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return sf2 * np.exp(-d2 / (2 * ell ** 2))

def gp_posterior(X, y, X_star, kernel, noise):
    K = kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = kernel(X, X_star)
    mean = Ks.T @ alpha
    V = np.linalg.solve(L, Ks)
    var = np.diag(kernel(X_star, X_star)) - np.sum(V ** 2, axis=0)
    return mean, var
~~~

~~~python tests
import numpy as np

X = np.array([[-2.0], [-1.0], [0.5], [2.0]])
y = np.sin(X[:, 0])
Xs = np.linspace(-4, 4, 9)[:, None]

def test_matches_direct_formula():
    """agrees with the textbook formulas using an explicit inverse"""
    mean, var = gp_posterior(X, y, Xs, rbf, 0.01)
    K = rbf(X, X) + 0.01 * np.eye(4); Ks = rbf(X, Xs)
    Ki = np.linalg.inv(K)
    assert approx(mean, Ks.T @ Ki @ y) and approx(var, np.diag(rbf(Xs, Xs) - Ks.T @ Ki @ Ks))

def test_interpolates():
    """with almost no noise the mean passes through the data and the variance there is ~0"""
    mean, var = gp_posterior(X, y, X, rbf, 1e-8)
    assert approx(mean, y, tol=1e-4) and np.all(var < 1e-4)

def test_far_away_returns_to_prior():
    """far from the data, mean -> 0 and variance -> sf2"""
    mean, var = gp_posterior(X, y, np.array([[30.0]]), rbf, 0.01)
    assert abs(mean[0]) < 1e-6 and abs(var[0] - 1.0) < 1e-6

def test_variance_non_negative():
    """posterior variances are non-negative (up to rounding)"""
    _, var = gp_posterior(X, y, Xs, rbf, 0.01)
    assert np.all(var > -1e-10)
~~~

<!-- hint: `L = np.linalg.cholesky(K)`; then `alpha = solve(L.T, solve(L, y))` and `V = solve(L, Ks)`. -->
<!-- /exercise -->

<!-- exercise: gp-evidence | Choose the length-scale by the marginal likelihood -->
Write `log_marginal_likelihood(X, y, ell, noise)` for the RBF kernel with $\sigma_f^2 = 1$, using the Cholesky factor for both the quadratic term and the log-determinant. Then `best_ell(X, y, ells, noise)` returns the length-scale in `ells` with the highest value.

~~~python starter
import numpy as np

def rbf(A, B, ell=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * ell ** 2))

def log_marginal_likelihood(X, y, ell, noise):
    raise NotImplementedError

def best_ell(X, y, ells, noise):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def rbf(A, B, ell=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * ell ** 2))

def log_marginal_likelihood(X, y, ell, noise):
    n = len(X)
    L = np.linalg.cholesky(rbf(X, X, ell) + noise * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(-0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi))

def best_ell(X, y, ells, noise):
    return ells[int(np.argmax([log_marginal_likelihood(X, y, e, noise) for e in ells]))]
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 10, 40))[:, None]
true_ell = 1.5
K = rbf(X, X, true_ell) + 1e-8 * np.eye(40)
y = np.linalg.cholesky(K) @ rng.normal(size=40) + 0.1 * rng.normal(size=40)

def test_formula():
    """matches the formula with an explicit inverse and determinant"""
    C = rbf(X, X, 1.0) + 0.01 * np.eye(40)
    ref = -0.5 * y @ np.linalg.solve(C, y) - 0.5 * np.linalg.slogdet(C)[1] - 20 * np.log(2 * np.pi)
    assert approx(log_marginal_likelihood(X, y, 1.0, 0.01), ref)

def test_recovers_length_scale():
    """picks a length-scale near the one that generated the data"""
    assert best_ell(X, y, [0.2, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0], 0.01) in (1.0, 1.5, 2.5)

def test_extremes_are_worse():
    """very short and very long length-scales explain the data worse"""
    lml = lambda e: log_marginal_likelihood(X, y, e, 0.01)
    assert lml(1.5) > lml(0.05) and lml(1.5) > lml(50.0)
~~~

<!-- hint: $\log\lvert C\rvert = 2\sum_i\log L_{ii}$, so the second term is `-np.sum(np.log(np.diag(L)))`. -->
<!-- /exercise -->

<!-- quiz -->
1. What does a GP's length-scale $\ell$ control?
   - [ ] The noise level
   - [x] How far apart two inputs must be before their function values become nearly independent
   - [ ] The number of training points
   > A short length-scale gives wiggly functions, a long one smooth, slowly varying functions.
2. What happens to the GP posterior variance at a training input when the noise is tiny?
   - [x] It drops to almost zero, because the function value there is known
   - [ ] It equals the prior variance
   - [ ] It becomes negative
   > Conditioning on an almost noise-free observation pins the function down at that point.
3. Why are exact GPs limited to a few thousand training points?
   - [ ] Kernels cannot be computed for more points
   - [x] Factorising the $n\times n$ kernel matrix costs $O(n^3)$ time and $O(n^2)$ memory
   - [ ] The marginal likelihood is undefined for large $n$
   > Sparse approximations with $m \ll n$ inducing points reduce the cost to $O(nm^2)$.
4. Why do GPs suit Bayesian optimisation of expensive experiments?
   - [x] They give good uncertainty estimates from few observations, which guides where to try next
   - [ ] They are the fastest models to train on big data
   - [ ] They need no kernel
   > Choosing points that are promising or uncertain needs reliable error bars, exactly what a GP provides.
<!-- /quiz -->
