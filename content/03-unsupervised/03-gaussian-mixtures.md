---
title: Gaussian mixtures and EM
level: advanced
minutes: 50
summary: Soft clustering with a probability model. Each cluster is a Gaussian, each point belongs to every cluster with some probability, and the expectation–maximisation algorithm fits it all.
---

## A generative model of clusters

A **Gaussian mixture model** (GMM) assumes each point was produced in two steps: pick a component $c$ with probability $\pi_c$, then draw the point from that component's Gaussian $\mathcal N(\boldsymbol\mu_c, \Sigma_c)$. The density of a point is the weighted sum:

<!-- formula: Gaussian mixture density -->
$$p(\mathbf x) = \sum_{c=1}^{K}\pi_c\,\mathcal N(\mathbf x\mid\boldsymbol\mu_c,\Sigma_c), \qquad \sum_c\pi_c = 1$$

Compared with [k-means](#/lesson/k-means), clusters can be elliptical and of different sizes, and membership is **soft**: a point between two clusters can be 60% one and 40% the other. In fact k-means is the limit of a GMM with equal spherical covariances shrinking to zero.

## Responsibilities

Given parameters, [Bayes' rule](#/lesson/probability) tells how likely each component is to have produced point $i$. This is its **responsibility**:

<!-- formula: Responsibilities -->
$$r_{ic} = \frac{\pi_c\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_c,\Sigma_c)}{\sum_{c'}\pi_{c'}\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_{c'},\Sigma_{c'})}$$

Compute it in log space with [log-sum-exp](#/lesson/probability): densities in many dimensions are tiny.

## The EM algorithm

Maximising the log-likelihood $\sum_i\log p(\mathbf x_i)$ directly is awkward because of the sum inside the log. **Expectation–maximisation** alternates two easy steps:

- **E-step**: with the current parameters, compute the responsibilities $r_{ic}$.
- **M-step**: re-estimate the parameters as responsibility-weighted versions of the usual [MLE](#/lesson/statistics-estimation) formulas, with $N_c = \sum_i r_{ic}$:

<!-- formula: EM M-step for a Gaussian mixture -->
$$\pi_c = \frac{N_c}{n}, \qquad \boldsymbol\mu_c = \frac{1}{N_c}\sum_i r_{ic}\,\mathbf x_i, \qquad \Sigma_c = \frac{1}{N_c}\sum_i r_{ic}\,(\mathbf x_i - \boldsymbol\mu_c)(\mathbf x_i - \boldsymbol\mu_c)^\top$$

Each EM iteration never decreases the log-likelihood, so it converges, but to a local maximum: run it from several starts (often initialised by k-means). Watch for a component collapsing onto a single point, where its variance goes to zero and the likelihood to infinity; adding a small constant to the covariance diagonal prevents it.

EM is a general recipe for models with hidden variables. Hidden Markov models, topic models and the training of [variational autoencoders](#/lesson/autoencoders-vae) all use the same idea.

## Choosing K

The log-likelihood keeps increasing with $K$. Penalise complexity with the **Bayesian information criterion**:

<!-- formula: Bayesian information criterion -->
$$\text{BIC} = -2\log L + p\log n$$

where $p$ is the number of free parameters. Pick the $K$ with the lowest BIC. A GMM also gives a density, so points with very low $p(\mathbf x)$ are natural [anomalies](#/lesson/anomaly-detection).

<!-- exercise: em-1d | EM for a one-dimensional mixture -->
Write `em_gmm_1d(x, K, means, n_iter=100)` for 1-D data. Initialise `pi = 1/K` for each component, means from the given list, and every variance to `x.var()`. Each iteration: E-step (responsibilities, computed in log space), M-step (update `pi`, `mu`, `var`; add `1e-6` to each variance). Return `(pi, mu, var)` as arrays, plus a function-level helper `log_likelihood(x, pi, mu, var)` returning $\sum_i\log p(x_i)$.

~~~python starter
import numpy as np

def log_likelihood(x, pi, mu, var):
    raise NotImplementedError

def em_gmm_1d(x, K, means, n_iter=100):
    x = np.asarray(x, float)
    pi = np.full(K, 1 / K)
    mu = np.array(means, float)
    var = np.full(K, x.var())
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def _log_comp(x, pi, mu, var):
    # (n, K): log pi_c + log N(x_i | mu_c, var_c)
    return np.log(pi) - 0.5 * np.log(2 * np.pi * var) - (x[:, None] - mu) ** 2 / (2 * var)

def _logsumexp_rows(L):
    m = L.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(L - m).sum(axis=1, keepdims=True)))[:, 0]

def log_likelihood(x, pi, mu, var):
    x = np.asarray(x, float)
    return float(_logsumexp_rows(_log_comp(x, pi, mu, var)).sum())

def em_gmm_1d(x, K, means, n_iter=100):
    x = np.asarray(x, float)
    pi = np.full(K, 1 / K)
    mu = np.array(means, float)
    var = np.full(K, x.var())
    for _ in range(n_iter):
        L = _log_comp(x, pi, mu, var)
        R = np.exp(L - _logsumexp_rows(L)[:, None])          # responsibilities (n, K)
        Nc = R.sum(axis=0)
        pi = Nc / len(x)
        mu = (R * x[:, None]).sum(axis=0) / Nc
        var = (R * (x[:, None] - mu) ** 2).sum(axis=0) / Nc + 1e-6
    return pi, mu, var
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2, 0.5, 300), rng.normal(3, 1.0, 700)])

def test_recovers_mixture():
    """finds weights 0.3/0.7, means -2/3 and variances 0.25/1"""
    pi, mu, var = em_gmm_1d(x, 2, [-1.0, 1.0])
    o = np.argsort(mu)
    assert approx(pi[o], [0.3, 0.7], tol=0.03)
    assert approx(mu[o], [-2.0, 3.0], tol=0.1)
    assert approx(var[o], [0.25, 1.0], tol=0.15)

def test_log_likelihood_value():
    """log_likelihood matches a direct computation"""
    pi, mu, var = np.array([0.4, 0.6]), np.array([0.0, 2.0]), np.array([1.0, 0.5])
    xs = np.array([-0.5, 1.0, 2.5])
    dens = sum(pi[c] * np.exp(-(xs - mu[c]) ** 2 / (2 * var[c])) / np.sqrt(2 * np.pi * var[c]) for c in range(2))
    assert approx(log_likelihood(xs, pi, mu, var), np.sum(np.log(dens)))

def test_likelihood_never_decreases():
    """more EM iterations never lower the log-likelihood"""
    ll = [log_likelihood(x, *em_gmm_1d(x, 2, [0.0, 0.5], n_iter=k)) for k in (1, 3, 10, 50)]
    assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:]))

def test_weights_sum_to_one():
    """mixing weights form a distribution"""
    pi, _, _ = em_gmm_1d(x, 3, [-2.0, 0.0, 3.0], n_iter=20)
    assert approx(pi.sum(), 1.0) and np.all(pi > 0)
~~~

<!-- hint: Build an `(n, K)` matrix of `log pi_c + log N(x_i | mu_c, var_c)`, normalise each row with log-sum-exp, and exponentiate to get the responsibilities. -->
<!-- hint: With `Nc = R.sum(0)`, the new mean is `(R * x[:, None]).sum(0) / Nc`. -->
<!-- /exercise -->

<!-- quiz -->
1. A point lies exactly halfway between two identical, equally weighted components. What are its responsibilities?
   - [ ] 1 for one component, 0 for the other
   - [x] 0.5 and 0.5
   - [ ] It cannot be computed
   > Both components explain the point equally well and have equal priors, so Bayes' rule splits it evenly.
2. Why does EM need several random restarts?
   - [ ] Each iteration can lower the likelihood
   - [x] It converges to a local maximum that depends on the starting parameters
   - [ ] The E-step is random
   > EM is guaranteed not to decrease the likelihood, but not to find the global maximum.
3. During EM the likelihood suddenly becomes enormous and one variance is almost zero. What happened?
   - [x] A component collapsed onto a single point; add a small value to the variances
   - [ ] The model found the true clusters
   - [ ] The learning rate is too high
   > A Gaussian centred on one point with variance going to zero has unbounded density there, an artefact a variance floor prevents.
4. How does a GMM relate to k-means?
   - [ ] They are unrelated
   - [x] k-means is like a GMM with equal spherical covariances and hard assignments
   - [ ] A GMM is k-means with $k = 1$
   > Shrinking equal spherical variances to zero turns the soft responsibilities into hard nearest-centre assignments.
<!-- /quiz -->
