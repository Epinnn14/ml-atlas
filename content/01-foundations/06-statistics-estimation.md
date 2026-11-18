---
title: Estimation, MLE and MAP
level: beginner
minutes: 40
summary: How to guess the parameters of a distribution from data, why maximum likelihood is the principle behind most losses, how a prior turns it into MAP, and the bootstrap for error bars.
---

## Estimators

We observe data $x_1, \dots, x_n$ drawn from a distribution with an unknown parameter $\theta$ (say the mean). An **estimator** $\hat\theta$ is any recipe that turns the data into a guess. Because the data are random, $\hat\theta$ is random too, and two properties describe how good it is:

<!-- formula: Bias and variance of an estimator -->
$$\operatorname{Bias}(\hat\theta) = \mathbb E[\hat\theta] - \theta, \qquad \operatorname{MSE}(\hat\theta) = \operatorname{Bias}(\hat\theta)^2 + \operatorname{Var}(\hat\theta)$$

An estimator can be slightly biased and still better overall if it has much lower variance. This trade-off returns, in a different form, in [bias and variance of models](#/lesson/bias-variance).

## Maximum likelihood

The **likelihood** is the probability of the observed data as a function of the parameter, $L(\theta) = \prod_i p(x_i\mid\theta)$ for independent data. Maximum likelihood estimation (MLE) picks the parameter that makes the data most probable. In practice we maximise the log:

<!-- formula: Maximum likelihood -->
$$\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

Two results that follow by setting the derivative to zero:

- Coin flips with $k$ heads in $n$ throws: $\hat p = k / n$.
- Gaussian data: the sample mean and the average squared deviation.

<!-- formula: Gaussian MLE -->
$$\hat\mu = \frac1n\sum_i x_i, \qquad \hat\sigma^2 = \frac1n\sum_i (x_i - \hat\mu)^2$$

The variance estimate divides by $n$ and is slightly biased low: it measures spread around the sample mean, which is always closer to the data than the true mean. Dividing by $n-1$ (Bessel's correction) removes the bias. NumPy's `np.var` divides by $n$ unless you pass `ddof=1`.

**Why this matters for ML:** minimising mean squared error is MLE under Gaussian noise, and minimising cross-entropy is MLE for Bernoulli or categorical labels. Most losses in this atlas are negative log-likelihoods in disguise.

## MAP: adding a prior

With little data MLE can be extreme: three heads in three throws gives $\hat p = 1$. If we have a prior belief $p(\theta)$, Bayes' rule gives a posterior, and its peak is the **maximum a posteriori** estimate:

<!-- formula: MAP estimate -->
$$\hat\theta_{\text{MAP}} = \arg\max_\theta \Big[\sum_i \log p(x_i\mid\theta) + \log p(\theta)\Big]$$

For a coin with a Beta$(\alpha, \beta)$ prior, which acts like $\alpha - 1$ imagined heads and $\beta - 1$ imagined tails:

<!-- formula: Beta-Bernoulli MAP -->
$$\hat p_{\text{MAP}} = \frac{k + \alpha - 1}{n + \alpha + \beta - 2}$$

The log-prior term is a penalty added to the loss. A Gaussian prior on weights gives exactly the L2 penalty of [ridge regression](#/lesson/regularization).

## The bootstrap

How uncertain is an estimate? If you could collect many fresh datasets you would see how much $\hat\theta$ varies. The bootstrap fakes this with the one dataset you have: draw $n$ points **with replacement**, recompute the estimate, repeat a few thousand times, and read off the spread. The 2.5% and 97.5% percentiles of the bootstrap estimates give an approximate 95% confidence interval. It works for almost any statistic, including ones with no formula, such as a median or an F1 score.

<!-- exercise: gaussian-mle | Fit a Gaussian and a coin -->
Write `gaussian_mle(x)` returning `(mu, sigma2)` with the MLE formulas (divide by $n$), and `coin_map(k, n, alpha, beta)` returning the Beta-Bernoulli MAP estimate.

~~~python starter
import numpy as np

def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    raise NotImplementedError

def coin_map(k, n, alpha, beta):
    raise NotImplementedError

print(gaussian_mle([2.0, 4.0, 6.0]))   # (4.0, 2.666...)
print(coin_map(3, 3, 2, 2))            # 0.8, not 1.0
~~~

~~~python solution
import numpy as np

def gaussian_mle(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))

def coin_map(k, n, alpha, beta):
    return (k + alpha - 1) / (n + alpha + beta - 2)

print(gaussian_mle([2.0, 4.0, 6.0]))
print(coin_map(3, 3, 2, 2))
~~~

~~~python tests
import numpy as np

def test_small():
    """[2, 4, 6] gives mean 4 and variance 8/3"""
    mu, s2 = gaussian_mle([2.0, 4.0, 6.0])
    assert approx(mu, 4.0) and approx(s2, 8 / 3)

def test_divides_by_n():
    """the MLE variance divides by n (it equals np.var with ddof=0)"""
    x = np.random.default_rng(0).normal(3, 2, 25)
    assert approx(gaussian_mle(x)[1], np.var(x))

def test_large_sample():
    """recovers the true parameters from a large sample"""
    x = np.random.default_rng(1).normal(-1.5, 0.5, 200_000)
    mu, s2 = gaussian_mle(x)
    assert abs(mu + 1.5) < 0.01 and abs(s2 - 0.25) < 0.01

def test_coin_map():
    """a Beta(2, 2) prior pulls 3 of 3 heads to 0.8"""
    assert approx(coin_map(3, 3, 2, 2), 0.8)

def test_uniform_prior_is_mle():
    """with Beta(1, 1), MAP equals the MLE k / n"""
    assert approx(coin_map(7, 10, 1, 1), 0.7)
~~~

<!-- /exercise -->

<!-- exercise: bootstrap | A bootstrap confidence interval -->
Write `bootstrap_ci(x, stat, n_boot=2000, level=0.95, seed=0)`. Use `rng = np.random.default_rng(seed)` and, `n_boot` times, draw indices with `rng.integers(0, n, size=n)`, compute `stat` on the resampled data and store it. Return the lower and upper percentiles `(100 - 100*level)/2` and `100 - that` of the stored values, as two floats.

~~~python starter
import numpy as np

def bootstrap_ci(x, stat, n_boot=2000, level=0.95, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    raise NotImplementedError

x = np.random.default_rng(42).exponential(2.0, 80)
print(bootstrap_ci(x, np.median))
~~~

~~~python solution
import numpy as np

def bootstrap_ci(x, stat, n_boot=2000, level=0.95, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = stat(x[rng.integers(0, n, size=n)])
    tail = (100 - 100 * level) / 2
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)

x = np.random.default_rng(42).exponential(2.0, 80)
print(bootstrap_ci(x, np.median))
~~~

~~~python tests
import numpy as np

x = np.random.default_rng(42).normal(10, 3, 100)

def test_contains_estimate():
    """the interval contains the sample mean"""
    lo, hi = bootstrap_ci(x, np.mean)
    assert lo < x.mean() < hi

def test_width_close_to_theory():
    """for the mean, the 95% width is close to 2 * 1.96 * s / sqrt(n)"""
    lo, hi = bootstrap_ci(x, np.mean)
    theory = 2 * 1.96 * x.std() / np.sqrt(len(x))
    assert abs((hi - lo) - theory) / theory < 0.15

def test_level():
    """a 50% interval is narrower than a 95% one"""
    a = bootstrap_ci(x, np.mean, level=0.5)
    b = bootstrap_ci(x, np.mean, level=0.95)
    assert a[1] - a[0] < b[1] - b[0]

def test_reproducible():
    """the same seed gives the same interval"""
    assert bootstrap_ci(x, np.median, seed=3) == bootstrap_ci(x, np.median, seed=3)

def test_follows_recipe():
    """uses rng.integers(0, n, size=n) and np.percentile as described"""
    rng = np.random.default_rng(5)
    s = [np.max(x[rng.integers(0, 100, size=100)]) for _ in range(300)]
    lo, hi = np.percentile(s, [5, 95])
    assert approx(bootstrap_ci(x, np.max, n_boot=300, level=0.9, seed=5), [lo, hi])
~~~

<!-- hint: Store each statistic in an array, then call `np.percentile(stats, [2.5, 97.5])` for level 0.95. -->
<!-- /exercise -->

<!-- quiz -->
1. You flip a coin 4 times and get 4 heads. What is the maximum likelihood estimate of $P(\text{heads})$?
   - [x] 1.0
   - [ ] 0.5
   - [ ] 0.8
   > MLE is $k/n = 4/4$. This overconfident answer is why a prior (MAP) helps with small samples.
2. Minimising mean squared error in linear regression is equivalent to maximum likelihood under which assumption?
   - [ ] The features are Gaussian
   - [x] The noise on the targets is Gaussian with constant variance
   - [ ] The weights have a Gaussian prior
   > The Gaussian log-density contains $-(y - \hat y)^2/(2\sigma^2)$, so maximising the likelihood minimises the squared error. A Gaussian prior on the weights would add a penalty (ridge).
3. Why does the bootstrap sample *with* replacement?
   - [ ] To make the computation faster
   - [x] Sampling n points without replacement would just reproduce the original dataset every time
   - [ ] Because the data must be independent
   > Drawing with replacement creates datasets that differ from the original, mimicking the variation of fresh samples.
4. An estimator is unbiased but has a huge variance. Another has a small bias and a tiny variance. Which has the lower mean squared error?
   - [ ] Always the unbiased one
   - [x] It depends, and often the slightly biased one wins, since MSE is bias squared plus variance
   - [ ] They have the same MSE
   > Accepting a little bias for a large drop in variance is the reason shrinkage and regularisation work.
<!-- /quiz -->
