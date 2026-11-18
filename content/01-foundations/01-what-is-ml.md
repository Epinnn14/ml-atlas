---
title: What machine learning is
level: beginner
minutes: 20
summary: Learning a function from examples, the four ingredients every method shares, and why we always test on data the model never saw.
---

## The idea

A normal program is a rule written by a person: *if the email contains "free money", mark it as spam*. Machine learning turns this around. We collect examples of inputs together with the answers we want, and let an algorithm find the rule.

Formally, we have a dataset of $n$ examples $\{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$. Each $\mathbf{x}_i \in \mathbb{R}^d$ is a vector of **features** (the size of a house, its age, its distance to the city) and $y_i$ is the **target** (its price). We want a function $f$ such that $f(\mathbf{x}) \approx y$ for inputs we have **not** seen yet.

That last part is the whole game. Memorising the training examples is easy; predicting new ones well is called **generalisation**.

## Three kinds of learning

| Kind | What you have | What you learn | Example |
|---|---|---|---|
| Supervised | inputs and the right answers | a mapping $\mathbf{x} \mapsto y$ | price from house features, spam or not |
| Unsupervised | inputs only | structure: groups, directions, density | customer segments, compressing images |
| Reinforcement | an environment that gives rewards | a policy: which action to take | game playing, robot control |

Supervised learning splits again by the kind of target: **regression** when $y$ is a number, **classification** when $y$ is one of a few classes.

## Four ingredients

Almost every method in this atlas is the same four pieces put together differently:

1. **Data**: the examples, split into a training part and a test part.
2. **Model**: a family of functions $f_{\boldsymbol\theta}$ with adjustable **parameters** $\boldsymbol\theta$. A line $f(x) = wx + b$ has parameters $w$ and $b$.
3. **Loss**: a number that says how wrong the model is on the data. For regression the usual one is the mean squared error.
4. **Optimiser**: a procedure that changes $\boldsymbol\theta$ to make the loss smaller.

<!-- formula: Mean squared error -->
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$$

Here $\hat{y}_i = f_{\boldsymbol\theta}(\mathbf{x}_i)$ is the prediction. Squaring makes every error positive and punishes big errors much more than small ones.

Training means solving

<!-- formula: Empirical risk minimisation -->
$$\hat{\boldsymbol\theta} = \arg\min_{\boldsymbol\theta} \; \frac{1}{n}\sum_{i=1}^{n} L\big(y_i, f_{\boldsymbol\theta}(\mathbf{x}_i)\big)$$

which is called **empirical risk minimisation**: we minimise the average loss on the data we have, hoping it is close to the loss on all the data we will ever see.

## Always keep a test set

If you judge a model on the same examples it learned from, a model that memorised everything looks perfect. So before training, put part of the data aside, typically 20%. Train on the rest. Report the score on the held-out part only.

Two habits that matter from day one:

- **Shuffle before splitting**, unless the order means something (time series, see [time series](#/lesson/time-series)). Data files are often sorted, and the last 20% of a sorted file is not a fair sample.
- **Compare with a baseline.** Predicting the mean of the training targets for everyone is the simplest possible regressor. A model that cannot beat it has learned nothing useful.

<!-- exercise: mse | Mean squared error and the mean baseline -->
Write `mse(y, y_hat)` that returns the mean squared error as a Python float, and `mean_baseline(y_train, n)` that returns an array of length `n` filled with the mean of `y_train`.

Use NumPy and no loops.

~~~python starter
import numpy as np

def mse(y, y_hat):
    # mean of the squared differences
    raise NotImplementedError

def mean_baseline(y_train, n):
    # the same prediction, the training mean, for n examples
    raise NotImplementedError

print(mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])))  # expect 1.333...
~~~

~~~python solution
import numpy as np

def mse(y, y_hat):
    y, y_hat = np.asarray(y, dtype=float), np.asarray(y_hat, dtype=float)
    return float(np.mean((y - y_hat) ** 2))

def mean_baseline(y_train, n):
    return np.full(n, np.mean(y_train), dtype=float)

print(mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])))
~~~

~~~python tests
import numpy as np

def test_small_example():
    """mse([1,2,3], [1,2,5]) is 4/3"""
    assert approx(mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])), 4 / 3)

def test_perfect():
    """a perfect prediction has error 0"""
    y = np.arange(10.0)
    assert mse(y, y) == 0.0

def test_returns_float():
    """mse returns a plain float"""
    assert isinstance(mse(np.ones(3), np.zeros(3)), float)

def test_baseline():
    """the baseline repeats the training mean"""
    b = mean_baseline(np.array([2.0, 4.0, 9.0]), 4)
    assert b.shape == (4,), f"shape is {np.shape(b)}, expected (4,)"
    assert approx(b, [5.0, 5.0, 5.0, 5.0])

def test_baseline_is_hard_to_beat_by_a_constant():
    """no other constant has a lower training MSE than the mean"""
    y = np.random.default_rng(0).normal(3, 2, 200)
    best = mse(y, mean_baseline(y, len(y)))
    for c in (2.5, 2.9, 3.1, 3.5):
        assert best <= mse(y, np.full(len(y), c)) + 1e-12
~~~

<!-- hint: `np.mean((y - y_hat) ** 2)` gives a NumPy scalar; wrap it in `float(...)`. -->
<!-- hint: `np.full(n, value)` builds an array of length n filled with value. -->
<!-- /exercise -->

<!-- exercise: split | A shuffled train/test split -->
Write `train_test_split(X, y, test_frac, seed)` that shuffles the rows with `np.random.default_rng(seed).permutation`, then puts the first `round(test_frac * n)` shuffled rows in the test set and the rest in the training set. Return `X_train, X_test, y_train, y_test`. Rows of `X` and `y` must stay paired.

~~~python starter
import numpy as np

def train_test_split(X, y, test_frac=0.2, seed=0):
    n = len(X)
    # 1. a random order of the row indices
    # 2. the first part of that order is the test set
    raise NotImplementedError

X = np.arange(20).reshape(10, 2)
y = np.arange(10)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.3, seed=1)
print(X_te, y_te)
~~~

~~~python solution
import numpy as np

def train_test_split(X, y, test_frac=0.2, seed=0):
    n = len(X)
    order = np.random.default_rng(seed).permutation(n)
    n_test = int(round(test_frac * n))
    test, train = order[:n_test], order[n_test:]
    return X[train], X[test], y[train], y[test]

X = np.arange(20).reshape(10, 2)
y = np.arange(10)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.3, seed=1)
print(X_te, y_te)
~~~

~~~python tests
import numpy as np

X = np.arange(40).reshape(20, 2)
y = np.arange(20)

def test_sizes():
    """20 rows with test_frac=0.25 give 15 train and 5 test rows"""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.25, seed=0)
    assert (len(X_tr), len(X_te), len(y_tr), len(y_te)) == (15, 5, 15, 5)

def test_pairs_kept():
    """each row of X still sits next to its own y"""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, 0.25, seed=3)
    assert np.all(X_tr[:, 0] == 2 * y_tr) and np.all(X_te[:, 0] == 2 * y_te)

def test_no_overlap():
    """train and test together hold every row exactly once"""
    _, _, y_tr, y_te = train_test_split(X, y, 0.25, seed=3)
    assert sorted(np.concatenate([y_tr, y_te]).tolist()) == list(range(20))

def test_shuffled_and_reproducible():
    """the split is shuffled, and the same seed gives the same split"""
    a = train_test_split(X, y, 0.25, seed=7)[3]
    b = train_test_split(X, y, 0.25, seed=7)[3]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, y[:5]), "the test set is just the first rows: shuffle first"

def test_uses_the_given_rng():
    """the test rows are the first ones of default_rng(seed).permutation(n)"""
    order = np.random.default_rng(11).permutation(20)
    assert np.array_equal(train_test_split(X, y, 0.25, seed=11)[3], y[order[:5]])
~~~

<!-- hint: `order = np.random.default_rng(seed).permutation(n)` gives the shuffled indices. -->
<!-- hint: Index both arrays with the same index arrays: `X[train], X[test], y[train], y[test]`. -->
<!-- /exercise -->

<!-- quiz -->
1. A model scores 99% accuracy on the data it was trained on and 61% on a held-out test set. What is the most likely explanation?
   - [ ] The test set is too small to trust
   - [x] The model memorised the training data instead of learning a pattern that generalises
   - [ ] The model needs more parameters
   > A large gap between training and test scores is the signature of overfitting. More parameters usually make it worse, not better.
2. You want to group customers by their buying habits and nobody has labelled the groups. Which kind of learning is this?
   - [ ] Supervised classification
   - [ ] Reinforcement learning
   - [x] Unsupervised learning
   > There are no targets $y$, only inputs. Finding groups in unlabelled data is clustering, a form of unsupervised learning.
3. Why does the mean squared error square the differences instead of just adding them up?
   - [ ] Squaring makes the computation faster
   - [x] So positive and negative errors do not cancel, and big errors count more
   - [ ] So the loss is always between 0 and 1
   > Raw differences can cancel to zero even for a terrible model. Squaring makes every term positive and weights a large error much more than several small ones.
4. A data file is sorted by date and you split off the last 20% of rows as a test set without shuffling. For an ordinary (not time series) problem, what goes wrong?
   - [x] The test set comes from a different period than the training set, so the score is biased
   - [ ] Nothing, the order of rows never matters
   - [ ] The model will train more slowly
   > Sorted data puts all the late examples in the test set. Unless you are forecasting the future on purpose, shuffle first so both parts are fair samples.
<!-- /quiz -->
