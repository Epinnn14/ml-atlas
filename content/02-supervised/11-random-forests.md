---
title: Bagging and random forests
level: intermediate
minutes: 40
summary: Average many unstable trees to cancel their mistakes. Bootstrap samples, random feature subsets, out-of-bag error, and the formula that says why decorrelating the trees matters.
---

## Averaging reduces variance

A single deep tree has low bias and high variance: train it on a slightly different sample and you get a different tree. Averaging many such models keeps the low bias and shrinks the variance. If $B$ models each have variance $\sigma^2$ and every pair has correlation $\rho$, their average has variance

<!-- formula: Variance of an average of correlated models -->
$$\operatorname{Var}\Big[\frac1B\sum_{b=1}^B f_b\Big] = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2$$

Adding more models removes the second term, but the first term $\rho\sigma^2$ stays. So averaging helps only as much as the models are **different from each other**. Both tricks below exist to lower $\rho$.

## Bagging

**Bootstrap aggregating** trains each tree on a [bootstrap sample](#/lesson/statistics-estimation): $n$ examples drawn with replacement. Each sample contains about 63% of the distinct examples, because

<!-- formula: Probability an example is left out of a bootstrap sample -->
$$\Big(1 - \frac1n\Big)^n \;\xrightarrow{\;n\to\infty\;}\; e^{-1} \approx 0.368$$

For classification, the ensemble takes a majority vote (or averages class probabilities); for regression, it averages predictions.

## Random forests

Trees grown on bootstrap samples are still correlated: if one feature is very strong, every tree splits on it first. A **random forest** adds a second source of randomness: at **each split**, only a random subset of $m$ features is considered, typically $m \approx \sqrt d$ for classification and $d/3$ for regression. Weaker features get their turn, the trees become more different, $\rho$ drops, and the average improves.

Practical properties:

- Few hyperparameters, and the defaults work well. More trees never overfits, it just costs time; a few hundred is usually enough.
- Trees are grown deep and not pruned: the averaging handles the variance.
- Works on raw, unscaled, mixed-type features. A standard strong baseline for tabular data.

## Out-of-bag error

Each tree never saw the roughly 37% of examples left out of its bootstrap sample. Predicting each example with only the trees that did not see it gives the **out-of-bag (OOB)** estimate of test error, almost for free and without a separate validation set.

## Feature importance

Forests report how much each feature reduced impurity, summed over all splits. This **impurity importance** is biased towards features with many distinct values. **Permutation importance**, the drop in validation score when one feature's column is shuffled, is more reliable ([interpretability](#/lesson/interpretability)).

<!-- exercise: bootstrap-oob | Bootstrap samples and out-of-bag sets -->
Write `bootstrap_with_oob(n, rng)` that draws `idx = rng.integers(0, n, size=n)` and returns `(idx, oob)`, where `oob` is the sorted array of indices in `range(n)` that do not appear in `idx`.

~~~python starter
import numpy as np

def bootstrap_with_oob(n, rng):
    raise NotImplementedError

idx, oob = bootstrap_with_oob(10, np.random.default_rng(0))
print(idx, oob)
~~~

~~~python solution
import numpy as np

def bootstrap_with_oob(n, rng):
    idx = rng.integers(0, n, size=n)
    oob = np.setdiff1d(np.arange(n), idx)
    return idx, oob

idx, oob = bootstrap_with_oob(10, np.random.default_rng(0))
print(idx, oob)
~~~

~~~python tests
import numpy as np

def test_recipe():
    """uses rng.integers(0, n, size=n)"""
    idx, _ = bootstrap_with_oob(20, np.random.default_rng(5))
    assert np.array_equal(idx, np.random.default_rng(5).integers(0, 20, size=20))

def test_oob_disjoint_and_complete():
    """oob holds exactly the indices that were not drawn"""
    idx, oob = bootstrap_with_oob(50, np.random.default_rng(1))
    assert set(oob) | set(idx) == set(range(50)) and not set(oob) & set(idx)
    assert np.all(np.diff(oob) > 0)

def test_about_37_percent_out():
    """on average about 36.8% of examples are out of bag"""
    rng = np.random.default_rng(2)
    frac = np.mean([len(bootstrap_with_oob(1000, rng)[1]) / 1000 for _ in range(200)])
    assert abs(frac - np.exp(-1)) < 0.01
~~~

<!-- hint: `np.setdiff1d(np.arange(n), idx)` returns the sorted values of the first array missing from the second. -->
<!-- /exercise -->

<!-- exercise: forest | A forest of random stumps -->
A **stump** is a tree with one split. Complete `fit_forest(X, y, n_trees, n_features, seed)`: with `rng = np.random.default_rng(seed)`, for each tree draw a bootstrap sample (`rng.integers(0, n, size=n)`), then a random subset of `n_features` feature indices (`rng.choice(d, n_features, replace=False)`), and fit a stump on that sample restricted to those features with the provided `fit_stump`. Store the stumps in a list. `predict_forest(forest, X)` returns the majority vote of the stumps (labels are 0/1; break ties towards 1, so predict 1 when the mean vote is at least 0.5).

~~~python starter
import numpy as np

def fit_stump(X, y, features):
    """Best single split (Gini) using only the given feature indices."""
    best = None
    for j in features:
        for t in np.unique(X[:, j])[:-1]:
            l = X[:, j] <= t
            score = sum(len(part) * (1 - np.mean(part) ** 2 - (1 - np.mean(part)) ** 2) for part in (y[l], y[~l]))
            if best is None or score < best[0]:
                best = (score, j, t, int(round(np.mean(y[l]))), int(round(np.mean(y[~l]))))
    if best is None:
        m = int(round(np.mean(y)))
        return (features[0], np.inf, m, m)
    return best[1:]

def predict_stump(stump, X):
    j, t, left, right = stump
    return np.where(X[:, j] <= t, left, right)

def fit_forest(X, y, n_trees=50, n_features=1, seed=0):
    raise NotImplementedError

def predict_forest(forest, X):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def fit_stump(X, y, features):
    """Best single split (Gini) using only the given feature indices."""
    best = None
    for j in features:
        for t in np.unique(X[:, j])[:-1]:
            l = X[:, j] <= t
            score = sum(len(part) * (1 - np.mean(part) ** 2 - (1 - np.mean(part)) ** 2) for part in (y[l], y[~l]))
            if best is None or score < best[0]:
                best = (score, j, t, int(round(np.mean(y[l]))), int(round(np.mean(y[~l]))))
    if best is None:
        m = int(round(np.mean(y)))
        return (features[0], np.inf, m, m)
    return best[1:]

def predict_stump(stump, X):
    j, t, left, right = stump
    return np.where(X[:, j] <= t, left, right)

def fit_forest(X, y, n_trees=50, n_features=1, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    forest = []
    for _ in range(n_trees):
        idx = rng.integers(0, n, size=n)
        feats = rng.choice(d, n_features, replace=False)
        forest.append(fit_stump(X[idx], y[idx], feats))
    return forest

def predict_forest(forest, X):
    votes = np.mean([predict_stump(s, X) for s in forest], axis=0)
    return (votes >= 0.5).astype(int)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 3))
y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)   # needs all three features
Xte, yte = X[300:], y[300:]
Xtr, ytr = X[:300], y[:300]

def test_forest_size():
    """builds n_trees stumps"""
    assert len(fit_forest(Xtr, ytr, n_trees=7, seed=1)) == 7

def test_beats_one_stump():
    """the forest of random stumps beats the single best stump"""
    single = fit_stump(Xtr, ytr, [0, 1, 2])
    acc_single = np.mean(predict_stump(single, Xte) == yte)
    acc_forest = np.mean(predict_forest(fit_forest(Xtr, ytr, 101, 1, seed=0), Xte) == yte)
    assert acc_forest > acc_single + 0.05

def test_uses_all_features():
    """random feature subsets let every feature be used"""
    used = {s[0] for s in fit_forest(Xtr, ytr, 60, 1, seed=2)}
    assert used == {0, 1, 2}

def test_recipe():
    """follows the order: bootstrap draw, then feature choice, per tree"""
    r = np.random.default_rng(9)
    idx = r.integers(0, 300, size=300); feats = r.choice(3, 2, replace=False)
    assert fit_forest(Xtr, ytr, 1, 2, seed=9)[0] == fit_stump(Xtr[idx], ytr[idx], feats)
~~~

<!-- hint: Inside the loop: `idx = rng.integers(0, n, size=n)`, then `feats = rng.choice(d, n_features, replace=False)`, then `fit_stump(X[idx], y[idx], feats)`. -->
<!-- hint: Stack the predictions of all stumps and average them along axis 0 to get the vote share for class 1. -->
<!-- /exercise -->

<!-- quiz -->
1. Why does a random forest consider only a random subset of features at each split?
   - [ ] To make each tree more accurate
   - [x] To make the trees less correlated, so averaging them removes more variance
   - [ ] To reduce memory use
   > The $\rho\sigma^2$ term in the variance of an average does not shrink with more trees; only decorrelation lowers it.
2. Can adding more trees to a random forest cause overfitting?
   - [ ] Yes, like adding layers to a network
   - [x] No, the error levels off as trees are added; it only costs computation
   - [ ] Only when trees are deep
   > More trees just make the average closer to its limit. Overfitting is controlled by the individual trees and the data.
3. What is the out-of-bag error?
   - [ ] The training error of the forest
   - [x] The error on each example using only trees whose bootstrap sample did not contain it
   - [ ] The error on a separate held-out set
   > Every tree leaves out about 37% of the data, so each example has a set of trees for which it is unseen.
4. Bagging helps a deep tree a lot but a linear regression very little. Why?
   - [x] Linear regression is a low-variance model, and averaging mainly removes variance
   - [ ] Linear models cannot be trained on bootstrap samples
   - [ ] Averaging linear models gives a non-linear model
   > Stable models change little between bootstrap samples, so their average is almost the same model.
<!-- /quiz -->
