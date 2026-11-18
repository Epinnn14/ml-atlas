---
title: Cross-validation and model selection
level: intermediate
minutes: 35
summary: Use every example for both training and validation with k-fold cross-validation, tune hyperparameters without fooling yourself, and know when ordinary folds are wrong.
---

## The problem with one validation split

A single validation set of 100 examples gives a noisy score: another random split could rank your models differently. Holding out more data for validation leaves less for training. **k-fold cross-validation** solves both problems.

## k-fold cross-validation

1. Shuffle the data and cut it into $k$ equal **folds** (typically $k = 5$ or $10$).
2. For each fold $j$: train on the other $k - 1$ folds, evaluate on fold $j$.
3. Report the mean of the $k$ scores, and their standard deviation as a measure of how stable it is.

<!-- formula: Cross-validation estimate -->
$$\text{CV}(k) = \frac1k\sum_{j=1}^{k} \text{score}\big(\text{model trained without fold } j,\ \text{fold } j\big)$$

Every example is used for validation exactly once and for training $k - 1$ times. The cost is training $k$ models. With $k = n$ it is **leave-one-out**: almost no bias, but expensive and high-variance.

## Tuning hyperparameters

Hyperparameters ($\lambda$ in [ridge](#/lesson/regularization), $k$ in [k-NN](#/lesson/knn), tree depth) are chosen by computing the CV score for each candidate and keeping the best, a **grid search**. Then retrain on all the training data with the chosen value.

A subtle trap: the best CV score of many candidates is optimistic, because you picked the luckiest one. The final, honest number must come from a test set that took no part in the search. If you also need an honest estimate of the whole tuning procedure with little data, use **nested cross-validation**: an inner CV loop chooses the hyperparameter, an outer loop measures how well that choice generalises.

## Every step inside the fold

Anything learned from data belongs inside the loop: scaling statistics, imputation values, feature selection, oversampling. Fit them on the training folds only and apply them to the validation fold. Selecting the "best 20 features" on the whole dataset and then cross-validating gives wildly optimistic scores, even on pure noise.

## When ordinary folds are wrong

Random folds assume examples are independent. They are not when:

- **Groups**: several rows per patient, customer or photo session. Rows of one group in both training and validation leak identity. Use **group k-fold**: whole groups go to one fold.
- **Time**: tomorrow cannot be used to predict today. Use **forward-chaining** splits: train on everything before a cut-off, validate on the period after, and move the cut-off forward ([time series](#/lesson/time-series)).
- **Imbalance**: a fold may end up with almost no positives. **Stratified** folds keep the class proportions the same in every fold.

<!-- exercise: kfold | k-fold indices -->
Write `kfold_indices(n, k, seed=0)` that shuffles `np.arange(n)` with `np.random.default_rng(seed).permutation(n)`, splits the shuffled order into `k` nearly equal folds with `np.array_split`, and returns a list of `k` pairs `(train_idx, val_idx)`, one per fold.

~~~python starter
import numpy as np

def kfold_indices(n, k, seed=0):
    raise NotImplementedError

for tr, va in kfold_indices(10, 3, seed=1):
    print(sorted(va))
~~~

~~~python solution
import numpy as np

def kfold_indices(n, k, seed=0):
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, k)
    pairs = []
    for j in range(k):
        train = np.concatenate([folds[i] for i in range(k) if i != j])
        pairs.append((train, folds[j]))
    return pairs

for tr, va in kfold_indices(10, 3, seed=1):
    print(sorted(va))
~~~

~~~python tests
import numpy as np

def test_count():
    """returns k pairs"""
    assert len(kfold_indices(20, 4)) == 4

def test_validation_partition():
    """validation folds cover every index exactly once"""
    vals = np.concatenate([va for _, va in kfold_indices(23, 5, seed=2)])
    assert sorted(vals.tolist()) == list(range(23))

def test_disjoint():
    """train and validation never overlap and together hold all indices"""
    for tr, va in kfold_indices(23, 5, seed=2):
        assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 23

def test_sizes_balanced():
    """fold sizes differ by at most one"""
    sizes = [len(va) for _, va in kfold_indices(23, 5)]
    assert max(sizes) - min(sizes) <= 1

def test_shuffled():
    """the first fold is not simply the first rows"""
    _, va = kfold_indices(100, 5, seed=0)[0]
    assert sorted(va.tolist()) != list(range(20))
~~~

<!-- /exercise -->

<!-- exercise: cv-ridge | Choose lambda for ridge by cross-validation -->
Write `cv_mse(X, y, lam, k=5, seed=0)` that returns the mean validation MSE of ridge regression over k folds, and `choose_lambda(X, y, lams)` that returns the value in `lams` with the lowest CV MSE. `ridge` and `kfold_indices` are provided. Standardisation is not needed here.

~~~python starter
import numpy as np

def ridge(X, y, lam):
    xm, ym = X.mean(0), y.mean()
    w = np.linalg.solve((X - xm).T @ (X - xm) + lam * np.eye(X.shape[1]), (X - xm).T @ (y - ym))
    return w, ym - xm @ w

def kfold_indices(n, k, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate(folds[:j] + folds[j + 1:]), folds[j]) for j in range(k)]

def cv_mse(X, y, lam, k=5, seed=0):
    raise NotImplementedError

def choose_lambda(X, y, lams):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ridge(X, y, lam):
    xm, ym = X.mean(0), y.mean()
    w = np.linalg.solve((X - xm).T @ (X - xm) + lam * np.eye(X.shape[1]), (X - xm).T @ (y - ym))
    return w, ym - xm @ w

def kfold_indices(n, k, seed=0):
    folds = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate(folds[:j] + folds[j + 1:]), folds[j]) for j in range(k)]

def cv_mse(X, y, lam, k=5, seed=0):
    errs = []
    for tr, va in kfold_indices(len(y), k, seed):
        w, b = ridge(X[tr], y[tr], lam)
        errs.append(np.mean((X[va] @ w + b - y[va]) ** 2))
    return float(np.mean(errs))

def choose_lambda(X, y, lams):
    return lams[int(np.argmin([cv_mse(X, y, lam) for lam in lams]))]
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n, d = 40, 30                    # few examples, many features: needs regularisation
X = rng.normal(size=(n, d))
w_true = np.zeros(d); w_true[:5] = [3, -2, 2, 1, -1]
y = X @ w_true + rng.normal(size=n)
lams = [1e-4, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4]

def test_returns_float():
    """cv_mse returns a non-negative float"""
    v = cv_mse(X, y, 1.0)
    assert isinstance(v, float) and v >= 0

def test_matches_manual_folds():
    """averages the validation MSE over the provided folds"""
    errs = []
    for tr, va in kfold_indices(n, 5, 0):
        w, b = ridge(X[tr], y[tr], 3.0)
        errs.append(np.mean((X[va] @ w + b - y[va]) ** 2))
    assert approx(cv_mse(X, y, 3.0), np.mean(errs))

def test_extremes_are_worse():
    """both tiny and huge lambdas have worse CV error than the chosen one"""
    best = choose_lambda(X, y, lams)
    assert best not in (lams[0], lams[-1])
    assert cv_mse(X, y, best) < cv_mse(X, y, lams[0]) and cv_mse(X, y, best) < cv_mse(X, y, lams[-1])
~~~

<!-- hint: Loop over `kfold_indices(len(y), k, seed)`, fit on `X[tr], y[tr]`, score on `X[va], y[va]`. -->
<!-- /exercise -->

<!-- quiz -->
1. You select the 50 most correlated features using all 1000 rows, then run 5-fold CV on a model using them. What is wrong?
   - [x] Feature selection saw the validation folds, so the CV score is optimistically biased
   - [ ] Nothing, feature selection is not training
   - [ ] 5 folds is too few
   > Choosing features is learning from data. Do it inside each fold, on the training part only.
2. Your dataset has 20 X-ray images per patient. How should you split folds?
   - [ ] Randomly by image
   - [x] By patient, so all images of one patient fall in the same fold
   - [ ] By image date
   > Images of the same patient are correlated; random splits let the model recognise patients instead of disease.
3. After picking the best of 30 hyperparameter settings by CV, the best score is 0.91. Is 0.91 a fair estimate of future performance?
   - [ ] Yes, cross-validation is unbiased
   - [x] It is likely optimistic, because it is the maximum of many noisy estimates; confirm on an untouched test set
   - [ ] It is pessimistic, because each model saw less data
   > Selecting the best of many noisy scores favours lucky ones. Report a separate test score or use nested CV.
4. Why not use random k-fold to validate a sales forecasting model?
   - [ ] Random folds are too slow
   - [x] The model would train on future days and validate on past ones, which is impossible in practice and leaks information
   - [ ] Forecasting models cannot be cross-validated
   > Use forward-chaining splits that always validate on a period after the training data.
<!-- /quiz -->
