---
title: Preparing data
level: beginner
minutes: 35
summary: Scaling numbers, encoding categories, filling gaps, and the one rule that prevents most silent bugs, fit every transformation on the training data only.
---

## Models only see numbers

Real tables mix numbers on wildly different scales, text categories and missing cells. Before a model can learn, every column has to become well-behaved numbers. The steps below are routine, but each hides a trap.

## Scaling numeric features

Distance-based methods ([k-NN](#/lesson/knn), [k-means](#/lesson/k-means), [SVMs](#/lesson/svm)) and anything trained by [gradient descent](#/lesson/gradient-descent) care about scale: a feature in the thousands drowns one between 0 and 1. Two common fixes:

<!-- formula: Standardisation -->
$$z = \frac{x - \mu}{\sigma}$$

<!-- formula: Min-max scaling -->
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Standardisation gives mean 0 and standard deviation 1 and is the default. Min-max squeezes into $[0, 1]$ but a single outlier squashes everything else. Trees and forests do not need scaling at all: they only compare a feature with thresholds.

## The golden rule: fit on training data only

$\mu$ and $\sigma$ are **learned** from data, just like model weights. Compute them on the training set, then apply the same numbers to the validation and test sets and to new data in production. If you standardise the whole dataset before splitting, information about the test set leaks into training, and your test score becomes optimistic. This is called **data leakage**, and it is one of the most common reasons a model that looked great fails in practice ([more in production ML](#/lesson/ml-in-production)).

So every transformer has two steps: `fit` (learn the numbers from training data) and `transform` (apply them to any data).

## Encoding categories

A column such as `city = {Jakarta, Bandung, Surabaya}` cannot be fed in as 0, 1, 2: that would claim Surabaya is "twice" Bandung. **One-hot encoding** gives each category its own 0/1 column:

| city | city=Bandung | city=Jakarta | city=Surabaya |
|---|---|---|---|
| Jakarta | 0 | 1 | 0 |
| Surabaya | 0 | 0 | 1 |

Points to watch:

- Learn the list of categories on the training set. A category that appears only later must still be handled, for example as all zeros.
- For a linear model with an intercept, the one-hot columns of one feature always sum to 1, which duplicates the intercept. Drop one column, or use regularisation.
- With thousands of categories (user ids, product ids) one-hot explodes; use target statistics, hashing, or learned [embeddings](#/lesson/word-embeddings) instead.

## Missing values

Options, from simplest to most careful:

1. Drop rows with missing values: fine if few and random, biased otherwise.
2. **Impute** with the training mean or median (for categories, the most frequent value).
3. Impute and add an indicator column `was_missing`. Missingness itself is often informative: an empty "previous loans" field may mean "no previous loans".

Imputation values are fitted parameters too, so they follow the golden rule.

## Outliers and skew

Money, counts and durations are usually skewed: most values small, a few huge. A log transform, $\log(1 + x)$, often makes such a feature far easier for linear models and neural networks. Check outliers before deleting them; a "wrong" value is sometimes the most important row in the data.

<!-- exercise: scaler | A standard scaler with fit and transform -->
Complete the class `StandardScaler` with `fit(X)` (store `self.mean_` and `self.std_` per column, replacing zero deviations by 1, and return `self`), `transform(X)` and `fit_transform(X)`.

~~~python starter
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X):
        raise NotImplementedError

train = np.array([[1.0, 100.0], [3.0, 300.0]])
test = np.array([[2.0, 500.0]])
s = StandardScaler().fit(train)
print(s.transform(test))   # [[0, 3]]
~~~

~~~python solution
import numpy as np

class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

train = np.array([[1.0, 100.0], [3.0, 300.0]])
test = np.array([[2.0, 500.0]])
s = StandardScaler().fit(train)
print(s.transform(test))
~~~

~~~python tests
import numpy as np

train = np.array([[1.0, 100.0], [3.0, 300.0]])

def test_fit_returns_self():
    """fit returns the scaler, so calls can be chained"""
    s = StandardScaler()
    assert s.fit(train) is s

def test_uses_training_statistics():
    """test data is scaled with the training mean and deviation"""
    s = StandardScaler().fit(train)
    assert approx(s.transform(np.array([[2.0, 500.0]])), [[0.0, 3.0]])

def test_fit_transform():
    """fit_transform on training data gives mean 0, std 1"""
    X = np.random.default_rng(0).normal(5, 2, size=(40, 3))
    Z = StandardScaler().fit_transform(X)
    assert approx(Z.mean(axis=0), np.zeros(3)) and approx(Z.std(axis=0), np.ones(3))

def test_constant_column():
    """a constant column does not produce nan"""
    Z = StandardScaler().fit_transform(np.array([[1.0, 4.0], [2.0, 4.0]]))
    assert not np.isnan(Z).any()

def test_attributes():
    """stores mean_ and std_"""
    s = StandardScaler().fit(train)
    assert approx(s.mean_, [2.0, 200.0]) and approx(s.std_, [1.0, 100.0])
~~~

<!-- /exercise -->

<!-- exercise: onehot | One-hot encoding with unseen categories -->
Write a class `OneHot` with `fit(values)` that stores the sorted unique categories in `self.categories_` (a Python list) and returns `self`, and `transform(values)` that returns an `(n, k)` float array with a 1 in the column of each value's category. A value not seen during `fit` gets a row of zeros.

~~~python starter
import numpy as np

class OneHot:
    def fit(self, values):
        raise NotImplementedError

    def transform(self, values):
        raise NotImplementedError

enc = OneHot().fit(["Jakarta", "Surabaya", "Bandung", "Jakarta"])
print(enc.categories_)
print(enc.transform(["Surabaya", "Medan"]))
~~~

~~~python solution
import numpy as np

class OneHot:
    def fit(self, values):
        self.categories_ = sorted(set(values))
        self._index = {c: i for i, c in enumerate(self.categories_)}
        return self

    def transform(self, values):
        out = np.zeros((len(values), len(self.categories_)))
        for row, v in enumerate(values):
            j = self._index.get(v)
            if j is not None:
                out[row, j] = 1.0
        return out

enc = OneHot().fit(["Jakarta", "Surabaya", "Bandung", "Jakarta"])
print(enc.categories_)
print(enc.transform(["Surabaya", "Medan"]))
~~~

~~~python tests
import numpy as np

enc_data = ["Jakarta", "Surabaya", "Bandung", "Jakarta"]

def test_categories_sorted():
    """categories are the sorted unique values"""
    assert OneHot().fit(enc_data).categories_ == ["Bandung", "Jakarta", "Surabaya"]

def test_transform():
    """each row has a single 1 in its category's column"""
    M = OneHot().fit(enc_data).transform(["Jakarta", "Bandung"])
    assert approx(M, [[0, 1, 0], [1, 0, 0]])

def test_unseen():
    """an unseen category becomes a row of zeros"""
    M = OneHot().fit(enc_data).transform(["Medan"])
    assert M.shape == (1, 3) and approx(M, [[0, 0, 0]])

def test_rows_sum_to_one():
    """known values always sum to exactly one per row"""
    M = OneHot().fit(enc_data).transform(enc_data)
    assert approx(M.sum(axis=1), np.ones(4))
~~~

<!-- hint: A dictionary from category to column index makes `transform` a simple loop. -->
<!-- /exercise -->

<!-- quiz -->
1. You standardise the whole dataset, then split it into training and test sets. What is wrong?
   - [ ] Nothing, standardising is a deterministic transformation
   - [x] The mean and deviation used the test rows, so test information leaked into training
   - [ ] Standardisation must always come after training
   > Scaling statistics are learned parameters. Fit them on the training rows only and reuse them for the test rows.
2. Which model does not need its features to be scaled?
   - [ ] k-nearest neighbours
   - [ ] Logistic regression trained with gradient descent
   - [x] A decision tree
   > A tree splits on thresholds of one feature at a time, so rescaling a feature moves the thresholds but changes nothing else.
3. Why not encode `city` as Jakarta = 0, Bandung = 1, Surabaya = 2 for a linear model?
   - [x] The model would treat the codes as quantities, as if Surabaya were twice Bandung
   - [ ] Integers cannot be used as features
   - [ ] The model would train too slowly
   > The numbers impose an order and spacing that do not exist. One-hot columns let each city have its own weight.
4. A loan application leaves "previous loans" empty for many applicants. What is a sensible first approach?
   - [ ] Delete those rows
   - [x] Impute a value and add a `was_missing` indicator, because missingness may itself carry information
   - [ ] Fill with a random number
   > Deleting can bias the data, and the fact that the field is empty may predict the target, so keep that information as a feature.
<!-- /quiz -->
