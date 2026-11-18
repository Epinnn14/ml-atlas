---
title: NumPy essentials
level: beginner
minutes: 30
summary: Arrays, shapes, axes and broadcasting, the four ideas that let you write every model in this atlas without a single Python loop over examples.
---

## Why NumPy

A dataset is a table of numbers: $n$ rows (examples) and $d$ columns (features). Python lists can hold it, but a loop over a million rows in pure Python is slow. NumPy stores the table as one contiguous block of memory and runs operations on the whole block in compiled code. Writing a model "the NumPy way" means expressing it as operations on whole arrays, which is called **vectorisation**.

By convention in this atlas:

- `X` has shape `(n, d)`: one row per example.
- `y` has shape `(n,)`: one target per example.
- `w` has shape `(d,)`: one weight per feature.

## Shapes and axes

Every array has a `shape`, a tuple with one number per dimension. Reductions such as `sum`, `mean` and `max` take an `axis` argument that says which dimension disappears:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)
X.sum(axis=0)                  # [5, 7, 9]   -> one value per column
X.sum(axis=1)                  # [6, 15]     -> one value per row
X.mean(axis=0, keepdims=True)  # shape (1, 3), handy for broadcasting
```

A trick to remember: `axis=0` collapses the rows, so you get **one result per column**.

## Broadcasting

When two arrays have different shapes, NumPy compares the shapes from the right. Two dimensions are compatible if they are equal or one of them is 1; a dimension of size 1 is stretched (without copying) to match the other.

| Left | Right | Result |
|---|---|---|
| `(n, d)` | `(d,)` | `(n, d)`: the vector is applied to every row |
| `(n, 1)` | `(1, m)` | `(n, m)`: every pair of a row and a column |
| `(n, d)` | `(n,)` | error: compared from the right, `d` and `n` differ |

The last case is the most common bug. To subtract a per-row value, give it shape `(n, 1)` with `v[:, None]` or `keepdims=True`.

## Matrix products

`A @ B` is the matrix product: shapes `(n, k) @ (k, m)` give `(n, m)`. A whole linear model applied to every example is one line:

<!-- formula: Linear model, all examples at once -->
$$\hat{\mathbf y} = X\mathbf w + b$$

which in NumPy is `X @ w + b`, shape `(n,)`.

## Boolean masks and fancy indexing

Comparisons give arrays of `True`/`False` that select elements:

```python
y = np.array([3, -1, 4, -1, 5])
y[y > 0]          # [3, 4, 5]
np.mean(y > 0)    # 0.6, the fraction of positives (True counts as 1)
idx = np.argsort(y)[:2]   # positions of the two smallest values
```

## A pattern you will use constantly: pairwise distances

Many methods ([k-nearest neighbours](#/lesson/knn), [k-means](#/lesson/k-means), kernels) need the squared distance between every row of `A` (shape `(n, d)`) and every row of `B` (shape `(m, d)`). Expanding the square avoids building a huge `(n, m, d)` array:

<!-- formula: Squared Euclidean distance, expanded -->
$$\lVert \mathbf a - \mathbf b \rVert^2 = \lVert \mathbf a\rVert^2 - 2\,\mathbf a^\top \mathbf b + \lVert \mathbf b \rVert^2$$

The middle term for all pairs at once is `A @ B.T`, shape `(n, m)`, and the squared norms broadcast as a column and a row.

<!-- exercise: standardise | Standardise the columns -->
Write `standardise(X)` that returns a copy of `X` where every **column** has mean 0 and standard deviation 1:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Use `X.mean(axis=0)` and `X.std(axis=0)` and broadcasting, no loops. If a column is constant ($\sigma_j = 0$), divide by 1 instead so you do not produce `nan`.

~~~python starter
import numpy as np

def standardise(X):
    X = np.asarray(X, dtype=float)
    raise NotImplementedError

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
print(standardise(X))
~~~

~~~python solution
import numpy as np

def standardise(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
print(standardise(X))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal([5, -2, 100], [1, 3, 20], size=(50, 3))

def test_shape():
    """keeps the shape (n, d)"""
    assert standardise(X).shape == (50, 3)

def test_mean_zero():
    """every column has mean 0"""
    assert approx(standardise(X).mean(axis=0), np.zeros(3))

def test_std_one():
    """every column has standard deviation 1"""
    assert approx(standardise(X).std(axis=0), np.ones(3))

def test_constant_column():
    """a constant column becomes zeros, not nan"""
    Z = standardise(np.array([[1.0, 7.0], [3.0, 7.0]]))
    assert not np.isnan(Z).any(), "got nan: guard against sigma == 0"
    assert approx(Z[:, 1], [0.0, 0.0])

def test_input_untouched():
    """the input array is not modified"""
    A = X.copy()
    standardise(A)
    assert np.array_equal(A, X)
~~~

<!-- hint: `X.mean(axis=0)` has shape `(d,)`, which broadcasts against `(n, d)` row by row. -->
<!-- hint: `np.where(sigma == 0, 1.0, sigma)` replaces zero deviations. -->
<!-- /exercise -->

<!-- exercise: pairwise | All pairwise squared distances -->
Write `pairwise_sq_dists(A, B)` that returns an `(n, m)` matrix whose entry `[i, j]` is the squared Euclidean distance between row `i` of `A` and row `j` of `B`. Use the expanded formula above and no Python loops. Rounding can make tiny values slightly negative; clip them at 0.

~~~python starter
import numpy as np

def pairwise_sq_dists(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    raise NotImplementedError

A = np.array([[0.0, 0.0], [1.0, 1.0]])
B = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
print(pairwise_sq_dists(A, B))   # [[1, 25, 0], [1, 13, 2]]
~~~

~~~python solution
import numpy as np

def pairwise_sq_dists(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    a2 = np.sum(A ** 2, axis=1)[:, None]   # (n, 1)
    b2 = np.sum(B ** 2, axis=1)[None, :]   # (1, m)
    D = a2 - 2 * A @ B.T + b2
    return np.maximum(D, 0.0)

A = np.array([[0.0, 0.0], [1.0, 1.0]])
B = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
print(pairwise_sq_dists(A, B))
~~~

~~~python tests
import numpy as np

def test_small():
    """matches the worked example"""
    A = np.array([[0.0, 0.0], [1.0, 1.0]])
    B = np.array([[1.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    assert approx(pairwise_sq_dists(A, B), [[1, 25, 0], [1, 13, 2]])

def test_against_loops():
    """agrees with a slow double loop on random data"""
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    slow = np.array([[np.sum((a - b) ** 2) for b in B] for a in A])
    assert approx(pairwise_sq_dists(A, B), slow, tol=1e-9)

def test_non_negative():
    """no negative distances, even for identical points"""
    A = np.random.default_rng(2).normal(size=(20, 3)) * 1e4
    assert (pairwise_sq_dists(A, A) >= 0).all()

def test_no_loops():
    """fast enough for 2000 x 2000 points (needs vectorised code)"""
    import time
    A = np.random.default_rng(3).normal(size=(2000, 8))
    t = time.perf_counter()
    pairwise_sq_dists(A, A)
    assert time.perf_counter() - t < 3.0, "too slow: replace Python loops with array operations"
~~~

<!-- hint: `np.sum(A**2, axis=1)` gives the squared norm of every row. Reshape it to a column with `[:, None]`. -->
<!-- hint: `A @ B.T` gives all the dot products at once, shape `(n, m)`. -->
<!-- /exercise -->

<!-- quiz -->
1. `X` has shape `(100, 5)`. What is the shape of `X.mean(axis=0)`?
   - [x] `(5,)`
   - [ ] `(100,)`
   - [ ] `(100, 5)`
   > `axis=0` collapses the 100 rows, leaving one mean per column.
2. `X` has shape `(100, 5)` and `v` has shape `(100,)`. Why does `X - v` fail, and what fixes it?
   - [ ] NumPy cannot subtract arrays of different sizes; use a loop
   - [x] Shapes are compared from the right, 5 and 100 differ; use `X - v[:, None]`
   - [ ] `v` must be converted to a list first
   > Broadcasting lines shapes up from the last dimension. Giving `v` shape `(100, 1)` makes it a column, which is then stretched across the 5 columns.
3. What is the shape of `A @ B` when `A` is `(32, 64)` and `B` is `(64, 10)`?
   - [ ] `(64, 64)`
   - [x] `(32, 10)`
   - [ ] It fails because the shapes differ
   > Matrix product shapes multiply as `(n, k) @ (k, m) -> (n, m)`; the shared inner size 64 disappears.
4. What does `np.mean(y > 0)` compute for a NumPy array `y`?
   - [ ] The mean of the positive values of `y`
   - [x] The fraction of entries of `y` that are positive
   - [ ] An error, because you cannot average booleans
   > `y > 0` is an array of booleans, which count as 1 and 0 in arithmetic, so their mean is the fraction of `True` values.
<!-- /quiz -->
