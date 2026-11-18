---
title: k-nearest neighbours
level: beginner
minutes: 30
summary: The model that does no training at all. Predict from the k most similar examples, choose k with validation, and see why distance stops meaning much in high dimensions.
---

## Predict like your neighbours

To classify a new point, find the $k$ training examples closest to it and take a majority vote of their labels. For regression, average their targets. There is no training step: the "model" is the stored data. Methods like this are called **instance-based** or **lazy** learners.

<!-- formula: k-NN regression -->
$$\hat y(\mathbf x) = \frac1k\sum_{i\in N_k(\mathbf x)} y_i$$

where $N_k(\mathbf x)$ are the indices of the $k$ nearest training points, usually by Euclidean distance (computed for all pairs with the [expanded formula](#/lesson/numpy-essentials)).

## Choosing k

$k$ controls the [bias–variance trade-off](#/lesson/bias-variance) directly:

- $k = 1$: the prediction copies the single closest example. Training error is zero and the boundary is jagged around every noisy point: high variance.
- Large $k$: predictions average over a wide region and become smooth, eventually just the overall majority class: high bias.

Pick $k$ by [cross-validation](#/lesson/cross-validation). For two classes, an odd $k$ avoids tied votes.

## Practical points

- **Scale the features.** Distance adds up squared differences, so a feature measured in thousands dominates. [Standardise](#/lesson/data-preprocessing) first.
- **Weighting.** Giving closer neighbours more say, for example weight $1/\text{distance}$, smooths the transition as $k$ grows.
- **Cost at prediction time.** Every prediction compares with all $n$ stored points, $O(nd)$. Tree structures (k-d trees, ball trees) help in low dimensions; approximate nearest-neighbour indexes (used in vector databases for [embeddings](#/lesson/word-embeddings) and retrieval) help in high dimensions.

## The curse of dimensionality

In high dimensions, intuition about distance breaks. Scatter points uniformly in a $d$-dimensional cube: as $d$ grows, the nearest and the farthest neighbour of a point end up at almost the same distance.

<!-- formula: Relative contrast of distances -->
$$\frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \;\xrightarrow{\;d\to\infty\;}\; 0$$

When everything is about equally far away, "nearest" carries little information, and you would need exponentially more data to fill the space. k-NN therefore works best in few dimensions, or on learned representations where similar things really are close.

<!-- exercise: knn-classify | k-NN classifier -->
Write `knn_predict(X_train, y_train, X_test, k)` returning the predicted integer label for every test row by majority vote among the `k` nearest training rows (Euclidean distance). Break ties by choosing the **smallest** label among those with the most votes (`np.bincount(...).argmax()` does exactly that).

~~~python starter
import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    d2 = (np.sum(X_test ** 2, 1)[:, None] - 2 * X_test @ X_train.T + np.sum(X_train ** 2, 1)[None, :])
    nearest = np.argsort(d2, axis=1, kind="stable")[:, :k]
    return np.array([np.bincount(y_train[idx]).argmax() for idx in nearest])
~~~

~~~python tests
import numpy as np

def test_tiny():
    """the label of the closest point when k = 1"""
    Xtr = np.array([[0.0, 0.0], [10.0, 10.0]]); ytr = np.array([0, 1])
    assert list(knn_predict(Xtr, ytr, np.array([[1.0, 1.0], [9.0, 8.0]]), 1)) == [0, 1]

def test_majority():
    """k = 3 outvotes a single nearby point of another class"""
    Xtr = np.array([[0.0], [1.0], [1.2], [1.3]]); ytr = np.array([0, 1, 1, 1])
    assert knn_predict(Xtr, ytr, np.array([[0.4]]), 3)[0] == 1

def test_tie_smallest_label():
    """a 1-1 tie picks the smaller label"""
    Xtr = np.array([[0.0], [2.0]]); ytr = np.array([3, 1])
    assert knn_predict(Xtr, ytr, np.array([[1.0]]), 2)[0] == 1

def test_blobs():
    """accurate on two separated blobs"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); X = np.array([[0, 0], [4, 4]])[y] + rng.normal(size=(200, 2))
    pred = knn_predict(X[:150], y[:150], X[150:], 5)
    assert np.mean(pred == y[150:]) > 0.95

def test_k1_memorises():
    """with k = 1, predicting the training set reproduces its labels"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)); y = rng.integers(0, 3, 50)
    assert np.array_equal(knn_predict(X, y, X, 1), y)
~~~

<!-- hint: Compute all test-to-train squared distances at once, then `np.argsort(d2, axis=1)[:, :k]` gives the neighbours of every test row. -->
<!-- /exercise -->

<!-- exercise: curse | Watch distances concentrate -->
Write `relative_contrast(d, n=500, seed=0)`: draw `n` points uniformly in the unit cube $[0,1]^d$ with `np.random.default_rng(seed).uniform(size=(n, d))`, take the first point as the query, compute its Euclidean distances to the other `n - 1` points, and return $(\max - \min)/\min$ as a float.

~~~python starter
import numpy as np

def relative_contrast(d, n=500, seed=0):
    raise NotImplementedError

for d in (2, 10, 100, 1000):
    print(d, round(relative_contrast(d), 3))
~~~

~~~python solution
import numpy as np

def relative_contrast(d, n=500, seed=0):
    P = np.random.default_rng(seed).uniform(size=(n, d))
    dist = np.linalg.norm(P[1:] - P[0], axis=1)
    return float((dist.max() - dist.min()) / dist.min())

for d in (2, 10, 100, 1000):
    print(d, round(relative_contrast(d), 3))
~~~

~~~python tests
import numpy as np

def test_value_formula():
    """follows the recipe exactly"""
    P = np.random.default_rng(3).uniform(size=(100, 5))
    dist = np.linalg.norm(P[1:] - P[0], axis=1)
    assert approx(relative_contrast(5, n=100, seed=3), (dist.max() - dist.min()) / dist.min())

def test_shrinks_with_dimension():
    """the contrast falls steadily as dimension grows"""
    c = [relative_contrast(d) for d in (2, 10, 100, 1000)]
    assert all(a > b for a, b in zip(c, c[1:]))

def test_high_dimension_small():
    """in 1000 dimensions nearest and farthest are within about 25% of each other"""
    assert relative_contrast(1000) < 0.25
~~~

<!-- /exercise -->

<!-- quiz -->
1. With $k = 1$, what is the training accuracy of k-NN (no duplicate points)?
   - [x] 100%, since each point's nearest neighbour is itself
   - [ ] About 50%
   - [ ] It depends on the test set
   > That perfect training score says nothing about generalisation; it is the high-variance extreme.
2. One feature is income in rupiah (millions) and another is age in years. What happens if you run k-NN on the raw values?
   - [ ] Nothing, k-NN is scale invariant
   - [x] Distances are dominated by income, so age is effectively ignored
   - [ ] The model trains more slowly
   > Euclidean distance adds squared differences; a feature with huge numbers swamps the rest. Standardise first.
3. What does increasing $k$ do?
   - [ ] Increases variance
   - [x] Smooths the decision boundary, raising bias and lowering variance
   - [ ] Has no effect on the predictions
   > Averaging over more neighbours ignores local noise but also blurs real local structure.
4. Why does k-NN struggle with 1000 raw features?
   - [x] Distances between points concentrate, so nearest neighbours are barely nearer than anyone else
   - [ ] NumPy cannot compute distances in 1000 dimensions
   - [ ] k must be larger than the number of features
   > This is the curse of dimensionality: "close" stops meaning "similar" unless the data lie on a low-dimensional structure.
<!-- /quiz -->
