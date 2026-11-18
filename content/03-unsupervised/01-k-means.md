---
title: k-means clustering
level: beginner
minutes: 35
summary: Group points around k centres by alternating two easy steps. Why it always converges, why it can converge badly, how k-means++ starts it well, and how to choose k.
---

## Clustering

Without labels, we can still ask which examples belong together: customer segments, groups of similar documents, colours in an image. **k-means** describes the data by $k$ centres (centroids) $\boldsymbol\mu_1, \dots, \boldsymbol\mu_k$ and assigns every point to its nearest centre. It looks for the centres that make points as close to their centre as possible:

<!-- formula: k-means objective (inertia) -->
$$J = \sum_{i=1}^n \min_{c}\ \lVert\mathbf x_i - \boldsymbol\mu_c\rVert^2$$

## Lloyd's algorithm

Start with $k$ centres, then repeat until nothing changes:

1. **Assign**: put each point in the cluster of its nearest centre.
2. **Update**: move each centre to the mean of the points assigned to it.

<!-- formula: k-means update -->
$$\boldsymbol\mu_c = \frac{1}{|C_c|}\sum_{i\in C_c}\mathbf x_i$$

Each step can only lower $J$: the assignment step picks the closest centre for every point, and the mean is the point that minimises the sum of squared distances to a group. Since there are finitely many ways to assign points, the algorithm stops after a finite number of steps. But it stops at a **local** minimum, which depends on the starting centres.

## k-means++

Bad starts (two centres in one real cluster, none in another) give bad results. **k-means++** spreads the initial centres out:

1. Pick the first centre uniformly at random from the data.
2. Pick each next centre from the data with probability proportional to $D(\mathbf x)^2$, the squared distance to the nearest centre chosen so far.

Far-away points are likely to be chosen, so the starting centres cover the data. Libraries also run k-means several times from different starts and keep the run with the lowest $J$.

## Choosing k

$J$ always decreases as $k$ grows (with $k = n$ it is 0), so it cannot choose $k$ by itself. Common approaches:

- **Elbow**: plot $J$ against $k$ and look for the bend after which adding clusters helps little.
- **Silhouette score**: for each point, compare the mean distance to its own cluster ($a$) with the mean distance to the nearest other cluster ($b$): $s = (b - a)/\max(a, b)$. Average $s$ near 1 means tight, separated clusters.
- The use: if the clusters feed a business decision with five customer teams, $k = 5$ may simply be the answer.

## Limits

k-means assumes roughly round, similarly sized clusters, because it only uses distance to a centre. It fails on elongated or ring-shaped clusters and is pulled around by outliers. Scale features first. For other shapes see [hierarchical clustering and DBSCAN](#/lesson/hierarchical-dbscan); for soft, elliptical clusters see [Gaussian mixtures](#/lesson/gaussian-mixtures).

<!-- exercise: lloyd | Lloyd's algorithm -->
Write `kmeans(X, init_centres, max_iter=100)` that runs Lloyd's algorithm from the given starting centres and returns `(centres, labels)`. Stop early when the labels stop changing. If a cluster becomes empty, leave its centre where it is. Ties in distance go to the lower centre index (`np.argmin` does that).

~~~python starter
import numpy as np

def kmeans(X, init_centres, max_iter=100):
    X = np.asarray(X, float)
    centres = np.array(init_centres, dtype=float)
    labels = None
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def kmeans(X, init_centres, max_iter=100):
    X = np.asarray(X, float)
    centres = np.array(init_centres, dtype=float)
    labels = None
    for _ in range(max_iter):
        d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for c in range(len(centres)):
            if np.any(labels == c):
                centres[c] = X[labels == c].mean(axis=0)
    return centres, labels
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
true = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X = np.vstack([c + rng.normal(size=(50, 2)) for c in true])

def inertia(X, C, lab):
    return np.sum((X - C[lab]) ** 2)

def test_finds_clusters():
    """recovers the three blob centres"""
    C, lab = kmeans(X, X[[0, 50, 100]])
    order = np.argsort(C[:, 0] + 10 * C[:, 1])
    assert approx(C[order], true[np.argsort(true[:, 0] + 10 * true[:, 1])], tol=0.4)

def test_labels_are_nearest_centre():
    """every point is assigned to its nearest final centre"""
    C, lab = kmeans(X, X[[0, 1, 2]])
    d2 = ((X[:, None] - C[None]) ** 2).sum(2)
    assert np.array_equal(lab, d2.argmin(1))

def test_centres_are_means():
    """every centre is the mean of its points"""
    C, lab = kmeans(X, X[[0, 50, 100]])
    for c in range(3):
        assert approx(C[c], X[lab == c].mean(0))

def test_one_step_never_increases_inertia():
    """more iterations never give a worse objective"""
    init = X[[3, 4, 5]]
    C1, l1 = kmeans(X, init, max_iter=1)
    C2, l2 = kmeans(X, init, max_iter=50)
    assert inertia(X, C2, l2) <= inertia(X, C1, l1) + 1e-9

def test_does_not_modify_input():
    """the initial centres array is not changed"""
    init = X[[0, 50, 100]].copy()
    kmeans(X, init)
    assert np.array_equal(init, X[[0, 50, 100]])
~~~

<!-- hint: `((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)` gives an `(n, k)` matrix of squared distances. -->
<!-- /exercise -->

<!-- exercise: kmeanspp | k-means++ initialisation -->
Write `kmeans_pp(X, k, seed=0)`. With `rng = np.random.default_rng(seed)`, choose the first centre as `X[rng.integers(len(X))]`. Then, for each next centre, compute `D2`, every point's squared distance to its nearest chosen centre, and pick index `rng.choice(len(X), p=D2 / D2.sum())`. Return a `(k, d)` array.

~~~python starter
import numpy as np

def kmeans_pp(X, k, seed=0):
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def kmeans_pp(X, k, seed=0):
    X = np.asarray(X, float)
    rng = np.random.default_rng(seed)
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        C = np.array(centres)
        D2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        centres.append(X[rng.choice(len(X), p=D2 / D2.sum())])
    return np.array(centres)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
blobs = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], float)
X = np.vstack([c + rng.normal(size=(40, 2)) for c in blobs])

def test_shape():
    """returns k centres taken from the data"""
    C = kmeans_pp(X, 4)
    assert C.shape == (4, 2)
    assert all(any(np.array_equal(c, x) for x in X) for c in C)

def test_spreads_out():
    """with far-apart blobs, the 4 centres land in 4 different blobs (most seeds)"""
    hits = 0
    for s in range(20):
        C = kmeans_pp(X, 4, seed=s)
        nearest_blob = ((C[:, None] - blobs[None]) ** 2).sum(2).argmin(1)
        hits += len(set(nearest_blob)) == 4
    assert hits >= 18

def test_recipe():
    """follows the sampling recipe exactly"""
    r = np.random.default_rng(7)
    first = X[r.integers(len(X))]
    D2 = ((X - first) ** 2).sum(1)
    second = X[r.choice(len(X), p=D2 / D2.sum())]
    C = kmeans_pp(X, 2, seed=7)
    assert np.array_equal(C[0], first) and np.array_equal(C[1], second)
~~~

<!-- hint: `D2` is the minimum over chosen centres of the squared distance: compute the `(n, c)` matrix and take `.min(axis=1)`. -->
<!-- /exercise -->

<!-- quiz -->
1. Why does Lloyd's algorithm always stop?
   - [x] Each step cannot increase the objective and there are finitely many possible assignments
   - [ ] It runs a fixed number of iterations
   - [ ] The centres always move towards the global optimum
   > A strictly decreasing objective cannot revisit an assignment, and there are only finitely many.
2. The inertia keeps dropping as you increase k from 2 to 20. Does that mean k = 20 is best?
   - [ ] Yes, lower inertia is always better
   - [x] No, inertia always decreases with more clusters; use the elbow, silhouette or the intended use
   - [ ] Only if the clusters are equal in size
   > With $k = n$ the inertia is zero and the clustering is useless.
3. Your data form two concentric rings. What will k-means with k = 2 do?
   - [ ] Separate the inner and outer ring
   - [x] Cut both rings in half with a straight boundary
   - [ ] Fail to converge
   > k-means boundaries are straight lines between centres; ring-shaped clusters need density- or connectivity-based methods.
4. What is the point of k-means++?
   - [ ] It makes each iteration faster
   - [x] It starts with well-spread centres, making poor local minima much less likely
   - [ ] It chooses k automatically
   > Choosing far-away points as later centres covers all clusters with high probability.
<!-- /quiz -->
