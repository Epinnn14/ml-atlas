---
title: Hierarchical clustering and DBSCAN
level: intermediate
minutes: 40
summary: Two alternatives to k-means. Agglomerative clustering builds a tree of merges you can cut at any level; DBSCAN finds clusters of any shape as dense regions and labels the rest as noise.
---

## Agglomerative clustering

Start with every point as its own cluster. Repeatedly merge the two closest clusters until one remains. The record of merges is a tree, the **dendrogram**; cutting it at a height gives a flat clustering, so you can look at the structure before choosing the number of clusters.

"Closest clusters" needs a **linkage** rule for the distance between two groups $A$ and $B$:

<!-- formula: Linkage rules -->
$$d_{\text{single}} = \min_{a\in A,\,b\in B} d(a,b), \quad d_{\text{complete}} = \max_{a\in A,\,b\in B} d(a,b), \quad d_{\text{average}} = \frac{1}{|A||B|}\sum_{a,b} d(a,b)$$

- **Single** linkage follows chains of close points: it can find long, thin clusters, but one bridge of points merges two groups ("chaining").
- **Complete** linkage prefers compact clusters of similar diameter.
- **Average** linkage is in between. **Ward** linkage merges the pair that increases the k-means objective least.

The naive algorithm needs the full $n\times n$ distance matrix, so it suits up to a few tens of thousands of points.

## DBSCAN

DBSCAN (density-based spatial clustering of applications with noise) defines clusters as dense regions. Two parameters: a radius $\varepsilon$ and a count `min_samples`.

- A **core point** has at least `min_samples` points (itself included) within distance $\varepsilon$.
- Points within $\varepsilon$ of a core point are **density-reachable** from it. A cluster is a maximal set of points connected through core points.
- **Border points** are in a cluster but not core. Points reachable from no core point are **noise**.

The algorithm grows clusters by breadth-first search from each unvisited core point. Advantages: no need to choose the number of clusters, arbitrary shapes (rings, moons), and outliers are labelled instead of distorting a centre. Disadvantages: one $\varepsilon$ for the whole space struggles when clusters have very different densities (HDBSCAN addresses this), and it suffers, like all distance methods, in high dimensions.

A rule of thumb for $\varepsilon$: plot every point's distance to its `min_samples`-th nearest neighbour, sorted; the knee of that curve is a good radius.

## Which to use

| Method | Shape of clusters | Number of clusters | Outliers |
|---|---|---|---|
| [k-means](#/lesson/k-means) | round, similar size | you choose $k$ | forced into a cluster |
| Agglomerative | depends on linkage | cut the dendrogram | forced into a cluster |
| DBSCAN | any shape, similar density | found automatically | labelled noise |
| [Gaussian mixture](#/lesson/gaussian-mixtures) | elliptical, soft membership | you choose $k$ | low probability |

<!-- exercise: single-linkage | Single-linkage clustering -->
Write `single_linkage(X, n_clusters)` that starts from singleton clusters and repeatedly merges the two clusters with the smallest single-linkage distance until `n_clusters` remain. Return an array of labels `0..n_clusters-1`, numbered in order of each cluster's smallest point index. A simple implementation that recomputes cluster distances from a precomputed point distance matrix is fine for the test sizes.

~~~python starter
import numpy as np

def single_linkage(X, n_clusters):
    X = np.asarray(X, float)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    clusters = [[i] for i in range(len(X))]
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def single_linkage(X, n_clusters):
    X = np.asarray(X, float)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > n_clusters:
        best = (np.inf, 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = D[np.ix_(clusters[a], clusters[b])].min()
                if d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] += clusters.pop(b)
    clusters.sort(key=min)
    labels = np.empty(len(X), dtype=int)
    for c, members in enumerate(clusters):
        labels[members] = c
    return labels
~~~

~~~python tests
import numpy as np

def test_two_groups():
    """two well-separated groups"""
    X = np.array([[0.0, 0], [0, 1], [1, 0], [10, 10], [10, 11]])
    assert list(single_linkage(X, 2)) == [0, 0, 0, 1, 1]

def test_chain():
    """single linkage follows a chain that k-means would cut"""
    line = np.c_[np.arange(0, 10, 0.5), np.zeros(20)]
    far = np.array([[4.75, 6.0], [5.25, 6.0]])
    lab = single_linkage(np.vstack([line, far]), 2)
    assert len(set(lab[:20])) == 1 and lab[20] == lab[21] != lab[0]

def test_n_clusters():
    """returns exactly n_clusters labels, numbered by first point"""
    X = np.random.default_rng(0).normal(size=(15, 2))
    lab = single_linkage(X, 4)
    assert sorted(set(lab)) == [0, 1, 2, 3] and lab[0] == 0

def test_all_singletons():
    """n_clusters = n keeps every point alone"""
    X = np.random.default_rng(1).normal(size=(6, 2))
    assert list(single_linkage(X, 6)) == [0, 1, 2, 3, 4, 5]
~~~

<!-- hint: `D[np.ix_(A, B)]` is the block of distances between the points of clusters A and B; its minimum is the single-linkage distance. -->
<!-- hint: After merging, sort the clusters by their smallest member (`clusters.sort(key=min)`) before numbering them. -->
<!-- /exercise -->

<!-- exercise: dbscan | DBSCAN -->
Write `dbscan(X, eps, min_samples)` returning labels: cluster numbers `0, 1, 2, ...` in the order clusters are discovered when scanning points by index, and `-1` for noise. A point is core if at least `min_samples` points (including itself) are within distance `<= eps`. Expand each new cluster from a core point by breadth-first search, adding all neighbours of core points.

~~~python starter
import numpy as np
from collections import deque

def dbscan(X, eps, min_samples):
    X = np.asarray(X, float)
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    neighbours = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    core = np.array([len(nb) >= min_samples for nb in neighbours])
    labels = np.full(n, -1)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np
from collections import deque

def dbscan(X, eps, min_samples):
    X = np.asarray(X, float)
    n = len(X)
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    neighbours = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    core = np.array([len(nb) >= min_samples for nb in neighbours])
    labels = np.full(n, -1)
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster
        queue = deque([i])
        while queue:
            p = queue.popleft()
            if not core[p]:
                continue
            for q in neighbours[p]:
                if labels[q] == -1:
                    labels[q] = cluster
                    queue.append(q)
        cluster += 1
    return labels
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
t = rng.uniform(0, 2 * np.pi, 200)
inner = np.c_[np.cos(t[:100]), np.sin(t[:100])] + 0.05 * rng.normal(size=(100, 2))
outer = 4 * np.c_[np.cos(t[100:]), np.sin(t[100:])] + 0.05 * rng.normal(size=(100, 2))
rings = np.vstack([inner, outer])

def test_rings():
    """separates two concentric rings"""
    lab = dbscan(rings, eps=1.0, min_samples=4)
    assert len(set(lab[:100])) == 1 and len(set(lab[100:])) == 1 and lab[0] != lab[100]

def test_noise():
    """an isolated point is noise"""
    X = np.vstack([rng.normal(size=(30, 2)) * 0.2, [[10.0, 10.0]]])
    lab = dbscan(X, eps=0.5, min_samples=3)
    assert lab[-1] == -1 and set(lab[:30]) == {0}

def test_order_of_discovery():
    """clusters are numbered in the order their first point appears"""
    X = np.array([[10.0, 0], [10.1, 0], [10.2, 0], [0, 0], [0.1, 0], [0.2, 0]])
    assert list(dbscan(X, eps=0.15, min_samples=2)) == [0, 0, 0, 1, 1, 1]

def test_border_point():
    """a border point joins the cluster but does not extend it"""
    X = np.array([[0.0, 0], [0.1, 0], [0.2, 0], [0.5, 0], [0.8, 0]])
    lab = dbscan(X, eps=0.3, min_samples=4)     # only 0.2 is core; 0.5 is its border; 0.8 is noise
    assert list(lab) == [0, 0, 0, 0, -1]
~~~

<!-- hint: Only core points add their neighbours to the queue; a border point gets a label but stops the expansion. -->
<!-- /exercise -->

<!-- quiz -->
1. Which method can find two interleaved half-moon clusters and flag scattered outliers?
   - [ ] k-means
   - [x] DBSCAN
   - [ ] Complete-linkage clustering
   > DBSCAN follows dense regions of any shape and labels low-density points as noise.
2. A few points form a bridge between two dense groups. Which linkage will most likely merge the groups early?
   - [x] Single linkage
   - [ ] Complete linkage
   - [ ] Ward linkage
   > Single linkage uses the closest pair, so a chain of close points connects the groups ("chaining").
3. DBSCAN finds one big cluster and a lot of noise in data with dense and sparse regions. What is the likely cause?
   - [ ] Too many points
   - [x] One radius $\varepsilon$ cannot suit clusters of very different densities
   - [ ] The data needs to be shuffled
   > A radius that fits the sparse cluster merges the dense ones, and one that fits the dense ones marks the sparse cluster as noise. HDBSCAN adapts the density level.
4. What is a dendrogram useful for?
   - [x] Seeing the whole hierarchy of merges and choosing where to cut it
   - [ ] Computing cluster centres
   - [ ] Speeding up k-means
   > It records every merge with its height, so different numbers of clusters can be read off one run.
<!-- /quiz -->
