---
title: Manifold learning, t-SNE and UMAP
level: advanced
minutes: 45
summary: When data lie on a curved surface, linear projections fail. Classical MDS from distances, the idea behind Isomap, and how t-SNE and UMAP make 2-D maps that keep neighbours together.
---

## The manifold hypothesis

Images of a face turning its head live in a space of millions of pixels, yet only a few numbers (the angles) really change. Such data lie near a low-dimensional **manifold**: a surface that is curved inside the high-dimensional space. [PCA](#/lesson/pca) finds the best flat subspace, which fails when the surface curls, like a rolled-up "swiss roll" sheet where points far apart along the sheet are close in straight-line distance.

## Classical MDS

**Multidimensional scaling** starts from a matrix of pairwise distances $D$ and finds coordinates whose distances match them. For Euclidean distances there is an exact recipe. Square the distances, then **double-centre**:

<!-- formula: Double centring in classical MDS -->
$$B = -\tfrac12\,J D^{(2)} J, \qquad J = I - \tfrac1n\mathbf 1\mathbf 1^\top$$

$B$ is the matrix of inner products of the centred points. Its top $k$ eigenvectors $\mathbf v_j$ with eigenvalues $\lambda_j$ give the coordinates $\mathbf v_j\sqrt{\lambda_j}$. With Euclidean input distances, classical MDS gives the same result as PCA.

**Isomap** makes this non-linear: build a nearest-neighbour graph, replace each distance by the shortest path through the graph (an approximation of distance along the manifold), and run classical MDS on those geodesic distances. The swiss roll unrolls.

## t-SNE

t-SNE is built for **visualisation**. It turns distances into neighbour probabilities in the original space, and tries to reproduce them in 2-D.

In the original space, point $j$'s probability of being picked as $i$'s neighbour uses a Gaussian with a width $\sigma_i$ tuned per point:

<!-- formula: t-SNE input affinities -->
$$p_{j\mid i} = \frac{\exp(-\lVert\mathbf x_i - \mathbf x_j\rVert^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\lVert\mathbf x_i - \mathbf x_k\rVert^2/2\sigma_i^2)}$$

Each $\sigma_i$ is found by binary search so that the distribution has a chosen **perplexity**, roughly the number of effective neighbours (typically 5–50):

<!-- formula: Perplexity -->
$$\text{Perp}(P_i) = 2^{H(P_i)}, \qquad H(P_i) = -\sum_j p_{j\mid i}\log_2 p_{j\mid i}$$

Dense regions get small $\sigma_i$, sparse regions large ones. In the 2-D map, similarities use a heavy-tailed Student-t kernel, $q_{ij} \propto (1 + \lVert\mathbf y_i - \mathbf y_j\rVert^2)^{-1}$, and the positions $\mathbf y_i$ are moved by gradient descent to minimise the [KL divergence](#/lesson/information-theory) $D_{\mathrm{KL}}(P\Vert Q)$. The heavy tail lets moderately distant points move far apart, which avoids crowding everything in the middle.

## UMAP

UMAP follows the same plan (a weighted neighbour graph in the original space, a layout optimised to match it) with a different theory and loss. It is much faster, scales to millions of points, keeps more of the global arrangement, and can embed new points.

## Reading these maps honestly

t-SNE and UMAP plots are seductive and easy to over-read:

- **Cluster sizes mean nothing**: the per-point bandwidth equalises density.
- **Distances between clusters mean little**, especially in t-SNE.
- **Results depend on perplexity (or `n_neighbors`) and the random seed.** Try several.
- **Random noise can show "clusters"** at low perplexity.

Use them to explore and to spot labelled groups, not as evidence on their own. For features to feed another model, prefer PCA or a learned representation ([autoencoders](#/lesson/autoencoders-vae)).

<!-- exercise: mds | Classical MDS -->
Write `classical_mds(D, k)` that takes an `(n, n)` matrix of Euclidean distances and returns `(n, k)` coordinates using double centring and the top-$k$ eigenpairs of $B$ (use `np.linalg.eigh`, which returns eigenvalues in ascending order). Clip negative eigenvalues to 0 before taking the square root.

~~~python starter
import numpy as np

def classical_mds(D, k):
    D = np.asarray(D, float)
    n = len(D)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def classical_mds(D, k):
    D = np.asarray(D, float)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:k]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))
~~~

~~~python tests
import numpy as np

def dists(P):
    return np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))

rng = np.random.default_rng(0)
P = rng.normal(size=(30, 2)) * np.array([3.0, 1.0])

def test_shape():
    """returns (n, k)"""
    assert classical_mds(dists(P), 2).shape == (30, 2)

def test_recovers_distances():
    """2-D points are recovered up to rotation: distances match exactly"""
    Y = classical_mds(dists(P), 2)
    assert approx(dists(Y), dists(P), tol=1e-8)

def test_centred():
    """the coordinates are centred"""
    assert approx(classical_mds(dists(P), 2).mean(axis=0), [0.0, 0.0], tol=1e-8)

def test_first_axis_has_most_spread():
    """the first coordinate carries the most variance"""
    Y = classical_mds(dists(P), 2)
    assert Y[:, 0].var() > Y[:, 1].var()
~~~

<!-- hint: `J = np.eye(n) - np.ones((n, n)) / n`; then `B = -0.5 * J @ (D**2) @ J`. -->
<!-- /exercise -->

<!-- exercise: perplexity | Calibrate t-SNE bandwidths -->
Write `conditional_p(d2_row, beta)` returning the distribution $p_{j\mid i} \propto \exp(-\beta\, d^2_{ij})$ over the given squared distances to the **other** points (here $\beta = 1/2\sigma^2$; subtract the minimum distance before exponentiating for stability). Then `find_beta(d2_row, perplexity, tol=1e-5)` that binary-searches $\beta$ so that $2^{H}$ (entropy in bits) matches the target within `tol`, starting from the interval $[10^{-10}, 10^{10}]$ and searching in log space (`beta = sqrt(lo * hi)`) for at most 200 steps. Return `beta`.

~~~python starter
import numpy as np

def conditional_p(d2_row, beta):
    raise NotImplementedError

def find_beta(d2_row, perplexity, tol=1e-5):
    lo, hi = 1e-10, 1e10
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def conditional_p(d2_row, beta):
    d2 = np.asarray(d2_row, float)
    w = np.exp(-beta * (d2 - d2.min()))
    return w / w.sum()

def find_beta(d2_row, perplexity, tol=1e-5):
    lo, hi = 1e-10, 1e10
    beta = 1.0
    for _ in range(200):
        beta = np.sqrt(lo * hi)
        p = conditional_p(d2_row, beta)
        nz = p > 0
        perp = 2 ** (-np.sum(p[nz] * np.log2(p[nz])))
        if abs(perp - perplexity) < tol:
            break
        if perp > perplexity:     # too spread out: sharpen
            lo = beta
        else:
            hi = beta
    return float(beta)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(60, 5))
d2 = ((X[1:] - X[0]) ** 2).sum(1)

def perp(p):
    nz = p > 0
    return 2 ** (-np.sum(p[nz] * np.log2(p[nz])))

def test_distribution():
    """conditional_p is a probability distribution favouring close points"""
    p = conditional_p(d2, 0.5)
    assert approx(p.sum(), 1.0) and p[np.argmin(d2)] == p.max()

def test_hits_target():
    """the found beta gives the requested perplexity"""
    for target in (5, 15, 30):
        assert abs(perp(conditional_p(d2, find_beta(d2, target))) - target) < 1e-3

def test_more_neighbours_needs_wider_kernel():
    """a larger perplexity needs a smaller beta (wider Gaussian)"""
    assert find_beta(d2, 30) < find_beta(d2, 5)

def test_stable_for_huge_distances():
    """no nan when distances are large"""
    p = conditional_p(d2 * 1e4, 10.0)
    assert not np.isnan(p).any() and approx(p.sum(), 1.0)
~~~

<!-- hint: Higher beta means a narrower kernel and a lower perplexity. If the perplexity is too high, raise `lo` to `beta`; otherwise lower `hi`. -->
<!-- /exercise -->

<!-- quiz -->
1. In a t-SNE plot, cluster A looks three times larger than cluster B. What can you conclude?
   - [ ] A has three times more spread in the original space
   - [x] Little: t-SNE adapts the bandwidth per point, so cluster sizes in the map are not meaningful
   - [ ] A has three times more points
   > Per-point bandwidths equalise local density, which erases information about cluster size and spread.
2. Why does Isomap use shortest paths through a neighbour graph?
   - [x] To approximate distances measured along the curved manifold rather than straight through space
   - [ ] To make the computation faster than PCA
   - [ ] To remove outliers
   > On a swiss roll, the straight-line distance between two layers is small but the distance along the sheet is large.
3. What does the perplexity parameter in t-SNE roughly control?
   - [ ] The number of output dimensions
   - [x] The effective number of neighbours each point considers
   - [ ] The learning rate
   > Perplexity is $2^H$ of each point's neighbour distribution, about the number of neighbours it treats as near.
4. You need 10 compact features for a downstream classifier. What is the better first choice?
   - [x] PCA, a deterministic linear projection that can be applied to new data
   - [ ] t-SNE with 10 output dimensions
   - [ ] A 2-D UMAP plot
   > t-SNE is designed for visualisation, is stochastic, and cannot project new points directly.
<!-- /quiz -->
