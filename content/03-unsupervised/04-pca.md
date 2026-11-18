---
title: Principal component analysis
level: intermediate
minutes: 45
summary: Find the directions along which data vary most, project onto a few of them, and reconstruct. PCA through the covariance matrix and through the SVD, explained variance, and when to use it.
---

## Fewer dimensions, most of the information

Data with many features often vary along only a few directions: height and arm span move together, pixels of a face are far from independent. **PCA** finds an orthonormal set of directions, the **principal components**, ordered by how much of the data's variance lies along each. Keeping the first $k$ gives the best $k$-dimensional linear summary of the data.

Uses: compressing features before another model, visualising high-dimensional data in 2-D, removing noise, and decorrelating features.

## Variance along a direction

Centre the data, $X_c = X - \bar{\mathbf x}$. The projection of the data onto a unit vector $\mathbf v$ has variance $\mathbf v^\top S\mathbf v$, where $S$ is the covariance matrix:

<!-- formula: Covariance matrix -->
$$S = \frac{1}{n-1}X_c^\top X_c$$

Maximising $\mathbf v^\top S\mathbf v$ subject to $\lVert\mathbf v\rVert = 1$ (with a Lagrange multiplier) gives $S\mathbf v = \lambda\mathbf v$: the best direction is the top [eigenvector](#/lesson/linear-algebra) of $S$, and the variance along it is the eigenvalue $\lambda$. The second component is the best direction orthogonal to the first, and so on: the components are the eigenvectors of $S$ sorted by eigenvalue.

## Through the SVD

In practice PCA is computed from the **singular value decomposition** of the centred data, which is more accurate than forming $S$:

<!-- formula: SVD of the centred data -->
$$X_c = U\Sigma V^\top$$

The columns of $V$ are the principal components, and the eigenvalues of $S$ are $\lambda_j = \sigma_j^2/(n-1)$. The projected data (the **scores**) are $X_cV_k = U_k\Sigma_k$.

## Projection and reconstruction

With the first $k$ components $V_k$ (shape $d\times k$):

<!-- formula: PCA projection and reconstruction -->
$$Z = X_cV_k, \qquad \hat X = ZV_k^\top + \bar{\mathbf x}$$

Among all rank-$k$ approximations, this reconstruction has the smallest squared error. The fraction of variance kept is the **explained variance ratio**:

<!-- formula: Explained variance ratio -->
$$\frac{\sum_{j=1}^k\lambda_j}{\sum_{j=1}^d\lambda_j}$$

A common choice is the smallest $k$ that keeps 90–95% of the variance, or the elbow of the **scree plot** of eigenvalues.

## Cautions

- **Scale matters.** PCA chases variance, so a feature measured in large units dominates. Standardise first unless the units are comparable.
- **Directions, not features.** Each component mixes all the original features, so they are harder to interpret.
- **Linear only.** Data on a curved surface (a "swiss roll") need [non-linear methods](#/lesson/manifold-learning).
- **Variance is not relevance.** The directions with most variance are not always the ones that predict the label; a low-variance direction can carry the signal.

<!-- exercise: pca-svd | PCA with the SVD -->
Write a class `PCA(k)` with `fit(X)` storing `self.mean_`, `self.components_` (shape `(k, d)`, rows are unit principal directions, sorted by variance) and `self.explained_variance_` (the top-$k$ eigenvalues $\sigma_j^2/(n-1)$), then `transform(X)` and `inverse_transform(Z)`. Use `np.linalg.svd(Xc, full_matrices=False)`.

~~~python starter
import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        X = np.asarray(X, float)
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, Z):
        raise NotImplementedError
~~~

~~~python solution
import numpy as np

class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        U, s, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[: self.k]
        self.explained_variance_ = s[: self.k] ** 2 / (len(X) - 1)
        return self

    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
latent = rng.normal(size=(500, 2)) * np.array([3.0, 1.0])
mix = np.array([[1.0, 0.5, 0.2, 0.0], [0.0, 0.3, -0.4, 1.0]])
X = latent @ mix + 0.05 * rng.normal(size=(500, 4)) + 7

def test_shapes():
    """components are (k, d) and transform gives (n, k)"""
    p = PCA(2).fit(X)
    assert p.components_.shape == (2, 4) and p.transform(X).shape == (500, 2)

def test_orthonormal():
    """components are orthonormal"""
    C = PCA(3).fit(X).components_
    assert approx(C @ C.T, np.eye(3), tol=1e-8)

def test_eigenvalues():
    """explained variances are the top eigenvalues of the covariance matrix"""
    ev = np.sort(np.linalg.eigvalsh(np.cov(X, rowvar=False)))[::-1]
    assert approx(PCA(3).fit(X).explained_variance_, ev[:3], tol=1e-8)

def test_reconstruction():
    """2 components reconstruct this nearly 2-D data almost perfectly"""
    p = PCA(2).fit(X)
    err = np.mean((p.inverse_transform(p.transform(X)) - X) ** 2)
    assert err < 0.01

def test_scores_uncorrelated():
    """the projected coordinates are uncorrelated, with variances = explained_variance_"""
    p = PCA(2).fit(X)
    Z = p.transform(X)
    C = np.cov(Z, rowvar=False)
    assert abs(C[0, 1]) < 1e-8 and approx(np.diag(C), p.explained_variance_, tol=1e-8)
~~~

<!-- hint: The rows of `Vt` from the SVD of the centred data are the principal directions, already sorted. -->
<!-- /exercise -->

<!-- exercise: choose-k | How many components? -->
Write `components_for(X, threshold)` returning the smallest $k$ whose cumulative explained variance ratio reaches at least `threshold` (for example 0.95). Compute all eigenvalues from the singular values of the centred data.

~~~python starter
import numpy as np

def components_for(X, threshold):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def components_for(X, threshold):
    X = np.asarray(X, float)
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    ratio = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(ratio, threshold) + 1)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(400, 6)) * np.array([10.0, 5.0, 1.0, 0.5, 0.1, 0.1])

def ratio_of(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return np.cumsum(s ** 2) / np.sum(s ** 2)

def test_matches_cumulative_ratio():
    """the answer is the first k reaching the threshold"""
    r = ratio_of(X)
    for th in (0.5, 0.8, 0.95, 0.999):
        k = components_for(X, th)
        assert r[k - 1] >= th and (k == 1 or r[k - 2] < th)

def test_two_big_directions():
    """two dominant directions hold over 95% here"""
    assert components_for(X, 0.95) == 2

def test_full_threshold():
    """a threshold of 1.0 can need every component"""
    assert components_for(X, 1.0) <= 6
~~~

<!-- hint: `np.cumsum(s**2) / np.sum(s**2)` is the cumulative ratio; `np.searchsorted` finds the first position reaching the threshold. -->
<!-- /exercise -->

<!-- quiz -->
1. You run PCA on features "income in rupiah" and "age in years" without scaling. What will the first component be?
   - [x] Almost entirely the income direction, because its variance is enormous in those units
   - [ ] An equal mix of both
   - [ ] Almost entirely age
   > PCA maximises variance in the units given. Standardise first so each feature starts with variance 1.
2. The first three components explain 98% of the variance of 50 features. What does that suggest?
   - [ ] Three of the features are the important ones
   - [x] The data lie close to a 3-dimensional linear subspace
   - [ ] The other 47 features are noise and must be deleted
   > Components are combinations of all features; a high ratio means three combinations capture almost all variation.
3. Why compute PCA from the SVD of the data rather than the eigenvectors of $X^\top X$?
   - [ ] They give different components
   - [x] It avoids squaring the data matrix, which loses numerical precision
   - [ ] The SVD is the only method that centres the data
   > Forming $X^\top X$ squares the condition number; the SVD works on $X$ directly.
4. For a classification task, why might the top principal components be a poor choice of features?
   - [x] Directions of largest variance need not be the ones that separate the classes
   - [ ] PCA cannot be used before classification
   - [ ] They are correlated with each other
   > PCA ignores the labels. A small-variance direction can be the one that matters.
<!-- /quiz -->
