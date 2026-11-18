---
title: Linear algebra for ML
level: beginner
minutes: 40
summary: Vectors, dot products, norms, matrices as transformations, solving linear systems, projections and eigenvectors, with the meaning each one has inside a model.
---

## Vectors and the dot product

A vector $\mathbf x \in \mathbb{R}^d$ is a list of $d$ numbers: one example's features, or a model's weights. The **dot product** of two vectors multiplies them entry by entry and adds:

<!-- formula: Dot product -->
$$\mathbf a^\top \mathbf b = \sum_{j=1}^{d} a_j b_j = \lVert\mathbf a\rVert\,\lVert\mathbf b\rVert \cos\theta$$

A linear model's prediction $\mathbf w^\top \mathbf x$ is exactly a dot product: a weighted sum of the features. The second form says the dot product also measures **alignment**: it is largest when the vectors point the same way, zero when they are perpendicular (**orthogonal**), negative when they point apart.

## Norms

A norm measures the length of a vector. Two appear everywhere:

<!-- formula: L2 and L1 norms -->
$$\lVert\mathbf x\rVert_2 = \sqrt{\textstyle\sum_j x_j^2}, \qquad \lVert\mathbf x\rVert_1 = \textstyle\sum_j |x_j|$$

The L2 norm is ordinary distance; the L1 norm adds absolute values. You will meet them again as penalties in [regularisation](#/lesson/regularization), where they behave very differently.

Dividing by the norms leaves only the angle, the **cosine similarity**, used to compare word embeddings and documents:

<!-- formula: Cosine similarity -->
$$\cos\theta = \frac{\mathbf a^\top\mathbf b}{\lVert\mathbf a\rVert_2\,\lVert\mathbf b\rVert_2}$$

## Matrices are functions

A matrix $A \in \mathbb{R}^{m \times n}$ maps a vector in $\mathbb{R}^n$ to one in $\mathbb{R}^m$ by $\mathbf x \mapsto A\mathbf x$. Every layer of a neural network starts with such a map. Useful facts:

- $(AB)^\top = B^\top A^\top$, and $(AB)C = A(BC)$, but in general $AB \neq BA$.
- The **identity** $I$ leaves vectors unchanged. The **inverse** $A^{-1}$, when it exists, undoes $A$: $A^{-1}A = I$.
- The **rank** of $A$ is the number of independent directions it can produce. If the columns of a data matrix are linearly dependent (one feature is the sum of two others), $X^\top X$ is not invertible.

## Solving linear systems

Many models end in "find $\mathbf w$ with $A\mathbf w = \mathbf b$". In code, never compute `inv(A) @ b`: it is slower and less accurate. Use `np.linalg.solve(A, b)`, which factorises $A$ instead.

## Projection

The projection of $\mathbf a$ onto the direction of $\mathbf b$ is the closest point to $\mathbf a$ on the line through $\mathbf b$:

<!-- formula: Projection onto a vector -->
$$\operatorname{proj}_{\mathbf b}\mathbf a = \frac{\mathbf a^\top \mathbf b}{\mathbf b^\top\mathbf b}\,\mathbf b$$

What is left, $\mathbf a - \operatorname{proj}_{\mathbf b}\mathbf a$, is orthogonal to $\mathbf b$. [Least squares](#/lesson/linear-regression) is exactly this idea in many dimensions: the predictions are the projection of $\mathbf y$ onto the space spanned by the feature columns.

## Eigenvectors

An eigenvector of a square matrix $A$ is a direction that $A$ only stretches:

<!-- formula: Eigenvalue equation -->
$$A\mathbf v = \lambda \mathbf v$$

$\lambda$ is the stretch factor, the eigenvalue. For a symmetric matrix such as a covariance matrix, the eigenvectors are orthogonal and the one with the largest eigenvalue points along the direction of greatest variance, which is the heart of [PCA](#/lesson/pca).

**Power iteration** finds that top eigenvector: start from a random vector and repeatedly multiply by $A$ and normalise. Each multiplication grows the component along the top eigenvector fastest, so the others fade away.

<!-- exercise: project | Projection and cosine similarity -->
Write `project(a, b)` returning the projection of `a` onto `b`, and `cosine(a, b)` returning the cosine similarity as a float.

~~~python starter
import numpy as np

def project(a, b):
    raise NotImplementedError

def cosine(a, b):
    raise NotImplementedError

a, b = np.array([3.0, 4.0]), np.array([1.0, 0.0])
print(project(a, b), cosine(a, b))   # [3, 0] and 0.6
~~~

~~~python solution
import numpy as np

def project(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a @ b) / (b @ b) * b

def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

a, b = np.array([3.0, 4.0]), np.array([1.0, 0.0])
print(project(a, b), cosine(a, b))
~~~

~~~python tests
import numpy as np

def test_projection_example():
    """projecting (3, 4) onto the x axis gives (3, 0)"""
    assert approx(project(np.array([3.0, 4.0]), np.array([1.0, 0.0])), [3.0, 0.0])

def test_residual_orthogonal():
    """a minus its projection is orthogonal to b"""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5), rng.normal(size=5)
    assert abs((a - project(a, b)) @ b) < 1e-9

def test_scale_of_b_does_not_matter():
    """projecting onto b or 10 b gives the same point"""
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, -1.0, 2.0])
    assert approx(project(a, b), project(a, 10 * b))

def test_cosine_values():
    """cosine is 1 for the same direction, 0 for perpendicular, -1 for opposite"""
    assert approx(cosine(np.array([1.0, 1.0]), np.array([2.0, 2.0])), 1.0)
    assert approx(cosine(np.array([1.0, 0.0]), np.array([0.0, 3.0])), 0.0)
    assert approx(cosine(np.array([1.0, 2.0]), np.array([-1.0, -2.0])), -1.0)
    assert isinstance(cosine(np.array([1.0, 0.0]), np.array([1.0, 1.0])), float)
~~~

<!-- hint: `a @ b` is the dot product of two 1-D arrays; `np.linalg.norm(a)` is the L2 norm. -->
<!-- /exercise -->

<!-- exercise: power-iteration | Power iteration -->
Write `power_iteration(A, steps=200, seed=0)` for a symmetric matrix `A`. Start from `np.random.default_rng(seed).normal(size=n)`, then repeat: multiply by `A`, divide by the norm. Return `(eigenvalue, eigenvector)`, where the eigenvalue is the **Rayleigh quotient** $\lambda = \mathbf v^\top A \mathbf v$ for the final unit vector $\mathbf v$.

~~~python starter
import numpy as np

def power_iteration(A, steps=200, seed=0):
    A = np.asarray(A, float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    raise NotImplementedError

A = np.array([[2.0, 1.0], [1.0, 2.0]])
print(power_iteration(A))   # eigenvalue 3, vector along (1, 1) / sqrt(2)
~~~

~~~python solution
import numpy as np

def power_iteration(A, steps=200, seed=0):
    A = np.asarray(A, float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(steps):
        v = A @ v
        v /= np.linalg.norm(v)
    return float(v @ A @ v), v

A = np.array([[2.0, 1.0], [1.0, 2.0]])
print(power_iteration(A))
~~~

~~~python tests
import numpy as np

def test_two_by_two():
    """finds eigenvalue 3 of [[2,1],[1,2]]"""
    lam, v = power_iteration(np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert approx(lam, 3.0, tol=1e-6)
    assert approx(abs(v), np.array([1.0, 1.0]) / np.sqrt(2), tol=1e-6)

def test_unit_length():
    """returns a unit vector"""
    _, v = power_iteration(np.diag([1.0, 5.0, 2.0]))
    assert approx(np.linalg.norm(v), 1.0)

def test_matches_numpy():
    """agrees with np.linalg.eigh on a random covariance matrix"""
    rng = np.random.default_rng(4)
    M = rng.normal(size=(30, 6))
    C = M.T @ M / 30
    lam, v = power_iteration(C, steps=500)
    vals, vecs = np.linalg.eigh(C)
    assert approx(lam, vals[-1], tol=1e-6)
    assert approx(abs(v @ vecs[:, -1]), 1.0, tol=1e-6)

def test_is_eigenvector():
    """A v equals lambda v"""
    C = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    lam, v = power_iteration(C, steps=1000)
    assert approx(C @ v, lam * v, tol=1e-6)
~~~

<!-- hint: Normalise with `v = v / np.linalg.norm(v)` after every multiplication, otherwise the numbers explode or vanish. -->
<!-- hint: For a unit vector, `v @ A @ v` is the Rayleigh quotient. -->
<!-- /exercise -->

<!-- quiz -->
1. The dot product of two non-zero vectors is 0. What does that tell you?
   - [ ] One of them must be the zero vector
   - [x] They are orthogonal (perpendicular)
   - [ ] They point in the same direction
   > $\mathbf a^\top\mathbf b = \lVert\mathbf a\rVert\lVert\mathbf b\rVert\cos\theta$; with both norms positive, the product is 0 only when $\cos\theta = 0$.
2. Why prefer `np.linalg.solve(A, b)` over `np.linalg.inv(A) @ b`?
   - [ ] `solve` works for non-square matrices and `inv` does not
   - [x] It factorises A instead of forming the inverse, which is faster and more accurate
   - [ ] They give different answers and only `solve` is correct
   > Mathematically both give $A^{-1}\mathbf b$, but explicitly inverting does more work and loses precision when $A$ is close to singular.
3. One feature in your data is exactly twice another feature. What happens to $X^\top X$?
   - [x] It becomes singular (not invertible), because its columns are linearly dependent
   - [ ] Nothing, duplicated information is harmless in linear algebra
   - [ ] Its eigenvalues all double
   > Linearly dependent columns lower the rank of $X$, and $X^\top X$ inherits that rank, so it has a zero eigenvalue and no inverse.
4. Power iteration converges slowly. What property of the matrix causes that?
   - [ ] The matrix is symmetric
   - [x] The two largest eigenvalues are close in size
   - [ ] The starting vector was random
   > Each step shrinks the second component relative to the first by the ratio $|\lambda_2/\lambda_1|$. When that ratio is near 1, many steps are needed.
<!-- /quiz -->
