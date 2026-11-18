---
title: Support vector machines
level: advanced
minutes: 50
summary: The widest possible margin between classes. Hinge loss and soft margins, training a linear SVM by sub-gradient descent, and kernels that draw curved boundaries without ever computing new features.
---

## The widest street

Many lines separate two linearly separable classes. A support vector machine picks the one with the largest **margin**: the widest empty street between the classes. Only the points on the edges of the street, the **support vectors**, determine it; moving any other point (without crossing the street) changes nothing.

With labels $y_i\in\{-1, +1\}$ and classifier $f(\mathbf x) = \mathbf w^\top\mathbf x + b$, scale $\mathbf w$ so the closest points satisfy $y_i f(\mathbf x_i) = 1$. The margin width is then $2/\lVert\mathbf w\rVert$, so maximising it means minimising $\lVert\mathbf w\rVert$:

<!-- formula: Hard-margin SVM -->
$$\min_{\mathbf w, b}\ \tfrac12\lVert\mathbf w\rVert^2 \quad\text{s.t.}\quad y_i(\mathbf w^\top\mathbf x_i + b) \ge 1\ \ \forall i$$

## Soft margins and the hinge loss

Real data overlap, so we allow violations and pay for them. The **hinge loss** is zero for points safely on the right side ($y f \ge 1$) and grows linearly otherwise:

<!-- formula: Hinge loss -->
$$\ell_{\text{hinge}}(y, f) = \max\big(0,\ 1 - y\,f\big)$$

<!-- formula: Soft-margin SVM objective -->
$$\min_{\mathbf w, b}\ \frac{\lambda}{2}\lVert\mathbf w\rVert^2 + \frac1n\sum_{i=1}^n \max\big(0,\ 1 - y_i(\mathbf w^\top\mathbf x_i + b)\big)$$

This is regularised empirical risk minimisation again, like [logistic regression](#/lesson/logistic-regression) with L2: only the loss differs. The hinge loss ignores points that are already correct with margin, which is why only support vectors matter. Small $\lambda$ (large $C = 1/(\lambda n)$ in library notation) fits the training data tightly; large $\lambda$ prefers a wide margin with more violations.

## Training by sub-gradient descent

The hinge loss has a kink at $yf = 1$, but a **sub-gradient** works fine. For each example with $y_i f(\mathbf x_i) < 1$, the loss contributes $-y_i\mathbf x_i$ (and $-y_i$ for $b$); other examples contribute nothing:

<!-- formula: SVM sub-gradient -->
$$\mathbf g_{\mathbf w} = \lambda\mathbf w - \frac1n\sum_{i:\,y_if_i<1} y_i\mathbf x_i, \qquad g_b = -\frac1n\sum_{i:\,y_if_i<1} y_i$$

The Pegasos algorithm does this with mini-batches and a step size $\eta_t = 1/(\lambda t)$. Classic libraries instead solve the **dual** problem, a quadratic program over one coefficient per example, which is what makes kernels possible.

## Kernels

In the dual, the data appear only through dot products $\mathbf x_i^\top\mathbf x_j$, and the decision function becomes $f(\mathbf x) = \sum_i \alpha_i y_i\, \mathbf x_i^\top\mathbf x + b$ with $\alpha_i > 0$ only for support vectors. Replacing every dot product by a **kernel** $k(\mathbf x, \mathbf x')$ fits a linear SVM in a (possibly infinite-dimensional) feature space without ever computing the features: the **kernel trick**. The most used is the Gaussian **RBF kernel**:

<!-- formula: RBF kernel -->
$$k(\mathbf x, \mathbf x') = \exp\!\big(-\gamma\,\lVert\mathbf x - \mathbf x'\rVert^2\big)$$

Large $\gamma$ makes each point's influence local and the boundary wiggly (risk of overfitting); small $\gamma$ makes it smooth. $\gamma$ and $C$ are tuned together by [cross-validation](#/lesson/cross-validation). A valid kernel's Gram matrix $K_{ij} = k(\mathbf x_i,\mathbf x_j)$ is always symmetric positive semi-definite. Kernels return in [Gaussian processes](#/lesson/gaussian-processes).

## Where SVMs stand today

Kernel SVMs were the state of the art for many tasks before deep learning. Their cost grows at least quadratically with $n$ (the Gram matrix is $n\times n$), so they fit small and medium datasets. Linear SVMs remain excellent for high-dimensional sparse data such as text.

<!-- exercise: linear-svm | Train a linear SVM -->
Write `fit_svm(X, y, lam=0.01, lr=0.1, steps=1000)` for labels in $\{-1, +1\}$: start from zeros and run full-batch sub-gradient descent with the formulas above. Return `(w, b)`. Also write `hinge_objective(X, y, w, b, lam)` returning the soft-margin objective as a float.

~~~python starter
import numpy as np

def hinge_objective(X, y, w, b, lam):
    raise NotImplementedError

def fit_svm(X, y, lam=0.01, lr=0.1, steps=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def hinge_objective(X, y, w, b, lam):
    margins = y * (X @ w + b)
    return float(lam / 2 * w @ w + np.mean(np.maximum(0, 1 - margins)))

def fit_svm(X, y, lam=0.01, lr=0.1, steps=1000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        active = y * (X @ w + b) < 1
        gw = lam * w - (y[active] @ X[active]) / n
        gb = -np.sum(y[active]) / n
        w -= lr * gw
        b -= lr * gb
    return w, b
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 300
y = rng.choice([-1, 1], n)
X = rng.normal(size=(n, 2)) + 1.8 * y[:, None] * np.array([1.0, 0.5])

def test_objective_at_zero():
    """at w = 0, b = 0 every hinge term is 1"""
    assert approx(hinge_objective(X, y, np.zeros(2), 0.0, 0.01), 1.0)

def test_accuracy():
    """separates the two clouds"""
    w, b = fit_svm(X, y)
    assert np.mean(np.sign(X @ w + b) == y) > 0.95

def test_objective_decreases():
    """training lowers the objective"""
    w1, b1 = fit_svm(X, y, steps=10)
    w2, b2 = fit_svm(X, y, steps=1000)
    assert hinge_objective(X, y, w2, b2, 0.01) < hinge_objective(X, y, w1, b1, 0.01) < 1.0

def test_stronger_penalty_smaller_weights():
    """a larger lambda gives a smaller w (a wider margin)"""
    wa, _ = fit_svm(X, y, lam=0.001)
    wb, _ = fit_svm(X, y, lam=0.5)
    assert np.linalg.norm(wb) < np.linalg.norm(wa)

def test_first_step():
    """the first step from zero uses every example (all margins are 0 < 1)"""
    w, b = fit_svm(X, y, lam=0.01, lr=1.0, steps=1)
    assert approx(w, y @ X / n) and approx(b, np.sum(y) / n)
~~~

<!-- hint: `active = y * (X @ w + b) < 1` selects the examples that contribute to the sub-gradient. -->
<!-- hint: `y[active] @ X[active]` sums $y_i\mathbf x_i$ over the active examples. -->
<!-- /exercise -->

<!-- exercise: rbf | The RBF Gram matrix -->
Write `rbf_kernel(A, B, gamma)` returning the `(n, m)` matrix $\exp(-\gamma\lVert\mathbf a_i - \mathbf b_j\rVert^2)$ without Python loops.

~~~python starter
import numpy as np

def rbf_kernel(A, B, gamma):
    A, B = np.asarray(A, float), np.asarray(B, float)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def rbf_kernel(A, B, gamma):
    A, B = np.asarray(A, float), np.asarray(B, float)
    d2 = np.sum(A ** 2, 1)[:, None] - 2 * A @ B.T + np.sum(B ** 2, 1)[None, :]
    return np.exp(-gamma * np.maximum(d2, 0.0))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
A, B = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))

def test_values():
    """matches the formula entry by entry"""
    K = rbf_kernel(A, B, 0.7)
    slow = np.array([[np.exp(-0.7 * np.sum((a - b) ** 2)) for b in B] for a in A])
    assert approx(K, slow)

def test_diagonal_ones():
    """k(x, x) = 1"""
    assert approx(np.diag(rbf_kernel(A, A, 2.0)), np.ones(6))

def test_psd():
    """the Gram matrix of one set is symmetric positive semi-definite"""
    K = rbf_kernel(A, A, 0.5)
    assert approx(K, K.T) and np.min(np.linalg.eigvalsh(K)) > -1e-10

def test_gamma_effect():
    """a larger gamma makes off-diagonal similarities smaller"""
    assert np.mean(rbf_kernel(A, B, 5.0)) < np.mean(rbf_kernel(A, B, 0.1))
~~~

<!-- hint: Reuse the expanded squared distance from the NumPy lesson, then apply `np.exp(-gamma * d2)`. -->
<!-- /exercise -->

<!-- quiz -->
1. You remove a training point that lies far from the margin, on the correct side. What happens to a fitted SVM?
   - [x] Nothing; only support vectors determine the boundary
   - [ ] The margin shrinks
   - [ ] The model must be retrained from scratch and changes completely
   > Points with $y f(\mathbf x) > 1$ have zero hinge loss and zero gradient, so they do not influence the solution.
2. What is the main difference between a linear SVM and L2-regularised logistic regression?
   - [ ] The SVM is non-linear
   - [x] The loss: hinge instead of logistic, so the SVM ignores well-classified points and gives no probabilities
   - [ ] The SVM has no regularisation
   > Both minimise a penalty plus an average loss over the same linear scores.
3. An RBF SVM has near-perfect training accuracy and poor validation accuracy. Which change is most likely to help?
   - [ ] Increase $\gamma$
   - [x] Decrease $\gamma$ and/or decrease $C$ (more regularisation)
   - [ ] Remove the kernel
   > A large $\gamma$ gives every point a tiny radius of influence, and a large $C$ punishes every training error: both overfit.
4. What does the kernel trick avoid?
   - [ ] Computing dot products
   - [x] Explicitly computing the high-dimensional feature map; only kernel values between pairs of points are needed
   - [ ] The need to choose hyperparameters
   > The RBF kernel corresponds to an infinite-dimensional feature space that could never be computed directly.
<!-- /quiz -->
