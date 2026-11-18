---
title: Derivatives and gradients
level: beginner
minutes: 35
summary: How a function changes when its input moves a little, the gradient as the direction of steepest increase, the chain rule behind backpropagation, and checking gradients numerically.
---

## The derivative

The derivative of $f$ at $x$ is the slope of $f$ there: how much the output changes per unit change of the input, for a very small change.

<!-- formula: Derivative -->
$$f'(x) = \lim_{h\to 0}\frac{f(x+h) - f(x)}{h}$$

Rules you will use all the time: $(x^n)' = n x^{n-1}$, $(e^x)' = e^x$, $(\ln x)' = 1/x$, and derivatives of sums are sums of derivatives.

Training a model is choosing parameters that make a loss small. The derivative says which way is downhill: if $f'(x) > 0$, decreasing $x$ decreases $f$.

## Partial derivatives and the gradient

A loss depends on many parameters $\mathbf w = (w_1, \dots, w_d)$. The **partial derivative** $\partial f / \partial w_j$ is the slope along $w_j$ with every other parameter held fixed. Collected into a vector they form the **gradient**:

<!-- formula: Gradient -->
$$\nabla f(\mathbf w) = \left(\frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_d}\right)^{\!\top}$$

The gradient points in the direction in which $f$ increases fastest, and its length is that rate. So $-\nabla f$ is the steepest way down, which is the whole idea of [gradient descent](#/lesson/gradient-descent).

## The chain rule

Models are compositions: a linear map, then a non-linearity, then a loss. The chain rule says how slopes multiply through a composition:

<!-- formula: Chain rule -->
$$\frac{d}{dx} f\big(g(x)\big) = f'\big(g(x)\big)\, g'(x)$$

For example $\frac{d}{dw}(wx - y)^2 = 2(wx - y)\cdot x$: the outer square contributes $2(wx-y)$ and the inner linear part contributes $x$. **Backpropagation** in neural networks is nothing more than the chain rule applied layer by layer, reusing the products already computed ([autograd](#/lesson/autograd) builds it from scratch).

## Gradients of common losses

Two results worth knowing by heart, for $X \in \mathbb{R}^{n\times d}$, $\mathbf y \in \mathbb{R}^n$:

<!-- formula: Gradient of the squared error -->
$$\nabla_{\mathbf w}\,\lVert X\mathbf w - \mathbf y\rVert^2 = 2X^\top(X\mathbf w - \mathbf y)$$

<!-- formula: Gradient of a quadratic form -->
$$\nabla_{\mathbf w}\,\mathbf w^\top A \mathbf w = (A + A^\top)\mathbf w$$

The shape check is a good habit: $X^\top$ is $(d, n)$, the residual $X\mathbf w - \mathbf y$ is $(n,)$, so the gradient is $(d,)$, one entry per weight, as it must be.

## Checking a gradient numerically

Hand-derived gradients are easy to get wrong. The **central difference** estimates each partial derivative from two function evaluations:

<!-- formula: Central difference -->
$$\frac{\partial f}{\partial w_j} \approx \frac{f(\mathbf w + h\,\mathbf e_j) - f(\mathbf w - h\,\mathbf e_j)}{2h}$$

where $\mathbf e_j$ is the $j$-th unit vector. Its error shrinks like $h^2$, much faster than the one-sided formula. With $h \approx 10^{-5}$ it agrees with a correct analytic gradient to about 6 digits. It needs $2d$ evaluations, so it is a test, not a training method.

<!-- exercise: numgrad | Numerical gradient -->
Write `numerical_gradient(f, w, h=1e-5)` that returns the central-difference estimate of the gradient of the scalar function `f` at the vector `w`. Do not change `w` itself.

~~~python starter
import numpy as np

def numerical_gradient(f, w, h=1e-5):
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)
    # for each j: nudge w[j] up and down by h
    raise NotImplementedError

f = lambda w: w[0] ** 2 + 3 * w[0] * w[1]
print(numerical_gradient(f, np.array([1.0, 2.0])))   # about [8, 3]
~~~

~~~python solution
import numpy as np

def numerical_gradient(f, w, h=1e-5):
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)
    for j in range(w.size):
        e = np.zeros_like(w)
        e.flat[j] = h
        grad.flat[j] = (f(w + e) - f(w - e)) / (2 * h)
    return grad

f = lambda w: w[0] ** 2 + 3 * w[0] * w[1]
print(numerical_gradient(f, np.array([1.0, 2.0])))
~~~

~~~python tests
import numpy as np

def test_polynomial():
    """f = w0^2 + 3 w0 w1 has gradient (2 w0 + 3 w1, 3 w0)"""
    f = lambda w: w[0] ** 2 + 3 * w[0] * w[1]
    assert approx(numerical_gradient(f, np.array([1.0, 2.0])), [8.0, 3.0], tol=1e-6)

def test_exp_sum():
    """the gradient of sum(exp(w)) is exp(w)"""
    w = np.array([-1.0, 0.0, 0.5])
    assert approx(numerical_gradient(lambda v: np.sum(np.exp(v)), w), np.exp(w), tol=1e-6)

def test_w_unchanged():
    """does not modify the input vector"""
    w = np.array([1.0, 2.0, 3.0])
    numerical_gradient(lambda v: np.sum(v ** 2), w)
    assert np.array_equal(w, [1.0, 2.0, 3.0])

def test_shape():
    """returns one partial derivative per entry"""
    assert numerical_gradient(lambda v: float(v @ v), np.ones(7)).shape == (7,)
~~~

<!-- hint: Build a vector `e` of zeros with `h` at position `j`, then evaluate `f(w + e)` and `f(w - e)`. -->
<!-- /exercise -->

<!-- exercise: lsq-grad | The gradient of the squared error -->
Write `loss(w, X, y)` returning $\lVert X\mathbf w - \mathbf y\rVert^2$ as a float, and `grad(w, X, y)` returning its gradient $2X^\top(X\mathbf w - \mathbf y)$. The tests compare your gradient with a numerical one.

~~~python starter
import numpy as np

def loss(w, X, y):
    raise NotImplementedError

def grad(w, X, y):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def loss(w, X, y):
    r = X @ w - y
    return float(r @ r)

def grad(w, X, y):
    return 2 * X.T @ (X @ w - y)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X, y, w = rng.normal(size=(30, 4)), rng.normal(size=30), rng.normal(size=4)

def _num(f, w, h=1e-5):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

def test_loss_value():
    """loss is the sum of squared residuals"""
    assert approx(loss(w, X, y), np.sum((X @ w - y) ** 2))

def test_gradient_matches_numerical():
    """the analytic gradient matches central differences"""
    assert approx(grad(w, X, y), _num(lambda v: loss(v, X, y), w), tol=1e-5)

def test_gradient_shape():
    """one gradient entry per weight"""
    assert np.shape(grad(w, X, y)) == (4,)

def test_zero_at_least_squares():
    """the gradient vanishes at the least-squares solution"""
    w_star = np.linalg.lstsq(X, y, rcond=None)[0]
    assert approx(grad(w_star, X, y), np.zeros(4), tol=1e-8)
~~~

<!-- hint: The residual `r = X @ w - y` has shape `(n,)`; the loss is `r @ r`. -->
<!-- /exercise -->

<!-- quiz -->
1. At a point where the gradient of the loss is the zero vector, what can you say?
   - [ ] The loss is at its global minimum
   - [x] It is a stationary point: a minimum, a maximum or a saddle point
   - [ ] The learning rate was too small
   > A zero gradient only means no direction is downhill to first order. For convex losses such as least squares it is the global minimum; in general it might not be.
2. What is $\frac{d}{dx}\, e^{3x^2}$?
   - [ ] $e^{3x^2}$
   - [ ] $3x^2 e^{3x^2 - 1}$
   - [x] $6x\, e^{3x^2}$
   > Chain rule: the outer exponential gives $e^{3x^2}$, the inner $3x^2$ gives $6x$.
3. Why use the central difference $\frac{f(x+h) - f(x-h)}{2h}$ instead of $\frac{f(x+h)-f(x)}{h}$?
   - [x] Its error shrinks like $h^2$ instead of $h$, so it is far more accurate for the same $h$
   - [ ] It needs fewer function evaluations
   - [ ] The one-sided formula does not work for vectors
   > Expanding both in a Taylor series, the first-order error terms cancel in the central difference.
4. $X$ is $(n, d)$. What is the shape of $X^\top (X\mathbf w - \mathbf y)$?
   - [ ] $(n,)$
   - [x] $(d,)$
   - [ ] $(n, d)$
   > $X^\top$ is $(d, n)$ times a vector of length $n$, which gives one entry per weight.
<!-- /quiz -->
