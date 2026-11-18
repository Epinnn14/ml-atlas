---
title: Overfitting and the bias–variance trade-off
level: beginner
minutes: 40
summary: Why a more flexible model can predict worse, how test error splits into bias, variance and noise, and how a validation set tells you how much flexibility the data supports.
---

## Underfitting and overfitting

Fit polynomials of increasing degree to twenty noisy points from a smooth curve:

- **Degree 1**: a straight line misses the curve everywhere. Training and test errors are both high. The model is too simple: **underfitting**.
- **Degree 4**: follows the curve, ignores most of the noise. Both errors are low.
- **Degree 15**: passes through almost every training point and swings wildly between them. Training error is near zero, test error is huge: **overfitting**.

Training error always falls as flexibility grows. Test error falls, then rises again. Everything in this lesson is about finding the bottom of that U.

## Polynomial features

A polynomial of degree $p$ in one variable is a linear model on the features $(1, x, x^2, \dots, x^p)$:

<!-- formula: Polynomial regression -->
$$\hat y = w_0 + w_1 x + w_2 x^2 + \dots + w_p x^p$$

So we can fit it with ordinary [least squares](#/lesson/linear-regression). The matrix of powers is called a **Vandermonde matrix**. For high degrees, scale $x$ to roughly $[-1, 1]$ first, or the columns differ by many orders of magnitude.

## The decomposition

Imagine drawing many training sets from the same source and fitting the same kind of model to each. At a fixed input $x$, with true function $f$ and noise variance $\sigma^2$, the expected squared error of the fitted model $\hat f$ splits into three parts:

<!-- formula: Bias–variance decomposition -->
$$\mathbb E\big[(y - \hat f(x))^2\big] = \underbrace{\big(\mathbb E[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}\big[\hat f(x)\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$$

- **Bias**: how far the average fitted model is from the truth. Simple models have high bias.
- **Variance**: how much the fit changes from one training set to another. Flexible models have high variance.
- **Noise**: the part no model can remove.

Making a model more flexible trades bias for variance. More data lowers variance without raising bias, which is why big datasets allow big models.

## Choosing flexibility with a validation set

You cannot pick the degree by training error (it always prefers the most flexible model) and you must not pick it by test error (then the test score is no longer an honest estimate). Split three ways:

1. **Training set**: fit the parameters.
2. **Validation set**: compare settings such as the degree, and pick the best.
3. **Test set**: touched once, at the very end, to report the result.

[Cross-validation](#/lesson/cross-validation) does step 2 more efficiently when data are scarce.

## Learning curves

Plot training and validation error against the number of training examples:

- Both high and close together: **high bias**. More data will not help; use a more flexible model or better features.
- Training low, validation much higher, gap shrinking as data grow: **high variance**. More data, [regularisation](#/lesson/regularization) or a simpler model will help.

<!-- exercise: poly-features | Polynomial features -->
Write `poly_features(x, degree)` that turns a 1-D array of length $n$ into the $(n, \text{degree}+1)$ matrix with columns $x^0, x^1, \dots, x^{\text{degree}}$.

~~~python starter
import numpy as np

def poly_features(x, degree):
    x = np.asarray(x, dtype=float)
    raise NotImplementedError

print(poly_features(np.array([1.0, 2.0, 3.0]), 2))
# [[1 1 1], [1 2 4], [1 3 9]]
~~~

~~~python solution
import numpy as np

def poly_features(x, degree):
    x = np.asarray(x, dtype=float)
    return x[:, None] ** np.arange(degree + 1)

print(poly_features(np.array([1.0, 2.0, 3.0]), 2))
~~~

~~~python tests
import numpy as np

def test_example():
    """degree 2 on [1, 2, 3]"""
    assert approx(poly_features(np.array([1.0, 2.0, 3.0]), 2), [[1, 1, 1], [1, 2, 4], [1, 3, 9]])

def test_shape():
    """shape is (n, degree + 1)"""
    assert poly_features(np.linspace(0, 1, 7), 5).shape == (7, 6)

def test_degree_zero():
    """degree 0 is a single column of ones"""
    assert approx(poly_features(np.array([3.0, -2.0]), 0), [[1.0], [1.0]])

def test_matches_vander():
    """matches np.vander with increasing powers"""
    x = np.array([-1.0, 0.5, 2.0])
    assert approx(poly_features(x, 3), np.vander(x, 4, increasing=True))
~~~

<!-- hint: Broadcasting a column `x[:, None]` against the row of powers `np.arange(degree + 1)` builds the whole matrix. -->
<!-- /exercise -->

<!-- exercise: best-degree | Pick the degree on a validation set -->
Write `validation_errors(x_tr, y_tr, x_val, y_val, degrees)` that, for each degree, fits least squares on the polynomial features of the training data and returns the list of **validation** mean squared errors, in the same order as `degrees`. A `poly_features` helper is provided. Then `best_degree(...)` returns the degree with the lowest validation error.

~~~python starter
import numpy as np

def poly_features(x, degree):
    return np.asarray(x, float)[:, None] ** np.arange(degree + 1)

def validation_errors(x_tr, y_tr, x_val, y_val, degrees):
    raise NotImplementedError

def best_degree(x_tr, y_tr, x_val, y_val, degrees):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def poly_features(x, degree):
    return np.asarray(x, float)[:, None] ** np.arange(degree + 1)

def validation_errors(x_tr, y_tr, x_val, y_val, degrees):
    errs = []
    for p in degrees:
        w = np.linalg.lstsq(poly_features(x_tr, p), y_tr, rcond=None)[0]
        errs.append(float(np.mean((poly_features(x_val, p) @ w - y_val) ** 2)))
    return errs

def best_degree(x_tr, y_tr, x_val, y_val, degrees):
    errs = validation_errors(x_tr, y_tr, x_val, y_val, degrees)
    return degrees[int(np.argmin(errs))]
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2.5 * x)
x_tr = rng.uniform(-1, 1, 25); y_tr = f(x_tr) + 0.15 * rng.normal(size=25)
x_val = rng.uniform(-1, 1, 200); y_val = f(x_val) + 0.15 * rng.normal(size=200)
degrees = [1, 2, 3, 4, 5, 6, 8, 12, 16]

def test_list_of_errors():
    """returns one validation error per degree"""
    e = validation_errors(x_tr, y_tr, x_val, y_val, degrees)
    assert len(e) == len(degrees) and all(v >= 0 for v in e)

def test_u_shape():
    """degree 1 underfits and degree 16 overfits compared with the best"""
    e = validation_errors(x_tr, y_tr, x_val, y_val, degrees)
    assert e[0] > min(e) * 2 and e[-1] > min(e) * 2

def test_training_error_keeps_falling():
    """the training error itself always falls (that is why we validate)"""
    e = validation_errors(x_tr, y_tr, x_tr, y_tr, degrees)
    assert all(a >= b - 1e-9 for a, b in zip(e, e[1:]))

def test_best_degree_moderate():
    """the chosen degree is moderate"""
    assert 3 <= best_degree(x_tr, y_tr, x_val, y_val, degrees) <= 8
~~~

<!-- hint: Fit with `np.linalg.lstsq(poly_features(x_tr, p), y_tr, rcond=None)[0]`, then evaluate on `poly_features(x_val, p)`. -->
<!-- /exercise -->

<!-- quiz -->
1. Training error 0.01, validation error 0.45. What is the most likely problem?
   - [ ] High bias
   - [x] High variance (overfitting)
   - [ ] Too much regularisation
   > A model that fits its training data almost perfectly but generalises badly has learned the noise in the training set.
2. Both training and validation errors are high and almost equal. What will help most?
   - [ ] Collecting much more data
   - [x] A more flexible model or more informative features
   - [ ] Stronger regularisation
   > The model is too simple to capture the pattern (high bias). More data mainly reduces variance, which is not the problem here.
3. Why not choose the polynomial degree by the error on the test set?
   - [x] The test score would then be part of model selection and stop being an unbiased estimate of real performance
   - [ ] The test set is too small to compute an error
   - [ ] Test error does not depend on the degree
   > Any data used to make choices becomes training data in disguise. Keep the test set for the final report.
4. Which part of the expected error can no model remove?
   - [ ] Bias
   - [ ] Variance
   - [x] The noise $\sigma^2$
   > Even the true function cannot predict the random part of $y$.
<!-- /quiz -->
