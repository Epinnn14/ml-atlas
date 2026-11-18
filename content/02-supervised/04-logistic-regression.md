---
title: Logistic regression
level: beginner
minutes: 40
summary: Classification with probabilities. Squash a linear score through the sigmoid, train with the cross-entropy loss, and read the weights as changes in log-odds.
---

## From scores to probabilities

For a yes/no target $y \in \{0, 1\}$ (spam or not, default or not), we want the probability $P(y = 1 \mid \mathbf x)$. A linear score $z = \mathbf w^\top\mathbf x + b$ can be any number, so we squash it into $(0, 1)$ with the **sigmoid**:

<!-- formula: Sigmoid -->
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$\sigma(0) = 0.5$, large positive scores give probabilities near 1, large negative ones near 0, and $\sigma(-z) = 1 - \sigma(z)$. The model is

<!-- formula: Logistic regression -->
$$P(y = 1\mid\mathbf x) = \sigma(\mathbf w^\top\mathbf x + b)$$

To classify, predict 1 when the probability is above a threshold, usually 0.5, which is the same as $z > 0$. The **decision boundary** $\mathbf w^\top\mathbf x + b = 0$ is a straight line (a hyperplane in more dimensions), so logistic regression is a linear classifier.

## Log-odds

Inverting the sigmoid gives the **logit**:

<!-- formula: Log-odds (logit) -->
$$\log\frac{p}{1-p} = \mathbf w^\top\mathbf x + b$$

So the model is linear in the log-odds. A weight $w_j = 0.7$ means one more unit of $x_j$ multiplies the odds by $e^{0.7} \approx 2$. This makes logistic regression a favourite in medicine and credit scoring, where effects must be explained.

## The loss: binary cross-entropy

Squared error on probabilities works badly here (its loss surface is not convex for this model). The principled choice is maximum likelihood: each label is a Bernoulli draw with probability $p_i = \sigma(z_i)$, and its negative log-likelihood is

<!-- formula: Binary cross-entropy -->
$$\mathcal L = -\frac1n\sum_{i=1}^n \Big[y_i\log p_i + (1 - y_i)\log(1 - p_i)\Big]$$

the [cross-entropy](#/lesson/information-theory) between labels and predictions. It is convex in $\mathbf w$, and its gradient is strikingly simple:

<!-- formula: Logistic regression gradient -->
$$\nabla_{\mathbf w}\mathcal L = \frac1n X^\top(\mathbf p - \mathbf y), \qquad \frac{\partial\mathcal L}{\partial b} = \frac1n\sum_i (p_i - y_i)$$

the same shape as for linear regression: features times errors. There is no closed form, so we use [gradient descent](#/lesson/gradient-descent).

## Numerical care

`1 / (1 + np.exp(-z))` overflows for large negative $z$ (NumPy warns and the result is still fine), and $\log(p)$ fails when $p$ rounds to exactly 0 or 1. Two robust habits:

- Compute the sigmoid with `np.exp(-np.abs(z))`, which never overflows, choosing the formula by the sign of $z$.
- Compute the loss directly from the scores: $\log(1 + e^{z}) - yz$ per example, with `np.logaddexp(0, z)` for $\log(1 + e^z)$.

## Separable data

If a line separates the classes perfectly, the loss keeps decreasing as the weights grow, and they run off to infinity, giving probabilities of exactly 0 and 1. [Regularisation](#/lesson/regularization) (an L2 penalty) keeps the weights finite and the probabilities sensible. More than two classes need the [softmax](#/lesson/softmax-regression) generalisation.

<!-- exercise: sigmoid-bce | A stable sigmoid and loss -->
Write `sigmoid(z)` that works element-wise without overflow warnings for any `z` (try `z = -1000`), and `bce_from_logits(z, y)` that returns the mean binary cross-entropy computed from the scores `z`, as $\frac1n\sum_i [\log(1 + e^{z_i}) - y_i z_i]$ using `np.logaddexp(0, z)`.

~~~python starter
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    raise NotImplementedError

def bce_from_logits(z, y):
    raise NotImplementedError

print(sigmoid(np.array([-1000.0, 0.0, 1000.0])))   # [0, 0.5, 1]
~~~

~~~python solution
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))                      # never overflows
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def bce_from_logits(z, y):
    z, y = np.asarray(z, float), np.asarray(y, float)
    return float(np.mean(np.logaddexp(0, z) - y * z))

print(sigmoid(np.array([-1000.0, 0.0, 1000.0])))
~~~

~~~python tests
import numpy as np
import warnings

def test_values():
    """sigmoid(0) = 0.5 and sigmoid(-z) = 1 - sigmoid(z)"""
    z = np.array([-3.0, -0.5, 0.0, 2.0])
    assert approx(sigmoid(z), 1 / (1 + np.exp(-z)))
    assert approx(sigmoid(-z), 1 - sigmoid(z))

def test_no_overflow():
    """extreme inputs give 0 and 1 without warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = sigmoid(np.array([-1000.0, 1000.0]))
    assert approx(s, [0.0, 1.0])

def test_bce_matches_probability_formula():
    """equals -mean(y log p + (1-y) log(1-p)) for moderate scores"""
    z = np.array([-2.0, 0.3, 1.5, -0.1]); y = np.array([0, 1, 1, 1.0])
    p = 1 / (1 + np.exp(-z))
    assert approx(bce_from_logits(z, y), -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def test_bce_finite_when_confidently_wrong():
    """a score of -800 for a positive example costs about 800, not inf"""
    assert approx(bce_from_logits(np.array([-800.0]), np.array([1.0])), 800.0)
~~~

<!-- hint: For `z >= 0` use `1 / (1 + exp(-z))`; for `z < 0` use `exp(z) / (1 + exp(z))`. Both equal the sigmoid, and `np.exp(-np.abs(z))` gives the needed exponential safely. -->
<!-- /exercise -->

<!-- exercise: fit-logistic | Train logistic regression -->
Write `fit_logistic(X, y, lr=0.1, steps=2000)` that starts from zero weights and bias and runs full-batch gradient descent with the gradients above, returning `(w, b)`. Then `predict_proba(X, w, b)` and `predict(X, w, b)` (labels 0/1 with threshold 0.5). A stable `sigmoid` is provided.

~~~python starter
import numpy as np

def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def fit_logistic(X, y, lr=0.1, steps=2000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    raise NotImplementedError

def predict_proba(X, w, b):
    raise NotImplementedError

def predict(X, w, b):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def sigmoid(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))

def fit_logistic(X, y, lr=0.1, steps=2000):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)
        w -= lr * X.T @ (p - y) / n
        b -= lr * float(np.mean(p - y))
    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict(X, w, b):
    return (predict_proba(X, w, b) >= 0.5).astype(int)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 400
X = rng.normal(size=(n, 2))
true_w, true_b = np.array([2.0, -1.0]), 0.5
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ true_w + true_b)))).astype(float)

def test_accuracy():
    """classifies the training data well"""
    w, b = fit_logistic(X, y)
    assert np.mean(predict(X, w, b) == y) > 0.8

def test_recovers_direction():
    """weights point the same way as the true ones"""
    w, b = fit_logistic(X, y, steps=4000)
    assert np.sign(w[0]) == 1 and np.sign(w[1]) == -1
    assert abs(w[0] / w[1] - (-2.0)) < 0.6

def test_gradient_is_zero_at_end():
    """after training, the average gradient is close to zero"""
    w, b = fit_logistic(X, y, lr=0.5, steps=5000)
    p = predict_proba(X, w, b)
    assert np.max(np.abs(X.T @ (p - y) / n)) < 1e-3 and abs(np.mean(p - y)) < 1e-3

def test_probabilities_and_labels():
    """probabilities lie in (0, 1) and labels are 0 or 1"""
    w, b = fit_logistic(X, y, steps=200)
    p = predict_proba(X, w, b)
    assert np.all((p > 0) & (p < 1))
    assert set(np.unique(predict(X, w, b))) <= {0, 1}
~~~

<!-- hint: Each step: `p = sigmoid(X @ w + b)`, then `w -= lr * X.T @ (p - y) / n` and `b -= lr * np.mean(p - y)`. -->
<!-- /exercise -->

<!-- quiz -->
1. A logistic regression weight for "has a guarantor" is 1.1. How does having a guarantor change the odds of repayment?
   - [ ] It adds 1.1 to the probability
   - [x] It multiplies the odds by about $e^{1.1} \approx 3$
   - [ ] It increases the probability by 110%
   > The model is linear in the log-odds, so a weight acts additively on log-odds and multiplicatively on odds.
2. What shape is the decision boundary of logistic regression on raw features?
   - [x] A straight line (a hyperplane)
   - [ ] An S-shaped curve
   - [ ] A circle around the positive class
   > The boundary is where $\mathbf w^\top\mathbf x + b = 0$. The sigmoid only maps scores to probabilities; it does not bend the boundary.
3. Why does the weight vector grow without limit on perfectly separable data?
   - [ ] The learning rate is too high
   - [x] Scaling up the weights always makes the separated points more confidently right, so the loss keeps decreasing towards zero
   - [ ] Cross-entropy is not convex
   > With no penalty, the infimum of the loss is reached only at infinite weights. An L2 penalty fixes this.
4. Which is the gradient of the mean binary cross-entropy with respect to $\mathbf w$?
   - [ ] $X^\top(\mathbf y - \mathbf p)\,\mathbf p(1 - \mathbf p)$
   - [x] $\frac1n X^\top(\mathbf p - \mathbf y)$
   - [ ] $\frac2n X^\top(X\mathbf w - \mathbf y)$
   > The sigmoid's derivative cancels against the derivative of the log, leaving features times prediction errors.
<!-- /quiz -->
