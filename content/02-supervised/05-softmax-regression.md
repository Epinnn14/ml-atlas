---
title: Softmax regression
level: intermediate
minutes: 35
summary: Logistic regression for many classes. One score per class, the softmax turns scores into a probability distribution, and the gradient is again features times errors.
---

## One score per class

With $K$ classes (digits 0–9, ten product categories) we give each class $k$ its own weight vector $\mathbf w_k$ and bias $b_k$, and compute a score, or **logit**, $z_k = \mathbf w_k^\top\mathbf x + b_k$. Stacking the weight vectors as columns of $W \in \mathbb{R}^{d\times K}$, all logits for all examples are one matrix product: $Z = XW + \mathbf b$, shape $(n, K)$.

## The softmax

The **softmax** turns a vector of logits into probabilities that are positive and sum to one:

<!-- formula: Softmax -->
$$\operatorname{softmax}(\mathbf z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

The largest logit gets the largest probability, and adding the same constant to every logit changes nothing. That second property gives the stable implementation: subtract the row maximum before exponentiating, exactly the [log-sum-exp trick](#/lesson/probability). With $K = 2$ the softmax reduces to the sigmoid of the difference of the two logits, so this is a true generalisation of [logistic regression](#/lesson/logistic-regression).

## Loss and gradient

The loss is the categorical [cross-entropy](#/lesson/information-theory): minus the log probability given to the correct class,

<!-- formula: Categorical cross-entropy -->
$$\mathcal L = -\frac1n\sum_{i=1}^n \log p_{i, y_i}, \qquad \log p_{i,k} = z_{i,k} - \log\sum_j e^{z_{i,j}}$$

Let $P$ be the $(n, K)$ matrix of predicted probabilities and $Y$ the one-hot matrix of labels. The gradient has the familiar form:

<!-- formula: Softmax regression gradient -->
$$\nabla_W\mathcal L = \frac1n X^\top(P - Y), \qquad \nabla_{\mathbf b}\mathcal L = \frac1n \sum_i (P - Y)_{i,:}$$

Each class's weights move towards the examples of that class (where $Y = 1$ and $P < 1$) and away from examples of other classes it wrongly claimed.

This exact layer, a linear map followed by softmax and cross-entropy, is the output layer of almost every neural network classifier ([neural networks](#/lesson/neural-networks)) and of every language model predicting the next token ([language models](#/lesson/language-models)).

## Temperature

Dividing the logits by a temperature $T$ before the softmax controls how peaked the distribution is: $T < 1$ sharpens it towards the top class, $T > 1$ flattens it towards uniform. It is used to [calibrate](#/lesson/calibration-uncertainty) probabilities and to control randomness when sampling text.

<!-- exercise: softmax | A stable softmax and the cross-entropy -->
Write `softmax(Z)` that works row by row on a 2-D array of logits without overflow, and `cross_entropy(Z, y)` that returns the mean categorical cross-entropy for integer labels `y`, computed from the logits with the log-sum-exp form.

~~~python starter
import numpy as np

def softmax(Z):
    Z = np.asarray(Z, dtype=float)
    raise NotImplementedError

def cross_entropy(Z, y):
    Z = np.asarray(Z, dtype=float)
    raise NotImplementedError

print(softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])))
~~~

~~~python solution
import numpy as np

def softmax(Z):
    Z = np.asarray(Z, dtype=float)
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def cross_entropy(Z, y):
    Z = np.asarray(Z, dtype=float)
    m = Z.max(axis=1, keepdims=True)
    log_norm = m[:, 0] + np.log(np.exp(Z - m).sum(axis=1))
    return float(np.mean(log_norm - Z[np.arange(len(y)), y]))

print(softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])))
~~~

~~~python tests
import numpy as np

def test_rows_sum_to_one():
    """every row is a probability distribution"""
    P = softmax(np.random.default_rng(0).normal(size=(5, 4)))
    assert approx(P.sum(axis=1), np.ones(5)) and np.all(P > 0)

def test_large_logits():
    """no overflow for logits around 1000"""
    assert approx(softmax(np.array([[1000.0, 1000.0, 1000.0]])), [[1 / 3, 1 / 3, 1 / 3]])

def test_shift_invariance():
    """adding a constant to a row changes nothing"""
    Z = np.array([[0.5, -1.0, 2.0]])
    assert approx(softmax(Z), softmax(Z + 37.0))

def test_two_classes_is_sigmoid():
    """for two classes, p(class 1) = sigmoid(z1 - z0)"""
    Z = np.array([[0.2, 1.7]])
    assert approx(softmax(Z)[0, 1], 1 / (1 + np.exp(-1.5)))

def test_cross_entropy():
    """matches -mean log p[true class], and is stable for huge logits"""
    Z = np.array([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]]); y = np.array([0, 1])
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    assert approx(cross_entropy(Z, y), -np.mean(np.log(P[[0, 1], y])))
    assert approx(cross_entropy(np.array([[0.0, 900.0]]), np.array([0])), 900.0)
~~~

<!-- hint: Subtract `Z.max(axis=1, keepdims=True)` before `np.exp`. -->
<!-- hint: `Z[np.arange(n), y]` picks the logit of the true class in every row. -->
<!-- /exercise -->

<!-- exercise: train-softmax | Train a three-class classifier -->
Write `fit_softmax(X, y, K, lr=0.5, steps=500)` that starts from `W = zeros((d, K))`, `b = zeros(K)` and runs full-batch gradient descent with the gradient above, returning `(W, b)`. Then `predict(X, W, b)` returns the most likely class per row. A stable `softmax` is provided.

~~~python starter
import numpy as np

def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def fit_softmax(X, y, K, lr=0.5, steps=500):
    n, d = X.shape
    W, b = np.zeros((d, K)), np.zeros(K)
    Y = np.eye(K)[y]          # one-hot labels, shape (n, K)
    raise NotImplementedError

def predict(X, W, b):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def softmax(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)

def fit_softmax(X, y, K, lr=0.5, steps=500):
    n, d = X.shape
    W, b = np.zeros((d, K)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(steps):
        G = softmax(X @ W + b) - Y
        W -= lr * X.T @ G / n
        b -= lr * G.mean(axis=0)
    return W, b

def predict(X, W, b):
    return np.argmax(X @ W + b, axis=1)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
centres = np.array([[0.0, 3.0], [-3.0, -2.0], [3.0, -2.0]])
y = rng.integers(0, 3, 300)
X = centres[y] + rng.normal(size=(300, 2))

def test_accuracy():
    """separates three blobs with over 90% accuracy"""
    W, b = fit_softmax(X, y, 3)
    assert np.mean(predict(X, W, b) == y) > 0.9

def test_shapes():
    """W is (d, K) and b is (K,)"""
    W, b = fit_softmax(X, y, 3, steps=5)
    assert W.shape == (2, 3) and b.shape == (3,)

def test_loss_decreases():
    """the cross-entropy falls with more steps"""
    def loss(W, b):
        P = softmax(X @ W + b)
        return -np.mean(np.log(P[np.arange(len(y)), y]))
    assert loss(*fit_softmax(X, y, 3, steps=200)) < loss(*fit_softmax(X, y, 3, steps=20)) < np.log(3)

def test_first_step():
    """one step from zeros matches the gradient formula"""
    W, b = fit_softmax(X, y, 3, lr=1.0, steps=1)
    G = np.full((300, 3), 1 / 3) - np.eye(3)[y]
    assert approx(W, -X.T @ G / 300) and approx(b, -G.mean(axis=0))
~~~

<!-- hint: `G = softmax(X @ W + b) - Y` has shape `(n, K)`; then `W -= lr * X.T @ G / n`. -->
<!-- /exercise -->

<!-- quiz -->
1. The logits for three classes are $(2, 2, 2)$. What does the softmax return?
   - [ ] $(2, 2, 2)$
   - [x] $(1/3, 1/3, 1/3)$
   - [ ] $(0, 0, 1)$
   > Equal logits mean equal probabilities, whatever their common value.
2. Why subtract the row maximum before exponentiating?
   - [x] It prevents overflow and does not change the result, because the softmax ignores constant shifts
   - [ ] It makes the probabilities sharper
   - [ ] It is required for the gradient to be correct
   > $e^{1000}$ overflows to infinity; after the shift the largest exponent is $e^0 = 1$.
3. Dividing logits by a temperature $T = 0.1$ before the softmax...
   - [x] makes the distribution much more peaked on the top class
   - [ ] makes all classes nearly equally likely
   - [ ] has no effect, like adding a constant
   > Scaling multiplies the gaps between logits by 10, and the exponential magnifies gaps.
4. With $K = 2$, softmax regression is equivalent to...
   - [ ] linear regression on the class index
   - [x] logistic regression on the difference of the two scores
   - [ ] a decision tree with one split
   > $e^{z_1}/(e^{z_0} + e^{z_1}) = \sigma(z_1 - z_0)$, so only the difference of the two weight vectors matters.
<!-- /quiz -->
