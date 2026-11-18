---
title: Entropy, cross-entropy and KL
level: intermediate
minutes: 35
summary: Measuring uncertainty in bits, why cross-entropy is the loss of every classifier, how KL divergence compares distributions, and mutual information as shared knowledge.
---

## Surprise and entropy

An event with probability $p$ carries $-\log_2 p$ bits of **surprise**: a fair coin landing heads is 1 bit, a 1-in-1024 event is 10 bits. **Entropy** is the average surprise of a distribution, the uncertainty before you look:

<!-- formula: Entropy -->
$$H(p) = -\sum_x p(x)\log p(x)$$

A fair coin has $H = 1$ bit, a coin that always lands heads has $H = 0$. Among distributions over $K$ outcomes, the uniform one has the highest entropy, $\log K$. By convention $0 \log 0 = 0$: impossible outcomes add nothing. With the natural log the unit is the **nat**; ML code almost always uses natural logs.

## Cross-entropy

Suppose outcomes really follow $p$ but you use a model $q$ to predict them. The average surprise you experience is the **cross-entropy**:

<!-- formula: Cross-entropy -->
$$H(p, q) = -\sum_x p(x)\log q(x)$$

It is never smaller than $H(p)$, and equals it only when $q = p$. For a classifier with one-hot labels, $p$ puts all its mass on the true class $y$, so the sum collapses to $-\log q(y)$: the **negative log-likelihood** of the correct class. Averaged over a dataset this is the **cross-entropy loss** used by [logistic](#/lesson/logistic-regression) and [softmax regression](#/lesson/softmax-regression) and almost every neural network classifier.

The log makes confident mistakes very expensive: predicting the true class with $q = 0.01$ costs $-\ln 0.01 \approx 4.6$ nats, while $q = 0.5$ costs only $0.69$.

## KL divergence

The extra cost of using $q$ instead of the truth $p$ is the **Kullback–Leibler divergence**:

<!-- formula: KL divergence -->
$$D_{\mathrm{KL}}(p\,\Vert\,q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Properties: $D_{\mathrm{KL}} \ge 0$, it is 0 exactly when $p = q$, and it is **not symmetric**, $D_{\mathrm{KL}}(p\Vert q) \ne D_{\mathrm{KL}}(q\Vert p)$ in general. If $q(x) = 0$ where $p(x) > 0$, it is infinite: the model calls something impossible that does happen. Because $H(p)$ does not depend on the model, minimising cross-entropy and minimising KL to the data distribution are the same thing. KL also appears inside [variational autoencoders](#/lesson/autoencoders-vae) and in fine-tuning language models with a penalty for drifting from the original.

## Mutual information

How much does knowing $Y$ reduce uncertainty about $X$? That is the **mutual information**:

<!-- formula: Mutual information -->
$$I(X; Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)\,p(y)} = H(X) - H(X\mid Y)$$

It is the KL divergence between the joint distribution and the product of the marginals, so it is 0 exactly when $X$ and $Y$ are independent. Unlike correlation, it catches any kind of dependence, not only linear. [Decision trees](#/lesson/decision-trees) choose splits by **information gain**, which is the mutual information between the split and the label.

<!-- exercise: entropies | Entropy, cross-entropy and KL -->
Write `entropy(p)`, `cross_entropy(p, q)` and `kl(p, q)` for 1-D probability arrays, in nats (natural log), returning floats. Treat $0\log 0$ as 0: only sum over entries where `p > 0`.

~~~python starter
import numpy as np

def entropy(p):
    raise NotImplementedError

def cross_entropy(p, q):
    raise NotImplementedError

def kl(p, q):
    raise NotImplementedError

print(entropy(np.array([0.5, 0.5])))   # log 2 = 0.693...
~~~

~~~python solution
import numpy as np

def entropy(p):
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(p[nz])))

def cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    nz = p > 0
    return float(-np.sum(p[nz] * np.log(q[nz])))

def kl(p, q):
    return cross_entropy(p, q) - entropy(p)

print(entropy(np.array([0.5, 0.5])))
~~~

~~~python tests
import numpy as np

def test_fair_coin():
    """a fair coin has entropy log 2"""
    assert approx(entropy([0.5, 0.5]), np.log(2))

def test_certain():
    """a certain outcome has entropy 0 (no nan from 0 log 0)"""
    assert approx(entropy([1.0, 0.0, 0.0]), 0.0)

def test_uniform_is_max():
    """the uniform distribution over 4 outcomes has entropy log 4"""
    assert approx(entropy(np.full(4, 0.25)), np.log(4))
    assert entropy([0.7, 0.1, 0.1, 0.1]) < np.log(4)

def test_cross_entropy_one_hot():
    """with a one-hot p, cross-entropy is -log q[true class]"""
    assert approx(cross_entropy([0, 1, 0], [0.2, 0.7, 0.1]), -np.log(0.7))

def test_kl_properties():
    """KL is 0 for equal distributions, positive otherwise, and not symmetric"""
    p, q = np.array([0.6, 0.3, 0.1]), np.array([0.2, 0.5, 0.3])
    assert approx(kl(p, p), 0.0)
    assert kl(p, q) > 0 and kl(q, p) > 0
    assert abs(kl(p, q) - kl(q, p)) > 1e-3
    assert approx(kl(p, q), np.sum(p * np.log(p / q)))
~~~

<!-- hint: `nz = p > 0` then `p[nz] * np.log(q[nz])` skips the zero-probability terms. -->
<!-- /exercise -->

<!-- exercise: mutual-info | Mutual information from a table -->
Write `mutual_information(joint)` where `joint` is a 2-D array of probabilities $p(x, y)$ (rows are $x$, columns are $y$, all entries sum to 1). Compute the marginals by summing rows and columns, and return $I(X;Y)$ in nats, skipping cells with $p(x,y) = 0$.

~~~python starter
import numpy as np

def mutual_information(joint):
    joint = np.asarray(joint, dtype=float)
    raise NotImplementedError

print(mutual_information(np.array([[0.5, 0.0], [0.0, 0.5]])))   # log 2: Y tells you X exactly
~~~

~~~python solution
import numpy as np

def mutual_information(joint):
    joint = np.asarray(joint, dtype=float)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ py)[nz])))

print(mutual_information(np.array([[0.5, 0.0], [0.0, 0.5]])))
~~~

~~~python tests
import numpy as np

def test_perfect_dependence():
    """Y equal to X (a fair bit) shares log 2 nats"""
    assert approx(mutual_information([[0.5, 0.0], [0.0, 0.5]]), np.log(2))

def test_independent():
    """independent variables have zero mutual information"""
    px, py = np.array([0.2, 0.8]), np.array([0.5, 0.3, 0.2])
    assert approx(mutual_information(np.outer(px, py)), 0.0)

def test_non_negative_and_symmetric():
    """I(X;Y) is non-negative and equals I(Y;X)"""
    J = np.array([[0.1, 0.2, 0.05], [0.3, 0.05, 0.3]])
    a, b = mutual_information(J), mutual_information(J.T)
    assert a > 0 and approx(a, b)

def test_equals_entropy_difference():
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    J = np.array([[0.1, 0.2, 0.05], [0.3, 0.05, 0.3]])
    H = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    assert approx(mutual_information(J), H(J.sum(1)) + H(J.sum(0)) - H(J.ravel()))
~~~

<!-- hint: With `keepdims=True` the marginals have shapes `(rows, 1)` and `(1, cols)`, so `px * py` (or `px @ py`) is the product table. -->
<!-- /exercise -->

<!-- quiz -->
1. A classifier assigns probability 0.001 to the class that turns out to be correct. Roughly what does that example add to the cross-entropy loss?
   - [ ] 0.001
   - [x] about 6.9 nats
   - [ ] 1
   > The loss is $-\ln 0.001 \approx 6.9$. Confidently wrong predictions dominate the loss, which pushes the model towards calibrated probabilities.
2. Which statement about KL divergence is true?
   - [ ] It is a distance, so $D(p\Vert q) = D(q\Vert p)$
   - [x] It is zero only when the two distributions are equal and is otherwise positive
   - [ ] It can be negative when q is a better model than p
   > KL is non-negative and zero only for identical distributions, but it is not symmetric, so it is a divergence, not a distance.
3. Why is minimising cross-entropy the same as minimising $D_{\mathrm{KL}}(p_{\text{data}}\,\Vert\,q_{\text{model}})$?
   - [x] They differ by the entropy of the data, which does not depend on the model
   - [ ] Cross-entropy and KL are two names for the same formula
   - [ ] Only for binary classification
   > $H(p, q) = H(p) + D_{\mathrm{KL}}(p\Vert q)$, and $H(p)$ is a constant with respect to the model parameters.
4. Two variables have zero correlation but a mutual information of 1.2 nats. Is that possible?
   - [x] Yes, they can depend on each other non-linearly, for example $Y = X^2$
   - [ ] No, zero correlation implies independence
   - [ ] Only if the variables are discrete
   > Correlation only measures linear dependence. Mutual information is zero only for truly independent variables.
<!-- /quiz -->
