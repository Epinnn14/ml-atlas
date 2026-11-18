---
title: Naive Bayes
level: intermediate
minutes: 35
summary: A generative classifier built from Bayes' rule and one bold assumption. Gaussian naive Bayes for numbers, multinomial naive Bayes with Laplace smoothing for text.
---

## Generative versus discriminative

[Logistic regression](#/lesson/logistic-regression) models $P(y\mid\mathbf x)$ directly: it is **discriminative**. A **generative** classifier instead models how each class produces its data, $p(\mathbf x\mid y)$, together with the class frequencies $P(y)$, and classifies with [Bayes' rule](#/lesson/probability):

<!-- formula: Bayes classifier -->
$$\hat y = \arg\max_c\; P(y = c)\, p(\mathbf x\mid y = c)$$

The denominator $p(\mathbf x)$ is the same for every class, so it can be dropped.

## The naive assumption

Modelling $p(\mathbf x\mid y)$ for many features is hard. Naive Bayes assumes the features are **independent given the class**:

<!-- formula: Naive Bayes factorisation -->
$$p(\mathbf x\mid y = c) = \prod_{j=1}^{d} p(x_j\mid y = c)$$

That is rarely true (in an email, "free" and "money" are not independent), but the classifier often works well anyway: it only has to rank the classes correctly, not get the probabilities right. Its probabilities are usually overconfident, so do not trust them without [calibration](#/lesson/calibration-uncertainty).

Always work with logs: a product of hundreds of small probabilities underflows, a sum of their logs does not.

<!-- formula: Naive Bayes in log space -->
$$\hat y = \arg\max_c\Big[\log P(y=c) + \sum_j \log p(x_j\mid y=c)\Big]$$

## Gaussian naive Bayes

For numeric features, model each one, within each class, as a Gaussian with its own mean $\mu_{cj}$ and variance $\sigma^2_{cj}$, estimated by [maximum likelihood](#/lesson/statistics-estimation) from the training examples of that class. Training is a single pass of averages. A tiny constant is usually added to the variances so a constant feature does not cause a division by zero.

## Multinomial naive Bayes for text

Represent a document by word counts $x_j$ (a **bag of words**). For class $c$, estimate how likely each word is, $\theta_{cj}$, from its total count in that class's documents. A word never seen in a class would get probability 0 and veto the class forever, so add a pseudo-count $\alpha$ (usually 1) to every word, **Laplace smoothing**:

<!-- formula: Laplace-smoothed word probabilities -->
$$\theta_{cj} = \frac{N_{cj} + \alpha}{\sum_{j'} N_{cj'} + \alpha V}$$

where $N_{cj}$ is the count of word $j$ in class $c$ and $V$ is the vocabulary size. The log-likelihood of a document is then $\sum_j x_j\log\theta_{cj}$, a linear function of the counts, so multinomial naive Bayes is a linear classifier in disguise. It trains in one pass, works with tiny datasets, and is a strong baseline for spam filtering and topic classification.

<!-- exercise: gaussian-nb | Gaussian naive Bayes -->
Complete `GaussianNB`. `fit(X, y)` stores, for each class in `self.classes_` (sorted), the log prior, the per-feature means and the per-feature variances (MLE, plus `var_eps = 1e-9`). `predict_log_joint(X)` returns an `(n, C)` array of $\log P(c) + \sum_j \log\mathcal N(x_j\mid\mu_{cj},\sigma^2_{cj})$. `predict(X)` returns the class with the largest value.

~~~python starter
import numpy as np

class GaussianNB:
    var_eps = 1e-9

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        raise NotImplementedError

    def predict_log_joint(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
~~~

~~~python solution
import numpy as np

class GaussianNB:
    var_eps = 1e-9

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        self.mean_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + self.var_eps
        return self

    def predict_log_joint(self, X):
        X = np.asarray(X, float)[:, None, :]                 # (n, 1, d)
        ll = -0.5 * np.log(2 * np.pi * self.var_) - (X - self.mean_) ** 2 / (2 * self.var_)
        return self.log_prior_ + ll.sum(axis=2)              # (n, C)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_joint(X), axis=1)]
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
y = rng.integers(0, 3, 300)
means = np.array([[0, 0], [3, 0], [0, 3]])
X = means[y] + rng.normal(size=(300, 2)) * np.array([[1.0, 0.5]])

def test_parameters():
    """stores per-class means, variances and log priors"""
    m = GaussianNB().fit(X, y)
    assert approx(m.mean_[1], X[y == 1].mean(axis=0))
    assert approx(m.var_[2], X[y == 2].var(axis=0) + 1e-9)
    assert approx(np.exp(m.log_prior_).sum(), 1.0)

def test_log_joint_shape_and_value():
    """log joint is (n, C) and matches the formula for one point"""
    m = GaussianNB().fit(X, y)
    L = m.predict_log_joint(X[:4])
    assert L.shape == (4, 3)
    x, c = X[0], 1
    v = np.log(np.mean(y == c)) + np.sum(-0.5 * np.log(2 * np.pi * m.var_[c]) - (x - m.mean_[c]) ** 2 / (2 * m.var_[c]))
    assert approx(L[0, 1], v)

def test_accuracy():
    """classifies the blobs well"""
    m = GaussianNB().fit(X[:200], y[:200])
    assert np.mean(m.predict(X[200:]) == y[200:]) > 0.85

def test_string_labels():
    """works with non-integer class labels"""
    lab = np.array(["cat", "dog"])[y % 2]
    assert set(GaussianNB().fit(X, lab).predict(X[:10])) <= {"cat", "dog"}
~~~

<!-- hint: Give `X` shape `(n, 1, d)` and the means shape `(C, d)`; broadcasting gives `(n, C, d)`, and summing over the last axis gives `(n, C)`. -->
<!-- /exercise -->

<!-- exercise: multinomial-nb | Spam filter with multinomial naive Bayes -->
Write `fit_mnb(counts, y, alpha=1.0)` for a document–word count matrix `counts` (shape `(n, V)`) and 0/1 labels, returning `(log_prior, log_theta)` with shapes `(2,)` and `(2, V)` using Laplace smoothing. Then `predict_mnb(counts, log_prior, log_theta)` returns the class with the larger $\log P(c) + \sum_j x_j\log\theta_{cj}$.

~~~python starter
import numpy as np

def fit_mnb(counts, y, alpha=1.0):
    counts, y = np.asarray(counts, float), np.asarray(y)
    raise NotImplementedError

def predict_mnb(counts, log_prior, log_theta):
    raise NotImplementedError

vocab = ["free", "money", "meeting", "report", "win"]
docs = np.array([[2, 1, 0, 0, 1], [0, 0, 1, 2, 0], [1, 2, 0, 0, 2], [0, 0, 2, 1, 0]])
labels = np.array([1, 0, 1, 0])   # 1 = spam
lp, lt = fit_mnb(docs, labels)
print(predict_mnb(np.array([[1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]), lp, lt))   # [1 0]
~~~

~~~python solution
import numpy as np

def fit_mnb(counts, y, alpha=1.0):
    counts, y = np.asarray(counts, float), np.asarray(y)
    log_prior = np.log(np.array([np.mean(y == 0), np.mean(y == 1)]))
    N = np.array([counts[y == 0].sum(axis=0), counts[y == 1].sum(axis=0)])
    theta = (N + alpha) / (N.sum(axis=1, keepdims=True) + alpha * counts.shape[1])
    return log_prior, np.log(theta)

def predict_mnb(counts, log_prior, log_theta):
    return np.argmax(np.asarray(counts, float) @ log_theta.T + log_prior, axis=1)

vocab = ["free", "money", "meeting", "report", "win"]
docs = np.array([[2, 1, 0, 0, 1], [0, 0, 1, 2, 0], [1, 2, 0, 0, 2], [0, 0, 2, 1, 0]])
labels = np.array([1, 0, 1, 0])
lp, lt = fit_mnb(docs, labels)
print(predict_mnb(np.array([[1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]), lp, lt))
~~~

~~~python tests
import numpy as np

docs = np.array([[2, 1, 0, 0, 1], [0, 0, 1, 2, 0], [1, 2, 0, 0, 2], [0, 0, 2, 1, 0]])
labels = np.array([1, 0, 1, 0])

def test_shapes():
    """log_prior is (2,), log_theta is (2, V)"""
    lp, lt = fit_mnb(docs, labels)
    assert lp.shape == (2,) and lt.shape == (2, 5)

def test_rows_are_distributions():
    """each class's word probabilities sum to one"""
    _, lt = fit_mnb(docs, labels)
    assert approx(np.exp(lt).sum(axis=1), [1.0, 1.0])

def test_smoothing():
    """'meeting' never appears in spam but still gets probability (0 + 1) / (9 + 5)"""
    _, lt = fit_mnb(docs, labels, alpha=1.0)
    assert approx(np.exp(lt[1, 2]), 1 / 14)

def test_predict():
    """classifies new spam-like and work-like documents"""
    lp, lt = fit_mnb(docs, labels)
    assert list(predict_mnb(np.array([[1, 0, 0, 0, 1], [0, 0, 1, 1, 0]]), lp, lt)) == [1, 0]

def test_unseen_word_does_not_veto():
    """a spam document with one 'meeting' is still spam"""
    lp, lt = fit_mnb(docs, labels)
    assert predict_mnb(np.array([[3, 2, 1, 0, 2]]), lp, lt)[0] == 1
~~~

<!-- hint: `N = np.array([counts[y == 0].sum(0), counts[y == 1].sum(0)])` gives the word totals per class. -->
<!-- hint: Prediction is one matrix product: `counts @ log_theta.T + log_prior`. -->
<!-- /exercise -->

<!-- quiz -->
1. What does "naive" refer to in naive Bayes?
   - [x] Assuming the features are independent of each other given the class
   - [ ] Using a uniform prior over the classes
   - [ ] Ignoring the class frequencies
   > The factorisation $p(\mathbf x\mid c) = \prod_j p(x_j\mid c)$ is the naive part; Bayes' rule itself is exact.
2. Without smoothing, a test email contains a word never seen in any spam training email. What happens?
   - [ ] The word is ignored
   - [x] The spam likelihood becomes exactly zero, so the email can never be classified as spam
   - [ ] The model crashes
   > A single zero factor wipes out the whole product. Laplace smoothing gives every word a small non-zero probability.
3. Naive Bayes outputs $P(\text{spam}) = 0.99999$ for many emails. Should you trust these probabilities?
   - [ ] Yes, Bayes' rule makes them exact
   - [x] Not without calibration; ignoring feature dependence double-counts evidence and makes them overconfident
   - [ ] Only if the dataset is balanced
   > Correlated words are counted as independent pieces of evidence, pushing probabilities to the extremes even when the ranking is good.
4. Why compute $\log P(c) + \sum_j \log p(x_j\mid c)$ instead of the product of probabilities?
   - [ ] Logs give a different, better classifier
   - [x] The product of many small numbers underflows to zero, while the sum of logs stays representable
   - [ ] Logs remove the independence assumption
   > The argmax is unchanged because the log is increasing.
<!-- /quiz -->
