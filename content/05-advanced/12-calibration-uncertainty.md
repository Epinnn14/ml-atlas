---
title: Calibration and uncertainty
level: advanced
minutes: 45
summary: When a model says 80%, is it right 80% of the time? Reliability diagrams and expected calibration error, fixing over-confidence with temperature scaling, the two kinds of uncertainty, and conformal prediction's distribution-free guarantee.
---

## Calibrated probabilities

A classifier is **calibrated** if, among all cases where it predicts probability 0.8, about 80% are positive. Ranking metrics like [AUC](#/lesson/evaluation-metrics) ignore this, but any decision that weighs probabilities against costs (approve a loan if the default risk is below 5%, call a doctor if the risk of sepsis is above 30%) depends on it.

Modern neural networks are often **over-confident**: trained to drive cross-entropy down, they push probabilities towards 0 and 1. [Naive Bayes](#/lesson/naive-bayes) is notoriously over-confident too; well-regularised logistic regression is usually close to calibrated.

## Reliability diagrams and ECE

Group predictions into bins by confidence (0–0.1, 0.1–0.2, …). In each bin compare the average confidence with the actual accuracy. A **reliability diagram** plots one against the other; a calibrated model lies on the diagonal. The **expected calibration error** summarises the gaps, weighted by how many predictions fall in each bin:

<!-- formula: Expected calibration error -->
$$\text{ECE} = \sum_{b=1}^{B}\frac{|B_b|}{n}\,\big|\operatorname{acc}(B_b) - \operatorname{conf}(B_b)\big|$$

For multi-class models, confidence is the top predicted probability and accuracy is whether the top class was right. ECE depends on the binning, so compare models with the same bins. The **Brier score**, the mean squared error of the probabilities, measures calibration and sharpness together.

## Temperature scaling

The simplest effective fix: divide all logits by one scalar $T > 0$ before the softmax, with $T$ chosen to minimise the negative log-likelihood on a **held-out** calibration set:

<!-- formula: Temperature scaling -->
$$\hat p = \operatorname{softmax}(\mathbf z / T)$$

$T > 1$ softens over-confident predictions. Because dividing by a positive constant never changes which logit is largest, accuracy is untouched; only the confidence moves. Platt scaling (a logistic regression on the score) and isotonic regression (a monotone step function) are the classic alternatives for binary scores.

## Two kinds of uncertainty

- **Aleatoric**: noise inherent in the data, such as two identical-looking X-rays with different diagnoses. More data does not reduce it.
- **Epistemic**: the model's ignorance, high far from the training data. More data reduces it.

[Bayesian regression](#/lesson/bayesian-regression) and [Gaussian processes](#/lesson/gaussian-processes) separate them in closed form. For neural networks, **deep ensembles** (train several networks from different random starts and look at their disagreement) are the most reliable practical method; MC dropout is a cheaper approximation. Watch for **distribution shift**: on data unlike the training set, even calibrated models can be confidently wrong.

## Conformal prediction

Conformal prediction wraps any model and turns its outputs into prediction sets or intervals with a guaranteed coverage, with no assumption on the distribution other than exchangeability (calibration and test data come from the same source). **Split conformal** for regression:

1. Fit the model on training data.
2. On a separate calibration set of size $n$, compute the scores $s_i = |y_i - \hat f(\mathbf x_i)|$.
3. Take $\hat q$, the $\lceil(n+1)(1-\alpha)\rceil$-th smallest score.
4. For a new input, predict the interval $[\hat f(\mathbf x) - \hat q,\ \hat f(\mathbf x) + \hat q]$.

<!-- formula: Conformal coverage guarantee -->
$$P\big(y_{\text{new}} \in \hat C(\mathbf x_{\text{new}})\big) \ge 1 - \alpha$$

The guarantee holds for any model, good or bad; a better model simply gives narrower intervals. For classification, the set contains every class whose score passes the threshold, so an uncertain input gets a larger set.

<!-- exercise: ece | Expected calibration error -->
Write `ece(confidence, correct, n_bins=10)` for arrays of top-class confidences in $[0, 1]$ and 0/1 correctness. Use equal-width bins: a confidence $c$ belongs to bin `min(int(c * n_bins), n_bins - 1)`. Skip empty bins.

~~~python starter
import numpy as np

def ece(confidence, correct, n_bins=10):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ece(confidence, correct, n_bins=10):
    conf = np.asarray(confidence, float)
    corr = np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(total)
~~~

~~~python tests
import numpy as np

def test_perfectly_calibrated():
    """confidence 0.75 with 3 of 4 correct gives ECE 0"""
    assert approx(ece([0.75] * 4, [1, 1, 1, 0]), 0.0)

def test_overconfident():
    """always 0.9 confident but right half the time gives ECE 0.4"""
    assert approx(ece([0.9] * 10, [1, 0] * 5), 0.4)

def test_weights_by_bin_size():
    """bins are weighted by their share of predictions"""
    conf = [0.95] * 8 + [0.55] * 2
    corr = [1] * 8 + [0, 0]
    assert approx(ece(conf, corr), 0.8 * 0.05 + 0.2 * 0.55)

def test_confidence_one():
    """a confidence of exactly 1.0 falls in the last bin"""
    assert approx(ece([1.0, 1.0], [1, 1]), 0.0)

def test_simulated_calibrated_model():
    """labels drawn with the stated probability give a small ECE"""
    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.0, 20000)
    assert ece(c, rng.uniform(size=20000) < c) < 0.02
~~~

<!-- /exercise -->

<!-- exercise: conformal | Split conformal intervals -->
Write `conformal_quantile(residuals, alpha)` returning the $\lceil(n+1)(1-\alpha)\rceil$-th smallest absolute residual (or `np.inf` if that rank exceeds $n$), and `conformal_interval(pred, q)` returning `(pred - q, pred + q)`.

~~~python starter
import numpy as np

def conformal_quantile(residuals, alpha):
    raise NotImplementedError

def conformal_interval(pred, q):
    raise NotImplementedError
~~~

~~~python solution
import math
import numpy as np

def conformal_quantile(residuals, alpha):
    s = np.sort(np.abs(np.asarray(residuals, float)))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= len(s) else float("inf")

def conformal_interval(pred, q):
    pred = np.asarray(pred, float)
    return pred - q, pred + q
~~~

~~~python tests
import numpy as np

def test_rank():
    """with 9 residuals and alpha = 0.2, the 8th smallest is used"""
    assert conformal_quantile([5, 1, 4, 2, 3, 9, 7, 6, 8], 0.2) == 8.0

def test_absolute_values():
    """uses absolute residuals"""
    assert conformal_quantile([-5.0, 1.0, -2.0], 0.5) == 2.0

def test_infinite_when_too_few():
    """too little calibration data for the level gives an infinite interval"""
    assert conformal_quantile([1.0, 2.0], 0.05) == float("inf")

def test_coverage():
    """a deliberately poor model still gets about 90% coverage on new data"""
    rng = np.random.default_rng(0)
    f = lambda x: 0.5 * x                          # true relation is 2x + noise
    xc = rng.uniform(0, 1, 500); yc = 2 * xc + rng.normal(size=500)
    q = conformal_quantile(yc - f(xc), 0.1)
    xt = rng.uniform(0, 1, 20000); yt = 2 * xt + rng.normal(size=20000)
    lo, hi = conformal_interval(f(xt), q)
    cover = np.mean((yt >= lo) & (yt <= hi))
    assert 0.88 < cover < 0.93
~~~

<!-- hint: Sort the absolute residuals; the rank is `ceil((n + 1) * (1 - alpha))`, 1-based. -->
<!-- /exercise -->

<!-- quiz -->
1. A model's predictions with confidence around 0.9 are correct only 70% of the time. What is wrong, and what is a cheap fix?
   - [x] It is over-confident; fit a temperature $T > 1$ on held-out data
   - [ ] It is under-confident; set $T < 1$
   - [ ] Its accuracy is too low; retrain from scratch
   > Temperature scaling softens the probabilities without changing which class is predicted.
2. Why does temperature scaling not change accuracy?
   - [ ] It is applied after the prediction is made
   - [x] Dividing all logits by the same positive number keeps their order, so the top class is unchanged
   - [ ] It only changes the training loss
   > Only the shape of the probability distribution changes, not its argmax.
3. Which uncertainty would more training data near the query point reduce?
   - [ ] Aleatoric
   - [x] Epistemic
   - [ ] Neither
   > Epistemic uncertainty is lack of knowledge; aleatoric is noise in the data itself.
4. What does split conformal prediction guarantee?
   - [ ] Narrow intervals for any model
   - [x] At least $1-\alpha$ coverage on new data from the same distribution, whatever the model
   - [ ] That the model is calibrated
   > The guarantee comes from the ranks of exchangeable scores; a better model only makes the intervals tighter.
<!-- /quiz -->
