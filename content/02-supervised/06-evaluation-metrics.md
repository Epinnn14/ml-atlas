---
title: Evaluating models
level: beginner
minutes: 45
summary: Accuracy is not enough. The confusion matrix, precision, recall and F1, threshold-free ROC-AUC, and the regression metrics, with advice on which to use when classes are imbalanced.
---

## The confusion matrix

For a binary classifier, every prediction lands in one of four cells:

| | predicted positive | predicted negative |
|---|---|---|
| **actually positive** | true positive (TP) | false negative (FN) |
| **actually negative** | false positive (FP) | true negative (TN) |

Every classification metric is a ratio of these counts.

## Accuracy and its trap

<!-- formula: Accuracy -->
$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

If 1% of transactions are fraud, a model that says "never fraud" is 99% accurate and completely useless. With **imbalanced classes**, accuracy hides exactly the cases you care about.

## Precision, recall and F1

<!-- formula: Precision and recall -->
$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}$$

- **Precision**: of everything flagged as positive, how much really is? Matters when false alarms are costly (spam filters deleting real mail).
- **Recall** (sensitivity, true positive rate): of all real positives, how many did we catch? Matters when misses are costly (cancer screening, fraud).

Raising the decision threshold usually raises precision and lowers recall. The **F1 score** combines them with the harmonic mean, which is low if either is low:

<!-- formula: F1 score -->
$$F_1 = \frac{2\cdot\text{precision}\cdot\text{recall}}{\text{precision} + \text{recall}} = \frac{2TP}{2TP + FP + FN}$$

When precision or recall is undefined (no predicted or no actual positives), libraries report 0; do the same.

For many classes, compute the metric per class and average: **macro** averaging weights every class equally (rare classes count as much as common ones), **micro** averaging pools all counts.

## ROC curve and AUC

A probabilistic classifier gives scores, and each threshold gives a different confusion matrix. The **ROC curve** plots the true positive rate against the **false positive rate** $FP/(FP+TN)$ for every threshold. The area under it, **ROC-AUC**, has a clean meaning:

<!-- formula: ROC-AUC as a ranking probability -->
$$\text{AUC} = P\big(\text{score}(\text{random positive}) > \text{score}(\text{random negative})\big)$$

(ties count as one half). 0.5 is random guessing, 1.0 is a perfect ranking. AUC does not depend on the threshold or on rescaling the scores, which makes it a good summary of ranking quality. With heavy imbalance, the **precision–recall curve** and its average precision are more informative, because the false positive rate stays small even when false positives swamp the true ones.

## Regression metrics

<!-- formula: MAE and RMSE -->
$$\text{MAE} = \frac1n\sum_i|y_i - \hat y_i|, \qquad \text{RMSE} = \sqrt{\frac1n\sum_i (y_i - \hat y_i)^2}$$

Both are in the units of the target. RMSE punishes large errors more; MAE is robust to outliers and is minimised by the median rather than the mean. $R^2$ ([linear regression](#/lesson/linear-regression)) is unit-free. For forecasts with uncertainty, see quantile loss in [time series](#/lesson/time-series).

## Choosing a metric

Decide **before** modelling, from the cost of each kind of mistake. Then compare every model with a simple baseline on the same metric, on data the models never saw.

<!-- exercise: prf | Precision, recall and F1 -->
Write `confusion(y_true, y_pred)` returning `(tp, fp, fn, tn)` as integers for 0/1 labels, and `precision_recall_f1(y_true, y_pred)` returning three floats. Return 0.0 for any metric whose denominator is zero.

~~~python starter
import numpy as np

def confusion(y_true, y_pred):
    raise NotImplementedError

def precision_recall_f1(y_true, y_pred):
    raise NotImplementedError

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
print(confusion(y_true, y_pred), precision_recall_f1(y_true, y_pred))
~~~

~~~python solution
import numpy as np

def confusion(y_true, y_pred):
    t, p = np.asarray(y_true) == 1, np.asarray(y_pred) == 1
    return int(np.sum(t & p)), int(np.sum(~t & p)), int(np.sum(t & ~p)), int(np.sum(~t & ~p))

def precision_recall_f1(y_true, y_pred):
    tp, fp, fn, _ = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return float(precision), float(recall), float(f1)

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])
print(confusion(y_true, y_pred), precision_recall_f1(y_true, y_pred))
~~~

~~~python tests
import numpy as np

y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 0])

def test_confusion():
    """2 TP, 1 FP, 1 FN, 4 TN"""
    assert confusion(y_true, y_pred) == (2, 1, 1, 4)

def test_metrics():
    """precision 2/3, recall 2/3, F1 2/3"""
    assert approx(precision_recall_f1(y_true, y_pred), [2 / 3, 2 / 3, 2 / 3])

def test_never_positive():
    """predicting no positives gives 0 everywhere, not a division error"""
    assert precision_recall_f1(y_true, np.zeros(8, int)) == (0.0, 0.0, 0.0)

def test_harmonic_mean():
    """F1 is the harmonic mean of precision and recall"""
    yt = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); yp = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    p, r, f = precision_recall_f1(yt, yp)
    assert approx(f, 2 * p * r / (p + r))
~~~

<!-- /exercise -->

<!-- exercise: auc | ROC-AUC by counting pairs -->
Write `roc_auc(y_true, scores)` using the ranking definition: over all pairs of one positive and one negative example, the fraction where the positive has the higher score, counting ties as one half. A vectorised comparison of all pairs is fine for the test sizes here.

~~~python starter
import numpy as np

def roc_auc(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    raise NotImplementedError

print(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))   # 0.75
~~~

~~~python solution
import numpy as np

def roc_auc(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores, float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)

print(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]))
~~~

~~~python tests
import numpy as np

def test_example():
    """the classic 4-point example gives 0.75"""
    assert approx(roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)

def test_perfect_and_reversed():
    """perfect ranking gives 1, reversed ranking gives 0"""
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    assert approx(roc_auc(y, s), 1.0) and approx(roc_auc(y, -s), 0.0)

def test_ties_count_half():
    """constant scores give 0.5"""
    assert approx(roc_auc([0, 1, 0, 1], [0.3, 0.3, 0.3, 0.3]), 0.5)

def test_scale_invariant():
    """a monotonic transform of the scores keeps the AUC"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); s = y + rng.normal(size=200)
    assert approx(roc_auc(y, s), roc_auc(y, np.exp(3 * s)))

def test_random_is_half():
    """random scores give about 0.5"""
    rng = np.random.default_rng(1)
    assert abs(roc_auc(rng.integers(0, 2, 3000), rng.uniform(size=3000)) - 0.5) < 0.03
~~~

<!-- hint: `pos[:, None] - neg[None, :]` gives the matrix of score differences for every positive–negative pair. -->
<!-- /exercise -->

<!-- quiz -->
1. A disease screening test should miss as few sick patients as possible. Which metric matters most?
   - [ ] Precision
   - [x] Recall
   - [ ] Accuracy
   > Recall is the fraction of real positives that are caught. Follow-up tests can sort out false alarms; a miss may go untreated.
2. 1% of emails are phishing. A model has 99.2% accuracy. What should you check before celebrating?
   - [x] Precision and recall on the phishing class, because predicting "never phishing" already gives 99%
   - [ ] Whether accuracy is above 99.5%
   - [ ] The model's training time
   > With rare positives, accuracy is dominated by the easy majority class.
3. A model's ROC-AUC is 0.85. What does that mean?
   - [ ] It is correct 85% of the time
   - [x] A randomly chosen positive gets a higher score than a randomly chosen negative 85% of the time
   - [ ] At the best threshold, 85% of positives are caught
   > AUC measures ranking quality across all thresholds, not accuracy at any one threshold.
4. Your house-price errors include a few huge misses on mansions. Which metric is least affected by them?
   - [x] MAE
   - [ ] RMSE
   - [ ] MSE
   > Squaring makes large errors dominate MSE and RMSE; the absolute error grows only linearly.
<!-- /quiz -->
