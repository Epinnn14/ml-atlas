---
title: Machine learning in production
level: senior
minutes: 50
summary: Most of the work starts after the notebook. Leakage and training–serving skew, monitoring for data drift with PSI and the KS test, safe rollouts, retraining, and documenting a model honestly.
---

## The model is the small part

A deployed model sits inside a system: data pipelines, feature computation, serving infrastructure, monitoring, retraining, and the people and processes that act on its outputs. Most failures in practice come from that system, not from the choice of algorithm.

## Leakage: too good to be true

**Data leakage** means the model trains on information it will not have at prediction time. Symptoms: validation scores that look excellent and production performance that collapses. Common sources:

- Features computed after the event: "number of support calls" when predicting churn, but counted up to the cancellation date.
- Preprocessing fitted on all data before splitting ([preparing data](#/lesson/data-preprocessing)).
- Duplicates or the same entity in training and test ([group splits](#/lesson/cross-validation)).
- Random splits on time-ordered data ([time series](#/lesson/time-series)).

The defence: for every feature, ask "at the moment of prediction, would I know this value?", and evaluate on a split that mimics deployment (the most recent period).

## Training–serving skew

The same feature computed by two different code paths, a batch pipeline for training and an online service for serving, will eventually disagree: different defaults for missing values, time zones, rounding, a renamed category. Share one feature definition (a feature store, or one library used in both places), log the features actually used at serving time, and train on those logs.

## Monitoring and drift

The world changes after deployment:

- **Data drift**: the input distribution $p(\mathbf x)$ shifts (a new customer segment, a new phone model).
- **Concept drift**: the relation $p(y\mid\mathbf x)$ changes (fraudsters adapt, prices change).
- **Label delay**: the true outcome (did the loan default?) arrives months later, so model quality cannot be measured directly for a while.

Monitor what you can see immediately: input feature distributions, the distribution of predictions, missing-value rates, latency and errors. Two common drift statistics per feature:

<!-- formula: Population stability index -->
$$\text{PSI} = \sum_{b=1}^{B}\big(p_b - q_b\big)\ln\frac{p_b}{q_b}$$

where $q_b$ and $p_b$ are the fractions of the reference (training) and current data in bin $b$, with bins taken from quantiles of the reference. A rule of thumb: below 0.1 stable, 0.1–0.25 worth a look, above 0.25 a significant shift. PSI is a symmetrised [KL divergence](#/lesson/information-theory).

<!-- formula: Kolmogorov–Smirnov statistic -->
$$D = \sup_x\big|F_{\text{ref}}(x) - F_{\text{cur}}(x)\big|$$

the largest gap between the two empirical cumulative distribution functions. With large samples even trivial shifts become "significant", so judge drift by its size and by its effect on the business metric, not by p-values alone.

## Rolling out safely

- **Shadow mode**: the new model receives live traffic and logs predictions without acting on them.
- **Canary or A/B rollout**: give it a small share of traffic, compare with the current model on the real metric ([A/B testing](#/lesson/causal-inference)), then expand.
- Keep a fast **rollback** and version everything: data snapshot, code, features, hyperparameters, model weights.

## Retraining

Retrain on a schedule or when monitoring triggers. Automate the whole path (fetch data, train, evaluate against the current model on a fixed benchmark and recent data, require approval if the gap is small) so that retraining is boring. Beware feedback loops: if the model's decisions shape the next training data (who got a loan determines whose repayment you observe), keep some randomised exploration or use [causal methods](#/lesson/causal-inference).

## Documentation and responsibility

A **model card** records what a model is for and not for, the training data and its period, evaluation results broken down by relevant subgroups (performance often differs across regions, languages or demographics), known limitations, and the owner. Decide in advance who reviews errors and how affected people can contest a decision.

<!-- exercise: psi | Population stability index -->
Write `psi(reference, current, n_bins=10, eps=1e-6)`. Take bin edges at the quantiles of `reference` (`np.quantile(reference, np.linspace(0, 1, n_bins + 1))`), set the first edge to `-inf` and the last to `+inf`, count both samples into those bins with `np.histogram`, convert to fractions, replace zero fractions by `eps`, and return the PSI.

~~~python starter
import numpy as np

def psi(reference, current, n_bins=10, eps=1e-6):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def psi(reference, current, n_bins=10, eps=1e-6):
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    q = np.histogram(reference, edges)[0] / len(reference)
    p = np.histogram(current, edges)[0] / len(current)
    q, p = np.maximum(q, eps), np.maximum(p, eps)
    return float(np.sum((p - q) * np.log(p / q)))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
ref = rng.normal(size=50_000)

def test_same_distribution():
    """a fresh sample from the same distribution has PSI near 0"""
    assert psi(ref, rng.normal(size=50_000)) < 0.01

def test_shift_detected():
    """a shift of half a standard deviation is a significant change"""
    assert psi(ref, rng.normal(0.5, 1, 50_000)) > 0.2

def test_grows_with_shift():
    """larger shifts give larger PSI"""
    vals = [psi(ref, rng.normal(m, 1, 50_000)) for m in (0.1, 0.3, 0.6)]
    assert vals[0] < vals[1] < vals[2]

def test_empty_bin_finite():
    """a current sample missing whole bins still gives a finite value"""
    assert np.isfinite(psi(ref, np.full(1000, 3.0)))
~~~

<!-- /exercise -->

<!-- exercise: ks | The two-sample KS statistic -->
Write `ks_statistic(a, b)` returning the maximum absolute difference between the empirical CDFs of the two samples. Evaluate both CDFs at every value in the combined sample with `np.searchsorted(sorted_x, values, side="right") / n`.

~~~python starter
import numpy as np

def ks_statistic(a, b):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ks_statistic(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    Fa = np.searchsorted(a, grid, side="right") / len(a)
    Fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(Fa - Fb)))
~~~

~~~python tests
import numpy as np

def test_identical():
    """identical samples have D = 0"""
    x = np.array([1.0, 2.0, 3.0])
    assert ks_statistic(x, x) == 0.0

def test_disjoint():
    """completely separated samples have D = 1"""
    assert ks_statistic([1.0, 2.0], [5.0, 6.0, 7.0]) == 1.0

def test_small_example():
    """a hand-checked case"""
    assert approx(ks_statistic([1.0, 2.0, 3.0, 4.0], [3.5, 4.5]), 0.75)

def test_shift():
    """for large normal samples shifted by 0.5, D is near 2 * Phi(0.25) - 1"""
    rng = np.random.default_rng(0)
    d = ks_statistic(rng.normal(size=40_000), rng.normal(0.5, 1, 40_000))
    assert abs(d - 0.197) < 0.015
~~~

<!-- /exercise -->

<!-- quiz -->
1. A churn model scores 0.99 AUC in validation but performs poorly in production. Which is the most likely cause?
   - [x] Leakage: a feature contained information only available after customers had already churned
   - [ ] The model is underfitting
   - [ ] AUC is the wrong metric
   > Near-perfect validation scores followed by a collapse are the classic sign of leakage.
2. Training features come from a nightly SQL job; serving features from a Python service written separately. What risk does this create?
   - [ ] Concept drift
   - [x] Training–serving skew: the two implementations can compute different values for the same input
   - [ ] Leakage
   > Share one feature definition and train on logged serving features.
3. Fraudsters change tactics, so the same transaction features now mean something different. Which kind of drift is this?
   - [ ] Data drift only
   - [x] Concept drift: the relation between inputs and the label has changed
   - [ ] Label delay
   > The input distribution may look stable while $p(y\mid\mathbf x)$ changes.
4. What is a shadow deployment?
   - [x] The new model scores live traffic and logs predictions without affecting any decisions
   - [ ] Deploying the model only at night
   - [ ] Training the model on production data
   > It tests the full serving path and compares predictions safely before any user is affected.
<!-- /quiz -->
