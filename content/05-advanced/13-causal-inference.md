---
title: Causal inference and A/B testing
level: senior
minutes: 55
summary: Prediction asks what will happen; decisions ask what would happen if we acted. Potential outcomes, why randomised experiments work, confounding and Simpson's paradox, and estimating effects from observational data with propensity scores.
---

## Correlation is not enough

A model that predicts well can still give terrible advice. Customers who receive a discount buy more, but maybe the discount was sent to loyal customers who would have bought anyway. Hospitals with more doctors per patient have higher death rates, because the sickest patients go there. To decide whether to act, we need the **causal effect** of the action, not the correlation.

## Potential outcomes

For each unit $i$ (a customer, a patient) imagine two outcomes: $Y_i(1)$ if treated and $Y_i(0)$ if not. The individual effect is $Y_i(1) - Y_i(0)$, but we only ever see one of the two: the **fundamental problem of causal inference**. We can still estimate the **average treatment effect**:

<!-- formula: Average treatment effect -->
$$\text{ATE} = \mathbb E\big[Y(1) - Y(0)\big]$$

## Randomised experiments

If treatment $T$ is assigned at random, it is independent of the potential outcomes, so the treated and control groups are comparable on everything, measured or not. Then the simple difference in means is an unbiased estimate of the ATE:

<!-- formula: Difference-in-means estimator and its standard error -->
$$\widehat{\text{ATE}} = \bar Y_{T=1} - \bar Y_{T=0}, \qquad \text{SE} = \sqrt{\frac{s_1^2}{n_1} + \frac{s_0^2}{n_0}}$$

A 95% confidence interval is roughly $\widehat{\text{ATE}} \pm 1.96\,\text{SE}$. This is an **A/B test**. Practical rules:

- Decide the metric, the minimum effect worth detecting and the sample size **before** starting (a power calculation). Stopping as soon as the result looks significant ("peeking") inflates false positives.
- Randomise the right unit (user, not page view), and check that the groups are balanced in size and pre-treatment characteristics.
- Testing 20 metrics at the 5% level will produce about one "significant" result by chance; correct for multiple comparisons or name one primary metric.

## Confounding and Simpson's paradox

Without randomisation, a **confounder**, a variable that affects both treatment and outcome, creates spurious association. **Simpson's paradox** is the extreme case: a treatment can look worse overall yet better within every subgroup, because the subgroups differ in size between arms. Causal **graphs** (DAGs) make assumptions explicit: draw arrows for what causes what, and condition on the variables that block the "back-door" paths between treatment and outcome, but never on consequences of the treatment.

## Observational data: propensity scores

When an experiment is impossible, we can estimate effects from observational data if we measured all confounders $\mathbf x$ (**no unmeasured confounding**, an assumption that cannot be checked from the data). The **propensity score** $e(\mathbf x) = P(T = 1\mid\mathbf x)$ is the probability of treatment given the confounders, usually fitted with [logistic regression](#/lesson/logistic-regression). **Inverse probability weighting** reweights each unit so the treated and control groups each resemble the whole population:

<!-- formula: Inverse probability weighting estimator -->
$$\widehat{\text{ATE}}_{\text{IPW}} = \frac1n\sum_{i=1}^{n}\Big[\frac{T_iY_i}{e(\mathbf x_i)} - \frac{(1-T_i)Y_i}{1 - e(\mathbf x_i)}\Big]$$

A treated unit that was unlikely to be treated counts more, because it stands in for many similar untreated ones. Propensities near 0 or 1 give huge weights and noisy estimates (a lack of **overlap**). Doubly robust estimators combine IPW with an outcome model and stay consistent if either one is right.

Other designs exploit natural experiments: difference-in-differences, regression discontinuity, instrumental variables. Machine learning enters through flexible outcome and propensity models ("double machine learning") and through estimating **heterogeneous** effects: who benefits most from the treatment (uplift modelling).

<!-- exercise: ab-test | Analyse an A/B test -->
Write `ab_test(y_treat, y_control)` returning `(ate, se, (lo, hi))`: the difference in means, its standard error using sample variances with `ddof=1`, and the 95% interval `ate ± 1.96 se`.

~~~python starter
import numpy as np

def ab_test(y_treat, y_control):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ab_test(y_treat, y_control):
    a, b = np.asarray(y_treat, float), np.asarray(y_control, float)
    ate = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float(ate), float(se), (float(ate - 1.96 * se), float(ate + 1.96 * se))
~~~

~~~python tests
import numpy as np

def test_values():
    """matches the formulas on a small example"""
    a, b = np.array([3.0, 5.0, 4.0]), np.array([1.0, 2.0, 3.0, 2.0])
    ate, se, (lo, hi) = ab_test(a, b)
    s = np.sqrt(a.var(ddof=1) / 3 + b.var(ddof=1) / 4)
    assert approx(ate, 2.0) and approx(se, s) and approx([lo, hi], [2 - 1.96 * s, 2 + 1.96 * s])

def test_detects_effect():
    """a real effect of 0.3 on 4000 users per arm is detected"""
    rng = np.random.default_rng(0)
    ate, se, (lo, hi) = ab_test(rng.normal(10.3, 2, 4000), rng.normal(10.0, 2, 4000))
    assert lo > 0 and abs(ate - 0.3) < 0.15

def test_coverage():
    """with no effect, the 95% interval covers 0 in about 95% of experiments"""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(1000):
        _, _, (lo, hi) = ab_test(rng.normal(size=200), rng.normal(size=200))
        hits += lo <= 0 <= hi
    assert 0.93 < hits / 1000 < 0.97
~~~

<!-- /exercise -->

<!-- exercise: ipw | Inverse probability weighting -->
In observational data, sicker patients are more likely to be treated. Write `naive_effect(T, Y)` (difference in means) and `ipw_effect(T, Y, e)` using the IPW formula with known propensities `e`. The tests generate data where the naive estimate is badly biased and IPW recovers the true effect.

~~~python starter
import numpy as np

def naive_effect(T, Y):
    raise NotImplementedError

def ipw_effect(T, Y, e):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def naive_effect(T, Y):
    T, Y = np.asarray(T), np.asarray(Y, float)
    return float(Y[T == 1].mean() - Y[T == 0].mean())

def ipw_effect(T, Y, e):
    T, Y, e = np.asarray(T, float), np.asarray(Y, float), np.asarray(e, float)
    return float(np.mean(T * Y / e - (1 - T) * Y / (1 - e)))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 200_000
severity = rng.normal(size=n)                          # confounder
e = 1 / (1 + np.exp(-1.5 * severity))                  # sicker -> more likely treated
T = (rng.uniform(size=n) < e).astype(int)
Y = 2.0 * T - 3.0 * severity + rng.normal(size=n)      # true effect +2, severity hurts

def test_naive_is_biased():
    """the naive comparison even gets the sign wrong"""
    assert naive_effect(T, Y) < 0

def test_ipw_recovers_effect():
    """IPW with the true propensities recovers the +2 effect"""
    assert abs(ipw_effect(T, Y, e) - 2.0) < 0.15

def test_randomised_case():
    """with constant propensity 0.5, IPW and the naive estimate agree in expectation"""
    T2 = rng.integers(0, 2, n); Y2 = 2.0 * T2 + rng.normal(size=n)
    assert abs(ipw_effect(T2, Y2, np.full(n, 0.5)) - 2.0) < 0.05 and abs(naive_effect(T2, Y2) - 2.0) < 0.05
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why does randomising treatment make the difference in means an unbiased causal estimate?
   - [x] Random assignment makes treatment independent of everything else, so both groups are comparable on all variables, measured or not
   - [ ] It increases the sample size
   - [ ] It removes all noise from the outcome
   > Any difference in outcomes can then only come from the treatment (plus chance, which the standard error quantifies).
2. You check an A/B test every morning and stop the first day the p-value drops below 0.05. What is the problem?
   - [ ] Nothing, earlier stopping saves money
   - [x] Repeated looks inflate the false-positive rate well above 5%
   - [ ] The test will never become significant
   > Random fluctuations will cross the threshold at some point; fix the sample size in advance or use sequential methods.
3. A treatment looks harmful overall but helpful in every age group. What is going on?
   - [x] Simpson's paradox: age is a confounder distributed differently between the treated and untreated groups
   - [ ] A calculation error
   - [ ] The treatment has no effect
   > Aggregating over a confounder can reverse the direction of an association.
4. What does inverse probability weighting require to give unbiased effects?
   - [ ] A randomised experiment
   - [x] That all confounders are measured and used in the propensity model, and every unit has some chance of either treatment
   - [ ] That the outcome is binary
   > Unmeasured confounding and lack of overlap both break it, and neither can be fixed by more data.
<!-- /quiz -->
