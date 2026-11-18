---
title: Anomaly detection
level: intermediate
minutes: 40
summary: Finding the rare, strange examples: fraud, failing machines, broken sensors. Robust z-scores, Gaussian density and Mahalanobis distance, isolation forests, and how to evaluate with almost no labels.
---

## The setting

Anomalies are rare and varied: each fraud looks different from the last. Usually there are few or no labelled anomalies, so instead of learning "what fraud looks like" we learn **what normal looks like** and flag what does not fit. Every method below produces an **anomaly score**; a threshold turns it into alarms.

## Univariate scores and why to make them robust

The classic rule flags values far from the mean in units of standard deviation, $z = (x - \bar x)/s$, with $|z| > 3$ as an alarm. But the mean and standard deviation are themselves pulled by the outliers you are trying to find, and a few huge values can hide each other (**masking**). Robust statistics fix this with the median and the **median absolute deviation**:

<!-- formula: Robust z-score -->
$$z_i = \frac{x_i - \operatorname{median}(x)}{1.4826\cdot\operatorname{MAD}}, \qquad \operatorname{MAD} = \operatorname{median}\big(|x_i - \operatorname{median}(x)|\big)$$

The factor 1.4826 makes the MAD match the standard deviation for Gaussian data. Half the data would have to be corrupted before the median and MAD break down.

## Density and Mahalanobis distance

With several features, fit a density to normal data and flag low-density points. For a multivariate Gaussian with mean $\boldsymbol\mu$ and covariance $\Sigma$, low density is equivalent to a large **Mahalanobis distance**:

<!-- formula: Mahalanobis distance -->
$$d_M(\mathbf x) = \sqrt{(\mathbf x - \boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x - \boldsymbol\mu)}$$

It measures distance in units of the data's own spread and correlation: a point can be unremarkable in each feature separately and still be anomalous in combination (tall **and** very light). For Gaussian data $d_M^2$ follows a $\chi^2$ distribution with $d$ degrees of freedom, which gives principled thresholds. Clustered normal data call for a [Gaussian mixture](#/lesson/gaussian-mixtures) density instead.

## Isolation forest

An elegant idea: anomalies are **easy to isolate**. Build random trees that split on a random feature at a random value between its minimum and maximum, until each point is alone. Normal points sit in dense regions and need many splits; an anomaly is cut off after a few. The average path length over many trees, normalised, is the anomaly score. It is fast, needs no distances, and handles many features.

Other families: distance to the $k$-th nearest neighbour, the **local outlier factor** (density compared with the neighbours' density), one-class SVMs, and the reconstruction error of an [autoencoder](#/lesson/autoencoders-vae) trained on normal data.

## Evaluating with few labels

- If some confirmed anomalies exist, rank all points by score and measure **precision at k** (how many of the top $k$ alerts are real) and the area under the [precision–recall curve](#/lesson/evaluation-metrics). Accuracy is meaningless at a 0.1% base rate.
- Set the threshold from the capacity to investigate: if analysts can check 50 cases a day, alert on the top 50.
- Normal behaviour drifts (new products, seasons), so refit regularly and monitor the alert rate ([ML in production](#/lesson/ml-in-production)).

<!-- exercise: robust-z | Robust z-scores -->
Write `robust_z(x)` returning the robust z-score of every value, and `zscore(x)` returning the ordinary one (with `np.std`). If the MAD is 0, use 1 instead.

~~~python starter
import numpy as np

def zscore(x):
    raise NotImplementedError

def robust_z(x):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()

def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (1.4826 * (mad if mad > 0 else 1.0))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
normal = rng.normal(50, 2, 200)
x = np.concatenate([normal, rng.uniform(120, 130, 20)])     # 20 outliers among 220 values

def test_gaussian_scale():
    """on clean Gaussian data, robust and ordinary z-scores agree roughly"""
    a, b = robust_z(normal), zscore(normal)
    assert np.max(np.abs(a - b)) < 0.5

def test_masking():
    """20 big outliers inflate the std so the ordinary z-score misses them; the robust one does not"""
    assert np.all(np.abs(zscore(x)[-20:]) < 4)
    assert np.all(robust_z(x)[-20:] > 20)

def test_values():
    """matches the formula"""
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert approx(robust_z(v), (v - 3.0) / (1.4826 * 1.0))

def test_zero_mad():
    """a mostly constant column does not divide by zero"""
    assert np.all(np.isfinite(robust_z(np.array([5.0, 5.0, 5.0, 9.0]))))
~~~

<!-- /exercise -->

<!-- exercise: mahalanobis | Mahalanobis anomaly scores -->
Write `fit_gaussian(X)` returning `(mu, cov)` with `np.cov(X, rowvar=False)`, and `mahalanobis(X, mu, cov)` returning $d_M$ for every row, without inverting `cov` explicitly: solve with `np.linalg.solve`.

~~~python starter
import numpy as np

def fit_gaussian(X):
    raise NotImplementedError

def mahalanobis(X, mu, cov):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def fit_gaussian(X):
    X = np.asarray(X, float)
    return X.mean(axis=0), np.cov(X, rowvar=False)

def mahalanobis(X, mu, cov):
    diff = np.asarray(X, float) - mu
    sol = np.linalg.solve(cov, diff.T).T
    return np.sqrt(np.sum(diff * sol, axis=1))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
cov_true = np.array([[1.0, 0.9], [0.9, 1.0]])            # strongly correlated
X = rng.multivariate_normal([0, 0], cov_true, size=2000)

def test_identity_is_euclidean():
    """with identity covariance it is the Euclidean distance"""
    P = np.array([[3.0, 4.0], [1.0, 0.0]])
    assert approx(mahalanobis(P, np.zeros(2), np.eye(2)), [5.0, 1.0])

def test_correlation_matters():
    """(2, -2) breaks the correlation and scores far higher than (2, 2), though both are equally far in Euclidean terms"""
    mu, cov = fit_gaussian(X)
    d = mahalanobis(np.array([[2.0, 2.0], [2.0, -2.0]]), mu, cov)
    assert d[1] > 3 * d[0]

def test_chi_square_mean():
    """for Gaussian data the mean of d^2 is about the number of features"""
    mu, cov = fit_gaussian(X)
    assert abs(np.mean(mahalanobis(X, mu, cov) ** 2) - 2) < 0.1

def test_fit():
    """fit_gaussian returns the sample mean and covariance"""
    mu, cov = fit_gaussian(X)
    assert approx(mu, X.mean(0)) and approx(cov, np.cov(X.T))
~~~

<!-- hint: Solve `cov @ S = diff.T` for all rows at once; then each row's squared distance is `sum(diff * S.T, axis=1)`. -->
<!-- /exercise -->

<!-- quiz -->
1. Why use the median and MAD instead of the mean and standard deviation to flag outliers?
   - [ ] They are faster to compute
   - [x] Outliers themselves inflate the mean and standard deviation and can hide each other; the median and MAD barely move
   - [ ] They work only on Gaussian data
   > Robust statistics have a high breakdown point, so a handful of extreme values cannot distort the baseline.
2. A person is 190 cm tall and weighs 50 kg. Each value alone is within the normal range. Which score can flag them?
   - [ ] Per-feature z-scores
   - [x] Mahalanobis distance, which accounts for the correlation between height and weight
   - [ ] Neither
   > The combination is unusual given how the two features normally move together.
3. Why do isolation forests give anomalies short paths?
   - [x] Random splits separate points in sparse regions after only a few cuts
   - [ ] Anomalies have extreme feature values only
   - [ ] The trees are trained on labelled anomalies
   > Points in dense regions need many random splits to be separated from their neighbours.
4. Anomalies are 0.05% of transactions. Which evaluation is most useful?
   - [ ] Accuracy
   - [x] Precision among the top-ranked alerts, or average precision
   - [ ] Mean squared error
   > With so few positives, accuracy is near 100% for any model. What matters is how many of the cases you investigate are real.
<!-- /quiz -->
