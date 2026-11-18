---
title: Time series forecasting
level: advanced
minutes: 50
summary: Data where order matters. Trend and seasonality, autocorrelation, autoregressive models, the naive baselines every forecast must beat, backtesting without peeking at the future, and quantile forecasts.
---

## What makes time series different

Observations $y_1, y_2, \dots$ arrive in order and depend on their past. The usual assumption of independent examples fails, and so do random [cross-validation](#/lesson/cross-validation) folds: a model must never be trained on the future of the period it is evaluated on.

Typical structure:

- **Trend**: long-term growth or decline.
- **Seasonality**: patterns that repeat with a fixed period: the hour of the day, the day of the week, the month, or holidays that move each year.
- **Autocorrelation**: today resembles yesterday.
- **Noise and shocks**: one-off events, outages, promotions.

## Autocorrelation

The **autocorrelation** at lag $k$ is the correlation between the series and itself shifted by $k$ steps:

<!-- formula: Autocorrelation function -->
$$\rho_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar y)(y_{t-k} - \bar y)}{\sum_{t=1}^{n}(y_t - \bar y)^2}$$

A plot of $\rho_k$ against $k$ reveals the memory of the series, and peaks at multiples of the seasonal period (lag 7 for daily data with a weekly pattern). A series whose statistical properties do not change over time is **stationary**; differencing ($y_t - y_{t-1}$, or $y_t - y_{t-7}$ for weekly seasonality) often makes a series closer to stationary.

## Autoregressive models

An **AR(p)** model predicts the next value as a linear combination of the last $p$ values:

<!-- formula: Autoregressive model AR(p) -->
$$y_t = c + \phi_1y_{t-1} + \phi_2y_{t-2} + \dots + \phi_py_{t-p} + \varepsilon_t$$

It is just [linear regression](#/lesson/linear-regression) on lagged copies of the series, fitted by least squares. To forecast several steps ahead, feed each prediction back in as an input (**recursive** forecasting), which lets errors accumulate. ARIMA adds differencing and moving-average terms; exponential smoothing (ETS) models level, trend and season explicitly.

Modern practice often treats forecasting as tabular [boosting](#/lesson/boosting) on lag features, calendar features (day of week, holidays, paydays) and known future inputs (planned promotions), or uses deep models trained on many related series at once.

## Baselines first

Two forecasts cost nothing and are surprisingly hard to beat:

- **Naive**: tomorrow equals today, $\hat y_{t+h} = y_t$.
- **Seasonal naive**: the same day last week, $\hat y_{t+h} = y_{t+h-m}$ for season length $m$.

The **mean absolute scaled error** compares a forecast with the in-sample seasonal naive error, so it is unit-free and comparable across series:

<!-- formula: Mean absolute scaled error -->
$$\text{MASE} = \frac{\frac1H\sum_{h=1}^H|y_{T+h} - \hat y_{T+h}|}{\frac{1}{n-m}\sum_{t=m+1}^{n}|y_t - y_{t-m}|}$$

Below 1 beats the seasonal naive rule; above 1 loses to it.

## Backtesting

Evaluate as you would forecast: pick a cut-off, train on everything before it, forecast the next horizon, record the error, move the cut-off forward and repeat (**rolling-origin** evaluation). Average over many cut-offs, and include unusual periods (holidays) on purpose.

## Forecasting uncertainty: quantiles

Decisions such as how much stock to buy need more than a single number: they need "what is the 90% upper level of demand?". A **quantile forecast** for level $\tau$ is trained with the **pinball loss**, which charges under- and over-prediction asymmetrically:

<!-- formula: Pinball (quantile) loss -->
$$L_\tau(y, \hat q) = \begin{cases}\tau\,(y - \hat q) & \text{if } y \ge \hat q\\ (1-\tau)\,(\hat q - y) & \text{if } y < \hat q\end{cases}$$

Its minimiser is the $\tau$-quantile of the distribution. With $\tau = 0.5$ it is half the absolute error, giving the median. A set of quantiles (10%, 50%, 90%) forms a prediction interval whose **coverage** should match its nominal level.

<!-- exercise: ar | Fit and forecast an AR(p) model -->
Write `fit_ar(y, p)` returning `(c, phi)` by least squares, where row $t$ of the design matrix holds a 1 and $y_{t-1}, \dots, y_{t-p}$ (in that order) for $t = p, \dots, n-1$. Then `forecast_ar(y, c, phi, h)` returns the next `h` values recursively, feeding predictions back in.

~~~python starter
import numpy as np

def fit_ar(y, p):
    y = np.asarray(y, float)
    raise NotImplementedError

def forecast_ar(y, c, phi, h):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def fit_ar(y, p):
    y = np.asarray(y, float)
    n = len(y)
    X = np.column_stack([np.ones(n - p)] + [y[p - k:n - k] for k in range(1, p + 1)])
    coef = np.linalg.lstsq(X, y[p:], rcond=None)[0]
    return float(coef[0]), coef[1:]

def forecast_ar(y, c, phi, h):
    hist = list(np.asarray(y, float))
    out = []
    for _ in range(h):
        nxt = c + sum(phi[k] * hist[-1 - k] for k in range(len(phi)))
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 3000
y = np.zeros(n)
for t in range(2, n):
    y[t] = 1.0 + 0.6 * y[t - 1] - 0.3 * y[t - 2] + rng.normal(scale=0.5)

def test_recovers_coefficients():
    """recovers c = 1, phi = (0.6, -0.3)"""
    c, phi = fit_ar(y, 2)
    assert abs(c - 1.0) < 0.1 and approx(phi, [0.6, -0.3], tol=0.05)

def test_lag_order():
    """phi[0] multiplies the most recent value"""
    yy = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])     # y_t = 2 y_{t-1}
    c, phi = fit_ar(yy, 1)
    assert approx(phi, [2.0], tol=1e-8) and abs(c) < 1e-8

def test_recursive_forecast():
    """forecasts feed back in"""
    f = forecast_ar(np.array([1.0, 2.0]), 0.0, np.array([2.0]), 3)
    assert approx(f, [4.0, 8.0, 16.0])

def test_long_horizon_goes_to_mean():
    """a stationary AR forecast converges to c / (1 - sum(phi))"""
    c, phi = fit_ar(y, 2)
    f = forecast_ar(y, c, phi, 200)
    assert abs(f[-1] - c / (1 - phi.sum())) < 1e-6
~~~

<!-- hint: For lag `k`, the column is `y[p - k : n - k]`, aligned with the targets `y[p:]`. -->
<!-- /exercise -->

<!-- exercise: pinball-mase | Pinball loss and MASE -->
Write `pinball(y, q, tau)` returning the mean pinball loss, and `mase(y_train, y_test, y_pred, m)` returning the MASE with season length `m`.

~~~python starter
import numpy as np

def pinball(y, q, tau):
    raise NotImplementedError

def mase(y_train, y_test, y_pred, m):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def pinball(y, q, tau):
    y, q = np.asarray(y, float), np.asarray(q, float)
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))

def mase(y_train, y_test, y_pred, m):
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(np.asarray(y_test) - np.asarray(y_pred))) / scale)
~~~

~~~python tests
import numpy as np

def test_pinball_cases():
    """under-prediction costs tau per unit, over-prediction 1 - tau"""
    assert approx(pinball([10.0], [8.0], 0.9), 1.8) and approx(pinball([10.0], [12.0], 0.9), 0.2)

def test_median_is_half_mae():
    """tau = 0.5 gives half the mean absolute error"""
    y, q = np.array([1.0, 5.0, 2.0]), np.array([2.0, 3.0, 2.0])
    assert approx(pinball(y, q, 0.5), 0.5 * np.mean(np.abs(y - q)))

def test_minimiser_is_quantile():
    """the constant minimising pinball loss at tau is the tau-quantile"""
    y = np.random.default_rng(0).exponential(size=5001)
    grid = np.linspace(0, 4, 2001)
    best = grid[np.argmin([pinball(y, np.full_like(y, g), 0.9) for g in grid])]
    assert abs(best - np.quantile(y, 0.9)) < 0.01

def test_mase():
    """a forecast as good as seasonal naive in-sample scores 1"""
    train = np.array([1.0, 2, 3, 4, 2, 3, 4, 5])
    scale = np.mean(np.abs(train[4:] - train[:-4]))     # = 1
    assert approx(mase(train, [5.0, 6.0], [4.0, 7.0], 4), 1.0 / scale)
~~~

<!-- hint: `np.maximum(tau * d, (tau - 1) * d)` with `d = y - q` handles both cases at once. -->
<!-- /exercise -->

<!-- quiz -->
1. Why is random k-fold cross-validation wrong for evaluating a sales forecaster?
   - [x] Folds would let the model train on days after the ones it is tested on, which is impossible in deployment
   - [ ] Time series have too few points
   - [ ] Forecasting models cannot be retrained
   > Use rolling-origin backtests that always test on a period after the training data.
2. Daily sales have a strong weekly pattern. What is the most sensible first baseline?
   - [ ] The overall mean
   - [x] Seasonal naive: the value on the same weekday last week
   - [ ] Yesterday's value
   > Seasonal naive captures the weekly cycle at zero cost; any model must beat it.
3. A model's MASE is 1.4. What does that mean?
   - [ ] It is 40% better than the baseline
   - [x] Its error is 40% larger than the in-sample seasonal naive error
   - [ ] It is 1.4 times more accurate than perfect
   > MASE above 1 means the simple seasonal rule did better.
4. To decide how much stock to hold so that you run out only 1 time in 10, which forecast do you need?
   - [ ] The mean
   - [ ] The median
   - [x] The 90% quantile of demand
   > Stock at the 90% quantile covers demand nine times out of ten; train it with the pinball loss at $\tau = 0.9$.
<!-- /quiz -->
