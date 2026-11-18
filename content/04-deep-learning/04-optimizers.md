---
title: Optimisers: momentum to Adam
level: intermediate
minutes: 45
summary: Plain SGD struggles with ravines, noise and badly scaled parameters. Momentum smooths the path, RMSProp and Adam adapt the step per parameter, and schedules shape the learning rate over training.
---

## What goes wrong with plain SGD

[Gradient descent](#/lesson/gradient-descent) uses one learning rate for every parameter. Two problems follow. In a long narrow valley it zig-zags across the steep walls while creeping along the floor. And with noisy mini-batch gradients, each step jitters in random directions.

## Momentum

Keep a running **velocity** that accumulates past gradients, like a ball rolling downhill:

<!-- formula: SGD with momentum -->
$$\mathbf v_t = \beta\,\mathbf v_{t-1} + \mathbf g_t, \qquad \boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \eta\,\mathbf v_t$$

with $\beta \approx 0.9$. Components that keep the same sign add up and speed along the valley floor; components that flip sign each step cancel, damping the zig-zag. The effective step along a consistent direction grows to about $\eta/(1-\beta)$, ten times larger for $\beta = 0.9$. **Nesterov momentum** evaluates the gradient at the look-ahead point $\boldsymbol\theta - \eta\beta\mathbf v$ and corrects a little earlier.

## Adaptive learning rates

Parameters that receive large gradients should take smaller steps, and rarely updated parameters larger ones. **RMSProp** divides each component by a running root-mean-square of its gradients:

<!-- formula: RMSProp -->
$$\mathbf s_t = \rho\,\mathbf s_{t-1} + (1-\rho)\,\mathbf g_t^2, \qquad \boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \frac{\eta\,\mathbf g_t}{\sqrt{\mathbf s_t} + \epsilon}$$

(squares and roots element-wise). AdaGrad is the same with a sum instead of a moving average; it suits sparse features but its steps shrink forever.

## Adam

Adam combines momentum (first moment $\mathbf m$) with RMSProp (second moment $\mathbf v$), and corrects both for starting at zero:

<!-- formula: Adam -->
$$\begin{aligned}
\mathbf m_t &= \beta_1\mathbf m_{t-1} + (1-\beta_1)\mathbf g_t, & \mathbf v_t &= \beta_2\mathbf v_{t-1} + (1-\beta_2)\mathbf g_t^2\\
\hat{\mathbf m}_t &= \frac{\mathbf m_t}{1-\beta_1^t}, & \hat{\mathbf v}_t &= \frac{\mathbf v_t}{1-\beta_2^t}\\
\boldsymbol\theta_t &= \boldsymbol\theta_{t-1} - \frac{\eta\,\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t} + \epsilon}
\end{aligned}$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta$ around $10^{-3}$ (smaller for large Transformers). Without bias correction, the first steps would be tiny because $\mathbf m$ and $\mathbf v$ start at zero. Because the step is roughly $\eta$ times a ratio of order one, Adam is forgiving about gradient scale, which is why it is the default for most deep learning.

**AdamW** applies weight decay directly to the weights ($\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\lambda\boldsymbol\theta$) instead of adding it to the gradient, where Adam's scaling would distort it ([regularisation](#/lesson/regularization-dl)).

## Learning-rate schedules

The best learning rate changes during training:

- **Warmup**: start near zero and increase linearly over the first few hundred or thousand steps. Early gradients are large and Adam's second-moment estimate is unreliable; warmup avoids early blow-ups, and Transformers rely on it.
- **Decay**: lower the rate later so the noisy iterates settle. Step decay, exponential decay, or the popular **cosine** schedule:

<!-- formula: Cosine decay -->
$$\eta_t = \eta_{\min} + \tfrac12(\eta_{\max} - \eta_{\min})\Big(1 + \cos\frac{\pi t}{T}\Big)$$

Gradient **clipping** (rescaling the gradient when its norm exceeds a limit) is the other standard safety net, especially for [recurrent networks](#/lesson/rnn-lstm).

<!-- exercise: adam | One Adam step -->
Write `adam_step(theta, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8)`. `state` is a dict with `m`, `v` (arrays) and `t` (an int, the number of steps done so far). Increment `t`, update `m` and `v`, apply bias correction, and return the new `theta` (do not modify the input array in place). Update `state` in place.

~~~python starter
import numpy as np

def adam_step(theta, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    raise NotImplementedError

theta = np.array([1.0, -2.0])
state = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
print(adam_step(theta, np.array([0.5, -0.1]), state, lr=0.1))   # each moves by about 0.1
~~~

~~~python solution
import numpy as np

def adam_step(theta, g, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    state["m"] = b1 * state["m"] + (1 - b1) * g
    state["v"] = b2 * state["v"] + (1 - b2) * g ** 2
    m_hat = state["m"] / (1 - b1 ** t)
    v_hat = state["v"] / (1 - b2 ** t)
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps)

theta = np.array([1.0, -2.0])
state = {"m": np.zeros(2), "v": np.zeros(2), "t": 0}
print(adam_step(theta, np.array([0.5, -0.1]), state, lr=0.1))
~~~

~~~python tests
import numpy as np

def fresh():
    return {"m": np.zeros(2), "v": np.zeros(2), "t": 0}

def test_first_step_is_lr_sign():
    """the first bias-corrected step moves each parameter by about lr * sign(g)"""
    th = adam_step(np.array([1.0, -2.0]), np.array([0.5, -0.1]), fresh(), lr=0.1)
    assert approx(th, [0.9, -1.9], tol=1e-6)

def test_state_updated():
    """t increments and the moments are stored"""
    s = fresh()
    adam_step(np.zeros(2), np.array([1.0, 2.0]), s)
    assert s["t"] == 1 and approx(s["m"], [0.1, 0.2]) and approx(s["v"], [0.001, 0.004])

def test_input_not_modified():
    """returns a new array"""
    th = np.array([1.0, 1.0])
    adam_step(th, np.array([1.0, 1.0]), fresh())
    assert approx(th, [1.0, 1.0])

def test_minimises_badly_scaled_quadratic():
    """reaches the minimum of f = 100 x^2 + 0.01 y^2 from (1, 1)"""
    th, s = np.array([1.0, 1.0]), fresh()
    for _ in range(3000):
        th = adam_step(th, np.array([200 * th[0], 0.02 * th[1]]), s, lr=0.01)
    assert np.all(np.abs(th) < 0.05)
~~~

<!-- /exercise -->

<!-- exercise: schedule | Warmup then cosine decay -->
Write `lr_at(step, max_lr, warmup, total, min_lr=0.0)`: for `step < warmup` the rate rises linearly as `max_lr * (step + 1) / warmup`, reaching `max_lr` at step `warmup - 1`; after that, cosine decay from `max_lr` at `step = warmup` to `min_lr` at `step = total`, using $t = \text{step} - \text{warmup}$ and $T = \text{total} - \text{warmup}$. For `step >= total` return `min_lr`.

~~~python starter
import math

def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    raise NotImplementedError

print([round(lr_at(s, 1.0, 4, 20), 3) for s in range(0, 22, 3)])
~~~

~~~python solution
import math

def lr_at(step, max_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    t, T = step - warmup, total - warmup
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t / T))

print([round(lr_at(s, 1.0, 4, 20), 3) for s in range(0, 22, 3)])
~~~

~~~python tests
import math

def test_warmup():
    """linear warmup: 0.25, 0.5, 0.75, 1.0 over 4 steps"""
    assert [lr_at(s, 1.0, 4, 20) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]

def test_peak_and_end():
    """max at the end of warmup, min_lr at the end"""
    assert approx(lr_at(4, 2.0, 4, 20), 2.0) and approx(lr_at(20, 2.0, 4, 20, 0.1), 0.1)

def test_halfway():
    """halfway through the decay the rate is the midpoint"""
    assert approx(lr_at(12, 1.0, 4, 20, 0.0), 0.5)

def test_monotone_decay():
    """after warmup the rate never increases"""
    r = [lr_at(s, 1.0, 10, 100) for s in range(10, 120)]
    assert all(a >= b for a, b in zip(r, r[1:]))
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why does momentum help in a long, narrow valley?
   - [x] Gradient components that flip sign across the valley cancel, while those along the valley add up
   - [ ] It increases the learning rate for every parameter equally
   - [ ] It computes the exact Hessian
   > The running average damps oscillation and accelerates consistent directions.
2. What is the bias correction in Adam for?
   - [ ] To regularise the weights
   - [x] The moment estimates start at zero, so early on they are biased towards zero and the steps would be too small
   - [ ] To make Adam converge to a different minimum
   > Dividing by $1 - \beta^t$ rescales the averages to unbiased estimates during the first steps.
3. Why is AdamW preferred over Adam with L2 added to the gradient?
   - [ ] It is faster
   - [x] In Adam, an L2 term in the gradient gets rescaled per parameter, so the decay is uneven; AdamW decays the weights directly
   - [ ] Adam cannot use weight decay
   > Decoupling restores the intended uniform shrinkage of all weights.
4. A Transformer's loss explodes in the first hundred steps. What is the standard fix?
   - [x] Learning-rate warmup (and gradient clipping)
   - [ ] Increase the learning rate
   - [ ] Remove the bias correction
   > Large early updates with unreliable second-moment estimates destabilise training; ramping the rate up avoids them.
<!-- /quiz -->
