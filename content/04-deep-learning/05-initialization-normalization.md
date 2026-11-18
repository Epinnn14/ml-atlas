---
title: Initialisation, normalisation and residuals
level: advanced
minutes: 45
summary: Why deep networks once refused to train: signals that vanish or explode with depth. Xavier and He initialisation, batch and layer normalisation, and the residual connections that made very deep networks routine.
---

## Signals through many layers

Push an input through $L$ layers. If each layer multiplies the typical size of the activations by a factor $a$, the output is scaled by $a^L$. With $a = 0.9$ and 50 layers that is $0.005$; with $a = 1.1$ it is $117$. The same happens to gradients on the way back. **Vanishing** gradients leave early layers untrained; **exploding** gradients produce `nan`. Three tools keep the scale near 1: careful initialisation, normalisation layers, and residual connections.

## Variance-preserving initialisation

For a layer $\mathbf z = W\mathbf x$ with $n_{\text{in}}$ inputs, independent zero-mean weights of variance $\sigma^2_w$ and inputs of variance $\sigma_x^2$:

<!-- formula: Variance through a linear layer -->
$$\operatorname{Var}[z_j] = n_{\text{in}}\,\sigma_w^2\,\sigma_x^2$$

To keep the variance unchanged, choose $\sigma_w^2 = 1/n_{\text{in}}$. Adjusting for the activation gives the two standard schemes:

<!-- formula: Xavier and He initialisation -->
$$\text{Xavier (tanh, sigmoid): } \sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}, \qquad \text{He (ReLU): } \sigma_w^2 = \frac{2}{n_{\text{in}}}$$

He initialisation has the extra factor 2 because ReLU zeroes half of its inputs, halving the second moment. Biases usually start at zero.

## Batch normalisation

Instead of hoping the scale stays right, **normalise** it. Batch norm standardises each feature over the mini-batch, then lets the network rescale it with learned parameters $\gamma$ and $\beta$:

<!-- formula: Batch normalisation -->
$$\hat x_{ij} = \frac{x_{ij} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}, \qquad y_{ij} = \gamma_j\hat x_{ij} + \beta_j$$

where $\mu_j, \sigma_j^2$ are the mean and variance of feature $j$ over the batch. It allows much larger learning rates and makes training less sensitive to initialisation. Complication: at test time there may be one example, so batch statistics do not exist. During training, keep **running averages** of $\mu$ and $\sigma^2$, and use them at inference. Forgetting to switch to inference mode (`model.eval()`) is a classic bug. Batch norm also behaves badly with very small batches.

## Layer normalisation

**Layer norm** normalises each example over its own features instead of over the batch:

<!-- formula: Layer normalisation -->
$$\hat x_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}, \quad \mu_i = \frac1d\sum_j x_{ij}, \quad \sigma_i^2 = \frac1d\sum_j (x_{ij}-\mu_i)^2$$

followed by the same learned $\gamma, \beta$. It needs no batch statistics, so training and inference behave identically, and it works for sequences of any length. That is why [Transformers](#/lesson/transformer) use layer norm (or its simpler variant RMSNorm, which skips the mean). Convolutional networks for images mostly use batch norm.

## Residual connections

A **residual block** adds its input to its output:

<!-- formula: Residual block -->
$$\mathbf y = \mathbf x + F(\mathbf x)$$

If a block is not useful, it can learn $F \approx 0$ and pass its input through unchanged, so adding layers cannot make the network much worse. On the way back, the gradient has a direct path through the identity: $\partial\mathbf y/\partial\mathbf x = I + \partial F/\partial\mathbf x$, so it does not have to pass through every layer's weights. Residual networks (ResNets, 2015) trained 150+ layers where plain networks of 30 layers did worse than shallow ones. Every Transformer layer has two residual connections.

<!-- exercise: layernorm | Layer normalisation -->
Write `layer_norm(X, gamma, beta, eps=1e-5)` that normalises each **row** of `X` (shape `(n, d)`) over its `d` features using the biased variance, then applies `gamma` and `beta` (shape `(d,)`).

~~~python starter
import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def layer_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True)
    var = X.var(axis=-1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(3, 5, size=(4, 8))

def test_rows_standardised():
    """with gamma = 1 and beta = 0 each row has mean 0 and std about 1"""
    Y = layer_norm(X, np.ones(8), np.zeros(8))
    assert approx(Y.mean(1), np.zeros(4)) and approx(Y.std(1), np.ones(4), tol=1e-4)

def test_affine():
    """gamma and beta scale and shift every feature"""
    g, b = np.arange(1.0, 9.0), np.full(8, 2.0)
    Y0 = layer_norm(X, np.ones(8), np.zeros(8))
    assert approx(layer_norm(X, g, b), Y0 * g + b)

def test_independent_rows():
    """each row is normalised on its own (batch size does not matter)"""
    full = layer_norm(X, np.ones(8), np.zeros(8))
    single = layer_norm(X[2:3], np.ones(8), np.zeros(8))
    assert approx(full[2:3], single)

def test_scale_invariance():
    """scaling and shifting an input row does not change the output"""
    a = layer_norm(X, np.ones(8), np.zeros(8))
    b = layer_norm(10 * X + 4, np.ones(8), np.zeros(8))
    assert approx(a, b, tol=1e-4)
~~~

<!-- /exercise -->

<!-- exercise: batchnorm | Batch norm with running statistics -->
Complete `BatchNorm(d, momentum=0.1, eps=1e-5)`. In `forward(X, training)`: when training, normalise with the batch mean and biased variance of each column, and update `self.running_mean = (1 - momentum) * running_mean + momentum * batch_mean` (same for `running_var`, using the batch's **biased** variance for simplicity). When not training, normalise with the running statistics. Apply `gamma` (init ones) and `beta` (init zeros).

~~~python starter
import numpy as np

class BatchNorm:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)
        self.running_mean, self.running_var = np.zeros(d), np.ones(d)
        self.momentum, self.eps = momentum, eps

    def forward(self, X, training):
        raise NotImplementedError
~~~

~~~python solution
import numpy as np

class BatchNorm:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma, self.beta = np.ones(d), np.zeros(d)
        self.running_mean, self.running_var = np.zeros(d), np.ones(d)
        self.momentum, self.eps = momentum, eps

    def forward(self, X, training):
        if training:
            mu, var = X.mean(axis=0), X.var(axis=0)
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_var + m * var
        else:
            mu, var = self.running_mean, self.running_var
        return self.gamma * (X - mu) / np.sqrt(var + self.eps) + self.beta
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)

def test_training_normalises_batch():
    """in training mode each column of the output has mean 0 and std 1"""
    bn = BatchNorm(3)
    Y = bn.forward(rng.normal(5, 3, size=(64, 3)), training=True)
    assert approx(Y.mean(0), np.zeros(3)) and approx(Y.std(0), np.ones(3), tol=1e-3)

def test_running_update():
    """one training step moves the running stats 10% towards the batch stats"""
    bn = BatchNorm(2)
    X = np.array([[1.0, 10.0], [3.0, 14.0]])
    bn.forward(X, training=True)
    assert approx(bn.running_mean, [0.2, 1.2]) and approx(bn.running_var, [0.9 * 1 + 0.1 * 1.0, 0.9 * 1 + 0.1 * 4.0])

def test_inference_uses_running_stats():
    """in eval mode a single example works and uses the running statistics"""
    bn = BatchNorm(2)
    for _ in range(300):
        bn.forward(rng.normal([4.0, -1.0], [2.0, 0.5], size=(128, 2)), training=True)
    assert approx(bn.running_mean, [4.0, -1.0], tol=0.15)
    y = bn.forward(np.array([[4.0, -1.0]]), training=False)
    assert approx(y, [[0.0, 0.0]], tol=0.1)

def test_eval_does_not_update():
    """eval mode leaves the running statistics untouched"""
    bn = BatchNorm(2)
    bn.forward(rng.normal(size=(5, 2)) + 100, training=False)
    assert approx(bn.running_mean, [0.0, 0.0]) and approx(bn.running_var, [1.0, 1.0])
~~~

<!-- hint: Branch on `training`: batch statistics plus the running update, or the stored running statistics. -->
<!-- /exercise -->

<!-- quiz -->
1. A 40-layer ReLU network initialised with weights of standard deviation 0.01 does not learn at all. What is the likely cause?
   - [x] Activations and gradients shrink at every layer and vanish; use He initialisation
   - [ ] The learning rate is too high
   - [ ] ReLU cannot be used in deep networks
   > With such small weights each layer scales the signal far below 1, so after 40 layers almost nothing is left.
2. Your model scores well in training but erratically when predicting one image at a time. It uses batch norm. What is the likely bug?
   - [ ] Batch norm cannot be used for images
   - [x] It is still in training mode, so it normalises each single image with its own statistics instead of the running averages
   - [ ] The running averages are too accurate
   > Switch to inference mode so the stored running mean and variance are used.
3. Why do Transformers use layer norm rather than batch norm?
   - [x] Layer norm normalises each example on its own, so it does not depend on batch size or padding and behaves the same in training and inference
   - [ ] Layer norm is the only normalisation that has learned parameters
   - [ ] Batch norm does not work with attention by definition
   > Sequences vary in length and batch statistics over tokens are awkward; per-example normalisation avoids that.
4. How do residual connections help gradients?
   - [ ] They remove the need for an optimiser
   - [x] The identity path gives the gradient a route back that does not shrink through every layer's weights
   - [ ] They double the learning rate
   > $\partial(\mathbf x + F(\mathbf x))/\partial\mathbf x = I + \partial F/\partial\mathbf x$ always contains the identity.
<!-- /quiz -->
