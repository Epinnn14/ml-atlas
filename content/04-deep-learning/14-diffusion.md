---
title: Diffusion models
level: senior
minutes: 55
summary: Generate data by learning to undo noise. The forward process that destroys an image step by step, its closed form, the simple noise-prediction loss, and the reverse sampling loop of DDPM.
---

## Destroy, then learn to repair

Take an image and add a little Gaussian noise, then a little more, for $T$ steps (typically 1000), until only noise remains. That **forward process** needs no learning. A **diffusion model** learns the reverse: given a noisy image and the step number, predict how to make it slightly less noisy. Starting from pure noise and repeating the reverse step $T$ times produces a new image.

## The forward process

Each step mixes in noise with a small variance $\beta_t$ from a fixed **noise schedule** (for example linear from $10^{-4}$ to $0.02$):

<!-- formula: Diffusion forward step -->
$$q(\mathbf x_t\mid\mathbf x_{t-1}) = \mathcal N\big(\sqrt{1-\beta_t}\,\mathbf x_{t-1},\ \beta_t I\big)$$

Because Gaussians compose, we can jump to any step directly. With $\alpha_t = 1 - \beta_t$ and the cumulative product $\bar\alpha_t = \prod_{s=1}^t\alpha_s$:

<!-- formula: Closed-form noising -->
$$\mathbf x_t = \sqrt{\bar\alpha_t}\,\mathbf x_0 + \sqrt{1 - \bar\alpha_t}\;\boldsymbol\epsilon, \qquad \boldsymbol\epsilon\sim\mathcal N(\mathbf 0, I)$$

$\bar\alpha_t$ falls from almost 1 to almost 0: the signal fades and noise takes over, while the total variance stays about 1. The ratio $\bar\alpha_t/(1-\bar\alpha_t)$ is the signal-to-noise ratio at step $t$.

## Training: predict the noise

A network $\boldsymbol\epsilon_\theta(\mathbf x_t, t)$, usually a U-Net or a Transformer, is given the noisy input and the step, and asked to predict the noise that was added. Training is remarkably simple:

1. Take a real example $\mathbf x_0$, a random step $t$, and noise $\boldsymbol\epsilon$.
2. Build $\mathbf x_t$ with the closed form.
3. Minimise the squared error of the noise prediction:

<!-- formula: Diffusion training loss -->
$$\mathcal L = \mathbb E_{\mathbf x_0, t, \boldsymbol\epsilon}\Big[\big\lVert\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\ t\big)\big\rVert^2\Big]$$

No adversary, no collapse: it is plain regression, which is why diffusion is so much more stable than [GANs](#/lesson/gans). This loss is a simplified, reweighted form of a variational bound like the [VAE's ELBO](#/lesson/autoencoders-vae), and it is equivalent to learning the **score** $\nabla_{\mathbf x}\log p(\mathbf x_t)$, the direction towards more probable data.

## Sampling

Given the predicted noise, the clean image estimate is $\hat{\mathbf x}_0 = (\mathbf x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$. DDPM takes one small step back towards it and adds a little fresh noise:

<!-- formula: DDPM reverse step -->
$$\mathbf x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol\epsilon_\theta(\mathbf x_t, t)\Big) + \sigma_t\mathbf z, \qquad \mathbf z\sim\mathcal N(\mathbf 0, I)$$

with $\sigma_t^2 = \beta_t$ and no noise at the last step. A thousand network evaluations per sample is slow, so practical samplers (DDIM, DPM-Solver) take 20–50 larger deterministic steps, and distillation reduces it to a handful.

## Conditioning and guidance

To generate from a text prompt, the network also receives the prompt embedding (through [cross-attention](#/lesson/attention)). **Classifier-free guidance** trains the same network with and without the condition, then pushes samples further in the direction the condition suggests: $\tilde{\boldsymbol\epsilon} = \boldsymbol\epsilon_\theta(\mathbf x_t, \varnothing) + w\,(\boldsymbol\epsilon_\theta(\mathbf x_t, c) - \boldsymbol\epsilon_\theta(\mathbf x_t, \varnothing))$ with $w > 1$, trading diversity for prompt fidelity. **Latent diffusion** runs the whole process on a compressed autoencoder code instead of pixels, making high-resolution generation affordable. Closely related **flow matching** models learn a velocity field that transports noise to data along nearly straight paths.

<!-- exercise: schedule | Noise schedule and closed-form noising -->
Write `linear_schedule(T, beta_start=1e-4, beta_end=0.02)` returning `(betas, alphas, alpha_bars)` as arrays of length `T` (use `np.linspace` and `np.cumprod`), and `q_sample(x0, t, alpha_bars, eps)` returning $\mathbf x_t$ by the closed form for a **0-based** step index `t`.

~~~python starter
import numpy as np

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    raise NotImplementedError

def q_sample(x0, t, alpha_bars, eps):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T)
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)

def q_sample(x0, t, alpha_bars, eps):
    ab = alpha_bars[t]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
~~~

~~~python tests
import numpy as np

def test_schedule():
    """betas rise linearly, alpha_bar falls from ~1 to ~0"""
    b, a, ab = linear_schedule(1000)
    assert len(b) == 1000 and approx(b[[0, -1]], [1e-4, 0.02]) and approx(a, 1 - b)
    assert ab[0] > 0.999 and ab[-1] < 1e-4 and np.all(np.diff(ab) < 0)

def test_closed_form_matches_steps():
    """jumping to step t equals applying t+1 single steps with the same total noise variance"""
    b, a, ab = linear_schedule(50)
    x0 = np.full(200_000, 2.0)
    rng = np.random.default_rng(0)
    x = x0.copy()
    for t in range(50):
        x = np.sqrt(a[t]) * x + np.sqrt(b[t]) * rng.normal(size=x.shape)
    direct = q_sample(x0, 49, ab, np.random.default_rng(1).normal(size=x0.shape))
    assert abs(x.mean() - direct.mean()) < 0.02 and abs(x.std() - direct.std()) < 0.02

def test_q_sample_formula():
    """x_t = sqrt(ab) x0 + sqrt(1 - ab) eps"""
    _, _, ab = linear_schedule(10)
    x0, eps = np.array([1.0, -1.0]), np.array([0.5, 0.5])
    assert approx(q_sample(x0, 3, ab, eps), np.sqrt(ab[3]) * x0 + np.sqrt(1 - ab[3]) * eps)

def test_variance_preserved():
    """unit-variance data stays unit variance at every step"""
    _, _, ab = linear_schedule(1000)
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=100_000)
    for t in (0, 300, 999):
        assert abs(q_sample(x0, t, ab, rng.normal(size=x0.shape)).std() - 1) < 0.02
~~~

<!-- /exercise -->

<!-- exercise: ddpm-step | One reverse step -->
Write `ddpm_step(x_t, t, eps_pred, betas, alphas, alpha_bars, z)` returning $\mathbf x_{t-1}$ by the DDPM formula with $\sigma_t = \sqrt{\beta_t}$ and noise `z`, except that at `t == 0` no noise is added. Also write `predict_x0(x_t, t, eps_pred, alpha_bars)` returning the clean estimate $\hat{\mathbf x}_0$.

~~~python starter
import numpy as np

def ddpm_step(x_t, t, eps_pred, betas, alphas, alpha_bars, z):
    raise NotImplementedError

def predict_x0(x_t, t, eps_pred, alpha_bars):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ddpm_step(x_t, t, eps_pred, betas, alphas, alpha_bars, z):
    mean = (x_t - betas[t] / np.sqrt(1 - alpha_bars[t]) * eps_pred) / np.sqrt(alphas[t])
    return mean if t == 0 else mean + np.sqrt(betas[t]) * z

def predict_x0(x_t, t, eps_pred, alpha_bars):
    return (x_t - np.sqrt(1 - alpha_bars[t]) * eps_pred) / np.sqrt(alpha_bars[t])
~~~

~~~python tests
import numpy as np

T = 100
betas = np.linspace(1e-4, 0.02, T); alphas = 1 - betas; ab = np.cumprod(alphas)

def test_predict_x0_inverts_noising():
    """with the true noise, predict_x0 recovers x0 exactly"""
    x0, eps = np.array([0.3, -1.2, 2.0]), np.array([1.0, 0.5, -0.4])
    xt = np.sqrt(ab[60]) * x0 + np.sqrt(1 - ab[60]) * eps
    assert approx(predict_x0(xt, 60, eps, ab), x0)

def test_step_formula():
    """matches the DDPM update"""
    xt, e, z = np.array([0.4, -0.2]), np.array([0.1, 0.3]), np.array([1.0, -1.0])
    m = (xt - betas[40] / np.sqrt(1 - ab[40]) * e) / np.sqrt(alphas[40])
    assert approx(ddpm_step(xt, 40, e, betas, alphas, ab, z), m + np.sqrt(betas[40]) * z)

def test_last_step_noiseless():
    """no noise is added at t = 0"""
    xt, e = np.array([0.4]), np.array([0.1])
    a = ddpm_step(xt, 0, e, betas, alphas, ab, np.array([5.0]))
    b = ddpm_step(xt, 0, e, betas, alphas, ab, np.array([-5.0]))
    assert approx(a, b)

def test_sampling_with_oracle():
    """with a perfect noise predictor for a point mass at 1.5, sampling ends near 1.5"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    for t in reversed(range(T)):
        eps_true = (x - np.sqrt(ab[t]) * 1.5) / np.sqrt(1 - ab[t])   # the noise that explains x_t
        x = ddpm_step(x, t, eps_true, betas, alphas, ab, rng.normal(size=x.shape))
    assert abs(x.mean() - 1.5) < 0.05 and x.std() < 0.05
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why can diffusion training pick a random step $t$ for each example without simulating all earlier steps?
   - [x] The forward process has a closed form: $\mathbf x_t = \sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$
   - [ ] The network ignores $t$
   - [ ] Every step adds the same noise
   > Composing Gaussian steps gives another Gaussian, so any step can be sampled directly.
2. What does the network in a standard DDPM learn to predict?
   - [ ] The class label of the image
   - [x] The noise that was added to produce $\mathbf x_t$
   - [ ] Whether the input is real or fake
   > The loss is the squared error between the true and predicted noise, a plain regression.
3. Why are diffusion models easier to train than GANs?
   - [ ] They have fewer parameters
   - [x] Training is a single regression objective with no adversarial game, so there is no oscillation or mode collapse
   - [ ] They need no data
   > A stable loss that can only go down is much easier to optimise than a two-player game.
4. What does a classifier-free guidance weight $w > 1$ do?
   - [x] Pushes samples more strongly towards the condition, improving prompt fidelity at the cost of diversity
   - [ ] Speeds up sampling
   - [ ] Removes the need for the noise schedule
   > It extrapolates from the unconditional to the conditional prediction.
<!-- /quiz -->
