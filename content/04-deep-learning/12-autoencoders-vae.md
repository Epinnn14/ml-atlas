---
title: Autoencoders and VAEs
level: advanced
minutes: 50
summary: Learn a compact code by reconstructing the input through a bottleneck, then make the code space a probability distribution you can sample from. The ELBO, the reparameterisation trick and the KL term in closed form.
---

## Autoencoders

An **autoencoder** has two parts: an encoder $\mathbf z = e(\mathbf x)$ that compresses the input into a short code, and a decoder $\hat{\mathbf x} = g(\mathbf z)$ that reconstructs it. It is trained only to reproduce its input:

<!-- formula: Autoencoder loss -->
$$\mathcal L = \frac1n\sum_i\lVert\mathbf x_i - g(e(\mathbf x_i))\rVert^2$$

The **bottleneck** (a code much smaller than the input) forces the network to keep the most important structure. A linear autoencoder with squared error learns the same subspace as [PCA](#/lesson/pca); non-linear encoders and decoders learn curved structure.

Variants: **denoising** autoencoders reconstruct a clean input from a corrupted one, which forces robust features; **sparse** autoencoders penalise active code units. Uses include compression, pretraining, and [anomaly detection](#/lesson/anomaly-detection): an autoencoder trained on normal data reconstructs anomalies poorly.

## Why a plain autoencoder cannot generate

To make new images we would like to pick a random code and decode it. But nothing forces the codes of real images to fill a known region: they may sit on scattered islands, and a random code between them decodes to garbage.

## Variational autoencoders

A **VAE** makes the code probabilistic and shapes the code space. The encoder outputs a distribution, usually a diagonal Gaussian $q(\mathbf z\mid\mathbf x) = \mathcal N(\boldsymbol\mu(\mathbf x), \operatorname{diag}\boldsymbol\sigma^2(\mathbf x))$, and a prior $p(\mathbf z) = \mathcal N(\mathbf 0, I)$ says where codes should live. Training maximises the **evidence lower bound** on $\log p(\mathbf x)$:

<!-- formula: Evidence lower bound (ELBO) -->
$$\log p(\mathbf x) \ge \underbrace{\mathbb E_{q(\mathbf z\mid\mathbf x)}\big[\log p(\mathbf x\mid\mathbf z)\big]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\big(q(\mathbf z\mid\mathbf x)\,\Vert\,p(\mathbf z)\big)}_{\text{stay close to the prior}}$$

The reconstruction term wants informative codes; the [KL term](#/lesson/information-theory) pulls every code distribution towards the standard normal, so codes overlap and fill the space smoothly. Afterwards, sampling $\mathbf z\sim\mathcal N(0, I)$ and decoding produces new data.

For two diagonal Gaussians the KL term has a closed form:

<!-- formula: KL from a diagonal Gaussian to the standard normal -->
$$D_{\mathrm{KL}}\big(\mathcal N(\boldsymbol\mu,\operatorname{diag}\boldsymbol\sigma^2)\,\Vert\,\mathcal N(\mathbf 0, I)\big) = \frac12\sum_j\Big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\Big)$$

Encoders output $\log\sigma^2$ rather than $\sigma$, so the variance is always positive.

## The reparameterisation trick

The reconstruction term is an expectation over a random $\mathbf z$; we cannot backpropagate through "draw a sample". Rewrite the sample as a deterministic function of the parameters and independent noise:

<!-- formula: Reparameterisation trick -->
$$\mathbf z = \boldsymbol\mu + \boldsymbol\sigma\odot\boldsymbol\epsilon, \qquad \boldsymbol\epsilon\sim\mathcal N(\mathbf 0, I)$$

Now $\mathbf z$ depends smoothly on $\boldsymbol\mu$ and $\boldsymbol\sigma$, and gradients flow through it. One sample per example is usually enough.

## Trade-offs and descendants

VAE samples tend to be blurry, partly because a Gaussian likelihood averages over plausible details. [GANs](#/lesson/gans) give sharper samples but are harder to train; [diffusion models](#/lesson/diffusion) now give the best of both. VAEs remain important as the compression stage of latent diffusion models, which run the expensive diffusion in a small VAE code space instead of on pixels.

<!-- exercise: kl-gauss | The KL term -->
Write `kl_standard_normal(mu, logvar)` for arrays of shape `(n, k)` returning the KL divergence of each row's diagonal Gaussian from $\mathcal N(0, I)$, shape `(n,)`, using the closed form with $\sigma^2 = e^{\text{logvar}}$.

~~~python starter
import numpy as np

def kl_standard_normal(mu, logvar):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def kl_standard_normal(mu, logvar):
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1, axis=1)
~~~

~~~python tests
import numpy as np

def test_zero_for_prior():
    """the standard normal itself has KL 0"""
    assert approx(kl_standard_normal(np.zeros((3, 4)), np.zeros((3, 4))), np.zeros(3))

def test_mean_shift():
    """shifting the mean by m in one dimension costs m^2 / 2"""
    assert approx(kl_standard_normal(np.array([[2.0, 0.0]]), np.zeros((1, 2))), [2.0])

def test_positive():
    """KL is non-negative"""
    rng = np.random.default_rng(0)
    assert np.all(kl_standard_normal(rng.normal(size=(50, 3)), rng.normal(size=(50, 3))) >= 0)

def test_monte_carlo():
    """agrees with a Monte Carlo estimate of E_q[log q - log p]"""
    rng = np.random.default_rng(1)
    mu, logvar = np.array([0.5, -1.0]), np.array([-0.3, 0.4])
    s = np.exp(0.5 * logvar)
    z = mu + s * rng.normal(size=(200_000, 2))
    logq = np.sum(-0.5 * np.log(2 * np.pi * s ** 2) - (z - mu) ** 2 / (2 * s ** 2), axis=1)
    logp = np.sum(-0.5 * np.log(2 * np.pi) - z ** 2 / 2, axis=1)
    assert abs(kl_standard_normal(mu[None], logvar[None])[0] - np.mean(logq - logp)) < 0.01
~~~

<!-- /exercise -->

<!-- exercise: elbo | Reparameterise and compute the loss -->
Write `reparameterise(mu, logvar, rng)` returning `mu + exp(0.5 * logvar) * eps` with `eps = rng.normal(size=mu.shape)`, and `vae_loss(x, x_logits, mu, logvar)` returning the mean over the batch of (Bernoulli reconstruction loss + KL), where the reconstruction loss per example is the binary cross-entropy summed over pixels, computed stably from logits as $\sum_j[\log(1 + e^{\ell_j}) - x_j\ell_j]$. A `kl_standard_normal` helper is provided.

~~~python starter
import numpy as np

def kl_standard_normal(mu, logvar):
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1, axis=1)

def reparameterise(mu, logvar, rng):
    raise NotImplementedError

def vae_loss(x, x_logits, mu, logvar):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def kl_standard_normal(mu, logvar):
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - logvar - 1, axis=1)

def reparameterise(mu, logvar, rng):
    return mu + np.exp(0.5 * logvar) * rng.normal(size=mu.shape)

def vae_loss(x, x_logits, mu, logvar):
    recon = np.sum(np.logaddexp(0, x_logits) - x * x_logits, axis=1)
    return float(np.mean(recon + kl_standard_normal(mu, logvar)))
~~~

~~~python tests
import numpy as np

def test_reparameterise_recipe():
    """z = mu + sigma * eps with eps from rng.normal"""
    mu, lv = np.array([[1.0, -2.0]]), np.array([[0.0, np.log(4.0)]])
    eps = np.random.default_rng(3).normal(size=(1, 2))
    assert approx(reparameterise(mu, lv, np.random.default_rng(3)), mu + np.array([[1.0, 2.0]]) * eps)

def test_reparameterise_distribution():
    """samples have the right mean and standard deviation"""
    z = reparameterise(np.full((100_000, 1), 3.0), np.full((100_000, 1), np.log(0.25)), np.random.default_rng(0))
    assert abs(z.mean() - 3.0) < 0.01 and abs(z.std() - 0.5) < 0.01

def test_loss_value():
    """reconstruction + KL, averaged over the batch"""
    x = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    lg = np.array([[2.0, -1.0, 0.5], [-3.0, 0.0, 4.0]])
    mu, lv = np.array([[0.5], [0.0]]), np.array([[0.0], [-1.0]])
    p = 1 / (1 + np.exp(-lg))
    rec = -np.sum(x * np.log(p) + (1 - x) * np.log(1 - p), axis=1)
    kl = 0.5 * np.sum(mu ** 2 + np.exp(lv) - lv - 1, axis=1)
    assert approx(vae_loss(x, lg, mu, lv), np.mean(rec + kl))

def test_stable_logits():
    """huge logits do not produce inf or nan"""
    v = vae_loss(np.array([[1.0, 0.0]]), np.array([[800.0, -800.0]]), np.zeros((1, 1)), np.zeros((1, 1)))
    assert np.isfinite(v) and approx(v, 0.0, tol=1e-9)
~~~

<!-- hint: `np.logaddexp(0, l)` computes $\log(1 + e^{l})$ without overflow. -->
<!-- /exercise -->

<!-- quiz -->
1. What forces a plain autoencoder to learn useful features instead of copying its input?
   - [x] The bottleneck: the code is too small to store everything, so only the main structure survives
   - [ ] The KL term
   - [ ] Random noise in the decoder
   > Without a bottleneck or other constraint, an autoencoder can learn the identity function.
2. What does the KL term in the VAE objective do?
   - [ ] Improves reconstruction sharpness
   - [x] Pulls each code distribution towards the prior, so codes fill a known region and random samples decode sensibly
   - [ ] Makes the encoder deterministic
   > It regularises the latent space so sampling from $\mathcal N(0, I)$ works.
3. Why is the reparameterisation trick needed?
   - [ ] To make sampling faster
   - [x] Sampling is not differentiable; writing $\mathbf z = \boldsymbol\mu + \boldsymbol\sigma\odot\boldsymbol\epsilon$ moves the randomness into $\boldsymbol\epsilon$ so gradients reach $\boldsymbol\mu$ and $\boldsymbol\sigma$
   - [ ] To keep the variance positive
   > The noise is independent of the parameters, so the chain rule applies through $\boldsymbol\mu$ and $\boldsymbol\sigma$.
4. Why does a VAE encoder usually output $\log\sigma^2$ instead of $\sigma$?
   - [x] Any real number is a valid log-variance, so no constraint is needed to keep the variance positive
   - [ ] It makes the KL term zero
   - [ ] Logs are faster to compute
   > A network output can be any real number; exponentiating it guarantees a positive variance.
<!-- /quiz -->
