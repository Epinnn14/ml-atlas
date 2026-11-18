---
title: Generative adversarial networks
level: advanced
minutes: 45
summary: Two networks in a game: a generator that forges samples and a discriminator that spots forgeries. The minimax objective, the optimal discriminator, the non-saturating loss used in practice, and why GAN training is famously unstable.
---

## A forger and a detective

A **generator** $G$ turns random noise $\mathbf z\sim p(\mathbf z)$ into a sample $G(\mathbf z)$, for example an image. A **discriminator** $D$ outputs the probability that its input is real rather than generated. They are trained against each other: $D$ to tell real from fake, $G$ to fool $D$. Neither needs a likelihood: the generator learns only from the discriminator's gradient.

## The minimax game

<!-- formula: GAN objective -->
$$\min_G\max_D\; V(D, G) = \mathbb E_{\mathbf x\sim p_{\text{data}}}\big[\log D(\mathbf x)\big] + \mathbb E_{\mathbf z\sim p(\mathbf z)}\big[\log\big(1 - D(G(\mathbf z))\big)\big]$$

For the discriminator this is just binary cross-entropy with label 1 for real and 0 for fake. Training alternates: a few steps of gradient ascent on $V$ for $D$, then a step of descent for $G$.

## The optimal discriminator

For a fixed generator with sample density $p_g$, maximising $V$ point by point gives

<!-- formula: Optimal discriminator -->
$$D^*(\mathbf x) = \frac{p_{\text{data}}(\mathbf x)}{p_{\text{data}}(\mathbf x) + p_g(\mathbf x)}$$

Plugging it back in shows that the generator then minimises the **Jensen–Shannon divergence** between the data and its own distribution:

<!-- formula: Jensen–Shannon divergence -->
$$\operatorname{JSD}(p\,\Vert\,q) = \tfrac12 D_{\mathrm{KL}}(p\,\Vert\,m) + \tfrac12 D_{\mathrm{KL}}(q\,\Vert\,m), \qquad m = \tfrac12(p + q)$$

with $V(D^*, G) = -\log 4 + 2\,\operatorname{JSD}(p_{\text{data}}\Vert p_g)$. The global optimum is $p_g = p_{\text{data}}$, where $D^* = \tfrac12$ everywhere: the detective can only guess.

## The non-saturating loss

Early in training the generator is poor, $D(G(\mathbf z)) \approx 0$, and $\log(1 - D(G(\mathbf z)))$ is nearly flat there: the generator gets almost no gradient. In practice the generator instead **maximises** $\log D(G(\mathbf z))$, the same fixed point with strong gradients when it is losing:

<!-- formula: Non-saturating generator loss -->
$$\mathcal L_G = -\,\mathbb E_{\mathbf z}\big[\log D(G(\mathbf z))\big]$$

With logits $s = \text{logit}(D)$, both losses are cross-entropies from logits: $\mathcal L_D = \text{softplus}(-s_{\text{real}}) + \text{softplus}(s_{\text{fake}})$ and $\mathcal L_G = \text{softplus}(-s_{\text{fake}})$, where $\text{softplus}(u) = \log(1 + e^u)$.

## Why GANs are hard to train

- **Non-convergence**: it is a game, not a minimisation; the two players can oscillate.
- **Mode collapse**: the generator finds a few outputs that fool $D$ and produces only those, ignoring the variety of the data.
- **Vanishing signal**: if $D$ becomes too good, the generator's gradients become useless.

Remedies include Wasserstein GANs with gradient penalties, spectral normalisation of $D$, careful architectures and learning rates. GANs produced the first photorealistic faces (StyleGAN) and remain fast at sampling, but [diffusion models](#/lesson/diffusion) have largely replaced them for image generation because they are far easier to train and cover the data distribution better.

<!-- exercise: gan-losses | Discriminator and generator losses from logits -->
Write `d_loss(real_logits, fake_logits)` and `g_loss(fake_logits)` returning the mean losses above as floats, computed stably with `np.logaddexp(0, u)` for softplus.

~~~python starter
import numpy as np

def softplus(u):
    return np.logaddexp(0, u)

def d_loss(real_logits, fake_logits):
    raise NotImplementedError

def g_loss(fake_logits):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def softplus(u):
    return np.logaddexp(0, u)

def d_loss(real_logits, fake_logits):
    return float(np.mean(softplus(-real_logits)) + np.mean(softplus(fake_logits)))

def g_loss(fake_logits):
    return float(np.mean(softplus(-fake_logits)))
~~~

~~~python tests
import numpy as np

sig = lambda s: 1 / (1 + np.exp(-s))

def test_d_loss_is_bce():
    """equals -mean log D(real) - mean log(1 - D(fake))"""
    r, f = np.array([2.0, 0.5, -0.3]), np.array([-1.0, 0.2])
    assert approx(d_loss(r, f), -np.mean(np.log(sig(r))) - np.mean(np.log(1 - sig(f))))

def test_g_loss():
    """equals -mean log D(fake)"""
    f = np.array([-1.0, 0.2, 3.0])
    assert approx(g_loss(f), -np.mean(np.log(sig(f))))

def test_confused_discriminator():
    """logits of 0 everywhere give d_loss = 2 log 2 and g_loss = log 2"""
    z = np.zeros(4)
    assert approx(d_loss(z, z), 2 * np.log(2)) and approx(g_loss(z), np.log(2))

def test_stable():
    """a hopeless generator (fake logits -500) still gets a finite loss"""
    assert approx(g_loss(np.array([-500.0])), 500.0)
~~~

<!-- /exercise -->

<!-- exercise: optimal-d | The optimal discriminator and JSD -->
For two discrete distributions `p_data` and `p_g` over the same outcomes, write `optimal_d(p_data, p_g)` returning $D^*$ for each outcome, and `jsd(p, q)` returning the Jensen–Shannon divergence in nats (skip zero terms inside each KL). Then the tests check that $V(D^*, G) = -\log 4 + 2\,\text{JSD}$.

~~~python starter
import numpy as np

def optimal_d(p_data, p_g):
    raise NotImplementedError

def jsd(p, q):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def optimal_d(p_data, p_g):
    p_data, p_g = np.asarray(p_data, float), np.asarray(p_g, float)
    return p_data / (p_data + p_g)

def _kl(p, q):
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))

def jsd(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
~~~

~~~python tests
import numpy as np

p = np.array([0.5, 0.3, 0.2, 0.0])
q = np.array([0.1, 0.2, 0.3, 0.4])

def test_equal_distributions():
    """when the generator matches the data, D* = 1/2 and JSD = 0"""
    assert approx(optimal_d(p[:3] / p[:3].sum(), p[:3] / p[:3].sum()), np.full(3, 0.5)) and approx(jsd(p, p), 0.0)

def test_d_values():
    """D* follows the density ratio"""
    assert approx(optimal_d(p, q), [5 / 6, 0.6, 0.4, 0.0])

def test_jsd_bounds():
    """JSD is symmetric and at most log 2"""
    assert approx(jsd(p, q), jsd(q, p)) and 0 < jsd(p, q) <= np.log(2)
    assert approx(jsd(np.array([1.0, 0.0]), np.array([0.0, 1.0])), np.log(2))

def test_value_at_optimum():
    """V(D*, G) = -log 4 + 2 JSD(p_data, p_g)"""
    D = optimal_d(p, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        V = np.nansum(p * np.log(D)) + np.nansum(q * np.log(1 - D))
    assert approx(V, -np.log(4) + 2 * jsd(p, q))
~~~

<!-- /exercise -->

<!-- quiz -->
1. At the theoretical optimum of the GAN game, what does the discriminator output?
   - [ ] 1 for real and 0 for fake
   - [x] 0.5 everywhere, because generated and real data have the same distribution
   - [ ] Random values
   > With $p_g = p_{\text{data}}$, $D^* = p/(p + p) = 1/2$.
2. Why do practitioners use the non-saturating generator loss $-\log D(G(\mathbf z))$?
   - [x] When the generator is poor, $\log(1 - D(G(\mathbf z)))$ is flat and gives almost no gradient
   - [ ] It changes the optimum of the game
   - [ ] It removes the need for a discriminator
   > Both versions share the same fixed point, but the non-saturating one gives strong gradients early.
3. Your GAN produces the same few faces regardless of the noise input. What is this called?
   - [ ] Overfitting
   - [x] Mode collapse
   - [ ] Vanishing gradients
   > The generator has found a small set of outputs that fool the discriminator and stopped covering the rest of the data.
4. Which divergence does the original GAN objective minimise when the discriminator is optimal?
   - [ ] The KL divergence from data to model
   - [x] The Jensen–Shannon divergence
   - [ ] The Euclidean distance between images
   > Substituting $D^*$ gives $-\log 4 + 2\,\operatorname{JSD}(p_{\text{data}}\Vert p_g)$.
<!-- /quiz -->
