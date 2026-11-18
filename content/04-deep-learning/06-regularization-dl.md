---
title: Regularising neural networks
level: intermediate
minutes: 40
summary: Big networks can memorise anything, even random labels. Weight decay, dropout, early stopping, data augmentation and label smoothing, what each does and how to combine them.
---

## Networks can memorise

A large network trained long enough reaches near-zero training loss even when the labels are shuffled at random: it simply memorises. Yet the same networks generalise well on real data. Regularisation tips the balance towards the patterns that generalise. The diagnosis is the same as in [the bias–variance lesson](#/lesson/bias-variance): a growing gap between training and validation loss.

## Weight decay

Add $\frac{\lambda}{2}\lVert\boldsymbol\theta\rVert^2$ to the loss, as in [ridge regression](#/lesson/regularization). With SGD this shrinks every weight by a constant factor each step:

<!-- formula: Weight decay -->
$$\boldsymbol\theta \leftarrow (1 - \eta\lambda)\,\boldsymbol\theta - \eta\,\nabla L$$

With Adam, apply it decoupled, as in AdamW ([optimisers](#/lesson/optimizers)). Typical values: $10^{-4}$ to $10^{-1}$ for AdamW. Biases and normalisation parameters are usually excluded.

## Dropout

During training, set each unit's output to zero with probability $p$, independently on every forward pass. The network cannot rely on any single unit and learns redundant features; it is like training an ensemble of many thinned networks that share weights. At test time all units are used.

The scales must match between training and test. **Inverted dropout** rescales the surviving units during training, so nothing changes at test time:

<!-- formula: Inverted dropout -->
$$\mathbf h_{\text{train}} = \frac{\mathbf m\odot\mathbf h}{1 - p}, \quad m_j\sim\text{Bernoulli}(1 - p), \qquad \mathbf h_{\text{test}} = \mathbf h$$

The expected value of each unit is then the same in both modes. Typical $p$: 0.1 in Transformers, 0.5 in older fully connected nets.

## Early stopping

Track validation loss every epoch and keep the weights from the best epoch. Stop when it has not improved for a number of epochs called the **patience**. Early stopping limits how far the weights travel from their small initial values, an effect similar to L2 regularisation, and it is nearly free.

## Data augmentation

The most effective regulariser is more data. Augmentation creates plausible new examples from existing ones by transformations that keep the label: flips, crops, small rotations and colour changes for images; noise, time shifts and speed changes for audio; synonym swaps or back-translation for text. It encodes knowledge of invariances ("a cat flipped is still a cat") directly into the data.

## Label smoothing

Train against softened targets: put $1 - \varepsilon$ on the correct class and spread $\varepsilon$ evenly over all $K$ classes:

<!-- formula: Label smoothing -->
$$y^{\text{smooth}}_k = (1 - \varepsilon)\,[k = y] + \frac{\varepsilon}{K}$$

The model is no longer rewarded for pushing logits towards infinity, which reduces over-confidence and often improves [calibration](#/lesson/calibration-uncertainty).

<!-- exercise: dropout | Inverted dropout -->
Write `dropout(h, p, training, rng)`. In training mode, draw `mask = rng.uniform(size=h.shape) >= p` and return `h * mask / (1 - p)`. In evaluation mode, or when `p == 0`, return `h` unchanged.

~~~python starter
import numpy as np

def dropout(h, p, training, rng):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def dropout(h, p, training, rng):
    if not training or p == 0:
        return h
    mask = rng.uniform(size=h.shape) >= p
    return h * mask / (1 - p)
~~~

~~~python tests
import numpy as np

def test_eval_identity():
    """evaluation mode returns the input"""
    h = np.arange(6.0)
    assert np.array_equal(dropout(h, 0.5, False, np.random.default_rng(0)), h)

def test_drop_rate():
    """about a fraction p of units become zero"""
    out = dropout(np.ones(100_000), 0.3, True, np.random.default_rng(1))
    assert abs(np.mean(out == 0) - 0.3) < 0.01

def test_expected_value_kept():
    """survivors are scaled by 1/(1 - p), so the mean is unchanged"""
    out = dropout(np.full(200_000, 2.0), 0.25, True, np.random.default_rng(2))
    assert abs(out.mean() - 2.0) < 0.02 and approx(out[out > 0][0], 2.0 / 0.75)

def test_recipe():
    """uses mask = rng.uniform(size=h.shape) >= p"""
    h = np.random.default_rng(3).normal(size=(4, 5))
    m = np.random.default_rng(9).uniform(size=(4, 5)) >= 0.4
    assert approx(dropout(h, 0.4, True, np.random.default_rng(9)), h * m / 0.6)
~~~

<!-- /exercise -->

<!-- exercise: early-stopping | Early stopping with patience -->
Write `early_stop(val_losses, patience)` that simulates training: walk through the validation losses epoch by epoch (epoch numbers start at 0), remember the best (lowest) loss and its epoch, and stop as soon as `patience` epochs in a row have not improved on the best. Return `(best_epoch, stopped_epoch)`, where `stopped_epoch` is the epoch at which training stopped, or the last epoch if it never stopped. Only a strictly lower loss counts as an improvement.

~~~python starter
def early_stop(val_losses, patience):
    raise NotImplementedError

print(early_stop([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.69, 0.9], patience=3))   # (2, 5)
~~~

~~~python solution
def early_stop(val_losses, patience):
    best, best_epoch, waited = float("inf"), -1, 0
    for epoch, loss in enumerate(val_losses):
        if loss < best:
            best, best_epoch, waited = loss, epoch, 0
        else:
            waited += 1
            if waited >= patience:
                return best_epoch, epoch
    return best_epoch, len(val_losses) - 1

print(early_stop([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.69, 0.9], patience=3))
~~~

~~~python tests
def test_example():
    """stops after 3 epochs without improvement, before seeing the later 0.69"""
    assert early_stop([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.69, 0.9], 3) == (2, 5)

def test_more_patience():
    """with patience 4 it survives the plateau and finds epoch 6"""
    assert early_stop([1.0, 0.8, 0.7, 0.72, 0.71, 0.75, 0.69, 0.9], 4) == (6, 7)

def test_always_improving():
    """never stops on a steadily falling curve"""
    assert early_stop([5, 4, 3, 2, 1], 2) == (4, 4)

def test_equal_is_not_improvement():
    """repeating the best value does not reset the counter"""
    assert early_stop([1.0, 0.5, 0.5, 0.5], 2) == (1, 3)
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why does inverted dropout divide by $1 - p$ during training?
   - [ ] To make training faster
   - [x] So the expected activation is the same in training and at test time, when no units are dropped
   - [ ] To prevent the gradients from vanishing
   > Without rescaling, the next layer would see larger inputs at test time than it was trained on.
2. Training loss keeps falling but validation loss started rising at epoch 12. With early stopping, which weights do you keep?
   - [x] The weights from the epoch with the lowest validation loss, around epoch 12
   - [ ] The weights from the final epoch
   - [ ] The initial weights
   > Early stopping restores the best checkpoint by validation loss.
3. You train an image classifier on photos of products. Which augmentation is risky?
   - [ ] Small crops and brightness changes
   - [x] Flipping images that contain text, such as labels on packaging
   - [ ] Small rotations
   > An augmentation must preserve the label. Mirrored text can change what the image shows or create unrealistic examples.
4. What does label smoothing discourage?
   - [ ] Learning the correct class
   - [x] Driving logits towards infinity to reach probability 1, which makes models over-confident
   - [ ] Using dropout
   > With smoothed targets the optimum is a finite logit gap.
<!-- /quiz -->
