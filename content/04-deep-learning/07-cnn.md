---
title: Convolutional networks
level: intermediate
minutes: 50
summary: Networks built for images. Convolution as a small sliding filter with shared weights, padding and stride, pooling, receptive fields, and the architecture pattern behind image classifiers.
---

## Why not a plain MLP for images

A 224×224 colour image has 150 528 numbers. A fully connected first layer with 1000 units would need 150 million weights, and it would have to learn separately that a cat's ear looks the same in the top-left and the bottom-right corner. Images have two properties a better layer should exploit:

- **Locality**: nearby pixels are related; distant ones much less.
- **Translation equivariance**: a pattern means the same thing wherever it appears.

## Convolution

A convolutional layer slides a small **kernel** (filter) of weights, say 3×3, over the image and computes a weighted sum at every position:

<!-- formula: 2-D convolution (cross-correlation) -->
$$Y[i, j] = \sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[u, v]\; X[i + u,\, j + v] + b$$

(Deep learning libraries compute this cross-correlation and call it convolution; true convolution flips the kernel, which makes no difference when the kernel is learned.) The same few weights are used at every position: **weight sharing**. A 3×3 kernel has 9 weights whatever the image size. The result is a **feature map** that is large where the local patch matches the kernel's pattern: an edge, a corner, a texture.

Real layers have many input channels (red, green, blue, or the maps of a previous layer) and many output channels. Each output channel has a kernel of shape $C_{\text{in}}\times k\times k$, so the layer has $C_{\text{out}}(C_{\text{in}}k^2 + 1)$ parameters.

## Padding and stride

Without padding, a $k\times k$ kernel shrinks the map by $k - 1$. **Padding** adds a border of zeros to keep the size ("same" padding $p = (k-1)/2$ for odd $k$). **Stride** $s$ moves the kernel $s$ pixels at a time, downsampling the output. For input size $n$:

<!-- formula: Convolution output size -->
$$n_{\text{out}} = \left\lfloor\frac{n + 2p - k}{s}\right\rfloor + 1$$

## Pooling

**Max pooling** takes the maximum over small windows, typically 2×2 with stride 2, halving width and height. It makes the representation smaller and slightly tolerant to small shifts. Many modern networks use strided convolutions instead, and end with **global average pooling** over each channel before the final classifier.

## Receptive field and hierarchy

Each unit sees a patch of the input called its **receptive field**. Stacking layers grows it: two 3×3 layers see 5×5, three see 7×7, and pooling multiplies the growth. So early layers detect edges, middle layers combine them into textures and parts, late layers respond to whole objects. A typical classifier repeats *convolution → normalisation → ReLU* blocks, halving resolution and doubling channels at each stage, then pools globally and applies a linear layer with [softmax](#/lesson/softmax-regression). ResNets add [residual connections](#/lesson/initialization-normalization) to go deep.

Convolutions also work on 1-D signals (audio, time series, text) and 3-D volumes (video, medical scans). Vision Transformers, which apply [attention](#/lesson/attention) to image patches, now compete with CNNs on large datasets, but convolution's built-in assumptions make it more data-efficient.

<!-- exercise: conv2d | A 2-D convolution -->
Write `conv2d(X, K, stride=1, pad=0)` for a single-channel image `X` (2-D array) and kernel `K` (2-D), computing the cross-correlation formula above with zero padding and stride, and no bias. Return the output feature map. Loops over output positions are fine.

~~~python starter
import numpy as np

def conv2d(X, K, stride=1, pad=0):
    X, K = np.asarray(X, float), np.asarray(K, float)
    raise NotImplementedError

edge = np.array([[1.0, 0.0, -1.0]] * 3)          # vertical edge detector
img = np.zeros((5, 5)); img[:, :2] = 1.0
print(conv2d(img, edge))
~~~

~~~python solution
import numpy as np

def conv2d(X, K, stride=1, pad=0):
    X, K = np.asarray(X, float), np.asarray(K, float)
    if pad:
        X = np.pad(X, pad)
    k1, k2 = K.shape
    H = (X.shape[0] - k1) // stride + 1
    W = (X.shape[1] - k2) // stride + 1
    out = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            patch = X[i * stride:i * stride + k1, j * stride:j * stride + k2]
            out[i, j] = np.sum(patch * K)
    return out

edge = np.array([[1.0, 0.0, -1.0]] * 3)
img = np.zeros((5, 5)); img[:, :2] = 1.0
print(conv2d(img, edge))
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(7, 6)); K = rng.normal(size=(3, 3))

def ref(X, K, s, p):
    X = np.pad(X, p)
    H = (X.shape[0] - 3) // s + 1; W = (X.shape[1] - 3) // s + 1
    return np.array([[np.sum(X[i*s:i*s+3, j*s:j*s+3] * K) for j in range(W)] for i in range(H)])

def test_valid():
    """no padding, stride 1: output shrinks by k - 1"""
    out = conv2d(X, K)
    assert out.shape == (5, 4) and approx(out, ref(X, K, 1, 0))

def test_same_padding():
    """pad 1 with a 3x3 kernel keeps the size"""
    out = conv2d(X, K, pad=1)
    assert out.shape == (7, 6) and approx(out, ref(X, K, 1, 1))

def test_stride():
    """stride 2 follows the output-size formula"""
    out = conv2d(X, K, stride=2, pad=1)
    assert out.shape == ((7 + 2 - 3) // 2 + 1, (6 + 2 - 3) // 2 + 1) and approx(out, ref(X, K, 2, 1))

def test_edge_detector():
    """a vertical edge detector responds only at the edge"""
    img = np.zeros((5, 5)); img[:, :2] = 1.0
    out = conv2d(img, np.array([[1.0, 0.0, -1.0]] * 3))
    assert approx(out, [[3, 3, 0]] * 3)

def test_translation_equivariance():
    """shifting the input shifts the output"""
    img = np.zeros((8, 8)); img[2, 2] = 1
    moved = np.roll(img, 2, axis=1)
    assert approx(np.roll(conv2d(img, K), 2, axis=1)[:, 2:], conv2d(moved, K)[:, 2:])
~~~

<!-- hint: Pad first with `np.pad(X, pad)`, compute the output size with the formula, then for each output cell multiply the matching patch by the kernel and sum. -->
<!-- /exercise -->

<!-- exercise: maxpool | Max pooling and parameter counts -->
Write `max_pool(X, size=2)` that takes the maximum over non-overlapping `size`×`size` windows of a 2-D array (drop leftover rows and columns that do not fill a window). Then write `conv_params(c_in, c_out, k)` returning the number of weights plus biases of a convolution layer.

~~~python starter
import numpy as np

def max_pool(X, size=2):
    raise NotImplementedError

def conv_params(c_in, c_out, k):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def max_pool(X, size=2):
    X = np.asarray(X)
    H, W = X.shape[0] // size, X.shape[1] // size
    return X[:H * size, :W * size].reshape(H, size, W, size).max(axis=(1, 3))

def conv_params(c_in, c_out, k):
    return c_out * (c_in * k * k + 1)
~~~

~~~python tests
import numpy as np

def test_pool_values():
    """2x2 max pooling on a 4x4 grid"""
    X = np.arange(16).reshape(4, 4)
    assert np.array_equal(max_pool(X), [[5, 7], [13, 15]])

def test_pool_drops_leftovers():
    """a 5x7 input gives a 2x3 output"""
    assert max_pool(np.random.default_rng(0).normal(size=(5, 7))).shape == (2, 3)

def test_pool_size3():
    """window size 3"""
    X = np.random.default_rng(1).normal(size=(6, 6))
    ref = np.array([[X[i:i+3, j:j+3].max() for j in (0, 3)] for i in (0, 3)])
    assert approx(max_pool(X, 3), ref)

def test_params():
    """3 -> 64 channels with 3x3 kernels has 1792 parameters"""
    assert conv_params(3, 64, 3) == 1792 and conv_params(64, 128, 1) == 8320
~~~

<!-- hint: Reshape to `(H, size, W, size)` and take the max over axes 1 and 3. -->
<!-- /exercise -->

<!-- quiz -->
1. A 32×32 input passes through a 5×5 convolution with padding 0 and stride 1. What is the output size?
   - [ ] 32×32
   - [x] 28×28
   - [ ] 27×27
   > $(32 + 0 - 5)/1 + 1 = 28$.
2. What does weight sharing buy a convolutional layer?
   - [x] Far fewer parameters, and the same detector applied at every position
   - [ ] Invariance to rotation
   - [ ] Faster gradient descent
   > One small kernel replaces a separate weight for every input–output pixel pair.
3. How large is the receptive field of a unit after three stacked 3×3 convolutions with stride 1?
   - [ ] 3×3
   - [ ] 9×9
   - [x] 7×7
   > Each extra 3×3 layer adds 2 pixels: 3, 5, 7.
4. How many parameters does a convolution from 16 to 32 channels with 3×3 kernels have, including biases?
   - [ ] 4 608
   - [x] 4 640
   - [ ] 288
   > $32 \times (16\cdot 9 + 1) = 4640$.
<!-- /quiz -->
