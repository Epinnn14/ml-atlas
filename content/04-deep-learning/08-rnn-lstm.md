---
title: Recurrent networks and LSTMs
level: advanced
minutes: 50
summary: Process sequences one step at a time with a hidden state that carries memory. The RNN recurrence, backpropagation through time and its vanishing gradients, and the gated LSTM and GRU cells that fix them.
---

## A network with memory

Text, speech, sensor readings and sales are sequences, and their length varies. A **recurrent neural network** reads one element $\mathbf x_t$ at a time and updates a hidden state $\mathbf h_t$ that summarises everything seen so far, using the **same weights** at every step:

<!-- formula: RNN recurrence -->
$$\mathbf h_t = \tanh\big(W_{xh}\mathbf x_t + W_{hh}\mathbf h_{t-1} + \mathbf b_h\big), \qquad \mathbf y_t = W_{hy}\mathbf h_t + \mathbf b_y$$

Depending on the task you read out every step (tagging each word), only the last (classifying a review), or feed outputs back in to generate a sequence (an encoder–decoder for translation).

## Backpropagation through time

Unroll the recurrence over $T$ steps and you get a deep feed-forward network with shared weights; backpropagation through it is called **BPTT**. The gradient flowing from step $T$ back to step $t$ is multiplied by the Jacobian $\partial\mathbf h_k/\partial\mathbf h_{k-1} = \operatorname{diag}(1 - \mathbf h_k^2)\,W_{hh}$ once per step:

<!-- formula: Gradient through time -->
$$\frac{\partial\mathbf h_T}{\partial\mathbf h_t} = \prod_{k=t+1}^{T}\operatorname{diag}(1 - \mathbf h_k^2)\,W_{hh}$$

A product of many matrices shrinks or grows exponentially, depending on whether $W_{hh}$'s largest singular values are below or above 1. **Exploding** gradients are tamed by gradient clipping. **Vanishing** gradients are worse: the network cannot learn that a word 50 steps ago matters. In practice plain RNNs remember about 10–20 steps.

## LSTM

The **long short-term memory** cell adds a separate **cell state** $\mathbf c_t$ updated additively and controlled by gates, sigmoid-valued vectors between 0 and 1:

<!-- formula: LSTM cell -->
$$\begin{aligned}
\mathbf f_t &= \sigma(W_f[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_f) &&\text{forget gate}\\
\mathbf i_t &= \sigma(W_i[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_i) &&\text{input gate}\\
\mathbf o_t &= \sigma(W_o[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_o) &&\text{output gate}\\
\tilde{\mathbf c}_t &= \tanh(W_c[\mathbf h_{t-1}, \mathbf x_t] + \mathbf b_c) &&\text{candidate}\\
\mathbf c_t &= \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\tilde{\mathbf c}_t\\
\mathbf h_t &= \mathbf o_t\odot\tanh(\mathbf c_t)
\end{aligned}$$

where $[\mathbf h, \mathbf x]$ is concatenation. When the forget gate is near 1 and the input gate near 0, the cell state passes through unchanged and so does its gradient: an information highway through time, the same idea as [residual connections](#/lesson/initialization-normalization). Initialising the forget-gate bias to about 1 helps the cell remember from the start.

The **GRU** merges the forget and input gates into one update gate and drops the separate cell state. It has fewer parameters and often performs about as well.

## Where recurrent networks stand

LSTMs dominated speech recognition, translation and language modelling until about 2017. Their weakness is sequential computation: step $t$ must wait for step $t-1$, which wastes parallel hardware. [Attention](#/lesson/attention) and [Transformers](#/lesson/transformer) process all positions at once and connect distant positions directly, so they replaced RNNs for most large-scale sequence tasks. Recurrent ideas live on in efficient modern sequence models (state space models) that combine fast parallel training with constant-memory step-by-step inference.

<!-- exercise: rnn-forward | An RNN over a sequence -->
Write `rnn_forward(X, Wxh, Whh, bh, h0)` for a sequence `X` of shape `(T, d)`. Apply the recurrence step by step and return all hidden states as an array of shape `(T, h)`.

~~~python starter
import numpy as np

def rnn_forward(X, Wxh, Whh, bh, h0):
    # Wxh: (h, d), Whh: (h, h), bh: (h,), h0: (h,)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def rnn_forward(X, Wxh, Whh, bh, h0):
    h, states = h0, []
    for x in X:
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        states.append(h)
    return np.array(states)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
d, hdim, T = 3, 4, 6
Wxh, Whh, bh = rng.normal(size=(hdim, d)) * 0.5, rng.normal(size=(hdim, hdim)) * 0.5, rng.normal(size=hdim) * 0.1
X = rng.normal(size=(T, d))

def test_shape():
    """returns one hidden state per step"""
    assert rnn_forward(X, Wxh, Whh, bh, np.zeros(hdim)).shape == (T, hdim)

def test_first_step():
    """the first state uses h0"""
    h0 = rng.normal(size=hdim)
    H = rnn_forward(X, Wxh, Whh, bh, h0)
    assert approx(H[0], np.tanh(Wxh @ X[0] + Whh @ h0 + bh))

def test_recurrence():
    """each state follows from the previous one"""
    H = rnn_forward(X, Wxh, Whh, bh, np.zeros(hdim))
    for t in range(1, T):
        assert approx(H[t], np.tanh(Wxh @ X[t] + Whh @ H[t - 1] + bh))

def test_causal():
    """changing the last input does not change earlier states"""
    X2 = X.copy(); X2[-1] += 10
    a, b = rnn_forward(X, Wxh, Whh, bh, np.zeros(hdim)), rnn_forward(X2, Wxh, Whh, bh, np.zeros(hdim))
    assert approx(a[:-1], b[:-1]) and not approx(a[-1], b[-1])
~~~

<!-- /exercise -->

<!-- exercise: lstm-cell | One LSTM step -->
Write `lstm_step(x, h_prev, c_prev, W, b)`. `W` has shape `(4h, h + d)` and `b` has shape `(4h,)`: one matrix for all four gates applied to the concatenation `[h_prev, x]`. Split the result into four chunks of size `h` in the order **forget, input, output, candidate**, apply the gate activations, and return `(h, c)`.

~~~python starter
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h_prev, c_prev, W, b):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h_prev, c_prev, W, b):
    hdim = len(h_prev)
    z = W @ np.concatenate([h_prev, x]) + b
    f = sigmoid(z[:hdim])
    i = sigmoid(z[hdim:2 * hdim])
    o = sigmoid(z[2 * hdim:3 * hdim])
    g = np.tanh(z[3 * hdim:])
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
d, hdim = 3, 2
W, b = rng.normal(size=(4 * hdim, hdim + d)), rng.normal(size=4 * hdim)
x, h0, c0 = rng.normal(size=d), rng.normal(size=hdim), rng.normal(size=hdim)
sig = lambda z: 1 / (1 + np.exp(-z))

def test_matches_equations():
    """follows the gate equations"""
    z = W @ np.concatenate([h0, x]) + b
    f, i, o, g = sig(z[:2]), sig(z[2:4]), sig(z[4:6]), np.tanh(z[6:])
    c = f * c0 + i * g
    h, cc = lstm_step(x, h0, c0, W, b)
    assert approx(cc, c) and approx(h, o * np.tanh(c))

def test_memory_highway():
    """forget gate open and input gate closed: the cell state passes through"""
    Wz = np.zeros((8, 5)); bz = np.r_[np.full(2, 50.0), np.full(2, -50.0), np.zeros(4)]
    _, c = lstm_step(x, h0, c0, Wz, bz)
    assert approx(c, c0, tol=1e-9)

def test_forget_everything():
    """forget gate closed and input gate closed: the cell resets to zero"""
    Wz = np.zeros((8, 5)); bz = np.r_[np.full(2, -50.0), np.full(2, -50.0), np.zeros(4)]
    _, c = lstm_step(x, h0, c0, Wz, bz)
    assert approx(c, np.zeros(2), tol=1e-9)

def test_h_bounded():
    """the hidden state stays in (-1, 1)"""
    h, _ = lstm_step(x * 100, h0, c0 * 100, W, b)
    assert np.all(np.abs(h) <= 1)
~~~

<!-- hint: Compute `z = W @ np.concatenate([h_prev, x]) + b` once, then slice it into the four gates. -->
<!-- /exercise -->

<!-- quiz -->
1. Why do plain RNNs struggle to use information from 100 steps earlier?
   - [ ] They have too few parameters
   - [x] The gradient is multiplied by a Jacobian at every step and typically shrinks exponentially
   - [ ] Their hidden state is reset every 10 steps
   > Vanishing gradients through time mean distant inputs barely influence the weight updates.
2. What lets the LSTM carry information over long spans?
   - [x] The additively updated cell state, which passes through unchanged when the forget gate is near 1
   - [ ] A larger hidden state
   - [ ] Using ReLU instead of tanh
   > The additive path avoids repeated multiplication by weight matrices, so gradients survive.
3. Training an RNN produces occasional huge loss spikes and `nan`. What is the standard fix?
   - [ ] Remove the tanh activation
   - [x] Clip the gradient norm
   - [ ] Use a smaller hidden state
   > Exploding gradients are capped by rescaling the gradient whenever its norm exceeds a threshold.
4. Why did Transformers replace RNNs for large-scale language modelling?
   - [ ] RNNs cannot model sequences
   - [x] RNNs must process positions one after another, while attention handles all positions in parallel and links distant ones directly
   - [ ] Transformers have no parameters to train
   > Parallel training on modern hardware and direct long-range connections were decisive.
<!-- /quiz -->
