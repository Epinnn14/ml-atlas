---
title: The perceptron
level: beginner
minutes: 30
summary: The first learning machine, a single artificial neuron. Its update rule, the guarantee that it finds a separating line when one exists, and the XOR problem that showed why networks need hidden layers.
---

## An artificial neuron

A neuron takes inputs $x_1,\dots,x_d$, weighs them, adds a bias, and fires if the total passes zero:

<!-- formula: Perceptron -->
$$\hat y = \operatorname{sign}(\mathbf w^\top\mathbf x + b) \in \{-1, +1\}$$

This is the same linear decision boundary as [logistic regression](#/lesson/logistic-regression), with a hard step instead of a sigmoid. Frank Rosenblatt's perceptron (1958) was built in hardware and learned to recognise simple shapes.

## The learning rule

Go through the examples one by one. If an example is classified correctly, do nothing. If it is wrong, nudge the weights towards it:

<!-- formula: Perceptron update -->
$$\text{if } y_i(\mathbf w^\top\mathbf x_i + b) \le 0:\quad \mathbf w \leftarrow \mathbf w + y_i\mathbf x_i, \qquad b \leftarrow b + y_i$$

After the update, the score of that example moves in the right direction by $\lVert\mathbf x_i\rVert^2 + 1$. It is sub-gradient descent on the loss $\max(0, -y f(\mathbf x))$, a close cousin of the [SVM's hinge loss](#/lesson/svm) without the margin.

## The convergence theorem

If the classes are **linearly separable** with margin $\gamma$ (some unit vector separates them with every point at least $\gamma$ from the boundary) and all points lie within radius $R$, the perceptron makes at most

<!-- formula: Perceptron mistake bound -->
$$\text{mistakes} \le \left(\frac{R}{\gamma}\right)^2$$

updates, no matter the order of the examples. If the data are **not** separable, it never settles: it keeps cycling. The averaged or "pocket" perceptron keeps the best weights seen.

## XOR and the first AI winter

XOR outputs 1 when exactly one of two inputs is 1. Plot the four points: the two positives sit on one diagonal, the negatives on the other. No straight line separates them, so a single perceptron cannot learn XOR. Minsky and Papert's 1969 book made this limitation famous and funding for neural networks dried up for years.

The fix is to stack neurons. A hidden layer can compute OR and AND (each linearly separable), and the output neuron combines them: XOR = OR and not AND. Learning the weights of such a multi-layer network needed [backpropagation](#/lesson/neural-networks), which became practical in the 1980s.

<!-- exercise: perceptron | Train a perceptron -->
Write `train_perceptron(X, y, epochs=100)` for labels in $\{-1, +1\}$. Start from zeros. Each epoch, go through the examples **in order** and apply the update rule to each mistake (including scores exactly 0). Stop early after an epoch with no mistakes. Return `(w, b, total_mistakes)`.

~~~python starter
import numpy as np

def train_perceptron(X, y, epochs=100):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    mistakes = 0
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def train_perceptron(X, y, epochs=100):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    mistakes = 0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w = w + yi * xi
                b = b + yi
                errors += 1
        mistakes += errors
        if errors == 0:
            break
    return w, float(b), mistakes
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(200, 2))
margin = X @ np.array([1.0, -2.0]) + 0.3
keep = np.abs(margin) > 0.1                          # leave a gap: separable with a margin
X, y = X[keep], np.sign(margin[keep])

def test_separates():
    """on separable data it classifies every training point correctly"""
    w, b, _ = train_perceptron(X, y)
    assert np.all(np.sign(X @ w + b) == y)

def test_mistake_bound():
    """total mistakes stay below (R / gamma)^2 for the true separator"""
    _, _, m = train_perceptron(X, y)
    u = np.array([1.0, -2.0, 0.3]); u /= np.linalg.norm(u)
    Xa = np.c_[X, np.ones(len(X))]
    gamma = np.min(y * (Xa @ u)); R = np.max(np.linalg.norm(Xa, axis=1))
    assert m <= (R / gamma) ** 2

def test_first_update():
    """the first mistake (score 0) adds y0 * x0"""
    w, b, m = train_perceptron(X[:1], y[:1], epochs=1)
    assert approx(w, y[0] * X[0]) and b == y[0] and m == 1

def test_xor_never_converges():
    """XOR keeps producing mistakes every epoch"""
    Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); yx = np.array([-1, 1, 1, -1])
    w, b, m = train_perceptron(Xx, yx, epochs=50)
    assert m >= 50 and not np.all(np.sign(Xx @ w + b) == yx)
~~~

<!-- /exercise -->

<!-- exercise: xor-net | Build XOR from three neurons -->
Neurons here output 1 if their score is **greater than 0**, else 0. Choose weights so that `xor_net(x1, x2)` computes XOR with a hidden OR neuron, a hidden AND neuron and an output neuron that fires for "OR and not AND". Fill in the six weights and three biases.

~~~python starter
import numpy as np

def step(z):
    return int(z > 0)

def xor_net(x1, x2):
    h_or = step(0 * x1 + 0 * x2 + 0)       # fire if at least one input is 1
    h_and = step(0 * x1 + 0 * x2 + 0)      # fire only if both inputs are 1
    return step(0 * h_or + 0 * h_and + 0)  # OR and not AND

for a in (0, 1):
    for b in (0, 1):
        print(a, b, xor_net(a, b))
~~~

~~~python solution
import numpy as np

def step(z):
    return int(z > 0)

def xor_net(x1, x2):
    h_or = step(1 * x1 + 1 * x2 - 0.5)
    h_and = step(1 * x1 + 1 * x2 - 1.5)
    return step(1 * h_or - 2 * h_and - 0.5)

for a in (0, 1):
    for b in (0, 1):
        print(a, b, xor_net(a, b))
~~~

~~~python tests
def test_truth_table():
    """returns 0, 1, 1, 0 for the four inputs"""
    assert [xor_net(a, b) for a in (0, 1) for b in (0, 1)] == [0, 1, 1, 0]

def test_outputs_are_bits():
    """outputs are 0 or 1"""
    assert {xor_net(a, b) for a in (0, 1) for b in (0, 1)} <= {0, 1}
~~~

<!-- hint: OR fires when $x_1 + x_2 > 0.5$; AND fires when $x_1 + x_2 > 1.5$. -->
<!-- hint: The output should fire for (OR=1, AND=0) only: weights +1 and -2 with bias -0.5 work. -->
<!-- /exercise -->

<!-- quiz -->
1. When does the perceptron learning rule change the weights?
   - [ ] After every example
   - [x] Only when an example is misclassified (or exactly on the boundary)
   - [ ] Only at the end of each epoch
   > Correct examples cause no update, which is why the rule stops changing once the data are separated.
2. What does the perceptron do on data that are not linearly separable?
   - [ ] It converges to the best possible line
   - [x] It keeps updating forever without settling
   - [ ] It raises an error
   > The convergence guarantee needs separability. Otherwise there is always some mistake to fix.
3. Why can a single perceptron not compute XOR?
   - [x] The positive and negative points of XOR cannot be separated by one straight line
   - [ ] XOR needs negative weights
   - [ ] The step function is not differentiable
   > The positives are on one diagonal and the negatives on the other. Two lines, or a hidden layer, are needed.
4. A larger margin $\gamma$ in the mistake bound $(R/\gamma)^2$ means...
   - [ ] more mistakes before converging
   - [x] fewer mistakes before converging
   - [ ] the bound no longer applies
   > Well-separated classes are found quickly; nearly touching ones can take many updates.
<!-- /quiz -->
