---
title: Automatic differentiation
level: advanced
minutes: 50
summary: How PyTorch and JAX compute gradients of any program. Build a computational graph while computing, then walk it backwards once. You will write a tiny reverse-mode autograd engine.
---

## Three ways to get derivatives

- **Symbolic**: manipulate formulas like a computer algebra system. Exact, but expressions blow up and programs with loops and branches do not fit.
- **Numerical**: [finite differences](#/lesson/calculus-gradients). Easy, but approximate and $O(\text{parameters})$ evaluations per gradient, hopeless for millions of weights.
- **Automatic**: apply the chain rule to the elementary operations the program actually executes. Exact up to floating point, and works through loops, branches and function calls.

## The computational graph

Every computation is a graph of elementary operations. For $L = (a\cdot b + c)^2$: a node $d = a\cdot b$, a node $e = d + c$, a node $L = e^2$. Each node knows its inputs and the **local derivative** of its output with respect to each input:

| Operation | Output | Local derivatives |
|---|---|---|
| add | $z = x + y$ | $\partial z/\partial x = 1,\ \partial z/\partial y = 1$ |
| multiply | $z = xy$ | $\partial z/\partial x = y,\ \partial z/\partial y = x$ |
| tanh | $z = \tanh x$ | $1 - z^2$ |
| exp | $z = e^x$ | $z$ |
| power | $z = x^k$ | $k x^{k-1}$ |

## Reverse mode

Forward mode pushes derivatives from one input towards all outputs; it needs one pass per input. **Reverse mode** pulls derivatives from one output (the loss) back to all inputs in a single pass, which is exactly what training needs: one scalar loss, millions of parameters.

Each node stores a `grad`, the derivative of the final output with respect to that node, called its **adjoint**. Set the output's grad to 1, then visit nodes in **reverse topological order** (every node after all the nodes that use it) and push gradient to the inputs:

<!-- formula: Reverse-mode accumulation -->
$$\bar x \mathrel{+}= \bar z\,\frac{\partial z}{\partial x} \quad\text{for every input } x \text{ of every node } z$$

The `+=` matters: when a value is used twice (for example $x\cdot x$), the contributions from both uses add up. This is the **multivariable chain rule**, and forgetting it is the classic autograd bug.

[Backpropagation](#/lesson/neural-networks) in a network is this algorithm applied to a graph whose nodes are whole matrix operations, with local rules such as "the gradient of $XW$ with respect to $W$ is $X^\top\bar Z$".

## In real frameworks

PyTorch records the graph dynamically as Python runs (define-by-run), then `loss.backward()` walks it. JAX traces a function and transforms it (`grad`, `jit`, `vmap`). The engine below is the same algorithm on scalars.

<!-- exercise: value | A scalar autograd engine -->
Complete the `Value` class: `__add__` and `__mul__` (also with plain numbers), `tanh`, and `backward()`. Each operation creates a new `Value` with `_prev` set to its inputs and a `_backward` function that adds the local contribution to the inputs' `grad` (use `+=`). `backward()` must build a topological order of the graph, set `self.grad = 1.0`, and call `_backward` on the nodes in reverse order.

~~~python starter
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        raise NotImplementedError

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

a, b = Value(2.0), Value(-3.0)
L = (a * b + a).tanh()
L.backward()
print(L.data, a.grad, b.grad)
~~~

~~~python solution
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._prev:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()

a, b = Value(2.0), Value(-3.0)
L = (a * b + a).tanh()
L.backward()
print(L.data, a.grad, b.grad)
~~~

~~~python tests
import math

def test_add_mul():
    """d(a*b + c)/da = b, /db = a, /dc = 1"""
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    L = a * b + c
    L.backward()
    assert L.data == 4.0 and (a.grad, b.grad, c.grad) == (-3.0, 2.0, 1.0)

def test_reuse_accumulates():
    """using a value twice adds both contributions: d(x*x)/dx = 2x"""
    x = Value(3.0)
    y = x * x
    y.backward()
    assert x.grad == 6.0

def test_constants():
    """works with plain numbers on either side"""
    x = Value(2.0)
    y = 3 * x + 1
    y.backward()
    assert y.data == 7.0 and x.grad == 3.0

def test_tanh_chain():
    """gradient through tanh matches the analytic derivative"""
    a, b = Value(0.5), Value(-1.2)
    L = (a * b + a).tanh()
    L.backward()
    z = 0.5 * -1.2 + 0.5
    dz = 1 - math.tanh(z) ** 2
    assert approx(a.grad, dz * (-1.2 + 1)) and approx(b.grad, dz * 0.5)

def test_neuron():
    """a 3-input neuron: gradients match finite differences"""
    xs, ws = [1.0, -2.0, 0.5], [0.3, 0.8, -1.1]
    def f(w):
        return math.tanh(sum(wi * xi for wi, xi in zip(w, xs)) + 0.2)
    W = [Value(w) for w in ws]
    out = (W[0] * xs[0] + W[1] * xs[1] + W[2] * xs[2] + 0.2).tanh()
    out.backward()
    for i in range(3):
        up = ws.copy(); up[i] += 1e-6
        dn = ws.copy(); dn[i] -= 1e-6
        assert approx(W[i].grad, (f(up) - f(dn)) / 2e-6, tol=1e-5)
~~~

<!-- hint: In `__mul__`, the closure `_backward` should do `self.grad += other.data * out.grad` and the symmetric line for `other`. -->
<!-- hint: For the topological order, do a depth-first search that appends a node after all its inputs; then walk the list backwards. -->
<!-- /exercise -->

<!-- exercise: more-ops | Add exp, log and power -->
The engine below already supports `+`, `*` and `backward`. Add `exp()`, `log()` and `__pow__(k)` for a number `k`, each with the correct local derivative. The tests use them to differentiate softplus $\log(1 + e^x)$, whose derivative is the sigmoid.

~~~python starter
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data, self.grad, self._prev = float(data), 0.0, tuple(_prev)
        self._backward = lambda: None

    def _wrap(self, o):
        return o if isinstance(o, Value) else Value(o)

    def __add__(self, o):
        o = self._wrap(o); out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out

    def __mul__(self, o):
        o = self._wrap(o); out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out

    __radd__, __rmul__ = __add__, __mul__

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def __pow__(self, k):
        raise NotImplementedError

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v)); [visit(p) for p in v._prev]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()
~~~

~~~python solution
import math

class Value:
    def __init__(self, data, _prev=()):
        self.data, self.grad, self._prev = float(data), 0.0, tuple(_prev)
        self._backward = lambda: None

    def _wrap(self, o):
        return o if isinstance(o, Value) else Value(o)

    def __add__(self, o):
        o = self._wrap(o); out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out

    def __mul__(self, o):
        o = self._wrap(o); out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out

    __radd__, __rmul__ = __add__, __mul__

    def exp(self):
        out = Value(math.exp(self.data), (self,))
        def bw(): self.grad += out.data * out.grad
        out._backward = bw; return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def bw(): self.grad += out.grad / self.data
        out._backward = bw; return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out

    def backward(self):
        order, seen = [], set()
        def visit(v):
            if id(v) not in seen:
                seen.add(id(v)); [visit(p) for p in v._prev]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()
~~~

~~~python tests
import math

def test_exp():
    """d exp(x)/dx = exp(x)"""
    x = Value(0.7); y = x.exp(); y.backward()
    assert approx(y.data, math.exp(0.7)) and approx(x.grad, math.exp(0.7))

def test_log():
    """d log(x)/dx = 1/x"""
    x = Value(2.5); y = x.log(); y.backward()
    assert approx(x.grad, 0.4)

def test_pow():
    """d x^3/dx = 3x^2 and d x^-1/dx = -1/x^2"""
    x = Value(2.0); (x ** 3).backward()
    assert approx(x.grad, 12.0)
    z = Value(4.0); (z ** -1).backward()
    assert approx(z.grad, -1 / 16)

def test_softplus_derivative_is_sigmoid():
    """the derivative of log(1 + e^x) is the sigmoid"""
    for v in (-2.0, 0.0, 1.5):
        x = Value(v)
        (1 + x.exp()).log().backward()
        assert approx(x.grad, 1 / (1 + math.exp(-v)))
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why does training use reverse-mode rather than forward-mode differentiation?
   - [x] There is one scalar loss and many parameters; reverse mode gets all gradients in one backward pass
   - [ ] Forward mode is less accurate
   - [ ] Reverse mode does not need the chain rule
   > Forward mode needs one pass per input variable; reverse mode needs one pass per output.
2. A value `x` is used by two different operations. How is its gradient computed?
   - [ ] From the last operation that used it
   - [x] The contributions from both uses are added
   - [ ] From the first operation only
   > The multivariable chain rule sums over every path from `x` to the output, hence `+=`.
3. Why must nodes be processed in reverse topological order?
   - [ ] To save memory
   - [x] A node's gradient is complete only after every node that uses it has pushed its contribution
   - [ ] Any order gives the same result
   > Calling a node's backward too early would pass on a partial gradient.
4. What is the local derivative used for $z = \tanh(x)$?
   - [ ] $\tanh(x)$
   - [x] $1 - z^2$
   - [ ] $1/\cosh(x)$
   > Expressing it through the output $z$, which is already stored, avoids recomputing.
<!-- /quiz -->
