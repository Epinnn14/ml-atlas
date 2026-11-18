---
title: Decision trees
level: intermediate
minutes: 45
summary: A flowchart learned from data. Measure impurity with Gini or entropy, choose the split that reduces it most, grow the tree recursively, and stop it before it memorises the training set.
---

## A model you can read

A decision tree asks a sequence of yes/no questions about one feature at a time, *is income below 5 million?*, *is age above 40?*, until it reaches a **leaf** that gives the prediction. Trees need no feature scaling, handle mixed feature types, capture interactions, and can be drawn and explained. Alone they overfit easily; as building blocks of [random forests](#/lesson/random-forests) and [boosting](#/lesson/boosting) they are among the most successful models for tabular data.

## Impurity

A good question splits the examples into groups that are each as **pure** as possible, mostly one class. Two common measures of impurity for a node where class $c$ has proportion $p_c$:

<!-- formula: Gini impurity -->
$$G = 1 - \sum_c p_c^2$$

<!-- formula: Entropy impurity -->
$$H = -\sum_c p_c\log_2 p_c$$

Both are 0 for a pure node and largest when classes are evenly mixed (for two classes, Gini 0.5 and entropy 1 bit). In practice they choose almost the same splits; Gini is a little cheaper.

## Choosing a split

A split sends the $n$ examples of a node into a left child ($n_L$) and a right child ($n_R$). Its quality is the drop in impurity, weighting each child by its size:

<!-- formula: Impurity decrease of a split -->
$$\Delta = I(\text{parent}) - \frac{n_L}{n}I(\text{left}) - \frac{n_R}{n}I(\text{right})$$

With entropy, $\Delta$ is the **information gain**, the [mutual information](#/lesson/information-theory) between the split and the label. For a numeric feature, the candidate thresholds are the midpoints between consecutive distinct sorted values. The tree tries every feature and every threshold and keeps the best: a **greedy** search, one level at a time.

For regression trees, impurity is the variance of the targets in the node and the leaf predicts their mean.

## Growing and stopping

Build recursively: split the root, then split each child the same way. Stop when a node is pure, or a limit is reached. Left unchecked, a tree keeps splitting until every leaf holds one example: zero training error and severe overfitting. Common limits:

- **max depth**: the number of questions on any path;
- **min samples per leaf**: no leaf smaller than, say, 5 examples;
- **min impurity decrease**: do not split unless it helps enough.

Alternatively grow a large tree and **prune** it back, removing splits that do not improve validation error. Trees are also unstable: a small change in the data can change the first split and the whole tree below it, which is the high variance that ensembles average away.

<!-- exercise: best-split | Gini and the best split -->
Write `gini(y)` for an array of integer labels, and `best_split(x, y)` for one numeric feature `x`: try every midpoint between consecutive distinct sorted values of `x` as a threshold (left: `x <= t`), and return `(threshold, decrease)` for the split with the largest Gini decrease. If `x` has a single distinct value, return `(None, 0.0)`.

~~~python starter
import numpy as np

def gini(y):
    raise NotImplementedError

def best_split(x, y):
    x, y = np.asarray(x, float), np.asarray(y)
    raise NotImplementedError

x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
y = np.array([0, 0, 0, 1, 1, 1])
print(best_split(x, y))   # (6.5, 0.5)
~~~

~~~python solution
import numpy as np

def gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return float(1 - np.sum(p ** 2))

def best_split(x, y):
    x, y = np.asarray(x, float), np.asarray(y)
    values = np.unique(x)
    if len(values) < 2:
        return None, 0.0
    parent, n = gini(y), len(y)
    best_t, best_d = None, -1.0
    for t in (values[:-1] + values[1:]) / 2:
        left = x <= t
        d = parent - left.sum() / n * gini(y[left]) - (~left).sum() / n * gini(y[~left])
        if d > best_d:
            best_t, best_d = float(t), d
    return best_t, float(best_d)

x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
y = np.array([0, 0, 0, 1, 1, 1])
print(best_split(x, y))
~~~

~~~python tests
import numpy as np

def test_gini_values():
    """pure node 0, 50/50 node 0.5, three equal classes 2/3"""
    assert approx(gini(np.array([1, 1, 1])), 0.0)
    assert approx(gini(np.array([0, 1, 0, 1])), 0.5)
    assert approx(gini(np.array([0, 1, 2])), 2 / 3)

def test_perfect_split():
    """separable data splits at the gap with decrease 0.5"""
    t, d = best_split(np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0]), np.array([0, 0, 0, 1, 1, 1]))
    assert approx(t, 6.5) and approx(d, 0.5)

def test_imperfect():
    """chooses the threshold with the largest decrease"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); y = np.array([0, 0, 1, 0, 1, 1])
    t, d = best_split(x, y)
    best = max(((tt, 0.5 - (x <= tt).mean() * gini(y[x <= tt]) - (x > tt).mean() * gini(y[x > tt])) for tt in [1.5, 2.5, 3.5, 4.5, 5.5]), key=lambda p: p[1])
    assert approx(t, best[0]) and approx(d, best[1])

def test_constant_feature():
    """a constant feature cannot be split"""
    assert best_split(np.ones(5), np.array([0, 1, 0, 1, 1])) == (None, 0.0)
~~~

<!-- hint: `np.unique(x)` is sorted; the candidate thresholds are `(values[:-1] + values[1:]) / 2`. -->
<!-- /exercise -->

<!-- exercise: tree | Grow a small classification tree -->
Write `build_tree(X, y, depth=0, max_depth=3)` returning nested dictionaries. A leaf is `{"leaf": label}` with the majority label (`np.bincount(y).argmax()`). Make a leaf when the node is pure, `depth == max_depth`, or no feature can be split. Otherwise choose the feature and threshold with the largest Gini decrease over all features (using the provided `best_split`; on ties keep the lower feature index), and return `{"feature": j, "threshold": t, "left": ..., "right": ...}` with left for `x[j] <= t`. Then write `predict_one(tree, x)`.

~~~python starter
import numpy as np

def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def best_split(x, y):
    values = np.unique(x)
    if len(values) < 2:
        return None, 0.0
    best = (None, -1.0)
    for t in (values[:-1] + values[1:]) / 2:
        l = x <= t
        d = gini(y) - l.mean() * gini(y[l]) - (~l).mean() * gini(y[~l])
        if d > best[1]:
            best = (float(t), float(d))
    return best

def build_tree(X, y, depth=0, max_depth=3):
    raise NotImplementedError

def predict_one(tree, x):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def best_split(x, y):
    values = np.unique(x)
    if len(values) < 2:
        return None, 0.0
    best = (None, -1.0)
    for t in (values[:-1] + values[1:]) / 2:
        l = x <= t
        d = gini(y) - l.mean() * gini(y[l]) - (~l).mean() * gini(y[~l])
        if d > best[1]:
            best = (float(t), float(d))
    return best

def build_tree(X, y, depth=0, max_depth=3):
    leaf = {"leaf": int(np.bincount(y).argmax())}
    if depth == max_depth or len(np.unique(y)) == 1:
        return leaf
    best_j, best_t, best_d = None, None, 0.0
    for j in range(X.shape[1]):
        t, d = best_split(X[:, j], y)
        if t is not None and d > best_d:
            best_j, best_t, best_d = j, t, d
    if best_j is None:
        return leaf
    left = X[:, best_j] <= best_t
    return {"feature": best_j, "threshold": best_t,
            "left": build_tree(X[left], y[left], depth + 1, max_depth),
            "right": build_tree(X[~left], y[~left], depth + 1, max_depth)}

def predict_one(tree, x):
    while "leaf" not in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["leaf"]
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(300, 2))
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)        # XOR: needs depth 2

def depth(t):
    return 0 if "leaf" in t else 1 + max(depth(t["left"]), depth(t["right"]))

def test_xor():
    """a depth-3 tree learns XOR, which no single split can"""
    t = build_tree(X, y, max_depth=3)
    acc = np.mean([predict_one(t, x) == yy for x, yy in zip(X, y)])
    assert acc > 0.95

def test_depth_limit():
    """never deeper than max_depth"""
    assert depth(build_tree(X, y, max_depth=2)) <= 2 and depth(build_tree(X, y, max_depth=0)) == 0

def test_depth_zero_is_majority():
    """max_depth = 0 gives a single leaf with the majority label"""
    assert build_tree(X, np.array([1] * 200 + [0] * 100), max_depth=0) == {"leaf": 1}

def test_pure_node_stops():
    """a pure dataset gives one leaf"""
    assert build_tree(X, np.zeros(300, int), max_depth=5) == {"leaf": 0}

def test_root_split():
    """on a one-feature step function the root splits at the step"""
    Xs = np.array([[0.0], [1.0], [2.0], [5.0], [6.0]]); ys = np.array([0, 0, 0, 1, 1])
    t = build_tree(Xs, ys, max_depth=1)
    assert t["feature"] == 0 and approx(t["threshold"], 3.5)
    assert t["left"] == {"leaf": 0} and t["right"] == {"leaf": 1}
~~~

<!-- hint: Use strict `d > best_d` while looping over features in order, which keeps the lower index on ties. -->
<!-- hint: `predict_one` walks down until it finds a dictionary with the key `"leaf"`. -->
<!-- /exercise -->

<!-- quiz -->
1. A node holds 8 examples of class A and 2 of class B. What is its Gini impurity?
   - [ ] 0.2
   - [x] 0.32
   - [ ] 0.5
   > $1 - (0.8^2 + 0.2^2) = 1 - 0.68 = 0.32$.
2. Why do trees not need standardised features?
   - [x] Splits compare one feature with a threshold, and rescaling a feature moves the threshold without changing which examples go left
   - [ ] Trees standardise internally
   - [ ] Trees only work with categorical features
   > Any monotonic transformation of a feature leaves the possible partitions unchanged.
3. A fully grown tree has 100% training accuracy and 70% test accuracy. What would most likely help?
   - [ ] Growing it deeper
   - [x] Limiting depth or leaf size, or averaging many trees in a forest
   - [ ] Standardising the features
   > A fully grown tree memorises noise (high variance). Constraints and ensembles reduce variance.
4. Why is tree building called greedy?
   - [ ] It uses all the features at once
   - [x] It picks the best split at each node without looking ahead at how later splits would turn out
   - [ ] It always splits on the most frequent class
   > Finding the optimal tree is computationally intractable, so each split is chosen locally. XOR shows the risk: no single first split looks useful.
<!-- /quiz -->
