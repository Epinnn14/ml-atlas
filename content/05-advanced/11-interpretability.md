---
title: Interpreting models
level: advanced
minutes: 45
summary: Why did the model say that? Permutation importance for global answers, partial dependence for the shape of an effect, and Shapley values for explaining a single prediction, with the traps that make explanations misleading.
---

## Global and local questions

- **Global**: which features matter to this model overall? How does the prediction change as one feature varies?
- **Local**: why did the model give *this* applicant *this* score?

Linear models and small trees can be read directly. For [forests](#/lesson/random-forests), [boosting](#/lesson/boosting) and neural networks we need model-agnostic tools that only query the model.

## Permutation importance

Measure the model's score on validation data. Then shuffle one feature's column, breaking its link with the target while keeping its distribution, and measure again. The drop is that feature's importance:

<!-- formula: Permutation importance -->
$$I_j = \text{score}(X) - \frac1R\sum_{r=1}^{R}\text{score}\big(X^{(j\text{ shuffled}, r)}\big)$$

Repeat the shuffle a few times and average. Use held-out data: importance on training data rewards features the model overfitted. Caveat: if two features are strongly correlated, shuffling one still leaves the other, so both can look unimportant even though together they matter; and shuffling creates unrealistic combinations (a 5-year-old with a mortgage).

## Partial dependence

To see the shape of a feature's effect, fix feature $j$ to a value $v$ for every row, average the predictions, and repeat over a grid of values:

<!-- formula: Partial dependence -->
$$\text{PD}_j(v) = \frac1n\sum_{i=1}^{n}f\big(v,\ \mathbf x_{i,-j}\big)$$

Plotting $\text{PD}_j$ shows whether the effect is increasing, saturating or U-shaped. Individual conditional expectation (ICE) curves show the same per row and reveal interactions that the average hides.

## Shapley values

For a single prediction, how much did each feature contribute? Cooperative game theory gives a unique answer satisfying sensible fairness rules. Treat the features as players and the prediction as the payout. Feature $j$'s **Shapley value** is its average marginal contribution over all orders in which features could be added:

<!-- formula: Shapley value -->
$$\phi_j = \sum_{S\subseteq F\setminus\{j\}}\frac{|S|!\,(|F| - |S| - 1)!}{|F|!}\Big[v(S\cup\{j\}) - v(S)\Big]$$

where $v(S)$ is the model's expected prediction when only the features in $S$ are known (the others are filled in from background data). The values add up exactly: $\sum_j\phi_j = f(\mathbf x) - \mathbb E[f(X)]$, the difference between this prediction and the average one (**efficiency**). For a linear model with independent features, $\phi_j = w_j(x_j - \bar x_j)$.

Exact computation needs $2^{|F|}$ evaluations of $v$. **SHAP** approximates it efficiently: KernelSHAP by weighted regression on sampled subsets, TreeSHAP exactly and fast for tree ensembles.

## Pitfalls

- **Explaining the model, not the world.** A feature can be important to the model because it leaks the label or because it is correlated with the true cause. Importance is not causation ([causal inference](#/lesson/causal-inference)).
- **Correlated features** split or swap credit unpredictably.
- **Saliency maps** for images can look convincing while being insensitive to the model's actual weights. Test explanations with sanity checks.
- An explanation is a simplification. Present it as such, especially when it is used to justify decisions about people.

<!-- exercise: perm-importance | Permutation importance -->
Write `permutation_importance(predict, X, y, score, n_repeats=5, seed=0)` returning an array with one importance per feature: the baseline score minus the mean score after shuffling that column. Use `rng = np.random.default_rng(seed)` and, for each feature in order and each repeat, shuffle a **copy** of `X` with `X_perm[:, j] = rng.permutation(X[:, j])`.

~~~python starter
import numpy as np

def permutation_importance(predict, X, y, score, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def permutation_importance(predict, X, y, score, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = score(y, predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(X[:, j])
            drops.append(base - score(y, predict(Xp)))
        imp[j] = np.mean(drops)
    return imp
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 3))
y = 3 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * rng.normal(size=500)    # feature 2 is useless
predict = lambda Z: 3 * Z[:, 0] + 0.5 * Z[:, 1]
neg_mse = lambda yt, yp: -np.mean((yt - yp) ** 2)

def test_ranking():
    """the strong feature matters most and the useless one not at all"""
    imp = permutation_importance(predict, X, y, neg_mse)
    assert imp[0] > imp[1] > 0 and abs(imp[2]) < 1e-12

def test_size():
    """shuffling feature 0 costs about 2 * 9 * var(x0) in MSE"""
    imp = permutation_importance(predict, X, y, neg_mse, n_repeats=20)
    assert abs(imp[0] - 18.0) < 2.5

def test_input_untouched():
    """X is not modified"""
    X0 = X.copy()
    permutation_importance(predict, X, y, neg_mse)
    assert np.array_equal(X, X0)
~~~

<!-- /exercise -->

<!-- exercise: shapley | Exact Shapley values -->
Write `shapley_values(value, n_features)` where `value(S)` returns the payout of a coalition `S` (a frozenset of feature indices). Sum over all subsets of the other features with the Shapley weights. Then the tests check it on a linear model, where $v(S)$ fills unknown features with their background mean.

~~~python starter
import itertools, math
import numpy as np

def shapley_values(value, n_features):
    raise NotImplementedError
~~~

~~~python solution
import itertools, math
import numpy as np

def shapley_values(value, n_features):
    F = range(n_features)
    phi = np.zeros(n_features)
    for j in F:
        others = [k for k in F if k != j]
        for size in range(len(others) + 1):
            w = math.factorial(size) * math.factorial(n_features - size - 1) / math.factorial(n_features)
            for S in itertools.combinations(others, size):
                S = frozenset(S)
                phi[j] += w * (value(S | {j}) - value(S))
    return phi
~~~

~~~python tests
import numpy as np

w = np.array([2.0, -1.0, 0.5, 3.0])
x = np.array([1.0, 2.0, -1.0, 0.5])
mean = np.array([0.0, 1.0, 1.0, 0.0])

def value(S):
    z = mean.copy()
    for k in S:
        z[k] = x[k]
    return float(w @ z)

def test_linear_model():
    """for a linear model, phi_j = w_j (x_j - mean_j)"""
    assert approx(shapley_values(value, 4), w * (x - mean))

def test_efficiency():
    """the values add up to f(x) - f(mean)"""
    phi = shapley_values(value, 4)
    assert approx(phi.sum(), value(frozenset(range(4))) - value(frozenset()))

def test_interaction_split_fairly():
    """for v = 1 only when both features are present, each gets 1/2"""
    v = lambda S: 1.0 if S >= {0, 1} else 0.0
    assert approx(shapley_values(v, 2), [0.5, 0.5])

def test_dummy_feature():
    """a feature that never changes the payout gets 0"""
    v = lambda S: float(len(S & {0, 1}))
    assert approx(shapley_values(v, 3), [1.0, 1.0, 0.0])
~~~

<!-- hint: For each feature and each subset size, the weight is `size! (n - size - 1)! / n!`; loop over `itertools.combinations` of the other features. -->
<!-- /exercise -->

<!-- quiz -->
1. Why compute permutation importance on validation data rather than training data?
   - [x] On training data it rewards features the model overfitted, which may not help on new data
   - [ ] Training data cannot be shuffled
   - [ ] Validation data has more features
   > Importance should reflect what helps generalisation.
2. Two features are near copies of each other. What does permutation importance tend to show?
   - [ ] Both very important
   - [x] Both less important than they really are, since the model can lean on the unshuffled copy
   - [ ] An error
   > Shuffling one leaves the information available through the other.
3. What does the efficiency property of Shapley values guarantee?
   - [x] The contributions add up exactly to the prediction minus the average prediction
   - [ ] Every feature gets a positive value
   - [ ] The computation is fast
   > $\sum_j\phi_j = f(\mathbf x) - \mathbb E[f]$, so the explanation accounts for the whole difference.
4. A model relies heavily on "number of hospital visits" to predict illness. Does that mean visits cause illness?
   - [ ] Yes, high importance proves causation
   - [x] No, importance describes the model's use of a feature, which may be correlated with or caused by the outcome
   - [ ] Only if the Shapley value is positive
   > Interpretability tools explain the model, not the causal structure of the world.
<!-- /quiz -->
