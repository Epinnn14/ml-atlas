---
title: Multi-armed bandits
level: advanced
minutes: 45
summary: Learning while deciding. The explore–exploit dilemma, regret, epsilon-greedy, the optimism of UCB, and Thompson sampling, the simplest setting of reinforcement learning and the engine behind adaptive experiments.
---

## The dilemma

A slot machine with several arms pays out with unknown probabilities. Each pull, you choose an arm and see only that arm's reward. Pull the arm that looks best so far (**exploit**) and you may never discover a better one; try other arms (**explore**) and you give up reward now. The same structure appears in choosing which headline, price, ad or recommendation to show, and in clinical trials.

Formally, arm $a$ has an unknown mean reward $\mu_a$; the best is $\mu^* = \max_a\mu_a$. A strategy is judged by its **regret** after $T$ rounds, the reward lost compared with always pulling the best arm:

<!-- formula: Regret -->
$$R_T = T\mu^* - \mathbb E\Big[\sum_{t=1}^{T} r_t\Big]$$

Always exploiting or always exploring both give regret growing linearly in $T$. Good strategies achieve **logarithmic** regret: the cost of learning grows ever more slowly.

## ε-greedy

With probability $\varepsilon$ pick a random arm, otherwise the arm with the highest average reward so far. Simple and robust, but it keeps exploring obviously bad arms at the same rate forever. Decaying $\varepsilon$ over time helps.

## UCB: optimism under uncertainty

Give each arm a bonus that is large when it has been tried rarely, and pick the arm with the highest **upper confidence bound**:

<!-- formula: UCB1 -->
$$a_t = \arg\max_a\ \Big[\hat\mu_a + \sqrt{\frac{2\ln t}{n_a}}\Big]$$

$\hat\mu_a$ is the arm's average reward and $n_a$ how often it was pulled. An arm is chosen either because it looks good or because we know little about it. As $n_a$ grows the bonus shrinks; as $t$ grows slowly, rarely tried arms are revisited occasionally. UCB1 achieves logarithmic regret with no tuning. Pull each arm once at the start so $n_a > 0$.

## Thompson sampling

Keep a [Bayesian posterior](#/lesson/statistics-estimation) over each arm's mean, draw one sample from each posterior, and pull the arm whose sample is highest. For 0/1 rewards with a Beta(1, 1) prior, the posterior after $s$ successes and $f$ failures is Beta$(1 + s, 1 + f)$:

<!-- formula: Thompson sampling for Bernoulli arms -->
$$\theta_a \sim \text{Beta}(1 + s_a,\ 1 + f_a), \qquad a_t = \arg\max_a\ \theta_a$$

Each arm is chosen with the probability that it is the best, given the evidence. It explores uncertain arms naturally, matches UCB's guarantees, often does better in practice, and extends easily to delays and batches.

## Contextual bandits and beyond

When each round comes with context (the user, the time), a **contextual bandit** learns a model of reward given context and action, for example a [Bayesian linear model](#/lesson/bayesian-regression) per arm with Thompson sampling (LinTS) or a confidence bound (LinUCB). Bandits differ from full [reinforcement learning](#/lesson/q-learning) in one way: actions do not change future states. Compared with a fixed A/B test, a bandit wastes less traffic on losing variants, at the cost of messier statistical inference ([causal inference](#/lesson/causal-inference)).

<!-- exercise: ucb | Choose an arm with UCB1 -->
Write `ucb1_choose(counts, sums, t)` returning the arm to pull at round `t` (1-based total number of pulls so far plus one). Any arm with `count == 0` is chosen first (the lowest such index). Otherwise return the arm with the largest $\hat\mu_a + \sqrt{2\ln t / n_a}$; ties go to the lowest index.

~~~python starter
import numpy as np

def ucb1_choose(counts, sums, t):
    counts, sums = np.asarray(counts, float), np.asarray(sums, float)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ucb1_choose(counts, sums, t):
    counts, sums = np.asarray(counts, float), np.asarray(sums, float)
    untried = np.flatnonzero(counts == 0)
    if len(untried):
        return int(untried[0])
    scores = sums / counts + np.sqrt(2 * np.log(t) / counts)
    return int(np.argmax(scores))
~~~

~~~python tests
import numpy as np

def test_untried_first():
    """an arm never pulled is chosen first"""
    assert ucb1_choose([3, 0, 2], [1.0, 0.0, 2.0], 6) == 1

def test_bonus():
    """a rarely pulled arm with a lower mean can win on its bonus"""
    # arm 0: mean 0.6 over 100 pulls; arm 1: mean 0.5 over 2 pulls
    assert ucb1_choose([100, 2], [60.0, 1.0], 103) == 1

def test_exploit():
    """with equal counts the higher mean wins"""
    assert ucb1_choose([50, 50], [20.0, 30.0], 101) == 1

def test_low_regret():
    """over 3000 rounds most pulls go to the best arm"""
    rng = np.random.default_rng(0)
    p = np.array([0.3, 0.5, 0.65])
    counts, sums = np.zeros(3), np.zeros(3)
    for t in range(1, 3001):
        a = ucb1_choose(counts, sums, t)
        counts[a] += 1; sums[a] += rng.uniform() < p[a]
    assert counts[2] > 0.7 * 3000
~~~

<!-- /exercise -->

<!-- exercise: thompson | Thompson sampling -->
Write `thompson(p_true, T, seed=0)` that simulates Thompson sampling for Bernoulli arms with true success probabilities `p_true`. With `rng = np.random.default_rng(seed)`, each round draw `theta = rng.beta(1 + s, 1 + f)` (vectors over arms), pull `argmax(theta)`, then draw the reward as `rng.uniform() < p_true[a]` and update `s` or `f`. Return the array of pull counts per arm.

~~~python starter
import numpy as np

def thompson(p_true, T, seed=0):
    p_true = np.asarray(p_true, float)
    rng = np.random.default_rng(seed)
    s, f = np.zeros(len(p_true)), np.zeros(len(p_true))
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def thompson(p_true, T, seed=0):
    p_true = np.asarray(p_true, float)
    rng = np.random.default_rng(seed)
    s, f = np.zeros(len(p_true)), np.zeros(len(p_true))
    for _ in range(T):
        a = int(np.argmax(rng.beta(1 + s, 1 + f)))
        if rng.uniform() < p_true[a]:
            s[a] += 1
        else:
            f[a] += 1
    return s + f
~~~

~~~python tests
import numpy as np

def test_counts_sum():
    """returns pull counts that add up to T"""
    c = thompson([0.2, 0.4], 500)
    assert c.shape == (2,) and c.sum() == 500

def test_finds_best():
    """the best arm gets most of the pulls"""
    c = thompson([0.10, 0.12, 0.30, 0.15], 4000, seed=1)
    assert np.argmax(c) == 2 and c[2] > 0.75 * 4000

def test_recipe():
    """follows the sampling order: beta draws, then the uniform reward draw"""
    rng = np.random.default_rng(5)
    a = int(np.argmax(rng.beta(np.ones(3), np.ones(3))))
    c = thompson([0.5, 0.5, 0.5], 1, seed=5)
    assert c[a] == 1 and c.sum() == 1

def test_equal_arms_shared():
    """identical arms are all tried"""
    assert np.all(thompson([0.5, 0.5, 0.5], 900, seed=2) > 50)
~~~

<!-- hint: `rng.beta(1 + s, 1 + f)` draws one sample per arm at once when `s` and `f` are arrays. -->
<!-- /exercise -->

<!-- quiz -->
1. What does regret measure?
   - [ ] The number of times the worst arm was pulled
   - [x] The reward lost compared with always pulling the best arm
   - [ ] The variance of the rewards
   > Regret captures the total price of not knowing the best arm in advance.
2. Why does UCB1 sometimes pick an arm whose average reward is not the highest?
   - [x] Its confidence bonus is large because it has been tried only a few times
   - [ ] It picks randomly with probability epsilon
   - [ ] Because of a bug in the formula
   > Optimism under uncertainty: an arm stays attractive until enough evidence shows it is worse.
3. In Thompson sampling, with what probability is each arm chosen?
   - [ ] Uniformly at random
   - [x] With the posterior probability that it is the best arm
   - [ ] Proportional to its average reward
   > Sampling one value per posterior and taking the maximum does exactly that.
4. How does a bandit differ from a full reinforcement learning problem?
   - [x] In a bandit, actions do not change the future state of the environment
   - [ ] Bandits have no rewards
   - [ ] Bandits cannot explore
   > RL must account for how an action affects future situations; a bandit faces the same choice every round.
<!-- /quiz -->
