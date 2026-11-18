---
title: MDPs, value iteration and Q-learning
level: senior
minutes: 55
summary: Reinforcement learning where actions shape the future. Markov decision processes, the Bellman equations, solving a known MDP by value iteration, and learning from experience alone with Q-learning.
---

## Markov decision processes

An agent observes a **state** $s$, takes an **action** $a$, receives a **reward** $r$ and lands in a new state $s'$. A **Markov decision process** (MDP) describes this with transition probabilities $P(s'\mid s, a)$, a reward function $R(s, a)$ and a **discount factor** $\gamma\in[0,1)$. "Markov" means the next state depends only on the current state and action, not on the history.

The agent wants a **policy** $\pi(a\mid s)$ that maximises the expected **return**, the discounted sum of future rewards:

<!-- formula: Discounted return -->
$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1}$$

$\gamma$ near 0 makes the agent short-sighted; near 1, far-sighted. Discounting also keeps infinite sums finite.

## Value functions and the Bellman equations

The **action value** $Q^\pi(s, a)$ is the expected return from taking $a$ in $s$ and following $\pi$ afterwards. The optimal values satisfy the **Bellman optimality equation**: the value of acting now equals the immediate reward plus the discounted value of acting optimally from wherever you land:

<!-- formula: Bellman optimality equation -->
$$Q^*(s, a) = R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)\,\max_{a'}Q^*(s', a')$$

and $V^*(s) = \max_aQ^*(s, a)$. Once $Q^*$ is known, the optimal policy is greedy: in each state, take the action with the highest $Q^*$.

## Value iteration: planning with a known model

If $P$ and $R$ are known, turn the Bellman equation into an update and repeat until nothing changes:

<!-- formula: Value iteration -->
$$V_{k+1}(s) = \max_a\Big[R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)\,V_k(s')\Big]$$

The update is a **contraction**: each sweep shrinks the error by a factor $\gamma$, so it converges to $V^*$ from any start. This is dynamic programming.

## Q-learning: learning from experience

Usually the model is unknown: you can only act and observe. **Q-learning** keeps a table $Q(s, a)$ and, after each observed transition $(s, a, r, s')$, moves the estimate towards a sampled Bellman target:

<!-- formula: Q-learning update -->
$$Q(s, a) \leftarrow Q(s, a) + \alpha\Big[r + \gamma\max_{a'}Q(s', a') - Q(s, a)\Big]$$

The bracket is the **temporal-difference error**: the surprise between what was predicted and what one step of experience suggests. At terminal states the target is just $r$. Q-learning is **off-policy**: it learns the value of the greedy policy while behaving with an exploratory one, usually ε-greedy ([bandits](#/lesson/bandits)). With every state–action pair visited infinitely often and a suitably decaying $\alpha$, it converges to $Q^*$.

## Deep Q-networks

With images or huge state spaces a table is impossible, so a neural network approximates $Q(s, a)$. DQN (2015, Atari games from pixels) made this stable with an **experience replay** buffer (train on random past transitions, breaking correlations) and a slowly updated **target network** for the $\max_{a'}$ term. The overestimation caused by maximising over noisy estimates is reduced by Double DQN. For continuous actions and stochastic policies, [policy gradient methods](#/lesson/policy-gradients) are the alternative.

<!-- exercise: value-iteration | Value iteration -->
Write `value_iteration(P, R, gamma, tol=1e-10)` where `P` has shape `(S, A, S)` (transition probabilities) and `R` shape `(S, A)`. Start from `V = 0` and apply the update until the largest change is below `tol`. Return `(V, policy)` where `policy[s]` is the greedy action under the final values (lowest index on ties).

~~~python starter
import numpy as np

def value_iteration(P, R, gamma, tol=1e-10):
    S, A, _ = P.shape
    V = np.zeros(S)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def value_iteration(P, R, gamma, tol=1e-10):
    S, A, _ = P.shape
    V = np.zeros(S)
    while True:
        Q = R + gamma * P @ V              # (S, A)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * P @ V
    return V, Q.argmax(axis=1)
~~~

~~~python tests
import numpy as np

# a 5-state corridor: action 0 = left, 1 = right; reward 1 for stepping right out of the last state (which then stays)
S, A = 5, 2
P = np.zeros((S, A, S)); R = np.zeros((S, A))
for s in range(S):
    P[s, 0, max(s - 1, 0)] = 1.0
    P[s, 1, min(s + 1, S - 1)] = 1.0
R[S - 1, 1] = 1.0

def test_policy_goes_right():
    """the optimal policy walks right everywhere"""
    _, pi = value_iteration(P, R, 0.9)
    assert list(pi) == [1, 1, 1, 1, 1]

def test_values():
    """the last state earns 1 forever: V = 1 / (1 - gamma); earlier states are discounted"""
    V, _ = value_iteration(P, R, 0.9)
    assert approx(V[4], 10.0, tol=1e-6) and approx(V[0], 0.9 ** 4 * 10.0, tol=1e-6)

def test_bellman_fixed_point():
    """V satisfies the Bellman optimality equation"""
    rng = np.random.default_rng(0)
    P2 = rng.uniform(size=(4, 3, 4)); P2 /= P2.sum(-1, keepdims=True)
    R2 = rng.normal(size=(4, 3))
    V, _ = value_iteration(P2, R2, 0.8)
    assert approx(V, (R2 + 0.8 * P2 @ V).max(axis=1), tol=1e-8)
~~~

<!-- hint: `P @ V` has shape `(S, A)`: the expected next-state value for every state and action. -->
<!-- /exercise -->

<!-- exercise: q-learning | Q-learning on the corridor -->
Write `q_update(Q, s, a, r, s_next, done, alpha, gamma)` that applies the Q-learning update in place (target `r` alone when `done`). Then `train_corridor(episodes, seed=0)` learns the corridor below: states 0..4, start at 0, action 1 moves right and 0 left (walls keep you in place), reaching state 4 gives reward 1 and ends the episode (at most 50 steps per episode). Use ε-greedy with `eps = 0.2`: with `rng = np.random.default_rng(seed)`, if `rng.uniform() < eps` take `rng.integers(2)`, otherwise `argmax(Q[s])`. Use `alpha = 0.5`, `gamma = 0.9`. Return `Q`.

~~~python starter
import numpy as np

def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    raise NotImplementedError

def train_corridor(episodes, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((5, 2))
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def q_update(Q, s, a, r, s_next, done, alpha, gamma):
    target = r if done else r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (target - Q[s, a])

def train_corridor(episodes, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((5, 2))
    for _ in range(episodes):
        s = 0
        for _ in range(50):
            a = int(rng.integers(2)) if rng.uniform() < 0.2 else int(np.argmax(Q[s]))
            s_next = min(s + 1, 4) if a == 1 else max(s - 1, 0)
            done = s_next == 4
            r = 1.0 if done else 0.0
            q_update(Q, s, a, r, s_next, done, 0.5, 0.9)
            s = s_next
            if done:
                break
    return Q
~~~

~~~python tests
import numpy as np

def test_update_rule():
    """a single update moves Q towards r + gamma max Q(s')"""
    Q = np.array([[0.0, 0.0], [2.0, 4.0]])
    q_update(Q, 0, 1, 1.0, 1, False, 0.5, 0.9)
    assert approx(Q[0, 1], 0.5 * (1.0 + 0.9 * 4.0))

def test_terminal():
    """at a terminal transition the target is just r"""
    Q = np.array([[0.0, 0.0], [5.0, 5.0]])
    q_update(Q, 0, 0, 1.0, 1, True, 1.0, 0.9)
    assert approx(Q[0, 0], 1.0)

def test_learns_to_go_right():
    """after training, the greedy policy goes right in states 0..3"""
    Q = train_corridor(300)
    assert list(Q[:4].argmax(axis=1)) == [1, 1, 1, 1]

def test_values_approach_optimum():
    """Q(s, right) approaches gamma^(3 - s) for the states before the goal"""
    Q = train_corridor(2000)
    assert approx(Q[:4, 1], [0.9 ** 3, 0.9 ** 2, 0.9, 1.0], tol=0.02)
~~~

<!-- hint: Draw `rng.uniform()` first; only if it is below 0.2 draw `rng.integers(2)` for the random action. -->
<!-- /exercise -->

<!-- quiz -->
1. What does the discount factor $\gamma$ control?
   - [x] How much the agent values future rewards compared with immediate ones
   - [ ] The learning rate
   - [ ] The probability of exploring
   > Rewards $k$ steps ahead are weighted by $\gamma^k$.
2. Why does value iteration converge from any starting values?
   - [ ] Because rewards are positive
   - [x] The Bellman update is a contraction that shrinks the error by a factor $\gamma$ each sweep
   - [ ] Because the policy is fixed
   > A contraction mapping has a unique fixed point, reached from any start.
3. What makes Q-learning "off-policy"?
   - [x] It learns the value of the greedy policy while acting with a different, exploratory policy
   - [ ] It needs no policy at all
   - [ ] It uses the transition probabilities directly
   > The $\max_{a'}$ in the target evaluates the greedy choice regardless of what was actually done.
4. Why did DQN introduce experience replay?
   - [ ] To reduce memory use
   - [x] Consecutive transitions are highly correlated; sampling random past transitions makes training far more stable
   - [ ] To avoid using a neural network
   > Training a network on strongly correlated, non-stationary data in sequence is unstable.
<!-- /quiz -->
