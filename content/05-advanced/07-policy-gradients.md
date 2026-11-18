---
title: Policy gradients and PPO
level: senior
minutes: 55
summary: Optimise the policy directly by gradient ascent on expected return. The log-derivative trick behind REINFORCE, baselines and advantages to tame variance, actor–critic methods, and the clipped objective of PPO used to fine-tune language models.
---

## Learning the policy itself

[Q-learning](#/lesson/q-learning) learns values and acts greedily. **Policy gradient** methods instead parameterise the policy directly, $\pi_\theta(a\mid s)$, for example a network with a softmax over actions, and adjust $\theta$ to make high-return behaviour more likely. They handle continuous actions and stochastic policies naturally, and they are the basis of reinforcement learning from human feedback for language models.

## The log-derivative trick

We want to maximise $J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[R(\tau)]$, the expected return over trajectories. The return does not depend on $\theta$ directly; only the probability of each trajectory does. Using $\nabla_\theta p = p\,\nabla_\theta\log p$:

<!-- formula: Policy gradient theorem -->
$$\nabla_\theta J(\theta) = \mathbb E_{\pi_\theta}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;G_t\Big]$$

where $G_t$ is the **reward-to-go**, the discounted return from step $t$ onwards (actions cannot affect rewards that came before them). The environment's dynamics drop out entirely: we never need to differentiate through the world.

**REINFORCE**: run the policy, compute $G_t$ for every step, and take a step of gradient **ascent** on $\sum_t\log\pi_\theta(a_t\mid s_t)\,G_t$. Actions followed by high returns become more likely.

## Reducing variance: baselines and advantages

The raw estimate is very noisy. Subtracting a **baseline** $b(s_t)$ that does not depend on the action keeps the gradient unbiased (since $\mathbb E[\nabla\log\pi] = 0$) while reducing its variance. The best common choice is the state value $V(s_t)$, which turns the weight into the **advantage**:

<!-- formula: Advantage -->
$$A(s_t, a_t) = G_t - V(s_t)$$

"Was this action better or worse than what I usually get from here?" **Actor–critic** methods train a second network, the critic, to estimate $V$, and use it both as the baseline and to bootstrap returns. Generalised advantage estimation (GAE) blends multi-step returns to trade bias against variance.

## PPO

Large policy updates are destructive: one bad step can collapse performance, and the data collected under the old policy no longer describes the new one. **Proximal policy optimisation** limits how far the policy can move per update. With the probability ratio $\rho_t = \pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$ it maximises

<!-- formula: PPO clipped objective -->
$$L^{\text{CLIP}}(\theta) = \mathbb E_t\Big[\min\big(\rho_tA_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big]$$

with $\epsilon \approx 0.2$. For a good action ($A > 0$) the gain stops growing once $\rho_t > 1 + \epsilon$; for a bad action ($A < 0$) the penalty stops shrinking below $1 - \epsilon$. Taking the minimum makes the objective a pessimistic bound, so the optimiser has no reason to push the ratio far from 1. PPO can then reuse each batch of experience for several epochs of minibatch updates.

## RL from human feedback

To align a language model, humans compare pairs of answers, a **reward model** is trained to predict their preferences, and the language model (the policy, whose actions are tokens) is fine-tuned with PPO to maximise that reward minus a [KL penalty](#/lesson/information-theory) towards the original model, which stops it from drifting into gibberish that fools the reward model. Simpler preference-optimisation methods such as DPO skip the explicit reward model and the RL loop.

<!-- exercise: returns | Reward-to-go -->
Write `rewards_to_go(rewards, gamma)` returning $G_t = \sum_{k\ge 0}\gamma^k r_{t+k}$ for every step of one episode, computed in a single backward pass.

~~~python starter
import numpy as np

def rewards_to_go(rewards, gamma):
    raise NotImplementedError

print(rewards_to_go([1.0, 0.0, 2.0], 0.5))   # [1.5, 1.0, 2.0]
~~~

~~~python solution
import numpy as np

def rewards_to_go(rewards, gamma):
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G

print(rewards_to_go([1.0, 0.0, 2.0], 0.5))
~~~

~~~python tests
import numpy as np

def test_example():
    """[1, 0, 2] with gamma 0.5 gives [1.5, 1, 2]"""
    assert approx(rewards_to_go([1.0, 0.0, 2.0], 0.5), [1.5, 1.0, 2.0])

def test_no_discount():
    """gamma = 1 gives suffix sums"""
    assert approx(rewards_to_go([1.0, 2.0, 3.0], 1.0), [6.0, 5.0, 3.0])

def test_matches_definition():
    """agrees with the direct sum on a random episode"""
    r = np.random.default_rng(0).normal(size=20)
    direct = [sum(0.9 ** k * r[t + k] for k in range(20 - t)) for t in range(20)]
    assert approx(rewards_to_go(r, 0.9), direct)
~~~

<!-- /exercise -->

<!-- exercise: ppo-clip | The PPO clipped objective -->
Write `ppo_objective(logp_new, logp_old, adv, eps=0.2)` returning the mean clipped surrogate objective (to be maximised) from per-step log-probabilities under the new and old policies and the advantages.

~~~python starter
import numpy as np

def ppo_objective(logp_new, logp_old, adv, eps=0.2):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def ppo_objective(logp_new, logp_old, adv, eps=0.2):
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    adv = np.asarray(adv, float)
    return float(np.mean(np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)))
~~~

~~~python tests
import numpy as np

def test_unchanged_policy():
    """with the same policy the objective is the mean advantage"""
    lp = np.log(np.array([0.2, 0.5, 0.3]))
    assert approx(ppo_objective(lp, lp, [1.0, -2.0, 0.5]), np.mean([1.0, -2.0, 0.5]))

def test_positive_advantage_capped():
    """for A > 0, raising the ratio beyond 1 + eps gains nothing"""
    old = np.log([0.5]); adv = [2.0]
    assert approx(ppo_objective(np.log([0.5 * 1.2]), old, adv), 2.4)
    assert approx(ppo_objective(np.log([0.5 * 3.0]), old, adv), 2.4)

def test_negative_advantage_floor():
    """for A < 0, lowering the ratio below 1 - eps gains nothing, but raising it is fully penalised"""
    old = np.log([0.5]); adv = [-1.0]
    assert approx(ppo_objective(np.log([0.5 * 0.1]), old, adv), -0.8)
    assert approx(ppo_objective(np.log([0.5 * 2.0]), old, adv), -2.0)
~~~

<!-- hint: The ratio is `exp(logp_new - logp_old)`; take the element-wise minimum of the unclipped and clipped terms, then the mean. -->
<!-- /exercise -->

<!-- quiz -->
1. Why does the policy gradient not need a model of the environment's dynamics?
   - [x] The log-derivative trick moves the gradient onto $\log\pi_\theta$, and the transition probabilities do not depend on $\theta$
   - [ ] Because rewards are always known in advance
   - [ ] It does need the dynamics
   > $\nabla\log p(\tau)$ is a sum of $\nabla\log\pi_\theta$ terms; the environment terms vanish.
2. Why subtract a baseline from the return?
   - [ ] To make the gradient biased towards good actions
   - [x] It reduces the variance of the gradient estimate without changing its expected value
   - [ ] To make all rewards negative
   > A baseline that does not depend on the action leaves the expectation unchanged.
3. What does the clipping in PPO prevent?
   - [ ] Exploration
   - [x] Policy updates that move too far from the policy that collected the data
   - [ ] Negative advantages
   > Once the ratio leaves $[1-\epsilon, 1+\epsilon]$ in the favourable direction, the objective stops rewarding further change.
4. In RLHF, why is a KL penalty towards the original model added?
   - [x] To keep the fine-tuned model from drifting into outputs that exploit the reward model
   - [ ] To make training faster
   - [ ] To increase the reward
   > The reward model is imperfect; staying close to the original model keeps outputs fluent and sensible.
<!-- /quiz -->
