---
title: Probability
level: beginner
minutes: 40
summary: Random variables, distributions, expectation and variance, conditional probability and Bayes' rule, and the log-sum-exp trick that keeps probabilities from underflowing.
---

## Random variables and distributions

A **random variable** is a quantity whose value is uncertain: tomorrow's sales, whether an email is spam. Its **distribution** says how likely each value is.

- A discrete variable has a **probability mass function** $p(x) = P(X = x)$, with $\sum_x p(x) = 1$.
- A continuous variable has a **density** $p(x)$; probabilities are areas, $P(a \le X \le b) = \int_a^b p(x)\,dx$, and $\int p(x)\,dx = 1$. A density can be larger than 1; only areas are probabilities.

Distributions you will meet in almost every lesson:

| Distribution | Values | Parameters | Used for |
|---|---|---|---|
| Bernoulli | 0 or 1 | $p$ | one yes/no outcome, [logistic regression](#/lesson/logistic-regression) |
| Categorical | one of $K$ classes | $p_1..p_K$ | classification, [softmax](#/lesson/softmax-regression) |
| Binomial | $0..n$ | $n, p$ | count of successes |
| Poisson | $0, 1, 2, \dots$ | rate $\lambda$ | counts per time window |
| Gaussian (normal) | real numbers | $\mu, \sigma^2$ | noise, measurements, almost everything |

<!-- formula: Gaussian density -->
$$\mathcal N(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

## Expectation and variance

The **expectation** is the probability-weighted average value; the **variance** is the expected squared distance from it:

<!-- formula: Expectation and variance -->
$$\mathbb E[X] = \sum_x x\,p(x), \qquad \operatorname{Var}[X] = \mathbb E\big[(X - \mathbb E[X])^2\big] = \mathbb E[X^2] - \mathbb E[X]^2$$

Expectation is linear: $\mathbb E[aX + bY] = a\,\mathbb E[X] + b\,\mathbb E[Y]$ always. Variances add, $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$, only when $X$ and $Y$ are independent (more precisely, uncorrelated).

## Joint, marginal and conditional

For two variables, $p(x, y)$ is the joint distribution. Summing out one variable gives a **marginal**, $p(x) = \sum_y p(x, y)$. The **conditional** distribution is the joint restricted to a known value and renormalised:

<!-- formula: Conditional probability and product rule -->
$$p(y \mid x) = \frac{p(x, y)}{p(x)}, \qquad p(x, y) = p(y\mid x)\,p(x)$$

$X$ and $Y$ are **independent** when $p(x, y) = p(x)\,p(y)$: knowing one tells you nothing about the other.

## Bayes' rule

Writing the product rule both ways and dividing gives Bayes' rule, which turns "how likely is the evidence if the hypothesis is true" into "how likely is the hypothesis given the evidence":

<!-- formula: Bayes' rule -->
$$p(h \mid e) = \frac{p(e \mid h)\, p(h)}{p(e)} = \frac{p(e \mid h)\, p(h)}{\sum_{h'} p(e \mid h')\, p(h')}$$

$p(h)$ is the **prior**, $p(e\mid h)$ the **likelihood**, $p(h\mid e)$ the **posterior**. The classic surprise: a test that is 99% accurate for a disease that 1 in 1000 people have still gives mostly false alarms, because the healthy people vastly outnumber the sick ones.

## Working in log space

Probabilities of long sequences are products of many numbers below 1, which underflow to exactly 0 in floating point. Models therefore work with **log probabilities**, which add instead of multiply. The one awkward operation is adding probabilities that are stored as logs, $\log\sum_i e^{a_i}$. Computing $e^{a_i}$ directly overflows or underflows, so subtract the maximum first:

<!-- formula: Log-sum-exp -->
$$\log\sum_i e^{a_i} = m + \log\sum_i e^{a_i - m}, \qquad m = \max_i a_i$$

Now the largest term is $e^0 = 1$, nothing overflows, and at least one term is not lost. You will see this inside [softmax](#/lesson/softmax-regression), [mixture models](#/lesson/gaussian-mixtures) and language models.

<!-- exercise: bayes | A medical test and Bayes' rule -->
Write `posterior(prior, sensitivity, specificity)` that returns $P(\text{sick} \mid \text{positive test})$.

- `prior` is $P(\text{sick})$
- `sensitivity` is $P(\text{positive} \mid \text{sick})$
- `specificity` is $P(\text{negative} \mid \text{healthy})$, so a healthy person tests positive with probability `1 - specificity`

~~~python starter
def posterior(prior, sensitivity, specificity):
    # P(positive) = P(pos | sick) P(sick) + P(pos | healthy) P(healthy)
    raise NotImplementedError

print(posterior(0.001, 0.99, 0.99))   # about 0.09
~~~

~~~python solution
def posterior(prior, sensitivity, specificity):
    p_pos_sick = sensitivity * prior
    p_pos_healthy = (1 - specificity) * (1 - prior)
    return p_pos_sick / (p_pos_sick + p_pos_healthy)

print(posterior(0.001, 0.99, 0.99))
~~~

~~~python tests
def test_rare_disease():
    """a 99% accurate test for a 1-in-1000 disease gives about 9% posterior"""
    assert approx(posterior(0.001, 0.99, 0.99), 0.00099 / (0.00099 + 0.00999))

def test_perfect_test():
    """a perfect test gives certainty"""
    assert approx(posterior(0.2, 1.0, 1.0), 1.0)

def test_useless_test():
    """a test that is positive for everyone leaves the prior unchanged"""
    assert approx(posterior(0.3, 1.0, 0.0), 0.3)

def test_in_range():
    """the posterior is a probability"""
    for p in (0.01, 0.5, 0.9):
        assert 0.0 <= posterior(p, 0.8, 0.7) <= 1.0
~~~

<!-- hint: The denominator is the total probability of a positive test, from sick and healthy people together. -->
<!-- /exercise -->

<!-- exercise: logsumexp | A stable log-sum-exp -->
Write `logsumexp(a)` for a 1-D array that computes $\log\sum_i e^{a_i}$ without overflow, and `log_gaussian(x, mu, sigma2)` that returns the **log** of the Gaussian density, element-wise for an array `x`.

~~~python starter
import numpy as np

def logsumexp(a):
    a = np.asarray(a, dtype=float)
    raise NotImplementedError

def log_gaussian(x, mu, sigma2):
    raise NotImplementedError

print(logsumexp(np.array([1000.0, 1000.0])))   # 1000 + log 2, not inf
~~~

~~~python solution
import numpy as np

def logsumexp(a):
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

def log_gaussian(x, mu, sigma2):
    x = np.asarray(x, dtype=float)
    return -0.5 * np.log(2 * np.pi * sigma2) - (x - mu) ** 2 / (2 * sigma2)

print(logsumexp(np.array([1000.0, 1000.0])))
~~~

~~~python tests
import numpy as np

def test_small_values():
    """matches the direct formula when nothing overflows"""
    a = np.array([0.1, -2.0, 1.5])
    assert approx(logsumexp(a), np.log(np.sum(np.exp(a))))

def test_huge_values():
    """no overflow for large inputs"""
    assert approx(logsumexp(np.array([1000.0, 1000.0])), 1000 + np.log(2))

def test_tiny_values():
    """no underflow to -inf for very negative inputs"""
    assert approx(logsumexp(np.array([-1000.0, -1001.0])), -1000 + np.log(1 + np.exp(-1.0)))

def test_log_gaussian():
    """log density of N(0, 1) at 0 is -0.5 log(2 pi), and it works on arrays"""
    assert approx(log_gaussian(0.0, 0.0, 1.0), -0.5 * np.log(2 * np.pi))
    x = np.array([-1.0, 2.0, 3.5])
    direct = np.log(np.exp(-(x - 1.0) ** 2 / (2 * 4.0)) / np.sqrt(2 * np.pi * 4.0))
    assert approx(log_gaussian(x, 1.0, 4.0), direct)
~~~

<!-- hint: Subtract `m = np.max(a)` before exponentiating, then add it back outside the log. -->
<!-- hint: Take the log of the density formula by hand: the exponential disappears. -->
<!-- /exercise -->

<!-- quiz -->
1. A density function takes the value 2.5 at some point. Is something wrong?
   - [ ] Yes, probabilities cannot exceed 1
   - [x] No, a density can exceed 1; only the area under it must equal 1
   - [ ] Only if the variable is discrete
   > A narrow Gaussian with small $\sigma$ has a tall peak above 1. Probabilities come from integrating the density over an interval.
2. In Bayes' rule, what is the prior?
   - [x] What you believed about the hypothesis before seeing the evidence
   - [ ] The probability of the evidence if the hypothesis is true
   - [ ] The normalising constant $p(e)$
   > The prior $p(h)$ is updated by the likelihood $p(e\mid h)$ into the posterior $p(h\mid e)$.
3. $X$ and $Y$ are independent with variances 4 and 9. What is $\operatorname{Var}[X - Y]$?
   - [ ] 5
   - [ ] -5
   - [x] 13
   > $\operatorname{Var}[X - Y] = \operatorname{Var}[X] + (-1)^2\operatorname{Var}[Y] = 4 + 9$. Variances of independent variables add even when you subtract the variables.
4. Why do models usually maximise the log-likelihood rather than the likelihood?
   - [ ] The log-likelihood has a different maximum
   - [x] Products of many small probabilities underflow, while the sum of their logs is stable and easier to differentiate
   - [ ] Logarithms make the model more accurate
   > The logarithm is increasing, so the maximiser is the same, and products become sums.
<!-- /quiz -->
