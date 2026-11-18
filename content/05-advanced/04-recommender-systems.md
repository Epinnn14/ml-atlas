---
title: Recommender systems
level: advanced
minutes: 45
summary: Predict what a user will like from what everyone liked. Collaborative filtering, matrix factorisation trained only on observed ratings, implicit feedback, cold start, and ranking metrics such as NDCG.
---

## The problem

A shop has users and items and a sparse record of interactions: ratings, purchases, clicks, watch time. Most user–item pairs were never observed. A recommender fills in the missing entries well enough to rank, for each user, the items they are most likely to want.

- **Content-based** methods use item and user features ("you liked thrillers, here is another thriller").
- **Collaborative filtering** uses only the interaction patterns: users who agreed in the past tend to agree again. It needs no features and discovers tastes nobody named.

## Matrix factorisation

Arrange ratings in a matrix $R$ (users × items), mostly empty. Model each user $u$ and item $i$ by a vector of $k$ latent factors, $\mathbf p_u$ and $\mathbf q_i$, and predict a rating by their dot product plus biases:

<!-- formula: Matrix factorisation prediction -->
$$\hat r_{ui} = \mu + b_u + b_i + \mathbf p_u^\top\mathbf q_i$$

$\mu$ is the global mean, $b_u$ a user's tendency to rate high or low, $b_i$ an item's general popularity. The factors are learned from the **observed** entries only, with L2 [regularisation](#/lesson/regularization) because each user has few ratings:

<!-- formula: Matrix factorisation objective -->
$$\min_{P, Q, \mathbf b}\ \sum_{(u,i)\in\Omega}\big(r_{ui} - \hat r_{ui}\big)^2 + \lambda\big(\lVert\mathbf p_u\rVert^2 + \lVert\mathbf q_i\rVert^2 + b_u^2 + b_i^2\big)$$

where $\Omega$ is the set of observed pairs. Treating missing entries as zeros would teach the model that every unseen item is disliked. Two standard solvers:

- **SGD**: loop over observed ratings; with error $e = r_{ui} - \hat r_{ui}$, update $\mathbf p_u \mathrel{+}= \eta(e\,\mathbf q_i - \lambda\mathbf p_u)$ and $\mathbf q_i \mathrel{+}= \eta(e\,\mathbf p_u - \lambda\mathbf q_i)$ (using the old $\mathbf p_u$), and similarly for the biases.
- **Alternating least squares**: with $Q$ fixed, each $\mathbf p_u$ is a small [ridge regression](#/lesson/regularization) with a closed form; alternate. Easy to parallelise.

The learned factors often align with interpretable axes (comedy versus drama, mainstream versus niche) and are item [embeddings](#/lesson/word-embeddings) in all but name.

## Implicit feedback

Most real data is implicit: a click or a purchase, never an explicit dislike. Unobserved does not mean negative, just unknown. Approaches weight observed interactions by confidence and sample unobserved items as weak negatives (as in word2vec's negative sampling), or optimise a ranking loss such as BPR that only requires a seen item to score above an unseen one.

## Cold start, popularity and feedback loops

- **Cold start**: a new user or item has no interactions. Fall back on content features, popularity, or ask a few onboarding questions.
- **Popularity bias**: models over-recommend already popular items; the long tail never gets shown.
- **Feedback loops**: users can only click what was recommended, so the next model is trained on the previous model's choices. Some exploration ([bandits](#/lesson/bandits)) keeps the data informative.

## Evaluating rankings

What matters is the order of the top few recommendations, not the squared error of every rating. Common metrics on held-out interactions:

- **Precision@k / recall@k**: how many of the top $k$ were relevant.
- **NDCG@k**: rewards putting relevant items near the top, discounting by position:

<!-- formula: Discounted cumulative gain -->
$$\text{DCG@}k = \sum_{j=1}^{k}\frac{\text{rel}_j}{\log_2(j + 1)}, \qquad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

where $\text{rel}_j$ is the relevance of the item at position $j$ and IDCG is the DCG of the best possible ordering. Hold out by time (the last interactions of each user), not at random, and validate online with A/B tests ([causal inference](#/lesson/causal-inference)).

<!-- exercise: mf-sgd | Matrix factorisation with SGD -->
Write `train_mf(ratings, n_users, n_items, k=4, lr=0.02, lam=0.05, epochs=60, seed=0)` where `ratings` is a list of `(u, i, r)`. Initialise `rng = np.random.default_rng(seed)`, `P = rng.normal(scale=0.1, size=(n_users, k))`, `Q = rng.normal(scale=0.1, size=(n_items, k))`, zero biases, and `mu` as the mean rating. Each epoch, visit the ratings in the order `rng.permutation(len(ratings))` and apply the SGD updates above (biases: `bu += lr * (e - lam * bu)`, same for `bi`). Return a dict with `mu, bu, bi, P, Q`, and write `predict(model, u, i)`.

~~~python starter
import numpy as np

def train_mf(ratings, n_users, n_items, k=4, lr=0.02, lam=0.05, epochs=60, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(scale=0.1, size=(n_users, k))
    Q = rng.normal(scale=0.1, size=(n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    mu = float(np.mean([r for _, _, r in ratings]))
    raise NotImplementedError

def predict(model, u, i):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def train_mf(ratings, n_users, n_items, k=4, lr=0.02, lam=0.05, epochs=60, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(scale=0.1, size=(n_users, k))
    Q = rng.normal(scale=0.1, size=(n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    mu = float(np.mean([r for _, _, r in ratings]))
    for _ in range(epochs):
        for idx in rng.permutation(len(ratings)):
            u, i, r = ratings[idx]
            e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])
            pu = P[u].copy()
            P[u] += lr * (e * Q[i] - lam * P[u])
            Q[i] += lr * (e * pu - lam * Q[i])
    return {"mu": mu, "bu": bu, "bi": bi, "P": P, "Q": Q}

def predict(model, u, i):
    return float(model["mu"] + model["bu"][u] + model["bi"][i] + model["P"][u] @ model["Q"][i])
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(1)
nu, ni, k = 40, 30, 3
Pt, Qt = rng.normal(size=(nu, k)), rng.normal(size=(ni, k))
full = 3 + Pt @ Qt.T * 0.7
mask = rng.uniform(size=(nu, ni)) < 0.5
obs = [(u, i, full[u, i] + rng.normal(scale=0.1)) for u in range(nu) for i in range(ni) if mask[u, i]]
heldout = [(u, i, full[u, i]) for u in range(nu) for i in range(ni) if not mask[u, i]]

def rmse(model, data):
    return np.sqrt(np.mean([(predict(model, u, i) - r) ** 2 for u, i, r in data]))

def test_fits_observed():
    """training error is small"""
    m = train_mf(obs, nu, ni)
    assert rmse(m, obs) < 0.35

def test_beats_mean_on_unseen():
    """on unobserved pairs it beats predicting the global mean"""
    m = train_mf(obs, nu, ni)
    base = np.sqrt(np.mean([(m["mu"] - r) ** 2 for _, _, r in heldout]))
    assert rmse(m, heldout) < 0.75 * base

def test_predict_formula():
    """predict uses mu + bu + bi + p.q"""
    m = {"mu": 3.0, "bu": np.array([0.5]), "bi": np.array([0, -1.0]), "P": np.array([[1.0, 2.0]]), "Q": np.array([[0, 0], [0.5, 0.5]])}
    assert approx(predict(m, 0, 1), 3.0 + 0.5 - 1.0 + 1.5)

def test_zero_epochs_is_init():
    """with no epochs nothing is learned yet: biases stay zero"""
    m = train_mf(obs, nu, ni, epochs=0)
    assert np.all(m["bu"] == 0) and np.all(m["bi"] == 0)
~~~

<!-- hint: Compute the error once, then update the biases, then `P[u]` and `Q[i]`, using a copy of the old `P[u]` for the `Q[i]` update. -->
<!-- /exercise -->

<!-- exercise: ndcg | NDCG at k -->
Write `ndcg_at_k(relevances, k)` where `relevances` lists the relevance of the recommended items in the order shown (position 1 first). Compute DCG@k, divide by the DCG@k of the same relevances sorted in descending order, and return 0.0 if that ideal DCG is 0.

~~~python starter
import numpy as np

def ndcg_at_k(relevances, k):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def _dcg(rel, k):
    rel = np.asarray(rel, float)[:k]
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))

def ndcg_at_k(relevances, k):
    ideal = _dcg(sorted(relevances, reverse=True), k)
    return 0.0 if ideal == 0 else _dcg(relevances, k) / ideal
~~~

~~~python tests
import numpy as np

def test_perfect_order():
    """the best ordering scores 1"""
    assert approx(ndcg_at_k([3, 2, 1, 0], 4), 1.0)

def test_value():
    """matches the formula on a small example"""
    rel = [0, 2, 0, 1]
    dcg = 2 / np.log2(3) + 1 / np.log2(5)
    idcg = 2 / np.log2(2) + 1 / np.log2(3)
    assert approx(ndcg_at_k(rel, 4), dcg / idcg)

def test_cutoff():
    """only the top k positions count"""
    assert approx(ndcg_at_k([0, 0, 5], 2), 0.0)

def test_nothing_relevant():
    """no relevant items gives 0, not a division error"""
    assert ndcg_at_k([0, 0, 0], 3) == 0.0

def test_earlier_is_better():
    """a relevant item at position 1 beats it at position 3"""
    assert ndcg_at_k([1, 0, 0], 3) > ndcg_at_k([0, 0, 1], 3)
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why are missing entries not treated as zero ratings in matrix factorisation?
   - [x] Missing means unknown, not disliked; zeros would teach the model that every unseen item is bad
   - [ ] Zeros make the matrix too large
   - [ ] The dot product cannot handle zeros
   > The loss is summed only over observed pairs.
2. A new item has no interactions yet. Which approach helps most?
   - [ ] Collaborative filtering alone
   - [x] Using its content features (category, description) or showing it to some users to gather data
   - [ ] Increasing the number of latent factors
   > This is the cold-start problem: collaborative filtering has nothing to work with until interactions exist.
3. Why is NDCG preferred over RMSE for evaluating a "top 10 for you" list?
   - [ ] RMSE cannot be computed for recommendations
   - [x] What matters is placing relevant items near the top, which NDCG rewards and RMSE ignores
   - [ ] NDCG is cheaper to compute
   > A model can have low rating error but rank poorly at the top.
4. What is a feedback loop in recommender systems?
   - [ ] The SGD update loop
   - [x] Future training data only contains reactions to what the current model chose to show, reinforcing its own biases
   - [ ] Users rating items twice
   > Without exploration the model never learns about items it never shows.
<!-- /quiz -->
