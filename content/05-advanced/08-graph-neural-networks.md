---
title: Graph neural networks
level: senior
minutes: 45
summary: Learning on networks of things: molecules, road maps, social graphs, transactions. Message passing, the graph convolution with its normalised adjacency, and the over-smoothing that limits depth.
---

## Data as graphs

Many datasets are not tables or sequences but **graphs**: nodes with features, connected by edges. Atoms and bonds, users and friendships, accounts and transfers, papers and citations. Tasks include classifying nodes (is this account fraudulent?), predicting links (will these users connect?) and classifying whole graphs (is this molecule toxic?).

A graph with $n$ nodes is described by an **adjacency matrix** $A$ ($A_{ij} = 1$ if $i$ and $j$ are connected) and a node feature matrix $X$ of shape $(n, d)$. The **degree** of a node is its number of neighbours, $d_i = \sum_jA_{ij}$.

A model for graphs must not depend on the arbitrary order in which nodes are numbered: relabelling the nodes should just relabel the outputs (**permutation equivariance**). That rules out flattening $A$ into an ordinary MLP.

## Message passing

Graph neural networks update each node's representation by combining its own state with **messages** aggregated from its neighbours, using the same learned functions everywhere:

<!-- formula: Message passing -->
$$\mathbf h_i^{(l+1)} = \phi\Big(\mathbf h_i^{(l)},\ \bigoplus_{j\in\mathcal N(i)}\psi\big(\mathbf h_i^{(l)}, \mathbf h_j^{(l)}\big)\Big)$$

where $\bigoplus$ is a permutation-invariant aggregation such as sum, mean or max. After $L$ layers each node has information from its $L$-hop neighbourhood. For graph-level tasks, pool all node representations (sum or mean) into one vector.

## The graph convolutional network

The GCN layer is the simplest instance: average over the neighbourhood (including the node itself) with a symmetric normalisation, then apply a shared linear map and a non-linearity:

<!-- formula: GCN layer -->
$$H^{(l+1)} = \sigma\big(\hat A\,H^{(l)}W^{(l)}\big), \qquad \hat A = \tilde D^{-1/2}\tilde A\,\tilde D^{-1/2}, \quad \tilde A = A + I$$

$\tilde D$ is the diagonal degree matrix of $\tilde A$. Adding self-loops keeps each node's own features; the normalisation keeps high-degree nodes from producing huge sums and gives entry $\hat A_{ij} = 1/\sqrt{\tilde d_i\tilde d_j}$ for connected nodes. Like a [convolution](#/lesson/cnn), the same small set of weights is shared across the whole graph.

Other popular layers: **GraphSAGE** samples neighbours and concatenates self and neighbour summaries, scaling to huge graphs; **GAT** learns [attention](#/lesson/attention) weights over neighbours instead of fixed ones. A [Transformer](#/lesson/transformer) is in fact a GNN on a fully connected graph.

## Over-smoothing and other limits

Each GCN layer is a form of averaging. Stack many and every node's representation converges to the same vector, erasing the differences you need: **over-smoothing**. Most GNNs therefore use only 2–4 layers, with residual connections when deeper. Other limits: message passing cannot tell apart some structurally different graphs, and information from distant nodes gets squeezed through bottlenecks ("over-squashing").

<!-- exercise: norm-adj | The normalised adjacency -->
Write `normalised_adjacency(A)` returning $\tilde D^{-1/2}(A + I)\tilde D^{-1/2}$ for a symmetric 0/1 adjacency matrix.

~~~python starter
import numpy as np

def normalised_adjacency(A):
    A = np.asarray(A, float)
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def normalised_adjacency(A):
    A = np.asarray(A, float)
    At = A + np.eye(len(A))
    d = At.sum(axis=1)
    inv_sqrt = 1 / np.sqrt(d)
    return inv_sqrt[:, None] * At * inv_sqrt[None, :]
~~~

~~~python tests
import numpy as np

# a path graph 0 - 1 - 2
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

def test_values():
    """entries are 1 / sqrt(d_i d_j) with self-loops counted"""
    N = normalised_adjacency(A)
    assert approx(N[0, 0], 1 / 2) and approx(N[0, 1], 1 / np.sqrt(2 * 3)) and approx(N[1, 1], 1 / 3)
    assert N[0, 2] == 0

def test_symmetric():
    """the result is symmetric"""
    N = normalised_adjacency(A)
    assert approx(N, N.T)

def test_eigenvalues_bounded():
    """eigenvalues lie in (-1, 1], so repeated application stays bounded"""
    rng = np.random.default_rng(0)
    B = (rng.uniform(size=(10, 10)) < 0.3).astype(float); B = np.triu(B, 1); B = B + B.T
    ev = np.linalg.eigvalsh(normalised_adjacency(B))
    assert ev.max() <= 1 + 1e-9 and ev.min() > -1
~~~

<!-- hint: Multiply on both sides by broadcasting: `inv_sqrt[:, None] * At * inv_sqrt[None, :]`. -->
<!-- /exercise -->

<!-- exercise: gcn | A two-layer GCN -->
Write `gcn_forward(A_hat, X, W1, W2)` computing `softmax(A_hat @ relu(A_hat @ X @ W1) @ W2)` row-wise (a stable softmax), the standard two-layer node classifier.

~~~python starter
import numpy as np

def gcn_forward(A_hat, X, W1, W2):
    raise NotImplementedError
~~~

~~~python solution
import numpy as np

def gcn_forward(A_hat, X, W1, W2):
    H = np.maximum(0, A_hat @ X @ W1)
    Z = A_hat @ H @ W2
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)
~~~

~~~python tests
import numpy as np

rng = np.random.default_rng(0)
n = 6
A = np.zeros((n, n))
for i, j in [(0, 1), (1, 2), (3, 4), (4, 5)]:
    A[i, j] = A[j, i] = 1
At = A + np.eye(n); d = At.sum(1); A_hat = At / np.sqrt(d[:, None] * d[None, :])
X = rng.normal(size=(n, 3)); W1 = rng.normal(size=(3, 4)); W2 = rng.normal(size=(4, 2))

def test_rows_are_distributions():
    """output rows are probability distributions"""
    P = gcn_forward(A_hat, X, W1, W2)
    assert P.shape == (n, 2) and approx(P.sum(1), np.ones(n))

def test_formula():
    """matches the two-layer formula"""
    Z = A_hat @ np.maximum(0, A_hat @ X @ W1) @ W2
    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
    assert approx(gcn_forward(A_hat, X, W1, W2), P)

def test_permutation_equivariance():
    """relabelling nodes relabels the output"""
    perm = rng.permutation(n); Pm = np.eye(n)[perm]
    out = gcn_forward(Pm @ A_hat @ Pm.T, Pm @ X, W1, W2)
    assert approx(out, Pm @ gcn_forward(A_hat, X, W1, W2))

def test_two_hops_only():
    """two layers only reach two hops: the components {0,1,2} and {3,4,5} do not influence each other"""
    X2 = X.copy(); X2[5] += 10
    a, b = gcn_forward(A_hat, X, W1, W2), gcn_forward(A_hat, X2, W1, W2)
    assert approx(a[:3], b[:3])
~~~

<!-- /exercise -->

<!-- quiz -->
1. Why can't you classify graphs by flattening the adjacency matrix into a vector for an ordinary MLP?
   - [x] The node numbering is arbitrary, so the same graph would give different inputs and outputs depending on how nodes are labelled
   - [ ] Adjacency matrices are too sparse to store
   - [ ] MLPs cannot take binary inputs
   > A graph model must be permutation-equivariant (node tasks) or invariant (graph tasks).
2. After 3 layers of message passing, what information can a node's representation contain?
   - [ ] Only its own features
   - [x] Information from nodes up to 3 hops away
   - [ ] Information from the entire graph, always
   > Each layer extends the receptive field by one hop.
3. Why are self-loops added to the adjacency matrix in a GCN?
   - [ ] To make the matrix invertible
   - [x] So each node's own features are included in its update, not only its neighbours'
   - [ ] To double the degree of every node
   > Without self-loops a node's new representation would ignore its own previous state.
4. A 20-layer GCN performs worse than a 2-layer one, and all node embeddings look alike. What is this called?
   - [ ] Overfitting
   - [x] Over-smoothing
   - [ ] Vanishing attention
   > Repeated neighbourhood averaging makes all representations converge.
<!-- /quiz -->
