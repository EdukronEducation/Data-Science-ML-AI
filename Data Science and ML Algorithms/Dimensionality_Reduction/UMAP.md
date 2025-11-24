# UMAP (Uniform Manifold Approximation and Projection): The Modern Mapper

## 📄 Page 1 — Introduction / Intuition

### Introduction
**UMAP** is a dimensionality reduction technique that is similar to t-SNE but is **faster**, **scalable**, and preserves more **global structure**. It is constructed from a theoretical framework based on Riemannian geometry and algebraic topology.

### What Problem It Solves
It solves the limitations of t-SNE:
1.  **Speed**: t-SNE is slow. UMAP is fast.
2.  **Global Structure**: t-SNE destroys global relationships (distance between clusters). UMAP preserves them better.
3.  **Transform**: t-SNE cannot transform new data. UMAP can.

### Core Idea
1.  **Manifold Assumption**: Data is uniformly distributed on a locally connected Riemannian manifold.
2.  **Fuzzy Simplicial Set**: It models the data as a fuzzy topological structure.
3.  **Optimization**: It finds a low-dimensional projection that has the closest possible fuzzy topological structure.

### Intuition: The "Paper Mache" Analogy
*   **t-SNE**: Tears the paper into confetti to flatten it.
*   **UMAP**: Stretches and compresses the paper (like rubber) to flatten it, keeping the pieces connected.

### Visual Interpretation
UMAP plots look similar to t-SNE (distinct clusters), but the clusters are often more compact, and the relative positions of the clusters mean something.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is currently considered the **Best-in-Class** algorithm for general non-linear dimensionality reduction. It combines the visualization power of t-SNE with the speed and utility of PCA.

### Domains & Use Cases
1.  **Genetics**: Analyzing millions of cells (Single-cell genomics).
2.  **Feature Engineering**: Unlike t-SNE, UMAP features *can* be used as input for clustering (DBSCAN) or classification.
3.  **Big Data**: Can handle millions of rows.

### Type of Algorithm
*   **Learning Type**: Unsupervised (can be Supervised!).
*   **Task**: Dimensionality Reduction.
*   **Linearity**: Non-Linear.
*   **Probabilistic/Deterministic**: Stochastic (but stable).
*   **Parametric**: Non-parametric (but learns a transform).

---

## 📄 Page 3 — Mathematical Foundation

### Graph Construction
1.  **Local Connectivity**: For each point $x_i$, find $k$ nearest neighbors.
2.  **Fuzzy Set**: Define probability of connection $p_{ij}$ based on distance, but adjust for local density (like t-SNE).
    $$ p_{ij} = \exp(-(d(x_i, x_j) - \rho_i) / \sigma_i) $$
    *   $\rho_i$: Distance to the nearest neighbor (ensures local connectivity).

### Optimization (Cross Entropy)
UMAP minimizes the **Cross Entropy** between the High-D graph and Low-D graph.
$$ C = \sum p_{ij} \log(\frac{p_{ij}}{q_{ij}}) + (1-p_{ij}) \log(\frac{1-p_{ij}}{1-q_{ij}}) $$
*   **Term 1**: Attractive force (Keep neighbors close).
*   **Term 2**: Repulsive force (Push non-neighbors apart).
*   *Note*: t-SNE only has the first term (KL Divergence). The second term is why UMAP preserves global structure better (it explicitly models gaps).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Nearest Neighbors**: Use Approximate Nearest Neighbor (ANN) descent to find neighbors efficiently.
2.  **Construct Graph**: Build the fuzzy simplicial set (weighted graph).
3.  **Initialize Low-D**: Use Spectral Embedding (Laplacian Eigenmaps) for initialization (Deterministic start).
4.  **Optimize**: Use Stochastic Gradient Descent to minimize Cross Entropy.

### Pseudocode
```python
# Conceptual
knn = NearestNeighbors(data, k)
graph = fuzzy_simplicial_set(knn)
embedding = spectral_layout(graph)
embedding = optimize_layout(embedding, graph)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Uniformity**: Assumes data is uniformly distributed on the manifold.

### Hyperparameters
1.  **`n_neighbors`**:
    *   *Description*: Size of local neighborhood.
    *   *Effect*:
        *   Low (5): Local structure (fine details).
        *   High (50): Global structure (big picture).
2.  **`min_dist`**:
    *   *Description*: Minimum distance between points in Low-D.
    *   *Effect*:
        *   Low (0.1): Clumpy clusters.
        *   High (0.5): Spread out points.
3.  **`n_components`**: Dimensions (2 for plot, 10+ for features).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**.

### 2. Encoding
*   **Required**.

### 3. Metric
*   UMAP supports many metrics (Cosine, Manhattan, etc.). Use 'cosine' for text/NLP.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Speed**: Much faster than t-SNE.
2.  **Global Structure**: Preserves distances between clusters.
3.  **Transform**: Can project new data (Train/Test split possible).
4.  **Supervised**: Can use labels to guide the reduction (Supervised UMAP).

### Limitations
1.  **New**: Less mature than PCA.
2.  **Hyperparameters**: `n_neighbors` and `min_dist` significantly change the plot.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Trustworthiness**: Measures preservation of local structure.

### Python Implementation
```python
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Data
data = load_digits()
X = data.data
y = data.target

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Train UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X_scaled)

# 4. Visualize
plt.figure(figsize=(10, 8))
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.colorbar()
plt.title('UMAP of MNIST Digits')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
*   **Clustering**: Use low `n_neighbors` and low `min_dist`.
*   **Global View**: Use high `n_neighbors`.

### Real-World Applications
*   **Genomics**: The standard tool for single-cell analysis (Scanpy).
*   **NLP**: Visualizing BERT embeddings.

### When to Use
*   **Always**. It is generally superior to t-SNE.
*   When you need to use the reduced features for downstream tasks (Clustering/Classification).

### When NOT to Use
*   When you need exact linear interpretability (Use PCA).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs t-SNE**: UMAP is faster, scales better, and preserves global structure.
*   **vs PCA**: UMAP is non-linear. PCA is linear.

### Interview Questions
1.  **Q: Why is UMAP faster than t-SNE?** (It uses Approximate Nearest Neighbors and simplifies the optimization function).
2.  **Q: Can UMAP be supervised?** (Yes. You can pass `y` labels to `fit`, and it will pull same-class points together).

### Summary
UMAP is the modern successor to t-SNE. It rests on deep mathematical foundations (Topology) but delivers practical, fast, and beautiful results. It is the current state-of-the-art for non-linear dimensionality reduction.

### Cheatsheet
*   **Type**: Non-linear / Topological.
*   **Key**: Manifold, Fuzzy Set.
*   **Param**: `n_neighbors`, `min_dist`.
*   **Pros**: Fast, Global Structure.
