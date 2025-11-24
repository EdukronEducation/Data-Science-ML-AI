# Spectral Clustering: The Graph Cutter

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Spectral Clustering** is a technique that uses the eigenvalues (spectrum) of a similarity matrix to perform dimensionality reduction before clustering in fewer dimensions. It treats clustering as a **Graph Partitioning** problem.

### What Problem It Solves
It solves the problem of **Non-Convex Clusters**.
*   K-Means fails on concentric circles (donuts) or intertwined spirals.
*   Spectral Clustering unrolls these shapes so K-Means can handle them.

### Core Idea
1.  **Graph View**: Treat data points as nodes in a graph. Connect similar points with edges.
2.  **Cut**: We want to cut the graph into $k$ pieces such that:
    *   Edges *within* a piece have high weights (strong connections).
    *   Edges *between* pieces have low weights (weak connections).
3.  **Spectrum**: We use the Laplacian Matrix of the graph and its eigenvectors to find these cuts.

### Intuition: The "Bridge" Analogy
Imagine two islands connected by a single narrow bridge.
*   Points on Island A are highly connected to each other.
*   Points on Island B are highly connected to each other.
*   The bridge is the "weak link".
Spectral Clustering finds this weak link (the bridge) and cuts it, separating the two islands perfectly, even if the islands are shaped like bananas.

### Visual Interpretation
*   **Original Space**: Two concentric circles. K-Means fails.
*   **Spectral Embedding**: The algorithm projects the data into a new space where the two circles become two separated blobs.
*   **Clustering**: K-Means is run on these blobs and works perfectly.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is one of the most powerful algorithms for complex, non-linear structures. It outperforms K-Means and DBSCAN on "connected" shapes (like spirals).

### Domains & Use Cases
1.  **Image Segmentation**: Cutting an image into foreground and background (Normalized Cuts).
2.  **Social Community Detection**: Finding groups of friends in a social graph.
3.  **Speech Separation**: Cocktail party problem.

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Clustering.
*   **Structure**: Graph-based.
*   **Probabilistic/Deterministic**: Deterministic (mostly).
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### The Similarity Graph
Construct an Adjacency Matrix $W$ where $W_{ij}$ is the similarity between $i$ and $j$ (e.g., Gaussian Kernel).

### The Laplacian Matrix
$$ L = D - W $$
*   $D$: Degree Matrix (Diagonal matrix where $D_{ii} = \sum_j W_{ij}$).
*   $L$: Unnormalized Laplacian.

### Eigenvectors
We compute the eigenvalues and eigenvectors of $L$.
*   The eigenvector corresponding to the **second smallest eigenvalue** (the Fiedler vector) contains the information needed to split the graph into two.
*   For $k$ clusters, we use the first $k$ eigenvectors.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Similarity Matrix**: Compute $W$ (e.g., using Nearest Neighbors or RBF Kernel).
2.  **Laplacian**: Compute $L = D - W$.
3.  **Eigen Decomposition**: Compute the first $k$ eigenvectors of $L$.
4.  **New Feature Space**: Form a matrix $U$ using these eigenvectors as columns. The rows of $U$ are the new representation of the data points.
5.  **Cluster**: Run K-Means on the rows of $U$.

### Pseudocode
```python
W = compute_similarity_graph(X)
D = diag(sum(W, axis=1))
L = D - W
eigenvals, eigenvecs = eig(L)
U = eigenvecs[:, :k] # Top k vectors
labels = KMeans(k).fit_predict(U)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Connectivity**: Assumes clusters are connected components in a graph.

### Hyperparameters
1.  **`n_clusters` (k)**:
    *   *Description*: Number of clusters.
    *   *Impact*: Must be known.
2.  **`affinity`**:
    *   *Options*: 'rbf' (Gaussian), 'nearest_neighbors'.
    *   *Recommendation*: 'nearest_neighbors' creates a sparse graph and is often more robust.
3.  **`gamma`**:
    *   *Description*: Kernel coefficient for RBF. Controls connectivity range.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**. Distance-based construction of the graph requires scaling.

### 2. Encoding
*   **Required**.

### 3. Complexity
*   **Warning**: Eigen decomposition is $O(N^3)$. Very slow for $N > 2,000$.
*   **Solution**: Use Nystroem method or other approximations for large data.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Complex Shapes**: Handles non-convex, connected shapes perfectly.
2.  **Graph Theory**: Theoretically sound foundation.

### Limitations
1.  **Scalability**: Extremely slow for large datasets.
2.  **Parameter Sensitive**: Sensitive to the choice of similarity graph (k-NN vs RBF) and parameters (gamma).
3.  **Needs k**: Must specify number of clusters.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Silhouette Score.

### Python Implementation
```python
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# 1. Generate Non-Linear Data (Circles)
X, _ = make_circles(n_samples=500, factor=0.5, noise=0.05)

# 2. Train
# affinity='nearest_neighbors' is usually robust
sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = sc.fit_predict(X)

# 3. Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Spectral Clustering')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Tune `gamma` (if RBF) or `n_neighbors` (if NN).
Visual inspection is often the best way for Spectral Clustering, as standard metrics (Silhouette) assume convex clusters and might penalize the correct "Ring" shapes.

### Real-World Applications
*   **VLSI Design**: Partitioning circuits to minimize wire length between blocks.
*   **Image Segmentation**: Normalized Cuts.

### When to Use
*   Small datasets ($N < 2000$).
*   Complex, non-linear shapes (Rings, Spirals).
*   Graph data (Social networks).

### When NOT to Use
*   Large datasets.
*   Simple blobs (K-Means is faster).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs K-Means**: Spectral Clustering projects data to a space where K-Means works. It is a wrapper around K-Means.
*   **vs DBSCAN**: DBSCAN is faster and finds $k$ automatically. Spectral is better if you know $k$ but the density varies.

### Interview Questions
1.  **Q: What is the Laplacian Matrix?** (It is a matrix representation of a graph, $L = D - W$. Its spectrum reveals graph properties).
2.  **Q: Why use the second smallest eigenvalue?** (The smallest is always 0. The second smallest (Fiedler value) approximates the minimum cut of the graph).

### Summary
Spectral Clustering is the bridge between Linear Algebra and Clustering. By changing the representation of the data (Spectral Embedding), it turns impossible clustering problems into trivial K-Means problems.

### Cheatsheet
*   **Type**: Graph-based.
*   **Key**: Eigenvalues of Laplacian.
*   **Param**: `n_clusters`, `affinity`.
*   **Pros**: Complex Shapes.
*   **Cons**: Slow ($O(N^3)$).
