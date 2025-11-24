# t-SNE (t-Distributed Stochastic Neighbor Embedding): The Visualizer

## 📄 Page 1 — Introduction / Intuition

### Introduction
**t-SNE** is a non-linear dimensionality reduction technique well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions.

### What Problem It Solves
It solves the **Visualization** problem for complex, non-linear data.
*   PCA squashes everything onto a flat sheet. It crushes complex manifolds (like a Swiss Roll) into a mess.
*   t-SNE unrolls the manifold. It preserves **Local Structure**.

### Core Idea
"Keep neighbors close."
1.  **High-D**: Measure similarity between points. If two points are close, probability is high. If far, probability is low.
2.  **Low-D**: Place points randomly on a 2D sheet.
3.  **Move**: Move the 2D points around until their 2D similarities match their High-D similarities.

### Intuition: The "Elastic Band" Analogy
Imagine all your data points are connected by elastic bands.
*   Close neighbors have strong, short bands pulling them together.
*   Far points have weak, long bands.
t-SNE lets the points snap into a comfortable arrangement on a 2D table where the tension in the bands is minimized.

### Visual Interpretation
*   **PCA**: A shadow of the object.
*   **t-SNE**: A deconstructed map of the object. It tears the data apart to flatten it, creating distinct clusters.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the state-of-the-art for **Data Visualization**. If you see a beautiful 2D plot of MNIST digits or gene clusters, it is almost certainly t-SNE (or UMAP).

### Domains & Use Cases
1.  **Bioinformatics**: Single-cell RNA sequencing visualization.
2.  **Computer Vision**: Visualizing the internal layers of a Deep Neural Network (CNN).
3.  **NLP**: Visualizing Word Embeddings (Word2Vec).

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Dimensionality Reduction (Visualization).
*   **Linearity**: Non-Linear.
*   **Probabilistic/Deterministic**: Stochastic (Random).
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### High-Dimensional Similarity (Gaussian)
We calculate the conditional probability $p_{j|i}$ that point $x_i$ would pick $x_j$ as its neighbor using a Gaussian distribution centered at $x_i$.
$$ p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)} $$

### Low-Dimensional Similarity (Student's t)
In the 2D map, we use a **Student's t-distribution** (with 1 degree of freedom) instead of Gaussian.
$$ q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}} $$
*   **Why t-distribution?** It has "Heavy Tails". This solves the **Crowding Problem**. It allows distant points to be placed very far apart in 2D without crushing the local clusters.

### Cost Function (KL Divergence)
We minimize the Kullback-Leibler divergence between $P$ (High-D) and $Q$ (Low-D).
$$ C = \sum P_i \log \frac{P_i}{Q_i} $$
We use Gradient Descent to move the points $y$ to minimize $C$.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Compute Pairwise Affinities (High-D)**: Calculate distances between all pairs of points in original space. Convert to probabilities $P$.
2.  **Initialize Low-D**: Randomly place points in 2D space.
3.  **Compute Pairwise Affinities (Low-D)**: Calculate distances in 2D. Convert to probabilities $Q$ using t-distribution.
4.  **Gradient Descent**:
    *   Compare $P$ and $Q$.
    *   Calculate gradient of KL Divergence.
    *   Move points to make $Q$ look more like $P$.
    *   Attraction: Pull similar points together.
    *   Repulsion: Push dissimilar points apart.
5.  **Repeat**: For 1000+ iterations.

### Pseudocode
```python
P = compute_high_d_probs(X, perplexity)
Y = random_init(N, 2)
for i in range(iterations):
    Q = compute_low_d_probs(Y)
    gradient = calc_gradient(P, Q)
    Y = Y - learning_rate * gradient
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Local Structure**: Assumes local neighbors matter more than global geometry.
*   **Distance**: Assumes Euclidean distance is valid in High-D (which is debatable).

### Hyperparameters
1.  **`perplexity`**:
    *   *Description*: The number of effective neighbors.
    *   *Impact*:
        *   Low (5): Small, fragmented clusters.
        *   High (50): Global structure is preserved better, clusters merge.
        *   *Default*: 30.
2.  **`learning_rate`**:
    *   *Description*: Step size for Gradient Descent.
    *   *Impact*: Too high = Ball of points. Too low = Compressed cloud.
3.  **`n_iter`**:
    *   *Description*: Number of iterations. Needs to be high (1000+).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**.

### 2. Dimensionality Reduction (PCA)
*   **Recommended**: t-SNE is $O(N^2)$. It is slow.
*   It is standard practice to run PCA first to reduce dimensions to ~50, and then run t-SNE on those 50 components.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Unmatched Visualization**: Produces the most distinct, readable clusters of any algorithm.
2.  **Non-Linear**: Unfolds complex manifolds.

### Limitations
1.  **Slow**: Very computationally expensive.
2.  **Stochastic**: Different runs give different plots.
3.  **No Transformation**: You cannot map *new* data to the existing t-SNE plot. You have to re-run the whole thing.
4.  **Misleading**:
    *   Cluster size in t-SNE means nothing.
    *   Distance between clusters means nothing.
    *   It hallucinates structure in random noise.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **KL Divergence**: The final loss value. Lower is better.

### Python Implementation
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load Data (Digits)
data = load_digits()
X = data.data
y = data.target

# 2. PCA First (Speed up)
X_pca = PCA(n_components=50).fit_transform(X)

# 3. t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# 4. Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=10)
plt.colorbar()
plt.title('t-SNE of MNIST Digits')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Tune `perplexity`.
Try values: 5, 30, 50, 100.
*   Watch how the clusters form.
*   If it looks like a "ball", increase perplexity/learning rate.
*   If it looks like "confetti", decrease perplexity.

### Real-World Applications
*   **Cybersecurity**: Visualizing malware signatures.
*   **Music Analysis**: Mapping songs by audio similarity.

### When to Use
*   **Visualization ONLY**.
*   When you want to show off distinct clusters in a presentation.

### When NOT to Use
*   **Feature Engineering**: Do not use t-SNE output as input to a classifier (use PCA/UMAP instead).
*   **Large Data**: Too slow (use UMAP).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs PCA**: PCA preserves global structure (variance). t-SNE preserves local structure (neighbors).
*   **vs UMAP**: UMAP is faster, preserves more global structure, and is deterministic. UMAP is generally replacing t-SNE.

### Interview Questions
1.  **Q: Why use t-distribution instead of Gaussian in Low-D?** (To solve the Crowding Problem. Heavy tails allow more space for distant points).
2.  **Q: Can you interpret the distance between clusters in t-SNE?** (No. t-SNE distorts global distances to preserve local ones).

### Summary
t-SNE is the artist of dimensionality reduction. It creates beautiful, informative maps of high-dimensional data. However, it is a "black box" visualization tool, not a feature engineering tool.

### Cheatsheet
*   **Type**: Non-linear Visualization.
*   **Key**: Perplexity, t-distribution.
*   **Pros**: Best Clusters.
*   **Cons**: Slow, Random, Misleading.
