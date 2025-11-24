# Affinity Propagation: The Democracy of Points

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Affinity Propagation (AP)** is a clustering algorithm based on the concept of "message passing" between data points. Unlike K-Means, which forces points to belong to a fixed number of centroids, AP lets the data points "vote" on who should be the representative (exemplar).

### What Problem It Solves
It solves the problem of **Exemplar-based Clustering** without specifying $k$.
It finds the most representative actual data points to serve as cluster centers.

### Core Idea
Imagine a room full of people trying to elect leaders.
1.  **Responsibility**: "Hey Bob, you would make a great leader for me because you are similar to me."
2.  **Availability**: "Thanks Alice, I am available to lead you because many other people also want me to lead them."
3.  They exchange these messages until a consensus is reached on who the leaders (exemplars) are.

### Intuition: The "Election" Analogy
*   Every data point is a candidate.
*   They send messages to each other.
*   Eventually, a few "Stars" emerge who have high support.
*   Everyone else attaches themselves to the nearest Star.

### Visual Interpretation
*   Lines (messages) are drawn between points.
*   Over time, the lines converge to a few central points.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is unique because the cluster centers are **actual data points** (exemplars), not imaginary means like in K-Means. This is useful when the "average" of a cluster doesn't make sense (e.g., the average of a "Cat" image and a "Dog" image is a blurry mess, but the exemplar is a specific Cat image).

### Domains & Use Cases
1.  **Gene Detection**: Finding representative genes.
2.  **Face Recognition**: Finding a representative face for a person from a set of photos.
3.  **Text Summarization**: Finding the most representative sentence in a document.

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Clustering.
*   **Structure**: Exemplar-based.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Matrices
We update two matrices:
1.  **Responsibility $R(i, k)$**: How well-suited point $k$ is to serve as the exemplar for point $i$.
2.  **Availability $A(i, k)$**: How appropriate it is for point $i$ to choose point $k$ as its exemplar.

### Update Rules
$$ R(i, k) \leftarrow S(i, k) - \max_{k' \neq k} \{ A(i, k') + S(i, k') \} $$
$$ A(i, k) \leftarrow \min \{ 0, R(k, k) + \sum_{i' \notin \{i, k\}} \max(0, R(i', k)) \} $$
*   $S(i, k)$: Similarity (negative squared distance).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Similarity Matrix**: Compute similarity between all pairs (usually negative Euclidean distance).
2.  **Preference**: Set the diagonal $S(k, k)$ to a "Preference" value. This controls how likely a point is to become an exemplar.
3.  **Message Passing**:
    *   Update Responsibility Matrix $R$.
    *   Update Availability Matrix $A$.
4.  **Convergence**: Repeat until matrices stabilize.
5.  **Assignment**:
    *   Exemplars are points where $R(k, k) + A(k, k) > 0$.
    *   Assign other points to the exemplar that maximizes $R + A$.

### Pseudocode
```python
S = -distance_matrix(data)
S[diag] = preference
while not converged:
    update_responsibility(S, A)
    update_availability(R)
exemplars = find_positive_diagonals(R + A)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Symmetry**: Assumes similarity is symmetric.

### Hyperparameters
1.  **`preference`**:
    *   *Description*: The most critical parameter. It controls the number of clusters.
    *   *Effect*:
        *   High Preference: Many clusters (Every point wants to be a leader).
        *   Low Preference: Few clusters.
        *   *Default*: Median of input similarities.
2.  **`damping`**:
    *   *Description*: Factor between 0.5 and 1.0.
    *   *Effect*: Prevents numerical oscillations during message passing.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**.

### 2. Encoding
*   **Required**.

### 3. Complexity
*   **Warning**: $O(N^2)$ memory and time. Do not use for $N > 10,000$.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **No k required**: Finds the number of clusters automatically based on `preference`.
2.  **Exemplars**: Returns actual data points as centers (Interpretability).
3.  **Initialization**: Not sensitive to initialization (Deterministic).

### Limitations
1.  **Speed**: Very slow. $O(N^2)$.
2.  **Memory**: Stores $N \times N$ matrices.
3.  **Tuning**: `preference` is hard to tune intuitively.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Silhouette Score.

### Python Implementation
```python
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generate Data
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)

# 2. Train
# preference=None defaults to median similarity
af = AffinityPropagation(preference=-50, random_state=0)
af.fit(X)

# 3. Get Exemplars
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

print(f"Estimated clusters: {n_clusters_}")

# 4. Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
for center_index in cluster_centers_indices:
    center = X[center_index]
    plt.scatter(center[0], center[1], c='red', marker='x', s=200)
plt.title('Affinity Propagation')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Tune `preference`.
*   Start with `preference = median(similarity)`.
*   If too many clusters -> Decrease preference (make it more negative).
*   If too few clusters -> Increase preference.

### Real-World Applications
*   **Stock Market**: Finding representative stocks for a sector.
*   **Image Search**: Grouping similar images and showing one "cover" image for each group.

### When to Use
*   Small to Medium datasets.
*   When you need "Exemplars" (real points) as centers.
*   When $k$ is unknown.

### When NOT to Use
*   Large datasets.
*   When you just need a simple partition (use K-Means).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs K-Medoids**: K-Medoids also finds exemplars but requires $k$. AP finds $k$.
*   **vs K-Means**: K-Means finds centroids (averages). AP finds exemplars (real points).

### Interview Questions
1.  **Q: What is the main advantage of AP over K-Means?** (It doesn't need $k$, and centers are real data points).
2.  **Q: Why is AP slow?** (Message passing involves matrix operations on $N \times N$ matrices).

### Summary
Affinity Propagation is a democratic clustering algorithm. It allows data points to elect their representatives. It is computationally heavy but produces high-quality, interpretable clusters for small datasets.

### Cheatsheet
*   **Type**: Exemplar-based.
*   **Key**: Message Passing.
*   **Param**: `preference`.
*   **Pros**: Finds k, Real centers.
*   **Cons**: Slow ($O(N^2)$).
