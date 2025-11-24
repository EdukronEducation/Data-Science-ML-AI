# Mean Shift Clustering: The Hill Climber

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Mean Shift** is a non-parametric, centroid-based clustering algorithm. Unlike K-Means, it does **not** require you to specify the number of clusters ($k$) beforehand. It works by shifting data points towards the mode (highest density) of the data distribution.

### What Problem It Solves
It solves the problem of finding the "natural" number of clusters in a dataset. It is particularly good for **Mode Seeking** — finding the peaks in the data density.

### Core Idea
Imagine the data points are hikers on a foggy mountain range.
1.  Every hiker looks around them (within a certain bandwidth).
2.  They take a step towards the area where the fog is densest (where most other hikers are).
3.  They keep repeating this.
4.  Eventually, all hikers will converge at the peaks of the mountains.
5.  All hikers who end up at the same peak belong to the same cluster.

### Intuition: The "Party" Analogy
Imagine a large room full of people.
1.  Everyone looks for the nearest group of people.
2.  They move towards the center of that group.
3.  Eventually, distinct "clumps" of people form around the most popular spots (the food table, the DJ, the bar).
4.  Mean Shift finds these clumps automatically.

### Visual Interpretation
*   **Density Estimation**: It uses Kernel Density Estimation (KDE) to create a smooth surface over the data.
*   **Gradient Ascent**: It climbs the hill of this surface to find the local maxima (modes).

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
The main selling point is that it is **k-agnostic**. You don't need to guess if there are 3 clusters or 5. It finds them for you. It is also robust to non-spherical shapes (to some extent).

### Domains & Use Cases
1.  **Computer Vision**: Image Segmentation. It is famous for smoothing images and grouping pixels of similar color/texture.
2.  **Object Tracking**: Tracking a moving object in a video sequence (CamShift algorithm).
3.  **Mode Seeking**: Finding the most probable outcomes in a distribution.

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Clustering.
*   **Structure**: Density-based / Centroid-based.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Kernel Density Estimation (KDE)
We place a kernel (window) $K$ on each point $x$. The kernel is usually a Gaussian window.
The density at point $x$ is the sum of these kernels.

### The Mean Shift Vector
For a point $x$, the Mean Shift vector $m(x)$ points towards the direction of maximum increase in density.
$$ m(x) = \frac{\sum_{x_i \in N(x)} K(x_i - x) x_i}{\sum_{x_i \in N(x)} K(x_i - x)} - x $$
*   **Numerator**: Weighted average of neighbors.
*   **Denominator**: Sum of weights.
*   **Result**: The difference between the weighted mean and the current position.

### Update Rule
$$ x_{new} = x_{old} + m(x_{old}) $$
We simply move the point to the weighted mean of its neighbors.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Initialization**: Start a window (kernel) at each data point.
2.  **Compute Mean**: Calculate the mean of all points inside the window.
3.  **Shift**: Move the window center to the calculated mean.
4.  **Repeat**: Keep shifting until the window stops moving (converges to a peak).
5.  **Cluster**:
    *   All points that converge to the same peak are assigned to the same cluster.
    *   Peaks that are very close to each other are merged.

### Pseudocode
```python
for point in data:
    centroid = point
    while True:
        neighbors = get_points_in_bandwidth(centroid)
        new_centroid = mean(neighbors)
        if distance(new_centroid, centroid) < threshold:
            break
        centroid = new_centroid
    assign_cluster(point, centroid)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Density Gradient**: Assumes clusters are regions of higher density separated by regions of lower density.
*   **Bandwidth**: The most critical assumption is the "scale" of the clusters.

### Hyperparameters
1.  **`bandwidth`**:
    *   *Description*: The radius of the kernel window.
    *   *Impact*:
        *   **Small Bandwidth**: Many small peaks. Over-segmentation. (Rugged terrain).
        *   **Large Bandwidth**: Few large peaks. Under-segmentation. (Smooth terrain).
    *   *Auto-detection*: Sklearn has `estimate_bandwidth` function.
2.  **`bin_seeding`**:
    *   *Description*: Optimization. Discretize points into bins to speed up neighbor search.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**. Since it uses a fixed bandwidth (radius), features must be on the same scale.

### 2. Encoding
*   **Required**.

### 3. Dimensionality
*   **Warning**: Mean Shift is computationally expensive ($O(N^2)$). It struggles with high dimensions because the volume of the sphere grows exponentially, making the "density" meaningless.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **No k required**: Automatically discovers the number of clusters.
2.  **Robust to Outliers**: Outliers form their own tiny clusters or get absorbed.
3.  **Flexible Shapes**: Can handle non-spherical clusters better than K-Means.

### Limitations
1.  **Scalability**: Very slow. $O(T \cdot N^2)$. Not suitable for large datasets (> 10k rows).
2.  **Bandwidth Selection**: The result is extremely sensitive to the bandwidth parameter.
3.  **High Dimensions**: Fails in high-dim space.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Silhouette Score.

### Python Implementation
```python
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. Generate Data
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=0)

# 2. Estimate Bandwidth (Critical Step)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=200)
print(f"Estimated Bandwidth: {bandwidth:.2f}")

# 3. Train
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 4. Count Clusters
n_clusters_ = len(np.unique(labels))
print(f"Number of estimated clusters: {n_clusters_}")

# 5. Visualize
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x', s=200)
plt.title('Mean Shift Clustering')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
The only parameter to tune is `bandwidth`.
Use `estimate_bandwidth` with different `quantile` values.
*   `quantile=0.2`: Local view (More clusters).
*   `quantile=0.5`: Global view (Fewer clusters).

### Real-World Applications
*   **Image Segmentation**: In Photoshop/GIMP, "Posterize" or "Smooth" often uses Mean Shift logic to group colors.

### When to Use
*   Small datasets.
*   When you have no clue about $k$.
*   Computer Vision tasks.

### When NOT to Use
*   Large datasets (It will hang forever).
*   High dimensions.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs K-Means**: K-Means is fast but needs $k$. Mean Shift is slow but finds $k$.
*   **vs DBSCAN**: Both find $k$ automatically. DBSCAN is faster and handles noise better. Mean Shift is better for finding the "mode" or center of density.

### Interview Questions
1.  **Q: What is the complexity of Mean Shift?** ( $O(T \cdot N^2)$ ).
2.  **Q: How does bandwidth affect the result?** (Small bandwidth = jagged surface, many clusters. Large bandwidth = smooth surface, few clusters).

### Summary
Mean Shift is a powerful, mathematically elegant algorithm for finding the "peaks" of a distribution. While too slow for big data, it is indispensable in image processing and small-scale pattern recognition.

### Cheatsheet
*   **Type**: Density/Centroid.
*   **Key**: Hill Climbing.
*   **Param**: `bandwidth`.
*   **Pros**: Finds k automatically.
*   **Cons**: Slow.
