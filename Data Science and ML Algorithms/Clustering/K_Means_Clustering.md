# K-Means Clustering: The Centroid Seeker

## 📄 Page 1 — Introduction / Intuition

### Introduction
**K-Means Clustering** is the most popular and widely used unsupervised learning algorithm for partitioning a dataset into a set of $k$ distinct, non-overlapping groups (clusters). It is a centroid-based algorithm, meaning it represents each cluster by a central point called a **centroid**.

### What Problem It Solves
It solves the problem of **grouping similar data points together** without any prior knowledge of the labels.
*   "How can I segment my customers into 3 distinct groups based on spending behavior?"
*   "How can I compress this image by reducing the number of colors?"
*   "How can I group news articles into topics?"

### Core Idea
The core idea is to find $k$ centroids such that every data point is closer to its own cluster's centroid than to any other cluster's centroid. It minimizes the **within-cluster variance** (also known as inertia).

### Intuition: The "Pizza Delivery" Analogy
Imagine you own a pizza chain and you want to open 3 delivery hubs in a city to serve all your customers efficiently.
1.  **Initialization**: You randomly place 3 hubs on the map.
2.  **Assignment**: Every customer calls the hub closest to them. Now you have 3 groups of customers.
3.  **Update**: You realize the hubs are not in the center of their customer groups. You move each hub to the exact geographic center (average location) of its customers.
4.  **Repeat**: Now that the hubs moved, some customers might be closer to a different hub. They switch. You move the hubs again.
5.  **Convergence**: Eventually, the hubs stop moving because they are perfectly centered.

### Visual Interpretation
*   **Step 0**: Data points are scattered. Centroids are random.
*   **Step 1**: Points are colored based on the nearest centroid. The map looks like a Voronoi diagram (polygons).
*   **Step 2**: Centroids migrate towards the dense regions of their color.
*   **Final**: Centroids sit in the middle of "blobs" of data.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
K-Means is the "Hello World" of clustering. It is mathematically simple, computationally efficient (linear complexity), and easy to interpret. It serves as a strong baseline for any clustering task.

### Domains & Use Cases
1.  **Customer Segmentation**: Grouping customers by Recency, Frequency, and Monetary value (RFM Analysis) to target marketing campaigns.
2.  **Image Compression**: Reducing an image with millions of colors to just $k$ colors (e.g., 64 colors) by clustering pixel values.
3.  **Anomaly Detection**: Points that are very far from their cluster centroid can be considered anomalies.
4.  **Document Clustering**: Grouping similar documents using TF-IDF features.
5.  **Astronomy**: Classifying stars or galaxies.

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Clustering.
*   **Structure**: Flat (not hierarchical).
*   **Probabilistic/Deterministic**: Deterministic (if initialization is fixed), but usually Stochastic (due to random initialization).
*   **Parametric**: Non-parametric (though $k$ is a hyperparameter).

---

## 📄 Page 3 — Mathematical Foundation

### Core Objective Function
K-Means aims to minimize the **Inertia** or **Within-Cluster Sum of Squares (WCSS)**.
$$ J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 $$
*   $k$: Number of clusters.
*   $C_i$: The set of points belonging to cluster $i$.
*   $\mu_i$: The centroid (mean) of cluster $i$.
*   $||x - \mu_i||^2$: Squared Euclidean distance between a point and its centroid.

### Distance Metric
The standard K-Means algorithm uses **Squared Euclidean Distance**.
$$ d(p, q)^2 = \sum_{j=1}^{D} (p_j - q_j)^2 $$
*   *Note*: If you use Manhattan distance, the centroid becomes the *median* (K-Medians). If you use Cosine distance, it becomes Spherical K-Means.

### Optimization
The problem of finding the global minimum for K-Means is **NP-Hard**. However, Lloyd's Algorithm (the standard implementation) is a heuristic that converges to a **local minimum**. This is why we run it multiple times with different starting points.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm (Lloyd's Algorithm)

1.  **Initialization**: Choose $k$ random points as initial centroids. (Or use K-Means++ for smarter initialization).
2.  **Assignment Step (Expectation)**:
    *   Loop through every data point $x$.
    *   Calculate the distance from $x$ to all $k$ centroids.
    *   Assign $x$ to the cluster of the closest centroid.
3.  **Update Step (Maximization)**:
    *   Loop through every cluster $i$.
    *   Calculate the new centroid $\mu_i$ by taking the **mean** of all points currently assigned to cluster $i$.
    $$ \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x $$
4.  **Convergence Check**:
    *   Did the centroids move?
    *   Did the assignments change?
    *   If No, stop. If Yes, repeat from Step 2.

### Pseudocode
```python
centroids = random_init(k)
while not converged:
    # Assignment
    clusters = []
    for point in data:
        closest_centroid = argmin(distance(point, centroids))
        clusters[closest_centroid].append(point)
    
    # Update
    new_centroids = []
    for i in range(k):
        new_centroids.append(mean(clusters[i]))
    
    if new_centroids == centroids:
        break
    centroids = new_centroids
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions (The "Spherical" Assumption)
K-Means makes strong assumptions about the shape of the clusters:
1.  **Spherical**: It assumes clusters are round blobs. It fails on elongated, crescent, or irregular shapes.
2.  **Similar Size**: It assumes clusters have roughly the same number of points.
3.  **Similar Density**: It assumes clusters have similar density.
4.  **Linearly Separable**: It assumes clusters can be separated by linear boundaries (Voronoi cells).

### Hyperparameters
1.  **`n_clusters` (k)**:
    *   *Description*: The number of clusters to form.
    *   *Impact*: The most critical parameter. Wrong $k$ = meaningless results.
2.  **`init`**:
    *   *Options*: 'random' or 'k-means++'.
    *   *Recommendation*: Always use **'k-means++'**. It spreads out initial centroids to ensure faster and better convergence.
3.  **`n_init`**:
    *   *Description*: Number of times to run the algorithm with different centroid seeds.
    *   *Default*: 10. The algorithm returns the best result (lowest Inertia) among these runs.
4.  **`max_iter`**:
    *   *Description*: Maximum number of iterations for a single run.
    *   *Default*: 300.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization (MANDATORY)
*   **Why?**: K-Means is a distance-based algorithm. If one feature is "Salary" (range 20k-200k) and another is "Age" (range 20-80), distance calculations will be completely dominated by Salary. Age will be ignored.
*   **Action**: Apply `StandardScaler` or `MinMaxScaler` to all features before clustering.

### 2. Encoding
*   **Categorical Data**: K-Means does **not** handle categorical data natively. Euclidean distance is meaningless for "Red" vs "Blue".
*   **Action**: Use One-Hot Encoding (if sparse) or switch to K-Modes / K-Prototypes algorithms which are designed for categorical data.

### 3. Outlier Treatment
*   **Sensitivity**: K-Means is **highly sensitive** to outliers. Since the centroid is the *mean*, a single massive outlier can pull the centroid miles away from the actual cluster.
*   **Action**: Remove outliers or use K-Medians.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Speed**: It is very fast. Complexity is $O(n \cdot k \cdot i \cdot d)$ where $n$=points, $k$=clusters, $i$=iterations, $d$=dimensions. This is linear with respect to data size.
2.  **Scalability**: Can handle large datasets (Mini-Batch K-Means scales to millions of rows).
3.  **Simplicity**: Easy to understand and implement.
4.  **Convergence**: Guaranteed to converge (though maybe to a local minimum).

### Limitations
1.  **Choosing k**: You must specify $k$ in advance.
2.  **Spherical Assumption**: Fails on non-convex shapes (e.g., concentric circles, moons).
3.  **Outliers**: Sensitive to noise.
4.  **Initialization**: Random initialization can lead to bad results (fixed by K-Means++).
5.  **Local Minima**: Can get stuck in suboptimal solutions.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics (How to choose k?)
Since we don't have labels, we can't use Accuracy.
1.  **Elbow Method**: Plot Inertia vs $k$. Look for the "elbow" point where the decrease in inertia slows down.
2.  **Silhouette Score**: Measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation).
    *   Range: -1 to +1.
    *   +1: Perfect clustering.
    *   0: Overlapping clusters.
    *   -1: Wrong assignment.

### Python Implementation
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. Generate Data
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# 2. Scale (Always!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# 4. Train Final Model
kmeans = KMeans(n_clusters=4, init='k-means++')
y_kmeans = kmeans.fit_predict(X_scaled)

# 5. Evaluate
print(f"Silhouette Score: {silhouette_score(X_scaled, y_kmeans):.2f}")

# 6. Visualize
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title('K-Means Clustering')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
We don't "tune" K-Means like a classifier, but we do search for the optimal $k$.
```python
# Automated k selection using Silhouette Analysis
best_k = 2
best_score = -1

for k in range(2, 6):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k={k}, Silhouette Score={score:.3f}")
    if score > best_score:
        best_k = k
        best_score = score
```

### Real-World Applications
1.  **Image Segmentation**: Self-driving cars use it to group pixels into "Road", "Sky", "Car".
2.  **Search Engines**: Clustering search results to present "topics" to the user.
3.  **Cybersecurity**: Clustering network traffic to find "normal" patterns vs anomalies.

### When to Use
*   When you have a general idea of how many groups exist ($k$).
*   When clusters are likely spherical and distinct.
*   When you need a fast algorithm for large data.

### When NOT to Use
*   When clusters have irregular shapes (use DBSCAN).
*   When clusters have varying sizes/densities (use GMM).
*   When there are many outliers (use DBSCAN).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Hierarchical**: K-Means is faster ($O(N)$) but requires $k$. Hierarchical is slower ($O(N^2)$) but produces a dendrogram and doesn't need $k$ upfront.
*   **vs DBSCAN**: K-Means assumes spheres and assigns every point. DBSCAN finds arbitrary shapes and handles noise (unassigned points).
*   **vs GMM**: K-Means is a "hard" clustering (point belongs to A). GMM is "soft" (point is 70% A, 30% B). K-Means is actually a special case of GMM.

### Interview Questions
1.  **Q: Will K-Means always give the same result?**
    *   A: No. It depends on the random initialization of centroids. That's why we run it multiple times (`n_init`).
2.  **Q: What happens if you have outliers?**
    *   A: The centroid will be pulled towards the outlier, potentially ruining the cluster.
3.  **Q: Can K-Means handle non-linear boundaries?**
    *   A: No, the decision boundaries are linear (Voronoi tessellations). You would need Kernel K-Means or Spectral Clustering for that.

### Summary
K-Means is the workhorse of clustering. It is simple, fast, and effective for many standard problems. While it relies on strong assumptions (spherical blobs), it remains the first algorithm you should try when exploring a new unlabeled dataset.

### Cheatsheet
*   **Type**: Centroid-based.
*   **Param**: `n_clusters` (k).
*   **Metric**: Inertia (WCSS).
*   **Assumption**: Spherical, Equal Variance.
*   **Pros**: Fast, Scalable.
*   **Cons**: Needs k, Sensitive to Outliers.
