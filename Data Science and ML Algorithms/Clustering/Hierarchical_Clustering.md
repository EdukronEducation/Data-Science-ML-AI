# Hierarchical Clustering: The Taxonomy Builder

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Hierarchical Clustering** is a method of cluster analysis which seeks to build a **hierarchy** of clusters. Unlike K-Means, which partitions data into a flat set of groups, Hierarchical Clustering builds a tree-like structure called a **Dendrogram**.

### What Problem It Solves
It solves the problem of **grouping data at different levels of granularity**.
*   "I want to group animals. At a high level, I want 'Mammals vs Reptiles'. At a low level, I want 'Cats vs Dogs'."
*   "I want to understand the evolutionary relationship between species."

### Core Idea
There are two main types:
1.  **Agglomerative (Bottom-Up)**: "The Merger". Start with every point as its own cluster. At each step, merge the two closest clusters until only one giant cluster remains.
2.  **Divisive (Top-Down)**: "The Splitter". Start with one giant cluster containing all points. At each step, split the cluster until every point is its own cluster. (Less common).

### Intuition: The "Family Tree" Analogy
Imagine tracing your ancestry.
*   **Step 1**: You and your sibling are a cluster.
*   **Step 2**: Your cluster merges with your cousins' cluster.
*   **Step 3**: That group merges with second cousins.
*   **Final Step**: Everyone is merged into the "Human Family".
The algorithm builds this family tree based on similarity (DNA).

### Visual Interpretation
*   **Dendrogram**: A tree diagram. The y-axis represents the distance (dissimilarity) at which clusters merge.
*   **Cutting the Tree**: You can draw a horizontal line across the dendrogram at any height to "cut" the tree and get a specific number of clusters.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
The biggest advantage is that **you do not need to specify the number of clusters ($k$) beforehand**. You can look at the dendrogram and decide what makes sense. It also reveals the nested structure of the data.

### Domains & Use Cases
1.  **Bioinformatics**: Phylogenetic trees (evolutionary history of organisms).
2.  **Taxonomy**: Classifying plants or animals into Kingdom, Phylum, Class, etc.
3.  **Social Network Analysis**: Finding communities within communities.
4.  **Customer Segmentation**: Finding broad segments (e.g., "High Spenders") and sub-segments (e.g., "High Spenders who like Tech").

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Clustering.
*   **Structure**: Hierarchical (Tree).
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Distance Metrics (Between Points)
First, we need to measure distance between two points $p$ and $q$.
*   **Euclidean**: Straight line.
*   **Manhattan**: Grid-like.
*   **Cosine**: Angle (good for text).

### Linkage Criteria (Between Clusters)
Crucially, we need to define the distance between **two clusters** (sets of points). This is called **Linkage**.
1.  **Single Linkage (Min)**: Distance between the two *closest* points in the clusters.
    *   *Effect*: Produces long, "chain-like" clusters. Good for non-spherical shapes.
2.  **Complete Linkage (Max)**: Distance between the two *farthest* points in the clusters.
    *   *Effect*: Produces compact, spherical clusters. Sensitive to outliers.
3.  **Average Linkage**: Average distance between all pairs of points.
    *   *Effect*: Robust compromise.
4.  **Ward's Method**: Minimizes the increase in variance (Inertia) when merging.
    *   *Effect*: Most similar to K-Means. Produces balanced, spherical clusters. **Default choice**.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of Agglomerative Clustering

1.  **Initialization**: Treat each of the $N$ data points as a single cluster. We have $N$ clusters.
2.  **Distance Matrix**: Calculate the $N \times N$ distance matrix between all clusters.
3.  **Merge**:
    *   Find the two clusters with the smallest distance (based on Linkage).
    *   Merge them into a new cluster.
    *   Now we have $N-1$ clusters.
4.  **Update**: Update the distance matrix to calculate distances between the new cluster and all remaining clusters.
5.  **Repeat**: Repeat steps 3-4 until only 1 cluster remains.
6.  **Output**: A Dendrogram representing the history of merges.

### Pseudocode
```python
clusters = [Point(i) for i in data]
dendrogram = []

while len(clusters) > 1:
    c1, c2 = find_closest_pair(clusters, linkage='ward')
    new_cluster = merge(c1, c2)
    
    dendrogram.record_merge(c1, c2, distance)
    
    clusters.remove(c1)
    clusters.remove(c2)
    clusters.append(new_cluster)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Connectivity**: Assumes points closer together belong together.
*   **Metric**: The choice of distance metric (Euclidean vs Manhattan) drastically changes the result.

### Hyperparameters
1.  **`n_clusters`**:
    *   *Description*: The number of clusters to find (by cutting the tree).
    *   *Note*: Optional. You can use `distance_threshold` instead.
2.  **`linkage`**:
    *   *Options*: 'ward' (default), 'complete', 'average', 'single'.
    *   *Impact*: Determines the shape of clusters. 'ward' is usually best for general use.
3.  **`affinity` (metric)**:
    *   *Options*: 'euclidean', 'manhattan', 'cosine'.
    *   *Impact*: Defines "similarity".

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization (MANDATORY)
*   **Why?**: Like K-Means, it relies entirely on distance. Unscaled features will distort the hierarchy.
*   **Action**: Use `StandardScaler`.

### 2. Encoding
*   **Required**. Distance metrics need numbers.

### 3. Dimensionality Reduction
*   **Recommended**: Calculating the distance matrix is $O(N^2)$. If you have thousands of features, this is slow. PCA is often used before Hierarchical Clustering.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **No k required**: You don't need to guess the number of clusters. You can visualize the tree and pick the best cut.
2.  **Interpretability**: The dendrogram is a powerful visualization tool.
3.  **Shape Flexibility**: With Single Linkage, it can find non-spherical shapes.

### Limitations
1.  **Scalability**: It is **Slow**. Complexity is $O(N^2 \log N)$ or $O(N^3)$. It is practically unusable for datasets > 10,000 rows.
2.  **Irreversible**: Once two clusters are merged, they can never be separated. If the algorithm makes a "bad merge" early on, it propagates to the end.
3.  **Sensitivity**: Sensitive to noise and outliers (especially Complete Linkage).

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
1.  **Cophenetic Correlation Coefficient**: Measures how faithfully the dendrogram preserves the pairwise distances between the original data points.
2.  **Silhouette Score**.

### Python Implementation
```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Generate Data
X, _ = make_blobs(n_samples=50, centers=3, random_state=42) # Small N for visualization

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Plot Dendrogram (Scipy)
plt.figure(figsize=(10, 7))
plt.title("Customer Dendrogram")
# linkage matrix contains the merge history
Z = linkage(X_scaled, method='ward') 
dendrogram(Z)
plt.axhline(y=5, color='r', linestyle='--') # Cut line
plt.show()

# 4. Train Model (Sklearn)
# Based on dendrogram, we choose k=3
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_hc = hc.fit_predict(X_scaled)

# 5. Visualize Clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_hc, cmap='rainbow')
plt.title('Hierarchical Clustering')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
The "Tuning" here is mostly visual.
1.  Run with `linkage='ward'`. Plot Dendrogram.
2.  Does the tree look balanced?
3.  Try `linkage='average'`. Does it look better?
4.  Choose the cut height (y-axis) that crosses the longest vertical lines (representing large distance jumps between merges).

### Real-World Applications
*   **Gene Expression Analysis**: Grouping genes with similar behavior.
*   **Document Hierarchy**: Organizing a library of books into topics and sub-topics.

### When to Use
*   **Small Datasets**: $N < 5,000$.
*   When the **hierarchy** matters (e.g., biological taxonomy).
*   When you have no idea how many clusters there are.

### When NOT to Use
*   **Large Datasets**: It will crash your RAM and take forever. Use K-Means or DBSCAN.
*   When you need a real-time application.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs K-Means**:
    *   K-Means: Fast, Flat, Needs k, Random.
    *   Hierarchical: Slow, Tree, No k needed, Deterministic.
*   **vs DBSCAN**:
    *   DBSCAN handles noise better and is faster, but has trouble with varying densities. Hierarchical captures all points.

### Interview Questions
1.  **Q: What is the difference between Single and Complete Linkage?**
    *   A: Single = Min distance (chaining effect). Complete = Max distance (compact clusters).
2.  **Q: Why is Hierarchical Clustering not used for Big Data?**
    *   A: Because calculating the distance matrix for $N$ points requires $N^2$ memory and operations. For 1M points, that's 1 Trillion entries.
3.  **Q: How do you choose the number of clusters from a dendrogram?**
    *   A: Look for the longest vertical distance that doesn't cross any horizontal line. Draw a horizontal line there. The number of vertical lines it intersects is $k$.

### Summary
Hierarchical Clustering is the historian of algorithms. It painstakingly reconstructs the relationships between every single data point, building a beautiful family tree. It is computationally expensive but offers unmatched insight into the structure of small datasets.

### Cheatsheet
*   **Type**: Agglomerative vs Divisive.
*   **Key**: Dendrogram.
*   **Linkage**: Ward (Variance), Single (Min), Complete (Max).
*   **Pros**: No k needed, Hierarchy.
*   **Cons**: Very Slow ($O(N^2)$).
