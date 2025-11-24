# Principal Component Analysis (PCA): The Compressor

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Principal Component Analysis (PCA)** is the most popular dimensionality reduction algorithm. It is a linear transformation technique that identifies the "Principal Components" (directions of maximum variance) in the data and projects the data onto these components.

### What Problem It Solves
It solves the **Curse of Dimensionality**.
*   **Visualization**: How do you visualize 50 variables? You can't. PCA reduces them to 2 or 3 so you can plot them.
*   **Efficiency**: Training a model on 1000 features is slow and prone to overfitting. PCA reduces this to 100 features with minimal loss of information.
*   **Noise Reduction**: By ignoring the components with low variance (noise), PCA cleans the data.

### Core Idea
"Find the best angle to take a photo."
Imagine a 3D object (a teapot). You want to take a 2D photo that captures the most detail.
*   If you take a photo from the top, it looks like a circle (Bad).
*   If you take a photo from the side, you see the handle and spout (Good).
PCA finds the "side view" automatically. It finds the axis where the data is most spread out (Variance).

### Intuition: The "Spread" Analogy
*   **Variance = Information**.
*   If everyone in a room is 25 years old, the variable "Age" has 0 variance and 0 information. We can drop it.
*   If ages range from 5 to 90, "Age" has high variance and high information. We keep it.
PCA constructs new variables (Principal Components) that maximize this variance.

### Visual Interpretation
1.  **PC1**: The line passing through the "length" of the data cloud (Maximum Variance).
2.  **PC2**: The line perpendicular (orthogonal) to PC1 that captures the next most variance (Width).
3.  **Projection**: We rotate the dataset so PC1 becomes the X-axis and PC2 becomes the Y-axis.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the standard first step in any pipeline with high-dimensional data. It is mathematically rigorous, deterministic, and reversible (mostly).

### Domains & Use Cases
1.  **Image Processing**: Eigenfaces. Compressing images by keeping only the top components.
2.  **Genomics**: Analyzing gene expression data (thousands of genes -> 2D plot).
3.  **Finance**: Yield curve analysis (Level, Slope, Curvature).
4.  **Preprocessing**: Speeding up regression/classification algorithms.

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Dimensionality Reduction.
*   **Linearity**: Linear.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Covariance Matrix
We start by calculating the Covariance Matrix $\Sigma$ of the centered data.
$$ \Sigma = \frac{1}{N-1} X^T X $$
This matrix tells us how every variable correlates with every other variable.

### Eigen Decomposition
We find the **Eigenvalues** ($\lambda$) and **Eigenvectors** ($v$) of the Covariance Matrix.
$$ \Sigma v = \lambda v $$
*   **Eigenvector ($v$)**: The direction of the new axis (Principal Component).
*   **Eigenvalue ($\lambda$)**: The amount of variance captured by that axis.

### Explained Variance Ratio
$$ \text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum \lambda} $$
This tells us "PC1 explains 40% of the variance, PC2 explains 20%..."

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Standardize**: Center the data (Mean=0, Variance=1). **Crucial Step**.
2.  **Covariance**: Compute the Covariance Matrix.
3.  **Decompose**: Calculate Eigenvalues and Eigenvectors.
4.  **Sort**: Sort Eigenvectors by Eigenvalues in descending order.
5.  **Select**: Choose the top $k$ Eigenvectors.
6.  **Project**: Transform the original data $X$ onto the new subspace $W$.
    $$ X_{new} = X \cdot W $$

### Pseudocode
```python
X_std = (X - mean(X)) / std(X)
cov_mat = cov(X_std)
eigen_vals, eigen_vecs = eig(cov_mat)
pairs = sort([(abs(val), vec) for val, vec in zip(eigen_vals, eigen_vecs)])
W = hstack([p[1] for p in pairs[:k]])
X_pca = X_std.dot(W)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Linearity**: Assumes the data lies on a linear subspace (plane). It cannot "unroll" a Swiss Roll shape (use Kernel PCA or t-SNE for that).
2.  **Orthogonality**: Assumes Principal Components are perpendicular.
3.  **Variance = Importance**: Assumes that the direction with most spread is the most interesting. (Not always true, e.g., in classification, the separation direction might have low variance).

### Hyperparameters
1.  **`n_components`**:
    *   *Integer*: Number of components to keep (e.g., 2).
    *   *Float*: Amount of variance to preserve (e.g., 0.95 means "Keep enough components to explain 95% of variance").
2.  **`svd_solver`**:
    *   'auto', 'full', 'arpack', 'randomized'. Randomized is faster for large data.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization (MANDATORY)
*   **CRITICAL**. PCA maximizes variance. If one variable is "Salary" (range 100,000) and another is "Age" (range 100), Salary has massive variance just because of the units. PCA will think Salary is the only important variable.
*   **Action**: You MUST use `StandardScaler` before PCA.

### 2. Encoding
*   **Required**.

### 3. Missing Values
*   Must be imputed. PCA cannot handle NaNs.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Noise Reduction**: Dropping lower components removes noise.
2.  **Visualization**: The only way to see 100D data.
3.  **Feature Independence**: The resulting Principal Components are uncorrelated (Orthogonal). This fixes Multicollinearity issues in Regression.

### Limitations
1.  **Linear Only**: Fails on non-linear manifolds.
2.  **Interpretability**: What does "0.5 * Age - 0.3 * Salary" mean? You lose the physical meaning of original features.
3.  **Outliers**: Sensitive to outliers (they stretch the variance).

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Cumulative Explained Variance**: The sum of explained variance ratios. We usually aim for 90% or 95%.

### Python Implementation
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Scale (MANDATORY)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train PCA
# Keep 2 components for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Explained Variance
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

# 5. Visualize
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Data')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning (Choosing k)
Plot the **Scree Plot**.
```python
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()
```
Find the number of components where the curve crosses 0.95 (95%).

### Real-World Applications
*   **Stock Portfolios**: Reducing hundreds of stock price movements to a few "Market Factors".
*   **Image Compression**: Storing only the top eigenvectors to reconstruct faces.

### When to Use
*   Before any regression/classification model to reduce overfitting.
*   To visualize high-dimensional data.
*   To remove multicollinearity.

### When NOT to Use
*   When interpretability is key (Lasso is better for feature selection).
*   When data is non-linear (Use t-SNE/UMAP).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs LDA**: PCA is Unsupervised (Max Variance). LDA is Supervised (Max Class Separation). LDA is better for classification if you have labels.
*   **vs t-SNE**: PCA is linear and global. t-SNE is non-linear and local. t-SNE makes better plots but preserves less global structure.

### Interview Questions
1.  **Q: Why must we standardize data before PCA?** (Because PCA is sensitive to the scale of variances).
2.  **Q: What is the relationship between PCA and Eigenvectors?** (PCs are the eigenvectors of the covariance matrix).
3.  **Q: Can PCA be used for feature selection?** (No, it does Feature *Extraction*. It creates new features, it doesn't select old ones).

### Summary
PCA is the Swiss Army Knife of Data Science. It simplifies complexity. It takes a messy, high-dimensional cloud of data and rotates it to reveal its most important structure.

### Cheatsheet
*   **Type**: Linear Dim Reduction.
*   **Key**: Eigenvalues, Variance.
*   **Param**: `n_components`.
*   **Must**: Scale Data.
