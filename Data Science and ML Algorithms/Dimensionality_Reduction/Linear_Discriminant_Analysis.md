# Linear Discriminant Analysis (LDA): The Class Separator

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Linear Discriminant Analysis (LDA)** is a dimensionality reduction technique (like PCA), but unlike PCA, it is **Supervised**. It takes class labels into account.

### What Problem It Solves
It solves the problem where PCA fails: **Class Separation**.
PCA maximizes variance (spread). Sometimes, the direction of maximum spread is *not* the direction that separates the classes. LDA finds the direction that separates the classes best.

### Core Idea
"Maximize the distance between class means, minimize the spread within each class."
We want the classes to be:
1.  **Far apart** (High Between-Class Variance).
2.  **Tight and compact** (Low Within-Class Variance).

### Intuition: The "Good Photo" Analogy
Imagine you want to take a photo of red and blue balls to show they are different.
*   **PCA**: Takes a photo from the angle where the cloud of balls looks widest. The red and blue might overlap.
*   **LDA**: Takes a photo from the angle where the red bunch and blue bunch are clearly separated.

### Visual Interpretation
*   **PCA Axis**: Along the length of the banana.
*   **LDA Axis**: Perpendicular to the boundary between the two bananas.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the best dimensionality reduction technique for **Classification Preprocessing**. If your goal is to classify, use LDA instead of PCA.

### Domains & Use Cases
1.  **Face Recognition**: Fisherfaces. Projecting faces into a space where "Person A" and "Person B" are far apart.
2.  **Bioinformatics**: Classifying cancer types based on gene data.
3.  **Marketing**: Segmenting customers.

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Dimensionality Reduction (and Classification).
*   **Linearity**: Linear.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Fisher's Criterion
We want to find a projection vector $w$ that maximizes:
$$ J(w) = \frac{w^T S_B w}{w^T S_W w} $$

### Scatter Matrices
1.  **Within-Class Scatter ($S_W$)**: Measures how spread out each class is.
    $$ S_W = \sum_{c} \sum_{x \in c} (x - \mu_c)(x - \mu_c)^T $$
2.  **Between-Class Scatter ($S_B$)**: Measures how far the class means are from the global mean.
    $$ S_B = \sum_{c} N_c (\mu_c - \mu)(\mu_c - \mu)^T $$

### Solution
The optimal $w$ is the eigenvector corresponding to the largest eigenvalue of the matrix $S_W^{-1} S_B$.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Compute Means**: Calculate the mean vector for each class $\mu_c$ and the global mean $\mu$.
2.  **Compute Scatter Matrices**: Calculate $S_W$ and $S_B$.
3.  **Eigen Decomposition**: Find eigenvalues and eigenvectors of $S_W^{-1} S_B$.
4.  **Sort**: Sort eigenvectors by eigenvalues (descending).
5.  **Select**: Choose the top $k$ eigenvectors.
    *   *Constraint*: $k \le C - 1$ (where $C$ is number of classes).
6.  **Project**: $X_{new} = X \cdot W$.

### Pseudocode
```python
mean_vectors = []
for cl in range(num_classes):
    mean_vectors.append(mean(X[y==cl]))

S_W = zeros((d, d))
for cl, mv in zip(range(num_classes), mean_vectors):
    class_sc_mat = zeros((d, d))
    for row in X[y==cl]:
        row, mv = row.reshape(d,1), mv.reshape(d,1)
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat

S_B = zeros((d, d))
overall_mean = mean(X).reshape(d,1)
for i, mean_vec in enumerate(mean_vectors):
    n = X[y==i].shape[0]
    mean_vec = mean_vec.reshape(d,1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

eig_vals, eig_vecs = eig(inv(S_W).dot(S_B))
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Normality**: Assumes data is Gaussian.
2.  **Homoscedasticity**: Assumes classes have identical covariance matrices.
3.  **Independence**.

### Hyperparameters
1.  **`n_components`**:
    *   *Constraint*: Must be less than $min(\text{features}, \text{classes} - 1)$.
    *   *Example*: If you have 100 features but only 3 classes (Iris), LDA can produce at most 2 components. This is a major limitation compared to PCA.
2.  **`solver`**: 'svd', 'lsqr', 'eigen'.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **Required**. Like PCA, it is sensitive to scale.

### 2. Encoding
*   **Required**.

### 3. Outlier Treatment
*   **Critical**. Outliers shift the means and variances, ruining the scatter matrices.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Supervised**: Uses label information, making it superior for classification tasks.
2.  **Separability**: Explicitly optimizes for class separation.

### Limitations
1.  **Component Limit**: Can only produce $C-1$ components. If you have 2 classes, you get 1D output. You cannot visualize 2 classes in 2D using LDA.
2.  **Linear**: Fails on non-linear boundaries.
3.  **Assumptions**: Fails if classes have different variances (e.g., one tight blob and one wide blob).

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Classification Accuracy (since LDA is also a classifier).

### Python Implementation
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Data (3 Classes)
data = load_wine()
X, y = data.data, data.target

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Train LDA
# n_components must be < n_classes (3). So max is 2.
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 4. Visualize
plt.figure(figsize=(8,6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='rainbow')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA of Wine Data')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Usually none. The number of components is fixed by the number of classes.

### Real-World Applications
*   **Brain Computer Interfaces (BCI)**: Classifying EEG signals (Left hand vs Right hand).
*   **Bankruptcy Prediction**: Separating "Bankrupt" vs "Solvent" companies.

### When to Use
*   When you have labels (Supervised).
*   When you want to visualize multiclass separation.
*   As a preprocessing step for simple classifiers (like Logistic Regression).

### When NOT to Use
*   When variance of classes is very different.
*   When you need more components than $C-1$.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs PCA**:
    *   PCA: Unsupervised. Max Variance. No limit on components.
    *   LDA: Supervised. Max Separation. Limit $C-1$.
    *   *Rule*: Use PCA for compression. Use LDA for classification.

### Interview Questions
1.  **Q: Why is the max number of components C-1?**
    *   A: Because the centroids of $C$ classes lie on a subspace of dimension $C-1$. (e.g., 2 points define a line (1D), 3 points define a plane (2D)).
2.  **Q: What happens if the classes have different covariances?**
    *   A: LDA fails. You should use QDA (Quadratic Discriminant Analysis).

### Summary
LDA is the supervised brother of PCA. It cares about *who* the data points are (labels), not just where they are. It is the gold standard for linear dimensionality reduction in classification.

### Cheatsheet
*   **Type**: Supervised Dim Reduction.
*   **Key**: Between/Within Scatter.
*   **Limit**: C-1 components.
*   **Pros**: Class Separation.
