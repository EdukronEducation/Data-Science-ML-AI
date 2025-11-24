# Linear Discriminant Analysis (LDA): The Class Separator

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Linear Discriminant Analysis (LDA)** is a classic statistical technique used for dimensionality reduction and classification. Developed by Ronald Fisher in 1936 (originally as Fisher's Linear Discriminant), it predates most modern machine learning algorithms but remains a powerful tool. Unlike PCA (Principal Component Analysis), which is an unsupervised method that maximizes variance, LDA is a **Supervised** method. It seeks to find a linear combination of features that maximizes the separation between multiple classes.
LDA assumes that the data is generated from Gaussian distributions and that the classes share the same covariance matrix (homoscedasticity). While these assumptions are strict, LDA is surprisingly robust and often outperforms more complex models like Logistic Regression when the classes are well-separated or when the sample size is small. It is widely used in pattern recognition, face recognition, and medical diagnosis.

### What Problem It Solves
LDA solves the problem of **"Class Separation in Lower Dimensions."** Imagine you have a dataset with two classes that overlap significantly when projected onto the X-axis or the Y-axis. PCA would project them onto the axis of highest variance, which might not separate the classes at all (it might just spread the mixed mess out).
LDA, however, looks for a specific axis (a new line) such that when you project the points onto this line, the classes are as distinct as possible. It tries to maximize the distance between the means of the classes while simultaneously minimizing the spread (variance) within each class. This ratio of "Between-Class Variance" to "Within-Class Variance" is the core objective of LDA.

### Core Idea
The core idea is **Fisher's Criterion**: Maximize the ratio of between-class variance ($S_B$) to within-class variance ($S_W$).
$$ J(w) = \frac{w^T S_B w}{w^T S_W w} $$
We want to find a vector $w$ (the projection axis) that pushes the centers of the classes far apart ($S_B$) but keeps the points in each class tightly clustered together ($S_W$). If we only maximized the distance between means, we might project onto an axis where the classes are spread out so much that they still overlap. By penalizing within-class spread, LDA ensures a clean, compact separation.

### Intuition: The "Party" Analogy
Imagine a room full of people from two different companies (Red Corp and Blue Corp) mixing together at a party. You want to take a photo where the two companies look distinct.
*   **PCA**: PCA tells everyone to spread out as much as possible to fill the room. The photo will show a big crowd, but Red and Blue will still be mixed.
*   **LDA**: LDA tells all the Red employees to huddle together tightly in one corner, and all the Blue employees to huddle tightly in the opposite corner. Then it takes the photo from an angle that emphasizes the gap between the two groups. The resulting image shows two clear, separated clusters.

### Visual Interpretation
In 2D, LDA finds a line. The decision boundary is a line perpendicular to this projection axis.
If you have 3 classes, LDA finds a 2D plane (subspace) that separates them best. The number of dimensions LDA can find is limited to $C - 1$, where $C$ is the number of classes. So for binary classification, LDA always projects to 1 dimension (a line).

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
LDA is the **Best Linear Classifier** when the assumption of normality holds. It is analytically solvable (closed-form solution), meaning there is no iterative optimization loop like Gradient Descent. This makes it instantaneous to train.
It is also a **Dimensionality Reduction** technique. You can use LDA to project 100 features down to 2 features (for 3 classes) and then visualize the data. Unlike PCA, this visualization is guaranteed to show the class structure, making it a superior tool for exploratory data analysis of labeled data.

### Domains & Use Cases
**Face Recognition (Fisherfaces)**: In computer vision, LDA is used to reduce the dimensionality of face images. It finds the features (pixels) that best discriminate between different people, ignoring features that are common to all faces (like lighting or skin texture).
**Medical Diagnosis**: Classifying a patient as "Healthy", "Mild", or "Severe" based on blood tests. Since biological markers often follow a Normal distribution, LDA is a natural fit and often outperforms "fancier" algorithms.

### Type of Algorithm
LDA is a **Supervised Learning** algorithm. It supports **Classification** (Multiclass natively) and **Dimensionality Reduction**.
It is a **Parametric** model (learns Means and Covariance). It is **Linear** (creates linear decision boundaries). It is **Generative** (models $P(X|Y)$ as Gaussian).

---

## 📄 Page 3 — Mathematical Foundation

### Scatter Matrices
LDA relies on two matrices:
1.  **Within-Class Scatter Matrix ($S_W$)**: Measures how much the data scatters around their respective class means.
    $$ S_W = \sum_{c=1}^C \sum_{x \in D_c} (x - \mu_c)(x - \mu_c)^T $$
2.  **Between-Class Scatter Matrix ($S_B$)**: Measures how much the class means scatter around the global mean.
    $$ S_B = \sum_{c=1}^C N_c (\mu_c - \mu)(\mu_c - \mu)^T $$

### Eigenvalue Problem
To maximize the ratio $J(w)$, we take the derivative and set it to zero. This leads to the **Generalized Eigenvalue Problem**:
$$ S_W^{-1} S_B w = \lambda w $$
The optimal projection vectors $w$ (discriminants) are the eigenvectors of the matrix $S_W^{-1} S_B$ corresponding to the largest eigenvalues.
This is mathematically elegant: the "best" direction is simply the principal eigenvector of the ratio of variances.

### Dimensionality Limit
The rank of $S_B$ is at most $C - 1$ (because the sum of class means minus global mean is zero). Therefore, there are at most $C - 1$ non-zero eigenvalues.
This means LDA can only project data to a maximum of $C - 1$ dimensions. If you have 1000 features and 2 classes, LDA projects to 1 dimension. If you have 10 classes, it projects to 9 dimensions. This is a hard constraint compared to PCA.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm
1.  **Compute Means**: Calculate the mean vector $\mu_c$ for each class and the global mean $\mu$.
2.  **Compute Scatter Matrices**: Calculate $S_W$ and $S_B$.
3.  **Solve Eigenvalues**: Compute eigenvectors and eigenvalues of $S_W^{-1} S_B$.
4.  **Sort**: Sort eigenvectors by decreasing eigenvalues.
5.  **Project**: Choose the top $k$ eigenvectors ($k \le C-1$) to form a matrix $W$. Transform data $X_{new} = X \cdot W$.

### Pseudocode
```python
def fit(X, y):
    n_features = X.shape[1]
    class_labels = unique(y)
    mean_overall = mean(X, axis=0)
    SW = zeros((n_features, n_features))
    SB = zeros((n_features, n_features))
    
    for c in class_labels:
        X_c = X[y == c]
        mean_c = mean(X_c, axis=0)
        # Update SW
        SW += (X_c - mean_c).T.dot(X_c - mean_c)
        # Update SB
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        SB += n_c * (mean_diff).dot(mean_diff.T)
        
    # Solve Eigen problem
    A = inv(SW).dot(SB)
    eigenvalues, eigenvectors = eig(A)
    # Sort and return top vectors
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Multivariate Normality**: Each class is generated from a Gaussian distribution.
2.  **Homoscedasticity**: All classes share the *same* covariance matrix $\Sigma$. (If they have different covariances, use **QDA - Quadratic Discriminant Analysis**).
3.  **Independence**: Features are statistically independent (though LDA handles correlation better than Naive Bayes).

### Hyperparameters
**`solver`**:
*   `svd` (Singular Value Decomposition): Default. Does not compute covariance matrix directly. Good for wide data.
*   `lsqr` (Least Squares): Supports shrinkage.
*   `eigen`: Computes covariance. Good for small data.
**`shrinkage`**: 'auto' or float (0-1). Used with `lsqr` or `eigen`. Shrinks the covariance matrix towards a diagonal matrix. Crucial for high-dimensional data where $S_W$ is singular (not invertible).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
**Required**. LDA involves variance calculations and matrix inversion. While theoretically scale-invariant for the projection direction, numerical stability is better with scaling (StandardScaler).

### 2. Encoding
**Required**. Target labels must be encoded. Features must be numerical.

### 3. Outliers
**Sensitive**. Since LDA relies on Mean and Variance, outliers can skew the class means and the scatter matrices, leading to a suboptimal projection. Remove outliers.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
**Dimensionality Reduction**: Best supervised reduction technique.
**Multiclass**: Handles multiple classes naturally (unlike Logistic Regression which uses One-vs-Rest).
**Simple**: No learning rates, no epochs.

### Limitations
**Assumptions**: Fails if data is not Gaussian or if covariances are very different (use QDA).
**Linearity**: Cannot separate concentric circles or non-linear shapes (unless kernelized).
**Dimension Limit**: Limited to $C-1$ components.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Python Implementation
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load Data
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train LDA
lda = LinearDiscriminantAnalysis(n_components=2) # Reduce to 2D
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# 3. Evaluate Classifier
y_pred = lda.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 4. Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.title('LDA Projection of Wine Dataset')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Tune **`shrinkage`** if you have many features and few samples. This regularizes the covariance matrix.
Tune **`priors`** if you want to manually adjust class weights (e.g., if you know the test set has a different distribution).

### Real-World Applications
**Preprocessing Step**: Use LDA to reduce dimensions before feeding data into a Neural Network or SVM. This removes noise and speeds up training.
**Customer Segmentation**: Visualizing customer groups in 2D.

### When to Use
Use LDA when **Classes are Well-Separated**. Logistic Regression can be unstable here; LDA is stable.
Use it for **Dimensionality Reduction** when you have labels.

### When NOT to Use
Do not use when **Variance is Shared Unequally** (use QDA).
Do not use when relationship is **Non-Linear**.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs PCA**: PCA is Unsupervised (max variance). LDA is Supervised (max separation). PCA is better for compression; LDA is better for classification.
*   **vs Logistic Regression**: LDA assumes Gaussian data; LR makes no assumption. LR is more robust to outliers. LDA is better for small sample sizes.

### Interview Questions
1.  **Q: Why is LDA limited to C-1 dimensions?**
    *   A: Because the Between-Class Scatter Matrix $S_B$ is constructed from $C$ class means. These $C$ points lie in a subspace of dimension at most $C-1$ (since they are all defined relative to the global mean).
2.  **Q: What is the difference between LDA and QDA?**
    *   A: LDA assumes all classes share the same covariance matrix (linear boundary). QDA assumes each class has its own covariance matrix (quadratic boundary). QDA is more flexible but requires more data to estimate the individual covariances.

### Summary
LDA is the "Classic Separator." It uses the geometry of the data (means and variances) to find the perfect viewing angle to distinguish classes. While its assumptions are strict, its elegance and analytical solution make it a timeless tool for both classification and visualization.

### Cheatsheet
*   **Type**: Linear, Parametric, Generative.
*   **Key**: Scatter Matrices ($S_W, S_B$), Eigenvalues.
*   **Pros**: Fast, Supervised Reduction.
*   **Cons**: Gaussian Assumption, $C-1$ limit.
