# Support Vector Regression (SVR): The Street

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Support Vector Regression (SVR)** is the regression version of the famous SVM classification algorithm. Instead of a line that minimizes error, SVR fits a "tube" or "street" around the data.

### What Problem It Solves
It solves Regression problems, especially when:
*   You want to be robust to outliers.
*   The relationship is non-linear (using Kernels).
*   You have high-dimensional data.

### Core Idea
In Linear Regression, we try to minimize the error of *every* point.
In **SVR**, we define a "street" of width $\epsilon$ (epsilon).
*   Points *inside* the street are considered "correct". We don't care about their error.
*   Points *outside* the street are "wrong". We try to minimize their distance to the street.
*   We also try to keep the street as "flat" as possible (Regularization).

### Intuition: The "Highway" Analogy
Imagine you are building a highway. You want the highway to cover as many houses as possible.
*   The **Lane Width** is $\epsilon$.
*   Houses inside the lane are safe.
*   Houses outside the lane are problematic.
*   You adjust the highway's path to minimize the number of houses outside the lane, while keeping the highway as straight as possible.

### Visual Interpretation
*   **Center Line**: The regression prediction.
*   **Tube**: The area around the line ($\pm \epsilon$).
*   **Support Vectors**: The points on the edge or outside the tube. These are the *only* points that influence the model.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
SVR is unique because it is **sparse**. It ignores most of the training data (the points inside the tube). This makes it robust to noise. It is also incredibly powerful for non-linear data thanks to the Kernel Trick.

### Domains & Use Cases
1.  **Stock Price Prediction**: Financial data is noisy. SVR ignores the small fluctuations (noise inside the tube) and captures the main trend.
2.  **Weather Forecasting**: Predicting temperature where small errors don't matter, but large deviations do.
3.  **Material Science**: Modeling properties of new alloys.

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Regression.
*   **Linearity**: Can be Linear or Non-Linear (Kernel).
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric (technically parametric in feature space, but effectively non-parametric).

---

## 📄 Page 3 — Mathematical Foundation

### Core Formula
We want to find a function $f(x) = <w, x> + b$.
We minimize:
$$ \frac{1}{2}||w||^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) $$

*   **$\frac{1}{2}||w||^2$**: Regularization term. Minimizing $w$ makes the function "flat" (simple).
*   **$C$**: Penalty parameter.
*   **$\xi_i$**: Slack variables. The distance of a point *outside* the tube.

### Constraints
$$ y_i - <w, x_i> - b \le \epsilon + \xi_i $$
$$ <w, x_i> + b - y_i \le \epsilon + \xi_i^* $$
Basically: The error must be less than $\epsilon$, unless we pay a penalty $\xi$.

### The Kernel Trick
If data is not linear, we map it to high dimensions using $\phi(x)$.
$$ K(x_i, x_j) = <\phi(x_i), \phi(x_j)> $$
Common Kernels: Linear, Polynomial, RBF (Radial Basis Function).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Input Data**: $X, y$.
2.  **Kernel Mapping**: Implicitly map data to higher dimensions (if using non-linear kernel).
3.  **Optimization**: Solve the Quadratic Programming problem to find the "tube" that contains most points while keeping $w$ small.
4.  **Support Vector Identification**: Identify the points that lie on or outside the tube boundaries.
5.  **Model Definition**: The final model is defined *only* by these Support Vectors.
6.  **Prediction**: For a new point, calculate its similarity (kernel) to the Support Vectors to predict $y$.

### Pseudocode
```python
# Conceptual
Minimize (Regularization + C * Sum(Errors_Outside_Tube))
Subject to: Errors_Inside_Tube = 0
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Scaling**: SVR is distance-based. Features must be scaled.
2.  **IID**: Independent and Identically Distributed data.

### Hyperparameters
1.  **`kernel`**: The most important choice.
    *   'linear': For simple data.
    *   'rbf': For complex, non-linear data (Default).
2.  **`C`** (Regularization):
    *   High C: Strict. Penalizes points outside tube heavily. Wiggles to fit data. (Overfitting).
    *   Low C: Loose. Allows more points outside. Smoother. (Underfitting).
3.  **`epsilon`** ($\epsilon$):
    *   Width of the tube.
    *   Large $\epsilon$: Fewer support vectors, simpler model.
    *   Small $\epsilon$: More support vectors, complex model.
4.  **`gamma`** (for RBF):
    *   High Gamma: Only close points affect prediction. (Spiky).
    *   Low Gamma: Far points affect prediction. (Smooth).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling (MANDATORY)
*   SVR calculates distances. If one feature is 1000x larger than another, it dominates.
*   Use `StandardScaler`.

### 2. Dummy Variables
*   Convert categorical data using One-Hot Encoding.

### 3. Size Limits
*   SVR is slow ($O(N^3)$). Do not use on datasets with > 100,000 rows. Use `LinearSVR` or `SGDRegressor` for large data.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Robust to Outliers**: Thanks to the $\epsilon$-tube, small outliers are ignored.
2.  **Non-Linearity**: RBF Kernel can model incredibly complex shapes.
3.  **Generalization**: High regularization usually leads to good generalization on unseen data.

### Limitations
1.  **Hard to Tune**: Requires tuning $C$, $\epsilon$, and $\gamma$ simultaneously.
2.  **Slow**: Very slow training on large datasets.
3.  **Black Box**: Hard to interpret the weights when using Kernels.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   MSE, RMSE, $R^2$.

### Python Implementation
```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8)) # Add noise

# 2. Scale (MANDATORY)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1, 1))

# 3. Train
regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
regressor.fit(X_scaled, y_scaled.ravel())

# 4. Predict
y_pred_scaled = regressor.predict(X_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1,1))

# 5. Visualize
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('SVR (RBF)')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
GridSearch is essential here.

```python
params = {
    'C': [0.1, 1, 100, 1000],
    'epsilon': [0.0001, 0.001, 0.01, 0.1, 1],
    'gamma': [0.001, 0.1, 1, 10]
}
grid = GridSearchCV(SVR(kernel='rbf'), params, cv=5)
```

### Real-World Applications
*   **Real Estate**: Predicting prices where "market value" is a range, not a point.
*   **Energy Consumption**: Forecasting load on the power grid.

### When to Use
*   When data is non-linear.
*   When dataset is small to medium (< 50k rows).
*   When you need robustness to noise.

### When NOT to Use
*   When dataset is huge (> 100k rows).
*   When you need probabilistic output.
*   When you need to explain the formula to a business user.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Linear Regression**: SVR is slower but handles non-linearity better.
*   **vs Random Forest**: SVR is better for smooth, continuous functions. Random Forest is better for tabular data with sharp cutoffs.

### Interview Questions
1.  **Q: What is the Kernel Trick?** (Mapping data to high dimensions without calculating coordinates).
2.  **Q: What is the role of Epsilon?** (Defines the width of the tube where error is ignored).
3.  **Q: Why scale data for SVR?** (Distance-based algorithm).

### Summary
SVR is a powerful, flexible algorithm that fits a "tube" to data. It ignores small errors and focuses on the big picture. It is excellent for small, complex datasets but hard to tune and slow to train.

### Cheatsheet
*   **Params**: `C` (Regularization), `epsilon` (Tube width), `kernel` (Shape).
*   **Kernels**: Linear, Poly, RBF.
*   **Key**: Scale your data!
