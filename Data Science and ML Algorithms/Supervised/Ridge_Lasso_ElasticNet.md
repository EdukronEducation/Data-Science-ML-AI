# Ridge, Lasso, and ElasticNet: The Regularizers

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Regularization** is the technique of adding a "penalty" to the model to prevent it from overfitting. Ridge, Lasso, and ElasticNet are the three most common variations of Linear Regression that use regularization.

### What Problem It Solves
It solves **Overfitting** and **Multicollinearity**.
*   **Overfitting**: When the model learns the noise in the training data (High Variance).
*   **Multicollinearity**: When features are highly correlated, causing the model to be unstable.

### Core Idea
"Keep it simple."
We modify the Cost Function: **Cost = Error + Penalty**.
The model now has two goals:
1.  Minimize the error (fit the data).
2.  Minimize the penalty (keep the coefficients small).

### Intuition: The "Budget" Analogy
Imagine you are hiring a team of experts (features) to solve a problem.
*   **Linear Regression**: You have an infinite budget. You hire everyone and pay them huge salaries (large coefficients).
*   **Ridge**: You have a budget. You can hire everyone, but you must pay them very little (shrink coefficients).
*   **Lasso**: You have a strict budget. You can only hire a few key people; you must fire the rest (set coefficients to 0).

### Visual Interpretation
*   **Ridge**: A circle constraint. The solution is pulled towards the center but rarely hits zero.
*   **Lasso**: A diamond constraint. The corners touch the axes, forcing some coefficients to be exactly zero.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
Standard Linear Regression fails when you have:
1.  More features than data points ($p > n$).
2.  Highly correlated features.
3.  Too much noise.
Regularization fixes all these problems.

### Domains & Use Cases
1.  **Genomics**: You have 20,000 genes (features) but only 100 patients (samples). Lasso picks the 5 important genes.
2.  **Finance**: Predicting stock returns using hundreds of economic indicators. Ridge prevents the model from overreacting to one noisy indicator.
3.  **Signal Processing**: Compressive Sensing (reconstructing signals from few measurements) uses Lasso.

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Regression (can also be applied to Logistic Regression for classification).
*   **Linearity**: Linear.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Yes.

---

## 📄 Page 3 — Mathematical Foundation

### Core Formula
We add a penalty term to the MSE.

**1. Ridge (L2 Regularization)**
$$ J(\beta) = MSE + \lambda \sum_{i=1}^{n} \beta_i^2 $$
*   $\lambda$ (Lambda/Alpha): Penalty strength.
*   $\beta_i^2$: Square of coefficients.

**2. Lasso (L1 Regularization)**
$$ J(\beta) = MSE + \lambda \sum_{i=1}^{n} |\beta_i| $$
*   $|\beta_i|$: Absolute value of coefficients.

**3. ElasticNet**
$$ J(\beta) = MSE + r \lambda \sum |\beta_i| + \frac{1-r}{2} \lambda \sum \beta_i^2 $$
*   Mixes both L1 and L2.

### Optimization
*   **Ridge**: Can be solved with a closed-form equation (Matrix math).
*   **Lasso**: Cannot be solved with closed-form (because Absolute Value is not differentiable at 0). Requires Coordinate Descent.

### Assumptions
Same as Linear Regression, but we assume that the "true" model is sparse (for Lasso) or has small coefficients (for Ridge).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Input Data**: Features $X$, Target $y$.
2.  **Scaling**: **MANDATORY**. All features must be on the same scale (e.g., 0 to 1).
3.  **Initialization**: Initialize weights.
4.  **Training Loop**:
    *   Calculate Prediction.
    *   Calculate Error.
    *   Calculate **Penalty** (Sum of weights).
    *   Update weights to minimize (Error + Penalty).
5.  **Result**:
    *   Ridge: All weights are small numbers.
    *   Lasso: Many weights are exactly 0.

### Pseudocode (Ridge Gradient Descent)
```python
gradient = (X.T * error) + (2 * lambda * beta) # Added penalty term
beta = beta - learning_rate * gradient
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Scaling**: The most critical assumption. If you don't scale, the penalty will unfairly crush features with smaller magnitudes.

### Hyperparameters
1.  **`alpha`** (Float):
    *   *Control*: Strength of regularization.
    *   *Effect*:
        *   0 = Linear Regression.
        *   Infinity = All weights become 0 (Flat line).
        *   High Alpha = High Bias (Underfitting).
        *   Low Alpha = High Variance (Overfitting).
2.  **`l1_ratio`** (Float, ElasticNet only):
    *   *Control*: Mix between Lasso (1.0) and Ridge (0.0).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization (CRITICAL)
*   You **MUST** use `StandardScaler` or `MinMaxScaler`.
*   Example: If Feature A is "Distance in km" (0-10) and Feature B is "Distance in meters" (0-10000), Feature B has a huge number. The penalty term $\lambda \beta^2$ will try to crush Feature B's coefficient simply because the number is big, not because it's unimportant. Scaling fixes this.

### 2. Missing Values
*   Handle before training (Impute/Drop).

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Prevents Overfitting**: The best way to handle high-variance models.
2.  **Feature Selection (Lasso)**: Automatically tells you which features are important.
3.  **Handles Multicollinearity (Ridge)**: If two features are correlated, Ridge shares the weight between them (stable). Lasso picks one and drops the other (unstable but sparse).

### Limitations
1.  **Feature Scaling**: Extra step required.
2.  **Tuning**: You must tune `alpha`. You can't just guess it.
3.  **Bias**: By design, it adds Bias to the model. If your model was already underfitting, Regularization makes it worse.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   MSE, RMSE, $R^2$.
*   **Number of Non-Zero Coefficients**: Useful metric for Lasso to see how much it simplified the model.

### Python Implementation
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Generate Data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale (CRITICAL)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

# 4. Train Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

# 5. Evaluate
print(f"Ridge MSE: {mean_squared_error(y_test, ridge.predict(X_test_scaled)):.2f}")
print(f"Lasso MSE: {mean_squared_error(y_test, lasso.predict(X_test_scaled)):.2f}")
print(f"Lasso Features Kept: {sum(lasso.coef_ != 0)}")
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
We use `RidgeCV` or `LassoCV`, which are efficient versions of GridSearchCV built into the library.

```python
from sklearn.linear_model import RidgeCV

# Automatically tests alphas: 0.1, 1.0, 10.0
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best Alpha: {ridge_cv.alpha_}")
```

### Real-World Applications
*   **Credit Scoring**: Selecting the 10 most important financial behaviors out of 1000 tracked variables (Lasso).
*   **Image Reconstruction**: Reconstructing an image from fuzzy data (Total Variation Regularization).

### When to Use
*   **Ridge**: When you have many features and they are all likely important (Dense). When you have Multicollinearity.
*   **Lasso**: When you have millions of features and you suspect only a tiny fraction are useful (Sparse).
*   **ElasticNet**: When you are not sure, or when you have correlated features and want to keep groups of them.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **Ridge vs Lasso**: Ridge shrinks to *near* zero. Lasso shrinks to *exactly* zero.
*   **vs Linear Regression**: Regularized models are almost always better than standard OLS unless $n >> p$.

### Interview Questions
1.  **Q: Why does Lasso select features?** (Because the L1 geometry is a diamond; the solution hits the corner where coefficients are 0).
2.  **Q: What happens to Ridge if Alpha is 0?** (It becomes Linear Regression).
3.  **Q: What happens if Alpha is Infinity?** (All coefficients become 0; the model predicts the mean of y).

### Summary
Regularization is the "brake" on your model. It stops the model from memorizing noise. Use Ridge by default. Use Lasso if you need feature selection. Always scale your data.

### Cheatsheet
*   **Ridge**: L2 ($\beta^2$), Shrinks, Handles Multicollinearity.
*   **Lasso**: L1 ($|\beta|$), Selects, Sparse Solution.
*   **Alpha**: Higher = More Regularization = Simpler Model.
