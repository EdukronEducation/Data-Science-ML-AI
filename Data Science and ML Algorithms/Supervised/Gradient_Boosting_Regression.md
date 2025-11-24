# Gradient Boosting Regression: The Improver

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Gradient Boosting** is an **Ensemble** technique that builds models sequentially. Unlike Random Forest, which builds trees independently, Gradient Boosting builds one tree at a time, where each new tree helps to correct errors made by the previously trained tree.

### What Problem It Solves
It solves the problem of **Bias**.
Random Forest reduces Variance (overfitting). Gradient Boosting reduces Bias (underfitting) and Variance. It is often the most accurate algorithm for tabular data.

### Core Idea
"Teamwork with a purpose."
1.  Model 1 makes a prediction. It makes some mistakes.
2.  Model 2 looks *only* at the mistakes of Model 1 and tries to fix them.
3.  Model 3 looks at the mistakes of Model 2...
4.  The final prediction is the sum of all models.

### Intuition: The "Golf" Analogy
Imagine you are playing golf.
1.  **Shot 1**: You hit the ball towards the hole. It lands 50 yards short. (Error = +50).
2.  **Shot 2**: You don't aim for the hole. You aim to hit the ball 50 yards. You hit it, but overshoot by 10 yards. (Error = -10).
3.  **Shot 3**: You aim to hit it back 10 yards.
4.  **Final Position**: Sum of all shots. You are very close to the hole.

### Visual Interpretation
*   **Tree 1**: Fits the general trend.
*   **Tree 2**: Fits the noise/residuals left over.
*   **Tree 3**: Refines the details.
*   **Final**: A highly detailed curve that hugs the data.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the "Gold Standard" for tabular data competitions (Kaggle). If you want the absolute highest accuracy and don't care about training time, this is it.

### Domains & Use Cases
1.  **Search Ranking**: Google/Bing use boosting to rank search results (Learning to Rank).
2.  **Credit Scoring**: Extremely precise risk assessment.
3.  **Fraud Detection**: Identifying subtle patterns in transaction data.
4.  **Insurance**: Claim severity modeling.

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Regression.
*   **Ensemble**: Boosting (Sequential).
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Core Formula
The final model $F(x)$ is a sum of weak learners $h(x)$:
$$ F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x) $$
*   $F_0(x)$: Initial prediction (usually the mean).
*   $\eta$: Learning Rate (Shrinkage).
*   $h_m(x)$: The $m$-th tree.

### Optimization: Gradient Descent in Function Space
We want to minimize a Loss Function $L(y, F(x))$.
Instead of updating parameters (like in Linear Regression), we update the **function** $F(x)$.
We calculate the **Negative Gradient** of the loss function:
$$ r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)} $$
For MSE Loss, the negative gradient is simply the **Residual** ($y - \hat{y}$).
So, each new tree fits the residuals of the previous model.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Initialize**: Predict the mean of the target for all samples ($F_0$).
2.  **Loop** for $M$ stages:
    *   **Calculate Residuals**: $r = y - Prediction$.
    *   **Train Tree**: Fit a Decision Tree $h_m$ to the residuals $r$ (NOT to target $y$).
    *   **Update Model**: $F_{new} = F_{old} + \eta \cdot h_m$.
3.  **Output**: Final prediction is the sum of the initial mean plus all the weighted tree predictions.

### Pseudocode
```python
prediction = mean(y)
learning_rate = 0.1

for i in range(n_estimators):
    residuals = y - prediction
    tree = train_tree(X, residuals) # Fit to errors!
    prediction = prediction + learning_rate * tree.predict(X)

return prediction
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   Sensitive to Outliers: Because it tries to fix errors, massive outliers look like massive errors, and the model will obsess over fixing them, ruining the fit elsewhere.

### Hyperparameters
1.  **`n_estimators`**:
    *   *Control*: Number of boosting stages.
    *   *Effect*: High = Overfitting. Unlike Random Forest, you CAN overfit with too many trees.
2.  **`learning_rate`**:
    *   *Control*: Contribution of each tree.
    *   *Effect*: Low (0.01) = Better accuracy, slower training. High (1.0) = Fast, high risk of overfitting.
    *   *Rule*: Decrease Learning Rate -> Increase n_estimators.
3.  **`max_depth`**:
    *   *Control*: Depth of individual trees.
    *   *Effect*: Usually kept shallow (3-5). These are "Weak Learners".

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization
*   **NOT REQUIRED**. Tree-based.

### 2. Encoding
*   **Required**: One-Hot or Ordinal.

### 3. Outlier Treatment (CRITICAL)
*   **Mandatory**: You must handle outliers.
*   **Solution**: Remove them, or use a robust loss function like **Huber Loss** or **Quantile Loss** instead of MSE.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Accuracy**: Often beats Random Forest.
2.  **Flexibility**: Can optimize different loss functions (MAE, MSE, Huber, Quantile).
3.  **Mixed Data**: Handles numerical and categorical data well.

### Limitations
1.  **Sequential**: Cannot be parallelized (Tree 2 needs Tree 1). Slow training.
2.  **Tuning**: Harder to tune than Random Forest. 3-4 interacting parameters.
3.  **Overfitting**: Will memorize noise if not stopped early.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   MSE, RMSE, $R^2$.

### Python Implementation
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Generate Data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Train
# Note: shallow trees (max_depth=3)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbr.fit(X_train, y_train)

# 3. Predict
y_pred = gbr.predict(X_test)

# 4. Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"Train Score: {gbr.score(X_train, y_train):.2f}")
print(f"Test Score: {gbr.score(X_test, y_test):.2f}")
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
GridSearch is painful here due to slowness. Use RandomizedSearch or Bayesian Optimization.

```python
from sklearn.model_selection import RandomizedSearchCV

params = {
    'n_estimators': [100, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0] # Stochastic Gradient Boosting
}
search = RandomizedSearchCV(GradientBoostingRegressor(), params, n_iter=10, cv=3)
search.fit(X_train, y_train)
print(search.best_params_)
```

### Real-World Applications
*   **Web Search**: Ranking algorithms (LambdaMART).
*   **Ecology**: Modeling species distribution.

### When to Use
*   When you want to win a Kaggle competition.
*   When you have clean data (outliers removed).
*   When accuracy is more important than training speed.

### When NOT to Use
*   When you need massive parallelization (Use XGBoost/LightGBM instead of standard sklearn GBR).
*   When data is extremely noisy.
*   When interpretability is key.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Random Forest**: Boosting reduces Bias. RF reduces Variance. Boosting is sequential; RF is parallel.
*   **vs XGBoost**: XGBoost is an optimized version of Gradient Boosting (faster, regularization, handling missing values).

### Interview Questions
1.  **Q: Why are the trees in Gradient Boosting shallow?** (Because we want "Weak Learners" with high bias. If they were deep, they would overfit immediately).
2.  **Q: What is the relationship between Learning Rate and Number of Trees?** (Inverse. Lower LR requires more Trees).
3.  **Q: How does it handle classification?** (It fits the trees to the Log-Odds residuals).

### Summary
Gradient Boosting is the sniper of ML algorithms. It iteratively corrects its mistakes to produce a highly accurate model. It requires careful tuning and outlier management, but the results are worth it.

### Cheatsheet
*   **Type**: Boosting Ensemble.
*   **Key Params**: `learning_rate`, `n_estimators`.
*   **Pros**: SOTA Accuracy.
*   **Cons**: Slow, Sensitive to Outliers.
