# Polynomial Regression: The Curve Fitter

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Polynomial Regression** is an extension of Linear Regression that allows us to model **non-linear** relationships. While Linear Regression fits a straight line, Polynomial Regression fits a curve (like a parabola or an S-curve) to the data.

### What Problem It Solves
It solves Regression problems where the data is **not linear**.
*   "How does the spread of a virus grow over time?" (Exponential/Curved).
*   "What is the relationship between driving speed and fuel efficiency?" (Inverted U-shape).

### Core Idea
The core idea is to "trick" the Linear Regression model. We don't change the model itself; we change the **data**. We create new features by raising the original features to a power ($x^2, x^3, ...$) and then run standard Linear Regression on these new features.

### High-Level Logic
1.  Take the original input $x$.
2.  Create new features: $x^2, x^3, ..., x^d$.
3.  Treat these as separate variables ($x_1, x_2, ...$).
4.  Fit a linear equation: $y = w_1 x + w_2 x^2 + ...$
5.  The result is a curve in the original 2D space, but it's a "linear" hyperplane in the higher-dimensional feature space.

### Intuition: The "Wire" Analogy
Linear Regression is a stiff metal rod. You can move it up/down or tilt it, but you can't bend it.
Polynomial Regression is a flexible wire.
*   **Degree 2**: You can bend it once (U-shape).
*   **Degree 3**: You can bend it twice (S-shape).
*   **Degree 10**: You can twist it into a pretzel (High flexibility).

### Visual Interpretation
*   **Degree 1**: Straight Line.
*   **Degree 2**: Parabola (U).
*   **Degree 3**: S-Curve.

### What Makes It Special?
It bridges the gap between simple linear models and complex non-linear ones without needing a completely new algorithm. It uses the same OLS math as Linear Regression.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
Real-world data is rarely perfectly straight. Growth rates, physics, and biology often follow curved paths. Polynomial Regression captures this complexity while keeping the model interpretable (it's just a formula).

### Domains & Use Cases
1.  **Epidemiology**: Modeling disease spread (early stages are exponential).
2.  **Physics**: Trajectory of a projectile (Parabolic).
3.  **Biology**: Enzyme reaction rates vs Temperature (Bell curve).
4.  **Economics**: Utility functions (Diminishing returns).

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Regression.
*   **Linearity**: **Linear Model** (Wait, what? Yes, it is linear in *parameters* $w$, even though it fits a non-linear curve to $x$).
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Yes.

---

## 📄 Page 3 — Mathematical Foundation

### Core Formula
Standard Linear Regression:
$$ y = \beta_0 + \beta_1 x $$

Polynomial Regression (Degree $d$):
$$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_d x^d + \epsilon $$

Notice that if we let $z_1 = x, z_2 = x^2, z_3 = x^3$, the equation becomes:
$$ y = \beta_0 + \beta_1 z_1 + \beta_2 z_2 + ... $$
This is exactly the same form as Multiple Linear Regression!

### Cost Function
Same as Linear Regression: **Mean Squared Error (MSE)**.
$$ J(\beta) = \frac{1}{n} \sum (y - \hat{y})^2 $$

### Optimization
We use **Ordinary Least Squares (OLS)** or **Gradient Descent** just like in Linear Regression. The math doesn't change; only the input matrix $X$ changes (it gets wider with more columns).

### Assumptions in Math
Same assumptions as Linear Regression (Linearity in parameters, Homoscedasticity, Normality of residuals).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Input Data**: Feature $x$.
2.  **Feature Transformation**:
    *   If Degree = 2: Transform $[x]$ into $[x, x^2]$.
    *   If Degree = 3: Transform $[x]$ into $[x, x^2, x^3]$.
3.  **Model Initialization**: Initialize Linear Regression.
4.  **Training**: Fit the Linear Regression model to the *transformed* features.
5.  **Prediction**: To predict for a new $x_{new}$:
    *   Calculate $x_{new}^2, x_{new}^3...$
    *   Plug into the learned equation.
6.  **Output**: A predicted value $y$.

### Pseudocode
```python
degree = 2
X_poly = [X, X^2] # Create new columns

# Now just do standard linear regression
beta = solve_OLS(X_poly, y)

return beta
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Polynomial Relationship**: The data actually follows a polynomial trend. If the data is sinusoidal (wave), a polynomial is a bad choice (use Fourier series).
2.  **No Extrapolation**: Polynomials go to $\pm \infty$ very quickly. You cannot trust predictions outside the training range.

### Hyperparameters
1.  **`degree`** (int):
    *   *Control*: The complexity of the curve.
    *   *Effect*:
        *   **1**: Linear (Underfitting).
        *   **2-3**: Sweet spot for many physical problems.
        *   **10+**: Extreme Overfitting (Wiggles to hit every noise point).
2.  **`include_bias`** (True/False):
    *   *Control*: Whether to include the intercept column (column of 1s).
3.  **`interaction_only`** (True/False):
    *   *Control*: If True, only produces interaction features ($x_1 x_2$) and not powers ($x_1^2$).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling / Normalization (CRITICAL)
*   **Why?** If $x = 100$, then $x^2 = 10,000$ and $x^3 = 1,000,000$.
*   The features will have vastly different scales. Gradient Descent will fail miserably without scaling.
*   **Action**: Apply `StandardScaler` *after* generating polynomial features.

### 2. Feature Generation
*   Use `PolynomialFeatures` from sklearn to automatically generate the powers and interaction terms.

### 3. Outlier Treatment
*   Polynomials are **even more sensitive** to outliers than lines. A single outlier can pull the "tail" of the curve drastically. Remove them aggressively.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Fits Non-Linear Data**: Can model complex curves.
2.  **Simple**: Uses the same well-understood math as Linear Regression.
3.  **Interactions**: Automatically models interactions between features (e.g., Length $\times$ Width = Area).

### Limitations
1.  **Overfitting**: High-degree polynomials are notorious for overfitting (Runge's Phenomenon). They oscillate wildly at the edges.
2.  **Extrapolation**: Terrible at predicting the future. The curve will shoot off to infinity.
3.  **Dimensionality**: The number of features explodes. If you have 10 original features and Degree=3, you generate hundreds of new features.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **RMSE**: Standard error metric.
*   **$R^2$**: How well the curve fits the data.

### Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 1. Generate Non-Linear Data
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) # Quadratic

# 2. Create Pipeline (Transform -> Scale -> Fit)
# Pipeline ensures we don't leak data during scaling
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('linear', LinearRegression())
])

# 3. Train
pipeline.fit(X, y)

# 4. Predict
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
y_new = pipeline.predict(X_new)

# 5. Evaluate
print(f"R2 Score: {r2_score(y, pipeline.predict(X)):.2f}")

# 6. Visualize
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_new, y_new, color='red', linewidth=2, label='Degree 2')
plt.legend()
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
The main parameter to tune is `degree`. We use a validation curve to find the sweet spot.

```python
from sklearn.model_selection import GridSearchCV

params = {'poly__degree': [1, 2, 3, 4, 5, 10]}
grid = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error')
grid.fit(X, y)

print("Best Degree:", grid.best_params_['poly__degree'])
```

### Real-World Applications
*   **Growth Curves**: Bacteria growth, viral spread.
*   **Engineering**: Stress-strain curves in materials.
*   **Marketing**: Response curves (Sales vs Discount % - usually diminishing returns).

### When to Use
*   When the scatter plot clearly shows a curve.
*   When you know the physical law governing the data is polynomial (e.g., Area = $r^2$).

### When NOT to Use
*   When the data is complex/irregular (use Decision Trees or Splines).
*   When you need to extrapolate far into the future.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs. Linear Regression**: Polynomial is more flexible but higher risk of overfitting.
*   **vs. Splines**: Splines are better because they fit piecewise polynomials (local curves) rather than one giant global curve. Splines don't go crazy at the edges like Polynomials do.

### Interview Questions
1.  **Q: Is Polynomial Regression linear or non-linear?** (Linear in parameters).
2.  **Q: What happens if you choose a degree that is too high?** (Overfitting, High Variance, Runge's Phenomenon).
3.  **Q: Why do we need to scale features in Polynomial Regression?** (Because $x^n$ grows extremely fast, creating massive scale differences).

### Summary
Polynomial Regression is a clever hack to make Linear Regression fit curved data. It works by creating powers of features. It is powerful but dangerous—use low degrees (2 or 3) and always visualize the result to ensure it's not wiggling uncontrollably.

### Cheatsheet
*   **Formula**: $y = \beta_0 + \beta_1 x + \beta_2 x^2 ...$
*   **Key Param**: `degree`
*   **Risk**: Overfitting (High Variance)
*   **Preprocessing**: Scaling is MANDATORY.
