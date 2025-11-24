# Decision Tree Regression: The Flowchart

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Decision Tree Regression** is a non-parametric algorithm that breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with **decision nodes** and **leaf nodes**.

### What Problem It Solves
It solves Regression problems where the relationship between features and target is **non-linear** and complex.
*   "Predicting house prices based on location, size, and age."
*   "Estimating the number of bike rentals based on weather and time of day."

### Core Idea
Divide and Conquer.
Instead of trying to fit one single line to the whole dataset (like Linear Regression), we split the data into groups.
*   Group 1: Houses < 1000 sqft.
*   Group 2: Houses > 1000 sqft.
Then we predict the **average** value for each group.

### Intuition: The "20 Questions" Game
Imagine you are trying to guess the price of a car.
1.  "Is it a Ferrari?" -> Yes -> Price is high ($200k+).
2.  "Is it new?" -> No -> Price is medium ($100k).
3.  "Is it damaged?" -> Yes -> Price is lower ($50k).
You ask a series of Yes/No questions to narrow down the possibilities until you reach a final estimate.

### Visual Interpretation
*   **Tree Structure**: Root node at top, branches going down, leaves at bottom.
*   **Decision Boundary**: Step-function (blocky). It looks like a staircase, not a smooth curve.

### What Makes It Special?
It is **white-box**. You can visualize the tree and see exactly *why* a prediction was made. "The price is $50k because it's a Ferrari, it's old, and it's damaged."

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
Use it when:
1.  **Interpretability** is crucial (e.g., medical diagnosis, loan approval).
2.  Data has **non-linear** relationships.
3.  You don't want to worry about data scaling or normalization.

### Domains & Use Cases
1.  **Real Estate**: Pricing models where rules are distinct (e.g., "Waterfront property adds $100k").
2.  **Operations**: Predicting delivery times based on traffic conditions (If rain -> add 10 mins).
3.  **HR**: Predicting employee salary based on experience and level.

### Type of Algorithm
*   **Learning Type**: Supervised Learning.
*   **Task**: Regression.
*   **Linearity**: Non-Linear (Piecewise constant).
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric (The number of parameters grows with the data).

---

## 📄 Page 3 — Mathematical Foundation

### Core Formula
The tree splits the data space into $M$ regions $R_1, R_2, ..., R_M$.
For any input $x$ that falls into region $R_m$, the prediction is simply the average of the training values in that region:
$$ \hat{y} = \text{avg}(y_i | x_i \in R_m) $$

### Cost Function (Splitting Criteria)
How do we decide where to split? We want to minimize the **Variance** (or MSE) within each child node.
$$ MSE = \frac{1}{N} \sum (y_i - \bar{y})^2 $$
We choose the split that results in the largest **Reduction in Variance**.

### Optimization
It uses a **Greedy Algorithm** (Recursive Binary Splitting).
1.  Consider all features.
2.  Consider all possible split points for each feature.
3.  Calculate the MSE for every possible split.
4.  Choose the split with the lowest weighted MSE.
5.  Repeat recursively for the child nodes.

### Assumptions in Math
*   No assumptions about data distribution (Normal, Linear, etc.).
*   Assumes data is IID.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Start**: Put all data in the Root Node.
2.  **Search**: Loop through every feature (Age, Size, etc.) and every unique value.
3.  **Evaluate**: "If I split Age at 30, what is the MSE of the left group and right group?"
4.  **Split**: Pick the best feature and value (e.g., Age < 30).
5.  **Recurse**: Now you have two nodes. Repeat steps 2-4 for each node.
6.  **Stop**: Stop if max depth is reached, or if a node has too few samples.
7.  **Predict**: For a new data point, traverse the tree. When you hit a leaf, return the average value of the training samples in that leaf.

### Pseudocode
```python
function BuildTree(data):
    if stopping_condition(data):
        return Leaf(average(data.y))
    
    best_split = FindBestSplit(data) # Minimizes MSE
    left_data, right_data = Split(data, best_split)
    
    node = Node(best_split)
    node.left = BuildTree(left_data)
    node.right = BuildTree(right_data)
    return node
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **None**: That's the beauty of it. No scaling, no normality, no linearity needed.

### Hyperparameters
1.  **`max_depth`**:
    *   *Control*: How deep the tree can grow.
    *   *Effect*: Low = Underfitting. High = Overfitting (Memorizes data).
2.  **`min_samples_split`**:
    *   *Control*: Minimum samples required to split a node.
    *   *Effect*: Higher values prevent overfitting (stops growing tiny branches).
3.  **`min_samples_leaf`**:
    *   *Control*: Minimum samples required to be at a leaf node.
    *   *Effect*: Smoothing. If set to 1, the tree can predict based on a single data point (Extreme Overfitting).
4.  **`ccp_alpha`**:
    *   *Control*: Cost Complexity Pruning parameter.
    *   *Effect*: Used to prune the tree after building it.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Missing Value Handling
*   **Sklearn**: Does NOT support missing values. You must impute.
*   **Other Libraries (XGBoost)**: Can handle them automatically.

### 2. Scaling / Normalization
*   **NOT REQUIRED**.
*   Since the tree only asks "Is X > 5?", it doesn't care if X is 5 or 5,000,000. The split point just moves.

### 3. Encoding
*   **One-Hot Encoding**: Required for sklearn.
*   **Label Encoding**: Can be used for Ordinal data.

### 4. Outlier Treatment
*   **Robust**: Trees are generally robust to outliers in the *features* (they just get isolated in a leaf).
*   **Sensitive**: Trees are sensitive to outliers in the *target* (because we use Mean Squared Error, and the mean is pulled by outliers).

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Interpretability**: The "White Box" nature is its biggest selling point.
2.  **No Preprocessing**: Works on raw data (mostly).
3.  **Non-Linear**: Captures complex patterns.
4.  **Feature Importance**: Automatically tells you which features are most useful for splitting.

### Limitations
1.  **Overfitting**: Trees love to memorize noise. Without pruning/limits, they overfit 100% of the time.
2.  **Instability**: Changing one point in the dataset can result in a completely different tree. (High Variance).
3.  **Extrapolation**: Cannot predict outside the range of training data. If training data goes up to $500k, the tree can never predict $1M. It just predicts the max value it saw.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   MSE, RMSE, $R^2$.

### Python Implementation
```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate Data
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16)) # Add noise

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train
# max_depth=2 to prevent overfitting
regr = DecisionTreeRegressor(max_depth=2)
regr.fit(X_train, y_train)

# 4. Predict
y_pred = regr.predict(X_test)

# 5. Evaluate
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# 6. Visualize Tree
plt.figure(figsize=(10,6))
plot_tree(regr, filled=True)
plt.show()

# 7. Visualize Regression Line
X_grid = np.arange(min(X), max(X), 0.01)[:, np.newaxis]
plt.scatter(X, y, color='red')
plt.plot(X_grid, regr.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
We tune `max_depth` and `min_samples_leaf`.

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [2, 4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid = GridSearchCV(DecisionTreeRegressor(), params, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
```

### Real-World Applications
*   **Credit Scoring**: Rules-based decisions are legally required in some places.
*   **Medical Triage**: Simple flowcharts for doctors.

### When to Use
*   When you need to explain the decision to a human.
*   When data is mixed (numerical and categorical).
*   When you don't want to do heavy preprocessing.

### When NOT to Use
*   When you need the absolute highest accuracy (Use Random Forest/XGBoost).
*   When the relationship is strictly linear (Linear Regression is better).
*   When you need to extrapolate.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Random Forest**: Tree is interpretable but unstable. Forest is accurate/stable but a black box.
*   **vs Linear Regression**: Tree fits blocks. Linear fits a line.

### Interview Questions
1.  **Q: Why doesn't a Decision Tree require scaling?** (Because it splits based on order/thresholds, not distance).
2.  **Q: What is Pruning?** (Cutting back branches to reduce complexity and overfitting).
3.  **Q: Can a Decision Tree predict a value it has never seen?** (No, it can only predict the average of a leaf node. It cannot extrapolate trends).

### Summary
Decision Trees are the building blocks of modern ML. They are simple, intuitive flowcharts. While weak on their own (due to overfitting), they are the foundation for powerful ensembles like Random Forest and XGBoost.

### Cheatsheet
*   **Criterion**: MSE (Variance Reduction).
*   **Key Params**: `max_depth`, `min_samples_leaf`.
*   **Pros**: Interpretable, No Scaling.
*   **Cons**: Overfitting, No Extrapolation.
