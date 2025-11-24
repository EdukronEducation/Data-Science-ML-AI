# Factor Analysis: The Latent Variable Finder

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Factor Analysis (FA)** is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called **factors**.

### What Problem It Solves
It solves the problem of **Latent Variable Discovery**.
*   **Psychology**: You can't measure "Intelligence" directly. You measure "Math Score", "Verbal Score", "Logic Score". Factor Analysis infers the hidden variable "Intelligence" that causes these scores.
*   **Marketing**: You can't measure "Brand Loyalty". You measure "Repeat Purchase", "Recommendation", "Satisfaction". FA finds the underlying loyalty factor.

### Core Idea
"Where there is smoke, there is fire."
*   **Observed Variables (Smoke)**: The things we measure ($X_1, X_2, ...$).
*   **Latent Factors (Fire)**: The hidden causes ($F_1, F_2$).
FA assumes that the observed variables are linear combinations of the potential factors, plus some "noise" or unique variance.

### Intuition: The "Student Grades" Analogy
Imagine a student takes 5 tests: Algebra, Calculus, Geometry, Literature, History.
*   A student good at Algebra is likely good at Calculus. Why? Because of a hidden factor: **Quantitative Ability**.
*   A student good at Literature is likely good at History. Why? Because of a hidden factor: **Verbal Ability**.
Factor Analysis groups the 5 tests into 2 Factors.

### Visual Interpretation
It looks like PCA (arrows pointing in directions of variance), but the interpretation is causal. The arrows represent the *underlying* constructs.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the standard tool in **Social Sciences**, **Psychology**, and **Survey Analysis**. Unlike PCA (which just compresses data), FA tries to *model* the data structure.

### Domains & Use Cases
1.  **Psychometrics**: IQ tests, Personality tests (Big Five).
2.  **Marketing**: Survey analysis. "Do people buy this car because of Price or Safety?"
3.  **Finance**: Identifying underlying economic factors driving stock returns (e.g., GDP, Inflation).

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Dimensionality Reduction / Latent Variable Modeling.
*   **Linearity**: Linear.
*   **Probabilistic/Deterministic**: Probabilistic.
*   **Parametric**: Parametric.

---

## 📄 Page 3 — Mathematical Foundation

### The Model
$$ X = \mu + L F + \epsilon $$
*   $X$: Observed variables (vector).
*   $\mu$: Mean.
*   $L$: **Loading Matrix**. How much Factor $F$ contributes to Variable $X$.
*   $F$: Latent Factors (Common variance).
*   $\epsilon$: Error term (Unique variance).

### Factor Loadings
A "Loading" is the correlation between a variable and a factor.
*   Loading of 0.8: Strong positive relationship.
*   Loading of 0.1: No relationship.

### Rotation
Unlike PCA, FA allows **Rotation**. We can rotate the axes to make the factors more interpretable.
*   **Varimax Rotation**: Orthogonal. Forces loadings to be high or low (sparse). Makes it easier to say "Factor 1 is purely Math".
*   **Promax Rotation**: Oblique. Allows factors to be correlated (e.g., Math ability and Verbal ability might be correlated).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Correlation Matrix**: Compute correlations between all variables.
2.  **Extract Factors**: Find the number of factors.
3.  **Calculate Loadings**: Estimate the matrix $L$.
4.  **Rotate**: Rotate the factors to maximize interpretability (Simple Structure).
5.  **Interpret**: Name the factors based on which variables load highly on them.

### Pseudocode
```python
# Conceptual
corr_matrix = correlation(data)
eigenvalues = eig(corr_matrix)
n_factors = select_factors(eigenvalues)
loadings = estimate_loadings(corr_matrix, n_factors)
rotated_loadings = varimax(loadings)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Normality**: Assumes data is multivariate normal.
2.  **Linearity**: Assumes linear relationship between factors and variables.
3.  **No Multicollinearity**: Actually, FA *needs* correlation to work, but singularity is bad.
4.  **Sample Size**: Needs a lot of data (Rule of thumb: 10 observations per variable).

### Hyperparameters
1.  **`n_components`**: Number of factors.
2.  **`rotation`**:
    *   'varimax' (Orthogonal, most common).
    *   'quartimax'.
    *   None.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**.

### 2. Adequacy Tests
Before running FA, you must check if it's suitable.
*   **Bartlett's Test of Sphericity**: Checks if variables are correlated at all. (p-value < 0.05).
*   **KMO Test (Kaiser-Meyer-Olkin)**: Measures sampling adequacy. (> 0.6 is good).

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Interpretability**: Rotation makes factors much easier to name than PCA components.
2.  **Theory-Driven**: Allows testing hypotheses about underlying structures.
3.  **Noise Handling**: Separates "Common Variance" (Signal) from "Unique Variance" (Noise). PCA mixes them.

### Limitations
1.  **Subjective**: Naming the factors is an art, not science.
2.  **Rotation Ambiguity**: Different rotations give different numbers. Which is "true"?
3.  **Linear**: Fails on non-linear relationships.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Factor Loadings**: High loadings (> 0.6) are good.
*   **Cronbach's Alpha**: Measures internal consistency of the factors.

### Python Implementation
```python
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
data = load_iris()
X = data.data
feature_names = data.feature_names

# 2. Scale
X_scaled = StandardScaler().fit_transform(X)

# 3. Train FA
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)

# 4. View Loadings
loadings = pd.DataFrame(fa.components_.T, 
                        index=feature_names, 
                        columns=['Factor 1', 'Factor 2'])
print(loadings)

# 5. Visualize
plt.figure(figsize=(8,6))
plt.scatter(X_fa[:, 0], X_fa[:, 1], c=data.target, cmap='viridis')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.title('Factor Analysis of Iris')
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning (Choosing Factors)
*   **Kaiser Criterion**: Keep factors with Eigenvalue > 1.
*   **Scree Plot**: Look for the elbow.

### Real-World Applications
*   **Employee Satisfaction**: Grouping 50 survey questions into "Management", "Work-Life Balance", "Pay".
*   **Dietary Patterns**: Grouping food items into "Western Diet", "Mediterranean Diet".

### When to Use
*   When you suspect **Hidden Variables**.
*   When you want to **Interpret** the dimensions (Naming them).
*   Survey data.

### When NOT to Use
*   When you just want to compress data for a classifier (Use PCA).
*   When variables are uncorrelated.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs PCA**:
    *   **PCA**: Summarizes variance. Components are mathematical abstractions. Includes noise.
    *   **FA**: Models covariance. Factors are real-world constructs. Excludes noise.
    *   *Analogy*: PCA is like compressing a JPEG. FA is like finding the original objects in the photo.

### Interview Questions
1.  **Q: What is the difference between Common Variance and Unique Variance?** (Common = shared with other variables. Unique = specific to this variable + error. FA models Common. PCA models Total).
2.  **Q: Why rotate factors?** (To achieve "Simple Structure" where each variable loads highly on only one factor, making it interpretable).

### Summary
Factor Analysis is the detective of statistics. It looks at the messy, correlated evidence (observed variables) to deduce the identity of the hidden suspects (latent factors).

### Cheatsheet
*   **Type**: Latent Variable Model.
*   **Key**: Loadings, Rotation.
*   **Param**: `n_components`, `rotation`.
*   **Pros**: Interpretable.
*   **Cons**: Subjective.
