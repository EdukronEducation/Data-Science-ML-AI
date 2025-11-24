# Anomaly Detection: The Outlier Hunter

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Anomaly Detection** (or Outlier Detection) is the identification of rare items, events, or observations which raise suspicions by differing significantly from the majority of the data.

### What Problem It Solves
It finds the "Needle in the Haystack".
*   "Is this credit card transaction fraudulent?"
*   "Is this machine about to fail?"
*   "Is this network traffic a hacker?"

### Core Idea
"Normal is common. Abnormal is rare and different."
The algorithm learns what "Normal" looks like. Anything that deviates too much from this definition is flagged as an anomaly.

### Intuition: The "Black Sheep" Analogy
Imagine a field of 1000 white sheep.
*   If you see one black sheep, it stands out.
*   If you see a wolf, it stands out even more.
Anomaly detection measures "how different" an object is from the flock.

### Visual Interpretation
*   **Density**: Normal points are clustered together (High Density). Anomalies are far away in low-density regions.
*   **Distance**: Anomalies are far from the center of the data.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is critical for safety, security, and quality control. Often, the "anomaly" is the most valuable or dangerous part of the data.

### Domains & Use Cases
1.  **Fraud Detection**: Banks (Stolen cards).
2.  **Intrusion Detection**: Cybersecurity (DDoS attacks).
3.  **Manufacturing**: Defect detection (Broken parts).
4.  **Health**: Tumor detection (MRI scans).

### Type of Algorithm
*   **Learning Type**: Unsupervised (mostly), but can be Supervised (if labeled) or Semi-Supervised.
*   **Task**: Classification (Normal vs Anomaly).
*   **Linearity**: N/A.
*   **Probabilistic/Deterministic**: Probabilistic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Approaches
1.  **Statistical (Z-Score)**:
    *   Assume data is Normal.
    *   If $x$ is $> 3$ standard deviations from mean, it's an outlier.
    $$ z = \frac{x - \mu}{\sigma} $$

2.  **Isolation Forest**:
    *   Randomly split data.
    *   Anomalies are "easy to isolate" (require few splits).
    *   Normal points are "hard to isolate" (require many splits).

3.  **Local Outlier Factor (LOF)**:
    *   Compare local density of a point to its neighbors.
    *   If density is much lower than neighbors, it's an outlier.

4.  **One-Class SVM**:
    *   Fits a tight boundary around the "Normal" data.

---

## 📄 Page 4 — Working Mechanism (Isolation Forest)

### Flow of Isolation Forest
1.  **Subsampling**: Select a random subset of data.
2.  **Tree Building**:
    *   Pick a random feature.
    *   Pick a random split value.
    *   Repeat until point is isolated.
3.  **Path Length**: Count how many splits it took.
4.  **Score**:
    *   Short Path = Anomaly (Easy to isolate).
    *   Long Path = Normal (Hard to isolate).
5.  **Aggregate**: Average path length over many trees.

### Pseudocode
```python
# Isolation Forest Logic
for tree in forest:
    splits = 0
    while not isolated(point):
        random_split()
        splits += 1
    path_lengths.append(splits)

score = 2 ^ (-mean(path_lengths) / c)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Anomalies are Rare**: If 50% of your data is anomalies, this won't work.
*   **Anomalies are Different**: They must lie in a different region of feature space.

### Hyperparameters (Isolation Forest)
1.  **`n_estimators`**: Number of trees (100 is usually enough).
2.  **`contamination`**:
    *   *Control*: Expected % of outliers in dataset.
    *   *Effect*: Sets the threshold for the decision function. e.g., 'auto' or 0.05.
3.  **`max_samples`**: Number of samples to draw for each tree.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **Distance-based (LOF, SVM)**: **REQUIRED**.
*   **Tree-based (Isolation Forest)**: **NOT REQUIRED**.

### 2. Encoding
*   **Required**.

### 3. Noise
*   The algorithm *finds* noise. So you don't remove it beforehand.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Unsupervised**: Doesn't need labeled "Fraud" data (which is rare).
2.  **Isolation Forest**: Very fast and effective on high-dimensional data.
3.  **Versatile**: Works on text, images, time-series.

### Limitations
1.  **Contamination Parameter**: You often need to guess how many outliers exist (e.g., 1%? 5%?).
2.  **False Positives**: Often flags weird but legitimate events.
3.  **Definition**: "Anomaly" is subjective.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Since we often don't have labels, we use qualitative analysis.
*   If we have labels: Precision, Recall, F1, ROC-AUC.

### Python Implementation (Isolation Forest)
```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import numpy as np

# 1. Generate Data (Normal + Outliers)
X, _ = make_blobs(n_samples=300, centers=1, cluster_std=1.0, random_state=42)
X_outliers = np.random.uniform(low=-10, high=10, size=(20, 2))
X = np.r_[X, X_outliers] # Combine

# 2. Train
# contamination=0.06 (approx 20/320)
clf = IsolationForest(n_estimators=100, contamination=0.06, random_state=42)
clf.fit(X)

# 3. Predict
# 1 = Normal, -1 = Anomaly
y_pred = clf.predict(X)

# 4. Count
n_errors = (y_pred == -1).sum()
print(f"Detected {n_errors} anomalies.")
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
Hard to tune without labels. Usually, we tune `contamination` based on domain knowledge ("We expect 1% fraud").

### Real-World Applications
*   **Server Monitoring**: CPU usage spikes.
*   **Credit Cards**: "You usually spend $50 in NY. Today you spent $5000 in Paris."

### When to Use
*   When you have no labels (Unsupervised).
*   When you have highly imbalanced labels (Supervised Anomaly Detection).

### When NOT to Use
*   When you have a balanced dataset of "Normal" vs "Bad" (Use Classification).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Classification**: Classification learns "What is Class A" and "What is Class B". Anomaly Detection learns "What is Normal" and rejects everything else.

### Interview Questions
1.  **Q: How does Isolation Forest work?** (It isolates anomalies by randomly splitting data. Anomalies are isolated quickly/shallowly).
2.  **Q: What is the difference between Novelty Detection and Outlier Detection?** (Outlier = Training data contains outliers. Novelty = Training data is clean, we want to find new anomalies in test data).

### Summary
Anomaly Detection is the security guard of Machine Learning. It watches the data stream and flags anything suspicious.

### Cheatsheet
*   **Algo**: Isolation Forest, LOF, One-Class SVM.
*   **Key**: Normal vs Rare.
*   **Param**: `contamination`.
