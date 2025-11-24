# Association Rule Learning: The Basket Analyst

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Association Rule Learning** is a rule-based machine learning method for discovering interesting relations between variables in large databases. It is most famous for "Market Basket Analysis".

### What Problem It Solves
It solves the problem of finding **hidden patterns** in transactional data.
*   "People who buy Bread also buy Butter."
*   "People who watch 'The Office' also watch 'Parks and Rec'."

### Core Idea
"If This, Then That."
It looks for combinations of items that occur together more often than you would expect by random chance.

### Intuition: The "Beer and Diapers" Legend
A famous (urban) legend says that Walmart discovered a strong association between **Beer** and **Diapers** on Friday evenings.
*   *Why?* Young dads were sent to buy diapers, and they grabbed a beer for the weekend.
*   *Action:* Walmart placed beer next to diapers and sales of both went up.

### Visual Interpretation
*   **Antecedent (Left Hand Side)**: The "If" part (e.g., {Bread}).
*   **Consequent (Right Hand Side)**: The "Then" part (e.g., {Butter}).
*   **Rule**: {Bread} -> {Butter}.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the foundation of recommendation engines (before Matrix Factorization took over). It is simple, interpretable, and actionable.

### Domains & Use Cases
1.  **Retail**: Market Basket Analysis (Cross-selling).
2.  **Web Design**: "Users who visited Page A also visited Page B."
3.  **Medicine**: "Patients with Symptom A and B often have Disease C."

### Type of Algorithm
*   **Learning Type**: Unsupervised Learning.
*   **Task**: Pattern Discovery.
*   **Linearity**: N/A.
*   **Probabilistic/Deterministic**: Deterministic.
*   **Parametric**: Non-parametric.

---

## 📄 Page 3 — Mathematical Foundation

### Core Metrics
To evaluate a rule $A \to B$, we use three metrics:

1.  **Support**: How popular is the itemset?
    $$ \text{Support}(A) = \frac{\text{Transactions containing } A}{\text{Total Transactions}} $$
    *   High Support = Popular item.

2.  **Confidence**: How likely is B given A?
    $$ \text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} $$
    *   "If I buy A, there is a 70% chance I buy B."

3.  **Lift**: Is the relationship real or coincidence?
    $$ \text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} $$
    *   **Lift = 1**: A and B are independent (Coincidence).
    *   **Lift > 1**: A and B are positively correlated (Real pattern).
    *   **Lift < 1**: A and B are negatively correlated (Substitutes).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Algorithms
1.  **Apriori Algorithm**:
    *   Uses a "bottom-up" approach.
    *   **Apriori Principle**: "If an itemset is frequent, then all of its subsets must also be frequent."
    *   This prunes the search space drastically.

2.  **Eclat Algorithm**:
    *   Uses a "depth-first" search.
    *   Faster than Apriori for dense datasets.

3.  **FP-Growth (Frequent Pattern Growth)**:
    *   Uses a tree structure (FP-Tree).
    *   Fastest. No candidate generation.

### Flow (Apriori)
1.  **Set Min Support**: Filter out rare items.
2.  **Find Frequent 1-itemsets**: Count individual items.
3.  **Join**: Combine frequent items to make 2-itemsets.
4.  **Prune**: Remove 2-itemsets that fall below Min Support.
5.  **Repeat**: Make 3-itemsets, 4-itemsets...
6.  **Generate Rules**: From frequent itemsets, generate rules that meet Min Confidence.

### Pseudocode
```python
L1 = find_frequent_1_itemsets(data)
for k in 2..N:
    Ck = generate_candidates(Lk-1)
    Lk = filter_by_support(Ck)
    if Lk is empty: break
return generate_rules(Lk)
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Static Data**: Assumes patterns don't change over time (unless you re-run it).

### Hyperparameters
1.  **`min_support`**:
    *   *Control*: Popularity threshold.
    *   *Effect*: High = Only bestsellers. Low = Computationally expensive.
2.  **`min_confidence`**:
    *   *Control*: Reliability threshold.
    *   *Effect*: High = Only strong rules.
3.  **`min_lift`**:
    *   *Control*: Correlation threshold.
    *   *Effect*: Set > 1 to find meaningful patterns.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **NOT REQUIRED**.

### 2. Encoding
*   **Transaction Format**: Data must be a list of lists (transactions) or a One-Hot Encoded DataFrame (True/False).

### 3. Missing Values
*   N/A.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Interpretability**: Rules are easy to understand ("If Bread then Butter").
2.  **Actionable**: Directly leads to business decisions (Store layout).
3.  **Unsupervised**: No labels needed.

### Limitations
1.  **Computation**: Exponential complexity $O(2^N)$. Very slow on large datasets with low min_support.
2.  **Spurious Rules**: Can find obvious or useless rules ("If Pregnant -> Female").
3.  **Rare Items**: Often misses rare but valuable items (Long Tail) because they fall below min_support.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   Support, Confidence, Lift.

### Python Implementation
```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Create Data (One-Hot Encoded)
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 2. Frequent Itemsets (Apriori)
# min_support=0.6 means item must appear in 60% of transactions
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# 3. Generate Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 4. View
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
We tune `min_support` and `min_confidence` iteratively.
*   Start high (Support=0.5). If no rules, lower it.
*   If too many rules, raise it.

### Real-World Applications
*   **Amazon**: "Frequently bought together".
*   **Spotify**: Playlist generation.

### When to Use
*   Transactional data.
*   When you want to find relationships between items.

### When NOT to Use
*   Prediction problems (Use Classification).
*   Continuous data (Must be discretized first).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs Collaborative Filtering**: Apriori looks at Items co-occurrence. Collaborative Filtering looks at User similarity.

### Interview Questions
1.  **Q: What is the difference between Confidence and Lift?** (Confidence is conditional probability. Lift adjusts for the popularity of the consequent).
2.  **Q: Why is Apriori slow?** (It has to scan the database many times).
3.  **Q: How does FP-Growth improve speed?** (It scans the DB only twice and uses a tree structure).

### Summary
Association Rule Learning is the detective of data mining. It finds the hidden links between items that drive sales and engagement.

### Cheatsheet
*   **Metrics**: Support, Confidence, Lift.
*   **Algo**: Apriori, FP-Growth.
*   **Goal**: Find {A} -> {B}.
