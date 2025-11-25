# 🌟 Gradient Descent Explained with Simple Calculations

Gradient Descent is a method to **find the minimum of a function** by taking repeated **small steps downhill**.

Think of it as:

👉 *Check slope → Move opposite the slope → Repeat until flat*

---

# 📘 Example Function

We use a very simple function:

\[
f(x) = x^2
\]

The minimum is clearly at **x = 0**.

---

# 🧮 Step 1: Pick a Starting Point

\[
x = 4
\]

---

# 🧮 Step 2: Compute the Gradient (Derivative)

For  
\[
f(x) = x^2,
\]
the derivative is:

\[
f'(x) = 2x
\]

At **x = 4**:

\[
f'(4) = 2 \cdot 4 = 8
\]

Slope is **+8** (uphill).

---

# 🧮 Step 3: Update the Value of x

Gradient Descent update rule:

\[
x_{\text{new}} = x - \alpha \cdot f'(x)
\]

Let learning rate:

\[
\alpha = 0.1
\]

So:

\[
x_{\text{new}} = 4 - 0.1 \cdot 8 = 3.2
\]

---

# 🔁 Repeat the Steps

## Iteration 2
Current:  
\[
x = 3.2
\]

Gradient:
\[
f'(3.2) = 2 \cdot 3.2 = 6.4
\]

Update:
\[
x_{\text{new}} = 3.2 - 0.1 \cdot 6.4 = 2.56
\]

---

## Iteration 3
Current:
\[
x = 2.56
\]

Gradient:
\[
f'(2.56) = 5.12
\]

Update:
\[
x_{\text{new}} = 2.56 - 0.1 \cdot 5.12 = 2.048
\]

---

## Iteration 4
Current:
\[
x = 2.048
\]

Gradient:
\[
f'(2.048) = 4.096
\]

Update:
\[
x_{\text{new}} = 2.048 - 0.1 \cdot 4.096 = 1.6384
\]

---

# 🎯 Summary Table

| Iteration | x Value | Gradient (2x) | Updated x |
|----------|---------|----------------|-----------|
| 1 | 4.00 | 8.00 | 3.20 |
| 2 | 3.20 | 6.40 | 2.56 |
| 3 | 2.56 | 5.12 | 2.048 |
| 4 | 2.048 | 4.096 | 1.6384 |

---

# ⭐ Final Intuition

- The gradient tells you **which direction is uphill**
- Move **opposite the gradient** (downhill)
- Learning rate (**α**) controls **step size**
- Repeating this will move you toward the minimum

---

Let me know if you want:
- A Python implementation  
- Visual graph  
- Multi-variable gradient descent explanation  
