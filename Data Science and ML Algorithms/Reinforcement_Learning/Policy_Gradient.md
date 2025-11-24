# Policy Gradient: The Intuitive Actor

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Policy Gradient** methods are a class of RL algorithms that optimize the **Policy** directly, without learning a Value function (Q-value) first.

### What Problem It Solves
It solves the limitations of Value-based methods (DQN):
1.  **Continuous Actions**: DQN can't handle "Steer 20.5 degrees". Policy Gradient can output a probability distribution over continuous actions.
2.  **Stochastic Policies**: DQN is deterministic (argmax). Policy Gradient can learn optimal stochastic policies (e.g., in Rock-Paper-Scissors, you *must* be random).

### Core Idea
"If an action leads to a good reward, increase its probability. If it leads to a bad reward, decrease its probability."
We parameterize the policy $\pi_\theta(a|s)$ with a neural network. We use Gradient Ascent to maximize the expected reward.

### Intuition: The "Trial" Analogy
*   You try a bunch of actions.
*   At the end of the episode, you see if you won or lost.
*   If you won, you say "Whatever I did in this episode, do it more."
*   If you lost, you say "Do it less."

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is more stable and principled for continuous control problems (Robotics).

### Domains & Use Cases
1.  **Robotics**: Controlling robot joints (continuous torque).
2.  **Self-Driving Cars**: Steering and acceleration.
3.  **LLMs**: RLHF (Reinforcement Learning from Human Feedback) uses PPO (a Policy Gradient method) to tune ChatGPT.

### Type of Algorithm
*   **Learning Type**: Reinforcement Learning.
*   **Strategy**: On-Policy.
*   **Model**: Model-Free.
*   **Structure**: Policy-based.

---

## 📄 Page 3 — Mathematical Foundation

### The Objective Function
Maximize expected return:
$$ J(\theta) = E_{\pi_\theta} [ \sum r ] $$

### The Policy Gradient Theorem
How do we find the gradient $\nabla J(\theta)$?
$$ \nabla_\theta J(\theta) = E [ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t ] $$
*   $\nabla \log \pi$: The direction to move weights to make action $a$ more likely.
*   $G_t$: The return (Score).
*   **Interpretation**: Scale the gradient by the score. High score -> Big push. Negative score -> Push opposite way.

---

## 📄 Page 4 — Working Mechanism (REINFORCE Algorithm)

### Flow (Monte Carlo Policy Gradient)

1.  **Run Episode**: Play a full game using current policy $\pi_\theta$. Generate trajectory $(s_0, a_0, r_1, ...)$.
2.  **Compute Returns**: For each step $t$, calculate total future reward $G_t$.
3.  **Update Weights**:
    $$ \theta \leftarrow \theta + \alpha \cdot \nabla \log \pi(a_t|s_t) \cdot G_t $$
4.  **Repeat**.

### The "Baseline" Trick
To reduce variance, we subtract a baseline $b(s)$ (usually the Value function $V(s)$) from the return.
$$ \text{Update} \propto \nabla \log \pi \cdot (G_t - V(s_t)) $$
This is the **Advantage**. "How much better was this action than average?"

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Full Episodes**: REINFORCE requires waiting until the end of the episode to calculate $G_t$. (Slow).

### Hyperparameters
1.  **`learning_rate`**: Very sensitive. Too high = Policy collapse.
2.  **`gamma`**: Discount.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Normalization
*   Returns $G_t$ should be normalized (mean 0, std 1) to stabilize training.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Continuous Actions**: Handles robotics naturally.
2.  **Convergence**: Guaranteed to converge to a local optimum.

### Limitations
1.  **High Variance**: The gradient estimate is very noisy. Training is slow and shaky.
2.  **Sample Inefficiency**: Needs huge amounts of data because it throws away data after one update (On-Policy).

---

## 📄 Page 8 — Performance Metrics + Python Code

### Python Implementation (Conceptual)
```python
# Loss function for Policy Gradient
def loss(action_probs, rewards):
    # log probability of the taken action
    log_prob = tf.log(action_probs)
    # maximize reward = minimize negative reward
    loss = - log_prob * rewards
    return loss
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Advanced Versions
*   **A2C / A3C**: Actor-Critic. Learns both Policy (Actor) and Value (Critic).
*   **PPO (Proximal Policy Optimization)**: The standard today. Limits how much the policy can change in one step to prevent collapse.

### When to Use
*   Robotics / Continuous Control.
*   When DQN fails.

---

## 📄 Page 10 — Summary + Cheatsheet

### Summary
Policy Gradient is the "Muscle Memory" of RL. It directly trains the neural network to output the best action. It is the engine behind the most advanced AI systems today (Robotics, LLMs).

### Cheatsheet
*   **Type**: Policy-based.
*   **Key**: $\nabla \log \pi \cdot R$.
*   **Pros**: Continuous Actions.
*   **Cons**: High Variance.
