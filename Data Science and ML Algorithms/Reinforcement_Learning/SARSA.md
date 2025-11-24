# SARSA (State-Action-Reward-State-Action): The Cautious Learner

## 📄 Page 1 — Introduction / Intuition

### Introduction
**SARSA** is a model-free, value-based Reinforcement Learning algorithm. It is very similar to Q-Learning, but with one key difference: it is **On-Policy**.

### What Problem It Solves
It solves the problem of learning a policy while **taking into account the current exploration strategy**.
Q-Learning assumes "I will act optimally in the future."
SARSA assumes "I will act according to my current (possibly random) behavior in the future."

### Core Idea
The name **SARSA** comes from the update tuple:
$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.
1.  You are in **S**.
2.  You take Action **A**.
3.  You get Reward **R**.
4.  You land in **S'**.
5.  You *actually* choose the next Action **A'** (using your epsilon-greedy policy).
6.  Only THEN do you update the value of (S, A).

### Intuition: The "Cliff Walking" Analogy
Imagine a path along a cliff.
*   **Optimal Path**: Right along the edge. (Fastest).
*   **Safe Path**: A few steps away from the edge. (Slower).
*   **Q-Learning**: Learns the Optimal Path. But since the agent explores (randomly moves), it often falls off the cliff during training.
*   **SARSA**: Realizes "Hey, I often move randomly. If I walk on the edge, I might fall. I better walk on the Safe Path."
SARSA learns a safer policy during training.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is preferred when **safety during training** is important. If falling off the cliff breaks the robot, you want SARSA, not Q-Learning.

### Domains & Use Cases
1.  **Robotics**: Where damage is expensive.
2.  **Online Learning**: Where the agent learns while performing the task in the real world.

### Type of Algorithm
*   **Learning Type**: Reinforcement Learning.
*   **Strategy**: On-Policy (Learns the value of the policy being followed).
*   **Model**: Model-Free.
*   **Structure**: Tabular.

---

## 📄 Page 3 — Mathematical Foundation

### The SARSA Update Rule
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [ r + \gamma Q(s', a') - Q(s, a) ] $$

### Comparison with Q-Learning
*   **Q-Learning**: $r + \gamma \max_{a'} Q(s', a')$
    *   Uses the *best possible* next action.
*   **SARSA**: $r + \gamma Q(s', a')$
    *   Uses the *actual* next action $a'$ that was chosen.

This subtle difference changes everything. SARSA's target depends on the policy (including epsilon). Q-Learning's target is greedy.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Initialize**: Q-Table.
2.  **Loop** (Episode):
    *   Reset $S$.
    *   Choose $A$ (Epsilon-Greedy).
    *   **Loop** (Step):
        *   Take action $A$, observe $R, S'$.
        *   Choose next action $A'$ (Epsilon-Greedy) based on $S'$.
        *   **Update**:
            $$ Q(S, A) = Q(S, A) + \alpha (R + \gamma Q(S', A') - Q(S, A)) $$
        *   $S = S'$.
        *   $A = A'$. (Pass the action to the next step).

### Pseudocode
```python
for episode in range(max_episodes):
    s = env.reset()
    a = epsilon_greedy(s) # Choose first action
    
    while not done:
        s_next, r, done = env.step(a)
        a_next = epsilon_greedy(s_next) # Choose next action
        
        # Update using ACTUAL next action
        Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
        
        s = s_next
        a = a_next
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   Same as Q-Learning (Markov Property, Discrete States).

### Hyperparameters
1.  **`alpha`**: Learning rate.
2.  **`gamma`**: Discount factor.
3.  **`epsilon`**: Exploration rate.
    *   *Note*: As $\epsilon \to 0$, SARSA converges to Q-Learning (because the "actual" action becomes the "greedy" action).

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Discretization
*   Required for tabular methods.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Safety**: Avoids dangerous states that are close to optimal states if exploration is high.
2.  **Convergence**: Converges to the optimal policy if $\epsilon$ decays to 0.

### Limitations
1.  **Performance**: The final policy might be suboptimal (too cautious) if $\epsilon$ is not decayed properly.
2.  **Tabular**: Suffers from Curse of Dimensionality.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Cumulative Reward**.

### Python Implementation
```python
import gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

alpha = 0.8
gamma = 0.95
epsilon = 0.1 # Constant exploration for demo
episodes = 2000

for i in range(episodes):
    state = env.reset()
    
    # Choose A
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
        
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        
        # Choose A'
        if np.random.uniform(0, 1) < epsilon:
            next_action = env.action_space.sample()
        else:
            next_action = np.argmax(Q[next_state, :])
            
        # SARSA Update
        target = reward + gamma * Q[next_state, next_action]
        Q[state, action] += alpha * (target - Q[state, action])
        
        state = next_state
        action = next_action # Critical Step

print("Trained Q-Table (SARSA):")
print(Q)
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
*   **Epsilon**: In SARSA, epsilon directly affects the learned value. High epsilon = Very safe/conservative policy.

### Real-World Applications
*   **Robot Navigation**: Avoiding obstacles where hitting one is costly.

### When to Use
*   When you care about the agent's performance *during* training.
*   When exploration is dangerous.

### When NOT to Use
*   When you want the absolute optimal path regardless of risk.
*   When training is done in a simulator where crashes don't matter.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **Q-Learning**: Optimistic. "I'll assume I stop making mistakes."
*   **SARSA**: Realistic. "I know I make mistakes, so I'll be careful."

### Interview Questions
1.  **Q: Is SARSA On-Policy or Off-Policy?** (On-Policy. It learns the value of the policy it is actually executing, including the random exploration steps).
2.  **Q: When do SARSA and Q-Learning converge to the same thing?** (When $\epsilon = 0$, because then the "actual" action is the "greedy" action).

### Summary
SARSA is the prudent sibling of Q-Learning. It looks before it leaps, considering its own clumsiness (exploration) when evaluating a path.

### Cheatsheet
*   **Type**: On-Policy.
*   **Key**: $Q(s', a')$.
*   **Pros**: Safe training.
