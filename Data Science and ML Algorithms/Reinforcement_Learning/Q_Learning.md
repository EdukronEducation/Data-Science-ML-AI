# Q-Learning: The Table Master

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Q-Learning** is a model-free, value-based Reinforcement Learning algorithm. It is one of the most fundamental algorithms in RL. It learns the **Quality (Q-Value)** of an action in a given state.

### What Problem It Solves
It solves the problem of finding the **Optimal Policy** without knowing the rules of the environment (Model-Free). It tells the agent: "If you are in State A, Action B is the best thing to do."

### Core Idea
"The Cheat Sheet."
Imagine a massive cheat sheet (Table) where:
*   **Rows** = Every possible State (e.g., every square on a grid).
*   **Columns** = Every possible Action (Up, Down, Left, Right).
*   **Cells** = The "Score" (Q-Value) of taking that action in that state.
Initially, the table is all zeros. As the agent explores, it updates the scores.
"Hey, I went Down in square (1,1) and fell in a pit. Let's write a negative score there."

### Intuition: The "Maze" Analogy
You are in a maze.
1.  You take a random turn. Nothing happens.
2.  You take another turn. You find cheese (+100).
3.  You remember: "The step *before* the cheese was good."
4.  Next time, you remember: "The step *before* the step before the cheese was also good."
The reward propagates back from the goal to the start.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is simple, mathematically guaranteed to converge to the optimal policy (given enough time), and forms the basis for Deep Q-Networks (DQN).

### Domains & Use Cases
1.  **Grid Worlds**: Pathfinding in simple mazes.
2.  **Tic-Tac-Toe**: Small games with limited states.
3.  **Inventory Management**: Deciding when to order stock.

### Type of Algorithm
*   **Learning Type**: Reinforcement Learning.
*   **Strategy**: Off-Policy (Learns the value of the *optimal* policy while acting randomly).
*   **Model**: Model-Free.
*   **Structure**: Tabular (Lookup Table).

---

## 📄 Page 3 — Mathematical Foundation

### The Bellman Equation
The core of Q-Learning is the Bellman Equation, which describes the relationship between the value of a state and the value of the next state.
$$ Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a') $$
*   The value of doing $a$ in $s$ is the **Immediate Reward** plus the **Best Future Reward** from the next state $s'$.

### The Q-Learning Update Rule
When we take action $a$ in state $s$, get reward $r$, and land in $s'$, we update our Q-table:
$$ Q(s, a) \leftarrow Q(s, a) + \alpha [ r + \gamma \max_{a'} Q(s', a') - Q(s, a) ] $$
*   $\alpha$: Learning Rate.
*   $\gamma$: Discount Factor.
*   $[...]$: **Temporal Difference (TD) Error**. (The difference between what we expected and what we actually got).

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Initialize**: Create a Q-Table of size [Num_States, Num_Actions] with zeros.
2.  **Loop** (for each episode):
    *   Reset state $S$.
    *   **Loop** (for each step):
        *   **Choose Action**: Use Epsilon-Greedy (Random or Max Q).
        *   **Step**: Take action $A$, observe Reward $R$ and Next State $S'$.
        *   **Update**: Apply the formula:
            $$ Q(S, A) = Q(S, A) + \alpha (R + \gamma \max Q(S', :) - Q(S, A)) $$
        *   **Transition**: $S = S'$.
        *   If done, break.

### Pseudocode
```python
Q = zeros(states, actions)
for episode in range(max_episodes):
    s = env.reset()
    while not done:
        # Epsilon Greedy
        if random() < epsilon:
            a = random_action()
        else:
            a = argmax(Q[s, :])
            
        s_next, r, done = env.step(a)
        
        # Update
        Q[s, a] = Q[s, a] + alpha * (r + gamma * max(Q[s_next, :]) - Q[s, a])
        
        s = s_next
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
1.  **Markov Property**: The future depends only on the current state, not the history.
2.  **Discrete States**: Q-Learning requires a table. If the state space is continuous (e.g., position = 1.2345), the table would be infinite. (We need DQN for that).

### Hyperparameters
1.  **`alpha` (Learning Rate)**:
    *   0: Learn nothing.
    *   1: Forget old knowledge instantly.
    *   *Typical*: 0.1.
2.  **`gamma` (Discount Factor)**:
    *   0: Myopic (Care only about now).
    *   1: Far-sighted.
    *   *Typical*: 0.95 or 0.99.
3.  **`epsilon` (Exploration Rate)**:
    *   Start at 1.0, decay to 0.01.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Discretization (Binning)
*   If your state is continuous (e.g., speed), you must "bin" it into buckets (Low, Medium, High) to make it fit in a table.

### 2. State Reduction
*   Remove irrelevant info to keep the table small.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Optimality**: Guaranteed to find the optimal policy for finite MDPs.
2.  **Off-Policy**: Can learn from observed data (e.g., watching someone else play) without following that policy.

### Limitations
1.  **Curse of Dimensionality**: If you have many states, the table becomes too big to store in RAM.
    *   Chess: $10^{120}$ states. Table is impossible.
2.  **Slow Propagation**: Rewards take a long time to propagate from the goal back to the start.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Total Reward per Episode**: Should increase over time.
*   **Steps to Goal**: Should decrease.

### Python Implementation (FrozenLake)
```python
import gym
import numpy as np

# 1. Init
env = gym.make('FrozenLake-v1', is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Params
alpha = 0.8
gamma = 0.95
epsilon = 1.0
decay = 0.999
episodes = 2000

# 2. Train
for i in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon Greedy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
            
        next_state, reward, done, _ = env.step(action)
        
        # Q-Update
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        
        state = next_state
        
    epsilon *= decay

print("Trained Q-Table:")
print(Q)
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
*   **Epsilon Decay**: Crucial. If you stop exploring too early, you get stuck. If you explore too long, you waste time.

### Real-World Applications
*   **Packet Routing**: Routers learn which path is fastest.
*   **Elevator Control**: Optimizing which floor to go to.

### When to Use
*   Small state spaces (Grid world, simple games).
*   Discrete actions.

### When NOT to Use
*   Large state spaces (Images, Real World). Use DQN.
*   Continuous action spaces (Robot arm movement). Use DDPG.

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs SARSA**:
    *   Q-Learning: $Q(s,a) \leftarrow r + \gamma \max Q(s', :)$. (Optimistic / Off-Policy).
    *   SARSA: $Q(s,a) \leftarrow r + \gamma Q(s', a')$. (Realistic / On-Policy).
    *   *Result*: Q-Learning learns the *optimal* path (even if it's risky). SARSA learns the *safest* path (considering exploration noise).

### Interview Questions
1.  **Q: Why is Q-Learning called "Off-Policy"?** (Because it updates its Q-values assuming the *best* action will be taken next ($\max Q$), even if the agent actually takes a random action ($\epsilon$-greedy)).
2.  **Q: What is the Q-function?** (Expected cumulative future reward for taking action $a$ in state $s$).

### Summary
Q-Learning is the spreadsheet of AI. It meticulously records the value of every move in every situation. While limited by memory, it is the foundational concept for modern Deep RL.

### Cheatsheet
*   **Type**: Value-based, Off-Policy.
*   **Key**: Q-Table, Bellman Eq.
*   **Update**: $Q_{new} = Q_{old} + \alpha(Target - Q_{old})$.
*   **Pros**: Optimal.
*   **Cons**: Table size.
