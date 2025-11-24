# Deep Q-Network (DQN): The Atari Player

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Deep Q-Network (DQN)** combines Q-Learning with Deep Neural Networks. It was the breakthrough algorithm by DeepMind (2013) that learned to play Atari games from raw pixels, surpassing human performance.

### What Problem It Solves
It solves the **Curse of Dimensionality** in Q-Learning.
*   **Q-Learning**: Uses a table. Works for 100 states. Fails for $10^{100}$ states (like an image).
*   **DQN**: Uses a Neural Network to *approximate* the Q-Table.
    $$ Q(s, a) \approx NN(s, a; \theta) $$

### Core Idea
Instead of looking up the Q-value in a table row, we feed the state (image) into a Neural Network. The Network outputs the Q-values for all possible actions.
*   **Input**: State (e.g., Screen pixels).
*   **Output**: Q-values for [Left, Right, Jump].

### Intuition: The "Function Approximator"
Imagine the Q-Table is a landscape.
*   Q-Learning stores the height of every single GPS coordinate.
*   DQN learns the *formula* for the shape of the mountains. It can predict the height of a coordinate it has never seen before (Generalization).

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It allows RL to work in complex, high-dimensional environments (Video games, Robotics, Visual inputs). It is the foundation of modern Deep RL.

### Domains & Use Cases
1.  **Atari Games**: Breakout, Pong, Space Invaders.
2.  **Autonomous Driving**: Mapping camera inputs to steering angles.
3.  **Robotics**: Hand-eye coordination.

### Type of Algorithm
*   **Learning Type**: Reinforcement Learning.
*   **Strategy**: Off-Policy.
*   **Model**: Model-Free.
*   **Structure**: Deep Neural Network (Value-based).

---

## 📄 Page 3 — Mathematical Foundation

### The Loss Function
We want the Neural Network to predict the Q-value.
Target (from Bellman Eq): $Y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
Prediction: $Q(s, a; \theta)$

We minimize the Mean Squared Error (MSE):
$$ L(\theta) = E [ (Y - Q(s, a; \theta))^2 ] $$

### Stability Issues
Training a NN with RL is unstable because:
1.  **Correlated Data**: Sequential frames are highly correlated. (NNs assume IID data).
2.  **Moving Target**: The target $Y$ depends on the network itself ($Q$). Updating the network changes the target immediately, leading to loops/oscillations.

---

## 📄 Page 4 — Working Mechanism (The 2 Tricks)

### 1. Experience Replay (Solving Correlation)
Instead of training on the latest step, we store experiences $(s, a, r, s')$ in a massive **Replay Buffer** (Memory).
*   During training, we sample a **random batch** from memory.
*   This breaks the correlation between consecutive samples.

### 2. Target Network (Solving Moving Target)
We use two networks:
*   **Policy Network ($Q$)**: The one we train.
*   **Target Network ($Q_{target}$)**: A copy of the policy network, frozen in time.
*   We calculate the target $Y$ using the frozen Target Network.
*   Every $C$ steps, we copy weights from Policy Network to Target Network.

### Flow
1.  Play step, store in Memory.
2.  Sample batch from Memory.
3.  Calculate Target using Target Net.
4.  Train Policy Net to match Target.
5.  Update Target Net weights occasionally.

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Hyperparameters
1.  **`buffer_size`**: Size of Replay Memory (e.g., 1,000,000).
2.  **`batch_size`**: e.g., 32 or 64.
3.  **`gamma`**: Discount factor (0.99).
4.  **`epsilon_decay`**: How fast to stop exploring.
5.  **`update_freq`**: How often to update Target Network.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Frame Stacking
*   A single frame (image) doesn't show motion. Is the ball moving Left or Right?
*   We stack 4 consecutive frames together to give the network temporal context.

### 2. Grayscale & Resizing
*   Convert RGB (210x160x3) to Grayscale (84x84x1) to reduce computation.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **High Dimensions**: Can handle raw pixels.
2.  **Generalization**: Can handle unseen states.

### Limitations
1.  **Slow**: Takes days to train on Atari.
2.  **Overestimation**: DQN tends to overestimate Q-values (fixed by Double DQN).
3.  **Discrete Actions**: Cannot handle continuous actions (like "Turn steering wheel 20.5 degrees"). It only does [Left, Right].

---

## 📄 Page 8 — Performance Metrics + Python Code

### Python Implementation (Keras/TensorFlow)
*Conceptual Snippet*

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Extensions
*   **Double DQN**: Reduces overestimation.
*   **Dueling DQN**: Splits Q into Value (V) and Advantage (A).
*   **Prioritized Experience Replay**: Learn from "surprising" events more often.

### When to Use
*   Discrete action spaces.
*   High-dimensional state spaces.

### When NOT to Use
*   Continuous action spaces (Use DDPG/PPO).
*   Simple tabular problems (Overkill).

---

## 📄 Page 10 — Summary + Cheatsheet

### Summary
DQN proved that Neural Networks could learn to play games from scratch. It bridged the gap between perception (CNNs) and action (RL).

### Cheatsheet
*   **Type**: Deep RL, Value-based.
*   **Key**: Experience Replay, Target Network.
*   **Pros**: Handles Images.
*   **Cons**: Unstable, Discrete Actions.
