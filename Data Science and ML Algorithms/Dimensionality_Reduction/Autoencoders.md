# Autoencoders: The Neural Compressor

## 📄 Page 1 — Introduction / Intuition

### Introduction
**Autoencoders** are a specific type of neural network designed to learn a compressed representation (encoding) of data. They are an unsupervised learning technique where the goal is to learn the identity function: $f(x) \approx x$.

### What Problem It Solves
It solves the problem of **Non-Linear Dimensionality Reduction** using the power of Deep Learning.
*   PCA is linear.
*   Autoencoders can learn complex, non-linear manifolds by stacking layers of neurons with non-linear activation functions (ReLU, Sigmoid).

### Core Idea
"Memorize the important stuff, forget the noise."
Imagine you have to send a photo to your friend, but your internet is extremely slow.
1.  **Encoder**: You describe the photo in 10 words ("Cat on a red sofa").
2.  **Bottleneck**: You send these 10 words.
3.  **Decoder**: Your friend tries to draw the photo based on those 10 words.
If the drawing looks like the original, your 10 words captured the essence of the image.

### Intuition: The "Hourglass" Shape
The network looks like an hourglass.
*   **Input Layer**: Wide (e.g., 784 pixels).
*   **Hidden Layers**: Get narrower.
*   **Bottleneck (Latent Space)**: The narrowest point (e.g., 32 neurons). This forces compression.
*   **Output Layer**: Wide again (784 pixels).

### Visual Interpretation
*   **Input**: A noisy image of a handwritten '2'.
*   **Latent**: A vector of 3 numbers [0.1, 0.9, -0.5].
*   **Output**: A clean, sharp image of a '2'.

---

## 📄 Page 2 — Why This Algorithm? (Use Cases) + Type

### Why This Algorithm?
It is the bridge between classical ML and Deep Learning. It is incredibly versatile: it can be used for compression, denoising, generation, and anomaly detection.

### Domains & Use Cases
1.  **Image Denoising**: Train on (Noisy Image) -> Target (Clean Image). The network learns to ignore noise.
2.  **Anomaly Detection**: Train on "Normal" data. If you feed it an "Anomaly", the reconstruction error will be huge (because it never learned how to compress/reconstruct anomalies).
3.  **Genomics**: Compressing thousands of gene expressions into a few latent features.
4.  **Image Generation**: Variational Autoencoders (VAEs) can generate new images.

### Type of Algorithm
*   **Learning Type**: Unsupervised (Self-Supervised).
*   **Task**: Dimensionality Reduction / Reconstruction.
*   **Linearity**: Non-Linear.
*   **Probabilistic/Deterministic**: Deterministic (Standard AE) or Probabilistic (VAE).
*   **Parametric**: Parametric (Weights of the network).

---

## 📄 Page 3 — Mathematical Foundation

### Architecture
1.  **Encoder**: Maps input $x$ to latent code $h$.
    $$ h = \sigma(W_e x + b_e) $$
2.  **Decoder**: Maps latent code $h$ back to reconstruction $x'$.
    $$ x' = \sigma(W_d h + b_d) $$

### Loss Function
We minimize the **Reconstruction Loss** (how different is Output from Input?).
*   **MSE (Mean Squared Error)**: For continuous data.
    $$ L = \frac{1}{N} \sum ||x - x'||^2 $$
*   **Binary Cross Entropy**: For binary data (pixels 0 or 1).
    $$ L = - \sum [x \log(x') + (1-x) \log(1-x')] $$

### Regularization
To prevent the network from just "copying" the data, we add constraints:
*   **Sparsity (L1)**: Force most neurons to be inactive.
*   **Noise (Denoising AE)**: Corrupt the input, force it to predict the clean output.

---

## 📄 Page 4 — Working Mechanism (Step-by-Step Logic)

### Flow of the Algorithm

1.  **Input**: Feed raw data $x$ (e.g., image).
2.  **Encode**: Pass through hidden layers. Dimensions reduce.
3.  **Bottleneck**: The data is now a compressed vector $z$.
4.  **Decode**: Pass through upsampling layers. Dimensions increase.
5.  **Reconstruction**: Output $x'$.
6.  **Backpropagation**: Calculate Loss ($x$ vs $x'$). Update weights to minimize loss.

### Pseudocode
```python
# Keras Style
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
bottleneck = Dense(32, activation='relu')(encoded) # Latent

decoded = Dense(64, activation='relu')(bottleneck)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train) # Note: x_train is both input and target
```

---

## 📄 Page 5 — Assumptions + Hyperparameters

### Assumptions
*   **Manifold Hypothesis**: Assumes high-dimensional data lies on a lower-dimensional manifold.
*   **Data Availability**: Needs a LOT of data to train effectively compared to PCA.

### Hyperparameters
1.  **`encoding_dim`**: Size of the bottleneck.
    *   *Small*: High compression, blurry reconstruction.
    *   *Large*: Low compression, risk of identity mapping (no learning).
2.  **`layers`**: Depth of the network. Deeper = more complex features.
3.  **`activation`**: ReLU is standard. Sigmoid/Tanh for output (depending on data range).
4.  **`epochs` & `batch_size`**: Standard DL params.

---

## 📄 Page 6 — Data Preprocessing Required

### 1. Scaling
*   **MANDATORY**. Neural networks require inputs to be scaled (0-1 or -1 to 1).
*   Use `MinMaxScaler` if using Sigmoid output.
*   Use `StandardScaler` if using Linear output.

### 2. Flattening
*   If using Dense layers, images must be flattened. If using CNN (Conv2D), keep them as tensors.

---

## 📄 Page 7 — Strengths + Limitations

### Strengths
1.  **Non-Linear**: Can capture much more complex relationships than PCA.
2.  **Flexible**: Can work on Images (CNN), Text (RNN/Transformer), Graphs (GNN).
3.  **Generative**: VAEs can generate new content.

### Limitations
1.  **Black Box**: The latent features are uninterpretable. "Neuron 5 activation" means nothing.
2.  **Training**: Harder to train than PCA. Needs GPU, tuning, more data.
3.  **Local Minima**: Can get stuck.

---

## 📄 Page 8 — Performance Metrics + Python Code

### Performance Metrics
*   **Reconstruction Error (MSE)**: Lower is better.

### Python Implementation
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist

# 1. Load Data
(x_train, _), (x_test, _) = mnist.load_data()

# 2. Preprocess
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 3. Build Model
encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded) # Separate encoder model

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 4. Train
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 5. Visualize
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

---

## 📄 Page 9 — Hyperparameter Tuning + Applications + When to Use

### Hyperparameter Tuning
*   **Bottleneck Size**: Plot Reconstruction Error vs Bottleneck Size. Pick the "Elbow".
*   **Regularization**: Add L1 activity regularization to learn sparse features.

### Real-World Applications
*   **Super-Resolution**: Upscaling low-res images.
*   **Semantic Hashing**: Compressing documents into binary codes for fast search.

### When to Use
*   Complex, non-linear data (Images, Audio).
*   When PCA fails to capture the structure.
*   Anomaly Detection.

### When NOT to Use
*   Simple tabular data (PCA is faster and interpretable).
*   Small datasets (Overfitting).

---

## 📄 Page 10 — Comparisons + Visualization + Interview Qs + Summary + Cheatsheet

### Comparison
*   **vs PCA**: An Autoencoder with 1 hidden layer and Linear activation is mathematically equivalent to PCA. Deep Autoencoders are "Non-Linear PCA".
*   **vs GANs**: Autoencoders try to reproduce input. GANs try to generate new realistic input. VAEs sit in the middle.

### Interview Questions
1.  **Q: Why do we want the bottleneck to be small?** (To force the network to learn meaningful features/compression. If it's too wide, it just memorizes the input).
2.  **Q: What is a Denoising Autoencoder?** (One where we add noise to the input but train it to predict the clean original. This forces it to learn robust features).

### Summary
Autoencoders are the "Self-Taught" students of Deep Learning. By trying to recreate what they see through a tiny keyhole (bottleneck), they learn to understand the fundamental structure of the data.

### Cheatsheet
*   **Type**: Neural Network.
*   **Key**: Encoder -> Bottleneck -> Decoder.
*   **Loss**: Reconstruction Error.
*   **Pros**: Non-linear, Flexible.
*   **Cons**: Black box, Needs Data.
