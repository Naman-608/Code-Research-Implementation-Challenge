import numpy as np

# Data (XOR), Hyperparameters & Seed
np.random.seed(42)
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
lr = 0.5 # Learning rate
max_epochs, early_stop = 20000, 1e-3

# Network Architecture Initialization
w1, b1 = np.random.randn(2, 4).astype(np.float32) * 0.1, np.zeros((1, 4), dtype=np.float32)
w2, b2 = np.random.randn(4, 1).astype(np.float32) * 0.1, np.zeros((1, 1), dtype=np.float32)

# Activation & Derivative
sig = lambda x: 1 / (1 + np.exp(-x))
dsig = lambda a: a * (1 - a)

# Training Loop
for epoch in range(max_epochs):
    # Forward Pass
    a1 = sig(np.dot(X, w1) + b1)
    y_hat = sig(np.dot(a1, w2) + b2)
    
    loss = np.mean((y - y_hat) ** 2) # Loss Calculation
    
    # Logging & Early Stopping
    if loss <= early_stop or epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        if loss <= early_stop: break

    # Backpropagation (Gradients calculation)
    dz2 = (y_hat - y) * dsig(y_hat)
    dz1 = np.dot(dz2, w2.T) * dsig(a1)
    
    # Weight & Bias Updates
    w2 -= lr * np.dot(a1.T, dz2)
    b2 -= lr * np.sum(dz2, axis=0, keepdims=True)
    w1 -= lr * np.dot(X.T, dz1)
    b1 -= lr * np.sum(dz1, axis=0, keepdims=True)

print("Final output:\n", y_hat)