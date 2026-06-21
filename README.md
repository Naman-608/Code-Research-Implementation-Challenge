# XOR Classifier using Neural Network From Scratch

The code implements a simple **feed-forward neural network from scratch using NumPy**, without using any machine-learning frameworks.
The network learns the **XOR logic function**, demonstrating forward propagation, loss computation, backpropagation, and gradient-descent weight updates. It helps us to understand how backpropagation actually works behind the hood.

---
##  Problem: XOR Logic

| Input A | Input B | Output |
| ------: | ------: | -----: |
|       0 |       0 |      0 |
|       0 |       1 |      1 |
|       1 |       0 |      1 |
|       1 |       1 |      0 |

The XOR problem cannot be solved by a linear model, so it’s a classic example for neural networks.

---

## 📌 Architecture Topology

Single-layer neural networks are mathematically incapable of solving the XOR problem because the coordinate space cannot be bisected by a single straight line. This implementation resolves this by mapping inputs into a higher-dimensional space using a hidden layer:

* **Input Layer ($X$):** 2 Neurons (representing the binary bit states).
* **Hidden Layer ($a_1$):** 4 Neurons with Sigmoid activation to extract non-linear feature combinations.
* **Output Layer ($\hat{y}$):** 1 Neuron yielding a continuous probability scalar between `0.0` and `1.0`.

```text
  Inputs (2)           Hidden Layer (4)         Output (1)
   ( Bit 1 )  ----->     [ Neuron 1 ] 
                         [ Neuron 2 ]  ----->   ( Prediction )
   ( Bit 2 )  ----->     [ Neuron 3 ] 
                         [ Neuron 4 ]
```
---

## 📘 How It Works (Summary)

1. **Forward Pass**

   * Inputs → Hidden Layer → Output Layer
   * Sigmoid squashes outputs between 0–1

2. **Loss Calculation**

   * Mean Squared Error compares predictions with expected outputs

3. **Backpropagation**

   * Gradients are computed using the chain rule
   * Each weight is updated based on its contribution to the error

4. **Gradient Descent Update**

   * Weights and biases are adjusted repeatedly to reduce loss

We can see the loss decreasing during training and final predictions close to:

```
Input: [0 0] → 0
Input: [0 1] → 1
Input: [1 0] → 1
Input: [1 1] → 0
```

---

## ▶️ How To Run

```bash
python src\main.py
```
