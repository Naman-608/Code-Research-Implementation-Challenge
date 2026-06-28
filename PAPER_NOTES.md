# Paper: Learning Representations by Back-Propagating Errors
#### Authors: Rumelhart, Hinton & Williams
---
### What problem are they solving?
The paper states that neural networks in the 1980s (called "perceptrons") could only learn when we directly connected inputs to outputs. When we add hidden layers the learning would break and nobody knew how to train them. This is when the authors proposed the idea of **Backpropagation**. Using Backpropagation the datasets having non linear relationship between the input and output nodes could be efficiently trained. The idea of connecting inputs directly to the outputs works only when there is linear relationship in them.

During Backpropagation, the network traces backwards from the output nodes to the input nodes. At each layer it first calculates the error of that layer with the expected output and then accordingly adjusts the weights of the previous layer so that the expected output is obtained in the current layer.

The network performs a forward pass calculating results at each layer according to previous layer inputs, then calculates error using loss function and then backpropagates the error. The network does this for some thousand iterations and then reduces the error to great extent. This method is better than previous ones as here the network is actually learning how to process a non linear dataset.

---
### Mathematical equations used:
- Equation 1 — : `xⱼ = Σᵢ yᵢ · wⱼᵢ` The total input to unit j is the weighted sum of all connected units
- Equation 2 — : `yⱼ = 1 / (1 + e^(−xⱼ))` The output is a smooth, nonlinear function (sigmoid). It converges the output in the range of 0 to 1
- Equation 3 - : The term `yⱼ(1 − yⱼ)` is the derivative of sigmoid function and is used during finding error.
--- 
### Datasets used by the Paper:
1.  Symmetry: 64 possible 6-bit binary input vectors; the weight set was adjusted over 1,425 iterations through all 64 examples
2.  Family trees: 104 possible (person, relationship) triples; the network was trained on 100 of them, with 4 held out to test generalization.

### Evaluation metrics used by Paper:
The paper uses the mean squared error function `E = ½ Σ (y − d)²`. For evaluation, they use a practical threshold such that, a unit is considered "ON" if its output > 0.8 and "OFF" if < 0.2. Output units that should be ON but score below 0.8, or should be OFF but score above 0.2, count as errors.

Training parameters: ε = learning rate, α = momentum 
