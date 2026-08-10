# A Custom Neural Network Framework from Scratch

A lightweight, educational deep learning framework written in **pure Python** using only the native `math` module. No NumPy, no PyTorch, no external dependencies — every forward pass, backward pass, and weight update is implemented by hand.

## Features

* **Zero Dependencies:** Runs natively on any basic Python 3 environment.
* **Full Training Loop:** Forward propagation, backpropagation, and gradient-descent weight updates are all implemented — not just inference.
* **Modern Activations:** Custom implementations of `ReLU`, `LeakyReLU`, `Sigmoid`, `TanH`, `Softmax`, `Swish`, `GELU`, `ELU`, `SELU`, and `Mish`, each paired with a matching derivative for backpropagation.
* **Swappable Loss Functions:** Mean Squared Error, Mean Absolute Error, and Huber Loss, each with a matching gradient function — set per-network and looked up dynamically, no hardcoding.
* **Modular, Object-Oriented Design:** `Neuron` → `Layer` → `NeuralNetwork`, with each layer's activation function chosen independently by name.

## Architecture

| Class | Responsibility |
| :--- | :--- |
| `Neuron` | Holds its own weights and bias; computes a weighted sum, and updates itself given a gradient. |
| `Layer` | A collection of neurons sharing one activation function; runs forward and backward passes across all of them at once. |
| `NeuralNetwork` | A sequence of `Layer`s; orchestrates the full forward → loss → backward → update training cycle. |

## Project Structure

```
src/nnframework/
├── core.py                    # Neuron, Layer, NeuralNetwork
└── functions/
    ├── activations.py         # Forward activation functions
    ├── bactivation.py         # Activation derivatives (for backprop)
    ├── losses.py               # Forward loss functions
    └── blosses.py               # Loss derivatives (for backprop)
└── examples/
    └── foodscience/
        └── database/
            ├── fooddata.db
        ├── datainspect.py
        ├── predict.py
        ├── train.py 
tests/
├── actcalc.py                 # Sanity-checks a forward pass + loss calculation
├── train.py
├── multivar.py 
└── losscalc.py                 # Sanity-checks loss functions against a random network
```

## Table of Implemented Activations

| Function | Formula | Best Used For |
| :--- | :--- | :--- |
| `ReLU` | $\max(0, x)$ | General-purpose hidden layers |
| `LeakyReLU` | $x$ if $x > 0$ else $\alpha x$ | Avoiding "dead" neurons in hidden layers |
| `Sigmoid` | $\frac{1}{1+e^{-x}}$ | Binary classification output, gating |
| `TanH` | $\tanh(x)$ | Hidden layers needing zero-centered output |
| `Softmax` | Normalized Exponentials | Multi-class classification output |
| `Swish` | $x \cdot \sigma(\beta x)$ | Deep hidden layers (smooth, non-monotonic) |
| `GELU` | Standard Transformer Approximation | Hidden Layers (Transformers/LLMs) |
| `ELU` | $x$ if $x > 0$ else $\alpha(e^x - 1)$ | Hidden layers wanting negative saturation |
| `SELU` | Self-normalizing variant of ELU | Deep networks without explicit normalization |
| `Mish` | $x \cdot \tanh(\text{softplus}(x))$ | Computer Vision / Modern Architectures |

## Table of Implemented Loss Functions

| Function | Best Used For |
| :--- | :--- |
| `mean_squared_error` | General regression, penalizes large errors heavily |
| `mean_absolute_error` | Regression when robustness to outliers matters |
| `huber_loss` | Regression wanting a balance of MSE and MAE behavior |

## Usage

```python
from nnframework import core

# Build layers: each neuron's weight count must match the previous layer's neuron count
hidden_layer = core.Layer([core.Neuron([...], bias) for _ in range(6)], function="Sigmoid")
output_layer = core.Layer([core.Neuron([...], bias) for _ in range(4)], function=None)

network = core.NeuralNetwork(
    inputs=[2.0, -3.0, 5.0, 0.6],
    layers=[hidden_layer, output_layer],
    loss_function="mean_squared_error",
)

# Train on a single example
prediction = network.train_step(true_outputs=[-1.0, 0.0, 4.0, 1.2], learning_rate=0.05)

# Predict on new, unseen input without affecting training state
result = network.predict([1.0, -2.0, 3.0, 0.5])
```

## Status

Actively in development. Core training loop (forward propagation, backpropagation, gradient descent) is functional end-to-end. Planned next steps include additional optimizers beyond plain gradient descent (e.g. Momentum, Adam) and expanded test coverage for multi-example datasets.
