# A Custom Neural Network Framework from Scratch

A lightweight, educational deep learning framework written in **pure Python** using only the native `math` module. No NumPy, no PyTorch, no external dependencies.

## Features
* **Zero Dependencies:** Runs natively on any basic Python 3 environment.
* **Modern Activations:** Includes custom implementations of `ReLU`, `GELU`, `Mish`, `Softmax`, and more.
* **Modular Design:** Dynamic object-oriented layers, sequential model piping, and built-in loss calculations.

## Table of Implemented Activations

| Function | Formula | Best Used For |
| :--- | :--- | :--- |
| `GELU` | Standard Transformer Approximation | Hidden Layers (Transformers/LLMs) |
| `Mish` | $x \cdot \tanh(\text{softplus}(x))$ | Computer Vision / Modern Architectures |
| `Softmax` | Normalized Exponentials | Multi-class Classification Output |
