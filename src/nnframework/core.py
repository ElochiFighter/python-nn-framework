from __future__ import annotations
import math
from .functions import activations as func
from .functions import losses as loss
from collections.abc import Sequence
from numbers import Number

class NeuralNetwork:
    def __init__(self, inputs: Sequence[Number], layers: Sequence[Layer]):
        self.inputs = inputs
        self.layers = layers

    def forward(self) -> tuple[Sequence[float], tuple]:
        current_signals = self.inputs
        cache = ()
        for layer in self.layers:
            current_signals, layer_cache = layer.forward(current_signals)
            cache += (layer_cache,)
        return current_signals, cache

class Layer:
    def __init__(self, neurons: Sequence[Neuron], function: str | None = None):
        self.neurons = neurons
        self.function = function

    def get_weights(self) -> Sequence[Sequence[float]]:
        return [neuron.get_weights() for neuron in self.neurons]

    def get_biases(self) -> Sequence[float]:
        return [neuron.get_bias() for neuron in self.neurons]

    def forward(self, inputs: Sequence[Number]) -> tuple[Sequence[float], tuple]:
        pre_activ = [neuron.compute(inputs) for neuron in self.neurons]

        if self.function is None:
            outputs = pre_activ
        elif self.function == "Softmax":
            # Softmax is a special case, because it needs to be applied to the entire output vector
            outputs = func.Softmax(pre_activ)
        else:
            function = getattr(func, self.function)
            outputs = list(map(function, pre_activ))

        cache = (inputs, pre_activ, outputs)
        return outputs, cache

    def get_function(self):
        return self.function

    def backward(self, output_gradient: Sequence[float]):
        pass  # Placeholder for backward propagation logic

class Neuron:
    def __init__(self, weights: Sequence[float], bias: float):
        self.weights = list(weights)
        self.bias = bias

    def set_weights(self, x: Sequence[float]):
        self.weights = list(x)

    def get_weights(self) -> Sequence[float]:
        return self.weights

    def get_bias(self) -> float:
        return self.bias

    def compute(self, inputs: Sequence[Number]) -> float:
        if len(inputs) != len(self.weights):
            raise ValueError(f"Input size ({len(inputs)}) must match weight size ({len(self.weights)})")
        return math.fsum(i * w for i, w in zip(inputs, self.weights)) + self.bias
