import random

from nnframework import core
from nnframework.functions import activations

random.seed(67)  # weight reproduction + funny number

inputs = [2, -3, 5, 0.6]

activation_names = [
    None,
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "TanH",
    "Softmax",
    "Swish",
    "GELU",
    "ELU",
    "SELU",
    "Mish",
]


def build_neurons(num_neurons, num_weights_per_neuron):
    neurons = []
    for _ in range(num_neurons):
        weights = [random.random() for _ in range(num_weights_per_neuron)]
        neurons.append(core.Neuron(weights, random.random()))
    return neurons
neurons = build_neurons(4, len(inputs))

for name in activation_names:
    layer = core.Layer(neurons, function=name)
    try:
        output = layer.forward(inputs)
        print(f"{name}: {output}")
    except Exception as e:
        print(f"{name}: FAILED -> {type(e).__name__}: {e}")