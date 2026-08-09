import random
from nnframework import core
from nnframework.functions import losses

inputs = [2, -3, 5, 0.6]
goal = [-1, 0, 4, 1.2]


def random_neuron_list(num_neurons, num_weights_per_neuron):
    neurons = []
    for _ in range(num_neurons):
        weights = [random.random() for _ in range(num_weights_per_neuron)]
        neurons.append(core.Neuron(weights, random.random()))
    return neurons


input_layer = core.Layer(random_neuron_list(4, 4))
output_layer = core.Layer(random_neuron_list(4, 4))

layers = [input_layer, output_layer]

network = core.NeuralNetwork(inputs, layers)
prediction, cache = network.forward()

print(f"Prediction: {prediction}")
print(f"Goal:       {goal}")
print()
print(f"Mean squared error:  {losses.mean_squared_error(goal, prediction)}")
print(f"Mean absolute error: {losses.mean_absolute_error(goal, prediction)}")
print(f"Huber loss:          {losses.huber_loss(goal, prediction)}")