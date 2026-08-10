import random
from nnframework import core
from nnframework.functions import losses

random.seed(42)  # remove this if you want different random weights each run

inputs = [2.0, -3.0, 5.0, 0.6]
goal = [-1.0, 0.0, 4.0, 1.2]


def random_neuron_list(num_neurons, num_weights_per_neuron):
    neurons = []
    for _ in range(num_neurons):
        weights = [random.uniform(-1, 1) for _ in range(num_weights_per_neuron)]
        neurons.append(core.Neuron(weights, random.uniform(-1, 1)))
    return neurons


hidden_layer = core.Layer(random_neuron_list(6, 4), function="Sigmoid")
output_layer = core.Layer(random_neuron_list(4, 6), function=None)

network = core.NeuralNetwork(
    inputs,
    [hidden_layer, output_layer],
    loss_function="mean_squared_error",
)

learning_rate = 0.05
epochs = 50

for epoch in range(epochs):
    prediction = network.train_step(goal, learning_rate)
    loss_value = losses.mean_squared_error(goal, prediction)
    rounded = [round(p, 3) for p in prediction]
    print(f"Epoch {epoch + 1:2d} | Loss: {loss_value:.6f} | Prediction: {rounded}")

print()
print(f"Goal:             {goal}")
print(f"Final prediction: {[round(p, 3) for p in prediction]}")