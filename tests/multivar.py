import random
from nnframework import core
from nnframework.functions import losses

random.seed(42)

dataset = [
    ([2.0, -3.0, 5.0, 0.6],   [-1.0, 0.0, 4.0, 1.2]),
    ([1.0, -2.0, 3.0, 0.5],   [-0.5, 0.5, 2.0, 0.8]),
    ([0.0, 1.0, -1.0, 2.0],   [0.2, -0.3, 1.0, -0.5]),
    ([-1.0, 0.5, 2.0, -0.5],  [0.8, 0.1, -0.6, 0.3]),
]


def random_neuron_list(num_neurons, num_weights_per_neuron):
    neurons = []
    for _ in range(num_neurons):
        weights = [random.uniform(-1, 1) for _ in range(num_weights_per_neuron)]
        neurons.append(core.Neuron(weights, random.uniform(-1, 1)))
    return neurons


input_layer = core.Layer(random_neuron_list(4, 4), function=None)
hidden_layer_1 = core.Layer(random_neuron_list(6, 4), function="Sigmoid")
hidden_layer_2 = core.Layer(random_neuron_list(6, 6), function="Sigmoid")
hidden_layer_3 = core.Layer(random_neuron_list(6, 6), function="Sigmoid")
hidden_layer_4 = core.Layer(random_neuron_list(6, 6), function="Sigmoid")
output_layer = core.Layer(random_neuron_list(4, 6), function=None)

network = core.NeuralNetwork(
    dataset[0][0],
    [input_layer, hidden_layer_1, hidden_layer_2, hidden_layer_3, hidden_layer_4, output_layer],
    loss_function="mean_squared_error",
)

learning_rate = 0.05
epochs = 50

for epoch in range(epochs):
    total_loss = 0.0
    for input_vec, target_vec in dataset:
        network.inputs = input_vec
        prediction = network.train_step(target_vec, learning_rate)
        total_loss += losses.mean_squared_error(target_vec, prediction)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch + 1:2d} | Avg loss: {avg_loss:.6f}")

print()
print("Testing on training examples")
for input_vec, target_vec in dataset:
    result = network.predict(input_vec)
    print(f"Input: {input_vec}")
    print(f"  Goal:      {target_vec}")
    print(f"  Predicted: {[round(r, 3) for r in result]}")

print()
print("Testing on an unseen input")
new_input = [3.0, -1.0, 0.0, 1.0]
result = network.predict(new_input)
print(f"Input: {new_input}")
print(f"  Predicted: {[round(r, 3) for r in result]}")