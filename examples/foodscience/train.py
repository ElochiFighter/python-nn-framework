import math
import random
import sqlite3
import pandas as pd
from nnframework import core

DB_PATH = "examples/foodscience/database/fooddata.db"
FEATURES = ["Water","Ash","Total lipid (fat)","Calcium, Ca","Iron, Fe","Magnesium, Mg",
            "Manganese, Mn","Phosphorus, P","Potassium, K","Sodium, Na","Zinc, Zn","Nitrogen"]
TARGET = "Vitamin B-6"
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
EPOCHS = 100
LEARNING_RATE = 0.001

random.seed(RANDOM_SEED)
connection = sqlite3.connect(DB_PATH)
data = pd.read_sql_query("SELECT * FROM food_ml", connection)
connection.close()
print(f"Dataset size: {len(data)}")

data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
split_index = int(len(data) * TRAIN_RATIO)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]
print(f"Training: {len(train_data)} | Testing: {len(test_data)}")

X_train = train_data[FEATURES].astype(float).values.tolist()
y_train = train_data[TARGET].astype(float).values.tolist()
X_test = test_data[FEATURES].astype(float).values.tolist()
y_test = test_data[TARGET].astype(float).values.tolist()

means, stds = [], []
for ci in range(len(FEATURES)):
    vals = [row[ci] for row in X_train]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)
    if std == 0: std = 1.0
    means.append(mean); stds.append(std)

def normalize_inputs(row):
    return [(row[i] - means[i]) / stds[i] for i in range(len(FEATURES))]

X_train = [normalize_inputs(r) for r in X_train]
X_test = [normalize_inputs(r) for r in X_test]

target_mean = sum(y_train) / len(y_train)
target_variance = sum((v - target_mean) ** 2 for v in y_train) / len(y_train)
target_std = math.sqrt(target_variance)
if target_std == 0: target_std = 1.0

def normalize_target(v): return (v - target_mean) / target_std
def denormalize_target(v): return v * target_std + target_mean

y_train_normalized = [normalize_target(v) for v in y_train]

def random_neuron(input_count):
    scale = math.sqrt(2.0 / input_count)
    weights = [random.gauss(0, scale) for _ in range(input_count)]
    return core.Neuron(weights, 0.0)

input_count = len(FEATURES)
hidden_layer_1 = core.Layer([random_neuron(input_count) for _ in range(16)], function="ReLU")
hidden_layer_2 = core.Layer([random_neuron(16) for _ in range(8)], function="ReLU")
output_layer = core.Layer([random_neuron(8)], function=None)

network = core.NeuralNetwork(
    inputs=X_train[0],
    layers=[hidden_layer_1, hidden_layer_2, output_layer],
    loss_function="mean_squared_error",
)

print("=" * 60)
print("TRAINING")
print("=" * 60)
for epoch in range(EPOCHS):
    indices = list(range(len(X_train)))
    random.shuffle(indices)
    total_loss = 0.0
    for index in indices:
        network.inputs = X_train[index]
        true_value = [y_train_normalized[index]]
        prediction = network.train_step(true_outputs=true_value, learning_rate=LEARNING_RATE)
        error = prediction[0] - true_value[0]
        total_loss += error ** 2
    average_loss = total_loss / len(X_train)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | MSE: {average_loss:.6f}")

print("=" * 60)
print("EVALUATION")
print("=" * 60)
predictions, actual_values = [], []
for inputs, actual in zip(X_test, y_test):
    normalized_prediction = network.predict(inputs)[0]
    prediction = denormalize_target(normalized_prediction)
    predictions.append(prediction)
    actual_values.append(actual)

errors = [p - a for p, a in zip(predictions, actual_values)]
mae = sum(abs(e) for e in errors) / len(errors)
mse = sum(e**2 for e in errors) / len(errors)
rmse = math.sqrt(mse)
actual_mean = sum(actual_values) / len(actual_values)
ss_total = sum((a - actual_mean) ** 2 for a in actual_values)
ss_residual = sum((a - p) ** 2 for a, p in zip(actual_values, predictions))
r2 = 1.0 - (ss_residual / ss_total) if ss_total != 0 else 0.0

print(f"MAE:  {mae:.6f} mg")
print(f"RMSE: {rmse:.6f} mg")
print(f"R^2:  {r2:.6f}")