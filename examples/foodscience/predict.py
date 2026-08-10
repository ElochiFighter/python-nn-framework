import math
import random
import sqlite3
import pandas as pd
from nnframework import core

DB_PATH = "examples/foodscience/database/fooddata.db"
FEATURES = ["Water", "Ash", "Total lipid (fat)", "Calcium, Ca", "Iron, Fe", "Magnesium, Mg",
            "Manganese, Mn", "Phosphorus, P", "Potassium, K", "Sodium, Na", "Zinc, Zn", "Nitrogen"]
TARGET = "Vitamin B-6"
RANDOM_SEED = 42
TRAIN_RATIO = 0.80
EPOCHS = 100
LEARNING_RATE = 0.001

random.seed(RANDOM_SEED)

# ============================================================
# LOAD + TRAIN (same as train.py)
# ============================================================

connection = sqlite3.connect(DB_PATH)
data = pd.read_sql_query("SELECT * FROM food_ml", connection)
connection.close()

data = data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
split_index = int(len(data) * TRAIN_RATIO)
train_data = data.iloc[:split_index]

X_train = train_data[FEATURES].astype(float).values.tolist()
y_train = train_data[TARGET].astype(float).values.tolist()

# Per-feature real min/max, used later to generate REALISTIC random foods
# (not just any random number — one within the range real foods actually have)
feature_min = {f: train_data[f].min() for f in FEATURES}
feature_max = {f: train_data[f].max() for f in FEATURES}
feature_mean = {f: train_data[f].mean() for f in FEATURES}
feature_std = {f: train_data[f].std() for f in FEATURES}

means, stds = [], []
for ci in range(len(FEATURES)):
    vals = [row[ci] for row in X_train]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)
    if std == 0:
        std = 1.0
    means.append(mean)
    stds.append(std)

def normalize_inputs(row):
    return [(row[i] - means[i]) / stds[i] for i in range(len(FEATURES))]

X_train_norm = [normalize_inputs(r) for r in X_train]

target_mean = sum(y_train) / len(y_train)
target_variance = sum((v - target_mean) ** 2 for v in y_train) / len(y_train)
target_std = math.sqrt(target_variance)
if target_std == 0:
    target_std = 1.0

def normalize_target(v):
    return (v - target_mean) / target_std

def denormalize_target(v):
    return v * target_std + target_mean

y_train_normalized = [normalize_target(v) for v in y_train]

def random_neuron(input_count):
    scale = math.sqrt(2.0 / input_count)
    weights = [random.gauss(0, scale) for _ in range(input_count)]
    return core.Neuron(weights, 0.0)

hidden_layer_1 = core.Layer([random_neuron(len(FEATURES)) for _ in range(16)], function="ReLU")
hidden_layer_2 = core.Layer([random_neuron(16) for _ in range(8)], function="ReLU")
output_layer = core.Layer([random_neuron(8)], function=None)

network = core.NeuralNetwork(
    inputs=X_train_norm[0],
    layers=[hidden_layer_1, hidden_layer_2, output_layer],
    loss_function="mean_squared_error",
)

print("Training...")
for epoch in range(EPOCHS):
    indices = list(range(len(X_train_norm)))
    random.shuffle(indices)
    for index in indices:
        network.inputs = X_train_norm[index]
        network.train_step([y_train_normalized[index]], LEARNING_RATE)
print("Done.\n")

# ============================================================
# GENERATE RANDOM, REALISTIC FOOD PROFILES
# ============================================================

def generate_realistic_food():
    """
    Builds one random nutrient profile. Each value is sampled from a
    normal distribution centered on the REAL mean/std for that nutrient
    (taken from actual foods), then clamped to the real min/max range
    seen in the dataset — so it lands somewhere a real food plausibly
    could, rather than an arbitrary or negative number.
    """
    profile = {}
    for f in FEATURES:
        value = random.gauss(feature_mean[f], feature_std[f])
        value = max(feature_min[f], min(feature_max[f], value))
        profile[f] = value
    return profile

NUM_RANDOM_FOODS = 5

print("=" * 70)
print(f"PREDICTING VITAMIN B-6 FOR {NUM_RANDOM_FOODS} RANDOM, REALISTIC FOODS")
print("=" * 70)

for i in range(NUM_RANDOM_FOODS):
    profile = generate_realistic_food()

    # Build the input vector in the SAME feature order the network trained on
    input_vec = [profile[f] for f in FEATURES]
    normalized_input = normalize_inputs(input_vec)

    normalized_prediction = network.predict(normalized_input)[0]
    predicted_b6 = denormalize_target(normalized_prediction)

    print(f"\nRandom food #{i + 1}:")
    for f in FEATURES:
        print(f"  {f:20s} {profile[f]:8.3f}")
    print(f"  --> Predicted Vitamin B-6: {predicted_b6:.4f} mg")