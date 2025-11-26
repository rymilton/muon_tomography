# train_model.py
import os
import numpy as np
import tensorflow as tf
from model import Agg3D  # import your model script

# --------------------------
# Config
# --------------------------
INPUT_DIR = "./"      # where your .npz files are
BATCH_SIZE = 2
EPOCHS = 2
LEARNING_RATE = 1e-3
RESOLUTION = 64

# --------------------------
# Load data
# --------------------------
# Load transmission (theta-phi) maps
transmission_data = np.load(os.path.join(INPUT_DIR, "theta_phi_transmission.npz"))
transmission = transmission_data["transmission"]  # shape: (theta_bins, phi_bins)
theta_edges = transmission_data["theta_edges"]
phi_edges = transmission_data["phi_edges"]

# For batch training, add batch and channel dimensions
transmission = transmission[np.newaxis, ..., np.newaxis]  # shape: (1, theta_bins, phi_bins, 1)

# Load 3D density
density_data = np.load(os.path.join(INPUT_DIR, "object_densities.npz"))
density = density_data["arr_0"]
density = density[np.newaxis, ..., np.newaxis]  # add batch and channel dims

# --------------------------
# Create tf.data.Dataset
# --------------------------
dataset = tf.data.Dataset.from_tensor_slices((transmission, density))
dataset = dataset.batch(BATCH_SIZE)

# --------------------------
# Build model
# --------------------------
model = Agg3D(**{
            'downward_convs': [1, 2, 3, 4, 5],
            'downward_filters': [8, 16, 32, 64, 128],
            'upward_convs': [4, 3, 2, 1],
            'upward_filters': [64, 32, 16, 8],
            'resolution': 64,
        })

# Compile
model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE), 
        loss="mse", metrics=["mse", "mae"]
    )

# --------------------------
# Train
# --------------------------
model.fit(dataset, epochs=EPOCHS)
model.evaluate(dataset)
y_pred = model.predict(dataset)
print(y_pred)
