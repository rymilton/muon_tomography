# train_model.py
import os
import numpy as np
import tensorflow as tf
from model import Transmission3DRecon  # import your model script

# --------------------------
# Config
# --------------------------
INPUT_DIR = "./"      # where your .npz files are
BATCH_SIZE = 2
EPOCHS = 5
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
dataset = dataset.shuffle(buffer_size=1).batch(BATCH_SIZE)

# --------------------------
# Build model
# --------------------------
model = Transmission3DRecon(resolution=RESOLUTION)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=["mae"]
)

# --------------------------
# Train
# --------------------------
model.fit(dataset, epochs=EPOCHS)

# --------------------------
# Save model
# --------------------------
model.save(os.path.join(INPUT_DIR, "transmission3d_model"))
print("Model saved.")
