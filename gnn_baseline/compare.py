import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *

DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "processed_data")
MODEL_NAME = "best_model_1.keras"


def main():
    test_data_files = os.listdir(os.path.join(DATA_DIRECTORY, "test"))
    X_tests, y_trues = [], []

    for name in test_data_files:
        # Load in data
        test = np.load(os.path.join(DATA_DIRECTORY, "test", name))
        X_tests.append(test['X'])
        y_trues.append(test['y'])

    X_test = np.concatenate(X_tests, axis=0)
    y_true = np.concatenate(y_trues, axis=0)

    scalers = []
    for i in range(3):
        with open(os.path.join(DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))

    X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
    X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, "checkpoints", MODEL_NAME))
    y_model_scaled = model.predict(X_test_scaled, verbose=1)

    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)
    y_model = y_scaler.inverse_transform(y_model_scaled).flatten()

    # Vectorized lorentz addition
    particle_masses = np.zeros((X_test.shape[0], X_test.shape[1]))
    y_lorentz = vectorized_lorentz_addition(X_test, particle_masses)

    model_metrics = calculate_metrics(y_true, y_model, "best_model_1.keras")
    lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz")

    metrics = {
        **model_metrics,
        **lorentz_metrics
    }
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()