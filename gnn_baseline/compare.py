import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
from graph_embeddings import GraphEmbeddings
from utils import *

DATA_DIRECTORY = os.path.join(ROOT_DIR, "processed_data")
MODEL_NAME = "best_model_1.keras"


def main():
    # Predictions given by model
    _, _, _, _, X_test, y_test = load_data(DATA_DIRECTORY)

    model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, "checkpoints", MODEL_NAME))

    y_model_scaled = model.predict(X_test, verbose=1)

    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    y_model = y_scaler.inverse_transform(y_model_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test).flatten()

    # Predictions given by naive lorentz addition
    _, _, _, _, X_orig, y_orig = load_data_original(DATA_DIRECTORY)
    particle_masses = np.zeros((X_orig.shape[0], X_orig.shape[1]))
    y_lorentz = vectorized_lorentz_addition(X_orig, particle_masses)

    model_metrics = calculate_metrics(y_true, y_model, "best_model_1.keras")
    lorentz_metrics = calculate_metrics(y_orig, y_lorentz, "lorentz")

    metrics = {
        **model_metrics,
        **lorentz_metrics
    }
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()