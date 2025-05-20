"""
Module Name: model_comparison/full_dataset

Description:
    This module compares each model's performance on the full, original test dataset. Performance metrics are
    logged to metrics.json.     

Usage:
Author:
Date:
License:
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import config
from graph_embeddings import GraphEmbeddings
from utils import *
from plotting import *

def main():
    # Predictions given by gnn_baseline model
    _, _, _, _, X_test, y_test = load_data(config.PROCESSED_DATA_DIRECTORY)

    gnn_baseline_model_path = os.path.join(config.ROOT_DIR, "gnn_baseline", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_baseline_model = tf.keras.models.load_model(gnn_baseline_model_path)

    y_gnn_baseline_scaled = gnn_baseline_model.predict(X_test, verbose=1)

    with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    y_gnn_baseline = y_scaler.inverse_transform(y_gnn_baseline_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test).flatten()

    # Prediction given by gnn_transformed model
    gnn_transformed_model_path = os.path.join(config.ROOT_DIR, "gnn_transformed", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_transformed_model = tf.keras.models.load_model(gnn_transformed_model_path)

    y_gnn_transformed_scaled = gnn_transformed_model.predict(X_test, verbose=1)
    y_gnn_transformed = y_scaler.inverse_transform(y_gnn_transformed_scaled).flatten()

    # Predictions given by naive lorentz addition
    _, _, _, _, X_orig, y_orig = load_data_original(config.PROCESSED_DATA_DIRECTORY)
    particle_masses = np.zeros((X_orig.shape[0], X_orig.shape[1]))
    y_lorentz = vectorized_lorentz_addition(X_orig, particle_masses)

    gnn_baseline_metrics = calculate_metrics(y_true, y_gnn_baseline, "gnn_baseline")
    gnn_transformed_metrics = calculate_metrics(y_true, y_gnn_transformed, "gnn_transformed")
    lorentz_metrics = calculate_metrics(y_orig, y_lorentz, "lorentz")

    metrics = {
        **gnn_baseline_metrics,
        **gnn_transformed_metrics,
        **lorentz_metrics
    }
    with open(os.path.join(SCRIPT_DIR, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()