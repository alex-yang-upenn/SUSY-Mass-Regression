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

import json
import os
import pickle

import numpy as np
import tensorflow as tf

from config_loader import load_config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from model_evaluation.helpers import MODEL_CONFIGS, load_all_models
from plotting import *
from simCLR_model import *
from utils import *


def process_full_dataset_predictions(models, X_test, y_scaler):
    predictions = {}
    for model_name, model in models.items():
        y_pred_scaled = model.predict(X_test, verbose=1)
        predictions[model_name] = y_scaler.inverse_transform(y_pred_scaled).flatten()
    return predictions


def process_lorentz_predictions(config):
    _, _, _, _, X_orig, y_orig = load_data_original(config["PROCESSED_DATA_DIRECTORY"])
    particle_masses = np.zeros((X_orig.shape[0], X_orig.shape[1]))
    y_lorentz = vectorized_lorentz_addition(X_orig, particle_masses)
    return y_lorentz, y_orig


def calculate_all_metrics(y_true, predictions, y_orig_for_lorentz, y_lorentz):
    metrics = {}
    for model_name, y_pred in predictions.items():
        model_metrics = calculate_metrics(y_true, y_pred, model_name)
        metrics.update(model_metrics)

    # Special handling for lorentz which uses different y_true
    lorentz_metrics = calculate_metrics(y_orig_for_lorentz, y_lorentz, "lorentz")
    metrics.update(lorentz_metrics)

    return metrics


def main():
    config = load_config()

    # Load in data and scaler
    _, _, _, _, X_test, y_test = load_data(config["PROCESSED_DATA_DIRECTORY"])
    with open(
        os.path.join(config["PROCESSED_DATA_DIRECTORY"], "y_scaler.pkl"), "rb"
    ) as f:
        y_scaler = pickle.load(f)
    y_true = y_scaler.inverse_transform(y_test).flatten()

    # Load all models and process predictions
    models = load_all_models(config)
    predictions = process_full_dataset_predictions(models, X_test, y_scaler)
    y_lorentz, y_orig = process_lorentz_predictions(config)

    variance_performance_dict = {}
    for model_name, y_pred in predictions.items():
        variance_performance_dict[model_name] = aggregate_predictions_by_bucket(
            y_true, y_pred, bucket_width=config["BUCKET_WIDTH"], min_samples=5
        )

    # Generate the variance comparison plot
    variance_plot_path = os.path.join(
        config["ROOT_DIR"],
        "model_evaluation",
        f"accuracy_plots{config['DATASET_NAME']}",
        "variance_comparison.png",
    )
    compare_variance_all(
        variance_performance_dict,
        variance_plot_path,
        err_bar_h_spacing=config["ERR_BAR_H_SPACING"],
    )

    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, predictions, y_orig, y_lorentz)
    metric_dir = os.path.join(
        config["ROOT_DIR"], "model_evaluation", f"json{config["DATASET_NAME"]}"
    )
    os.makedirs(metric_dir, exist_ok=True)
    with open(os.path.join(metric_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
