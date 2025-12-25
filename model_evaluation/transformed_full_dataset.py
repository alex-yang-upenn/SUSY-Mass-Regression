"""
Module Name: model_comparison/full_dataset

Description:
    Identical evaluation as full_dataset, a comparison of models' performance on the full
    test dataset. However, this module first applies an augmentation to the input data.

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
from tqdm import tqdm

from config_loader import load_config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from model_evaluation.helpers import MODEL_CONFIGS, load_all_models
from plotting import *
from simCLR_model import *
from transformation import TransformationType, create_transformed_dataset
from utils import *


def process_transformed_full_dataset_predictions(models, X, y, y_scaler, config):
    test_size = len(X)
    test_batches = int(test_size // config["BATCHSIZE"])

    predictions = {}
    y_true = None

    for model_name, model in models.items():
        # Create transformed dataset for each model since it's a one-time iterator
        test_transformed = create_transformed_dataset(
            X,
            y,
            batchsize=config["BATCHSIZE"],
            n_features=config["N_FEATURES"],
            transformations=[TransformationType.DELETE_PARTICLE],
            transformation_kwargs={
                "num_particles_to_delete": config["NUM_PARTICLES_TO_DELETE"]
            },
        )
        test_transformed = test_transformed.take(test_batches)

        y_pred_list = []
        y_true_list = []
        for features, labels in tqdm(
            test_transformed,
            desc=f"{model_name.replace('_', ' ').title()}",
            total=test_batches,
        ):
            y_pred_batch_scaled = model.predict(features, verbose=0)
            y_pred_batch = y_scaler.inverse_transform(y_pred_batch_scaled)
            y_pred_list.append(y_pred_batch)
            y_true_batch = y_scaler.inverse_transform(labels.numpy())
            y_true_list.append(y_true_batch)

        predictions[model_name] = np.concatenate(y_pred_list).flatten()
        if y_true is None:  # Only set once, all models should have same y_true
            y_true = np.concatenate(y_true_list).flatten()

    return predictions, y_true


def process_transformed_lorentz_predictions(config):
    _, _, _, _, X_orig, y_orig = load_data_original(config["PROCESSED_DATA_DIRECTORY"])

    X_orig = X_orig.transpose(0, 2, 1)
    y_orig = y_orig.reshape(-1, 1)

    test_size = len(X_orig)
    test_batches = int(test_size // config["BATCHSIZE"])

    test_orig_transformed = create_transformed_dataset(
        X_orig,
        y_orig,
        batchsize=config["BATCHSIZE"],
        n_features=config["N_FEATURES"],
        transformations=[TransformationType.DELETE_PARTICLE],
    )
    test_orig_transformed = test_orig_transformed.take(test_batches)

    y_lorentz_list = []
    y_lorentz_true_list = []
    for features, labels in tqdm(
        test_orig_transformed, desc="Lorentz Addition", total=test_batches
    ):
        particles = features.numpy().transpose(0, 2, 1)
        particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
        y = vectorized_lorentz_addition(particles, particle_masses)
        y_lorentz_list.append(y)
        y_lorentz_true_list.append(labels.numpy())

    y_lorentz = np.concatenate(y_lorentz_list).flatten()
    y_lorentz_true = np.concatenate(y_lorentz_true_list).flatten()

    return y_lorentz, y_lorentz_true


def calculate_all_metrics(y_true, predictions, y_lorentz_true, y_lorentz):
    metrics = {}
    for model_name, y_pred in predictions.items():
        model_metrics = calculate_metrics(y_true, y_pred, model_name)
        metrics.update(model_metrics)

    # Special handling for lorentz which uses different y_true
    lorentz_metrics = calculate_metrics(y_lorentz_true, y_lorentz, "lorentz_addition")
    metrics.update(lorentz_metrics)

    return metrics


def main():
    config = load_config()

    # Load data and scaler
    _, _, _, _, X, y = load_data(config["PROCESSED_DATA_DIRECTORY"])

    with open(
        os.path.join(config["PROCESSED_DATA_DIRECTORY"], "y_scaler.pkl"), "rb"
    ) as f:
        y_scaler = pickle.load(f)

    # Load all models and process predictions
    models = load_all_models(config)
    predictions, y_true = process_transformed_full_dataset_predictions(
        models, X, y, y_scaler, config
    )

    # Process lorentz predictions
    y_lorentz, y_lorentz_true = process_transformed_lorentz_predictions(config)

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
        "transformed_variance_comparison.png",
    )
    compare_variance_all(
        variance_performance_dict,
        variance_plot_path,
        err_bar_h_spacing=config["ERR_BAR_H_SPACING"],
    )

    # Calculate all metrics
    metrics = calculate_all_metrics(y_true, predictions, y_lorentz_true, y_lorentz)

    metric_dir = os.path.join(
        config["ROOT_DIR"], "model_evaluation", f"json{config["DATASET_NAME"]}"
    )
    os.makedirs(metric_dir, exist_ok=True)
    with open(os.path.join(metric_dir, "transformed_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
