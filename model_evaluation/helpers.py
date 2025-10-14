"""
Module Name: model_evaluation/helpers

Description:
    Shared helper functions for model evaluation scripts to avoid code duplication.
    Contains common utilities for model loading, performance tracking, and metrics calculation.

Author:
Date:
License:
"""

import os

import numpy as np
import tensorflow as tf

from downstream_model import FinetunedNN
from graph_embeddings import GraphEmbeddings
from simCLR_model import SimCLRNTXentLoss
from utils import calculate_metrics

# Global model configuration
MODEL_CONFIGS = {
    "gnn_baseline": ("gnn_baseline", "model_{RUN_ID}{DATASET_NAME}"),
    "gnn_transformed": ("gnn_transformed", "model_{RUN_ID}{DATASET_NAME}"),
    "siamese_finetune": ("siamese", "model_{RUN_ID}_finetune{DATASET_NAME}"),
    "siamese_no_finetune": ("siamese", "model_{RUN_ID}_no_finetune{DATASET_NAME}"),
    "lorentz_addition": (None, None),  # Special case for lorentz
}


def create_performance_dict():
    """
    Creates an empty performance dictionary for tracking model performance metrics.

    Returns:
        dict: Dictionary with keys for true mass values, mean predictions,
              standard deviations, and sample counts.
    """
    return {
        "M_x(true)": [],
        "mean": [],
        "+1σ": [],
        "-1σ": [],
        "sample_count": [],
    }


def load_all_models(config):
    """
    Loads all models defined in MODEL_CONFIGS from their respective directories.

    Args:
        config (dict): Configuration dictionary containing ROOT_DIR, RUN_ID, and DATASET_NAME.

    Returns:
        dict: Dictionary mapping model names to loaded Keras models.
    """
    custom_objects = {
        "SimCLRNTXentLoss": SimCLRNTXentLoss,
        "GraphEmbeddings": GraphEmbeddings,
        "FinetunedNN": FinetunedNN,
    }

    models = {}
    for model_key, (subdir, model_name_template) in MODEL_CONFIGS.items():
        if subdir is None:  # Skip lorentz_addition
            continue

        model_name = model_name_template.format(
            RUN_ID=config["RUN_ID"], DATASET_NAME=config["DATASET_NAME"]
        )
        model_path = os.path.join(
            config["ROOT_DIR"], subdir, model_name, "best_model.keras"
        )
        models[model_key] = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects
        )

    return models


def calculate_all_metrics(y_true, predictions):
    """
    Calculates evaluation metrics for all model predictions.

    Args:
        y_true (np.ndarray): True target values.
        predictions (dict): Dictionary mapping model names to their predictions.

    Returns:
        dict: Dictionary containing metrics for each model.
    """
    metrics = {}
    for model_name, y_pred in predictions.items():
        model_metrics = calculate_metrics(y_true, y_pred, model_name)
        metrics.update(model_metrics)
    return metrics
