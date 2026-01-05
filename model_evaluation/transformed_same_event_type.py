"""
Module Name: model_comparison/transformed_same_event_type

Description:
    Identical evaluation as same_event_type, a file by file comparison of models' performance. However, this module
    first applies an augmentation to the input data.

Usage:
Author:
Date:
License:
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config_loader import load_config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from loss_functions import *
from model_evaluation.helpers import (
    MODEL_CONFIGS,
    calculate_all_metrics,
    create_performance_dict,
    load_all_models,
)
from plotting import *
from siamese import *
from transformation import (
    TransformationType,
    ViewTransformedGenerator,
    create_transformed_dataset,
)
from utils import *


def process_transformed_model_predictions(
    model, test_transformed, y_scaler, model_name, y_true, performance_dict
):
    y_pred_list = []
    for features, labels in test_transformed:
        y_pred_batch_scaled = model.predict(features, verbose=0)
        y_pred_batch = y_scaler.inverse_transform(y_pred_batch_scaled).flatten()
        y_pred_list.append(y_pred_batch)

    y_pred = np.concatenate(y_pred_list)
    pred_mean = np.mean(y_pred)
    pred_std = np.std(y_pred)

    performance_dict[model_name]["M_x(true)"].append(y_true[0])
    performance_dict[model_name]["mean"].append(pred_mean)
    performance_dict[model_name]["+1σ"].append(pred_mean + pred_std)
    performance_dict[model_name]["-1σ"].append(pred_mean - pred_std)
    performance_dict[model_name]["sample_count"].append(len(y_pred))

    return y_pred


def process_transformed_lorentz_predictions(
    orig_test_transformed, y_true, performance_dict
):
    y_lorentz_list = []
    for features, labels in orig_test_transformed:
        particles = features.numpy().transpose(0, 2, 1)
        particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
        y_lorentz_batch = vectorized_lorentz_addition(particles, particle_masses)
        y_lorentz_list.append(y_lorentz_batch)

    y_lorentz = np.concatenate(y_lorentz_list).flatten()
    lorentz_mean = np.mean(y_lorentz)
    lorentz_std = np.std(y_lorentz)

    performance_dict["lorentz_addition"]["M_x(true)"].append(y_true[0])
    performance_dict["lorentz_addition"]["mean"].append(lorentz_mean)
    performance_dict["lorentz_addition"]["+1σ"].append(lorentz_mean + lorentz_std)
    performance_dict["lorentz_addition"]["-1σ"].append(lorentz_mean - lorentz_std)
    performance_dict["lorentz_addition"]["sample_count"].append(len(y_lorentz))

    return y_lorentz


def create_transformed_histograms_for_file(config, name, predictions, y_true):
    base_path = os.path.join(
        config["ROOT_DIR"],
        "model_evaluation",
        f"transformed_dual_histograms{config['DATASET_NAME']}",
    )

    create_2var_histogram_with_marker(
        data1=predictions["gnn_baseline"],
        data_label1="GNN Prediction",
        data2=predictions["lorentz_addition"],
        data_label2="Lorentz Addition Prediction",
        marker=y_true[0],
        marker_label="True Mass",
        title=f"Mass Regression for {name}",
        x_label="Mass (GeV / c^2)",
        filename=os.path.join(base_path, f"{name[5:-4]}.png"),
    )

    create_2var_histogram_with_marker(
        data1=predictions["gnn_transformed"],
        data_label1="GNN Transformed Prediction",
        data2=predictions["gnn_baseline"],
        data_label2="GNN Baseline Prediction",
        marker=y_true[0],
        marker_label="True Mass",
        title=f"Mass Regression for {name}",
        x_label="Mass (GeV / c^2)",
        filename=os.path.join(base_path, f"{name[5:-4]}_gnn.png"),
    )

    create_2var_histogram_with_marker(
        data1=predictions["siamese_finetune"],
        data_label1="Siamese Finetune Prediction",
        data2=predictions["siamese_no_finetune"],
        data_label2="Siamese No Finetune Prediction",
        marker=y_true[0],
        marker_label="True Mass",
        title=f"Mass Regression for {name}",
        x_label="Mass (GeV / c^2)",
        filename=os.path.join(base_path, f"{name[5:-4]}_finetuning.png"),
    )


def main():
    config = load_config()

    # Make directories
    os.makedirs(
        os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"accuracy_plots{config["DATASET_NAME"]}",
        ),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"transformed_dual_histograms{config["DATASET_NAME"]}",
        ),
        exist_ok=True,
    )

    # Load in scalers and models
    scalers = []
    for i in range(3):
        with open(
            os.path.join(config["PROCESSED_DATA_DIRECTORY"], f"x_scaler_{i}.pkl"), "rb"
        ) as f:
            scalers.append(pickle.load(f))

    with open(
        os.path.join(config["PROCESSED_DATA_DIRECTORY"], "y_scaler.pkl"), "rb"
    ) as f:
        y_scaler = pickle.load(f)

    models = load_all_models(config)

    # Setup model performance dictionary
    model_performance_dict = {
        model_key: create_performance_dict() for model_key in MODEL_CONFIGS.keys()
    }

    # Setup metrics tracker for selected event types
    same_event_type_metrics = {}

    # Iterate across each file
    progress_bar = tqdm(
        os.listdir(os.path.join(config["PROCESSED_DATA_DIRECTORY"], "test"))
    )
    for name in progress_bar:
        progress_bar.set_description(f"Processing file {name}")

        # Load in unscaled/untransposed data
        test = np.load(os.path.join(config["PROCESSED_DATA_DIRECTORY"], "test", name))
        X_test = test["X"]
        y_test_flattened = test["y"]

        # Scale and transpose
        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)
        y_test = y_test_flattened.reshape(-1, 1)

        # Apply augmentations
        test_batches = int(len(X_test_scaled) // config["BATCHSIZE"])
        test_transformed = create_transformed_dataset(
            X_test_scaled,
            y_test,
            batchsize=config["BATCHSIZE"],
            n_features=config["N_FEATURES"],
            transformations=[TransformationType.DELETE_PARTICLE],
        )
        test_transformed = test_transformed.take(test_batches)

        # Extract y_true from the first batch for consistent reference
        y_true_list = []
        for features, labels in test_transformed:
            y_true_list.append(labels.numpy().flatten())
        y_true = np.concatenate(y_true_list)

        # Process all model predictions
        predictions = {}
        for model_name in MODEL_CONFIGS.keys():
            if model_name == "lorentz_addition":
                # Create separate transformed dataset for lorentz
                X_orig = X_test.transpose(0, 2, 1)
                orig_test_transformed = create_transformed_dataset(
                    X_orig,
                    y_test,
                    batchsize=config["BATCHSIZE"],
                    n_features=config["N_FEATURES"],
                    transformations=[TransformationType.DELETE_PARTICLE],
                )
                orig_test_transformed = orig_test_transformed.take(test_batches)
                predictions[model_name] = process_transformed_lorentz_predictions(
                    orig_test_transformed, y_true, model_performance_dict
                )
            else:
                # Recreate test_transformed for each model since it's a one-time iterator
                test_transformed_model = create_transformed_dataset(
                    X_test_scaled,
                    y_test,
                    batchsize=config["BATCHSIZE"],
                    n_features=config["N_FEATURES"],
                    transformations=[TransformationType.DELETE_PARTICLE],
                )
                test_transformed_model = test_transformed_model.take(test_batches)
                predictions[model_name] = process_transformed_model_predictions(
                    models[model_name],
                    test_transformed_model,
                    y_scaler,
                    model_name,
                    y_true,
                    model_performance_dict,
                )

        # Log metrics and visualize selected event types
        if name in config["EVAL_DATA_FILES"]:
            metrics = calculate_all_metrics(y_true, predictions)
            same_event_type_metrics[name[5:-4]] = metrics
            create_transformed_histograms_for_file(config, name, predictions, y_true)

    # Aggregate data by M_x values to avoid multiple points per M_x
    aggregated_model_performance_dict = aggregate_model_performance_by_mx(
        model_performance_dict
    )

    compare_performance_all(
        model_performance_dict=aggregated_model_performance_dict,
        filename=os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"accuracy_plots{config["DATASET_NAME"]}/transformed_inputs.png",
        ),
        err_bar_h_spacing=config["ERR_BAR_H_SPACING"],
    )

    # Create subset with only siamese_finetune, siamese_no_finetune, and gnn_baseline metrics
    best_models = ["siamese_finetune", "siamese_no_finetune", "gnn_baseline"]
    subset_model_performance_dict = {
        model: aggregated_model_performance_dict[model] for model in best_models
    }

    compare_performance_all(
        model_performance_dict=subset_model_performance_dict,
        filename=os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"accuracy_plots{config["DATASET_NAME"]}/transformed_inputs_best.png",
        ),
    )

    with open(
        os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"json{config["DATASET_NAME"]}/transformed_same_event_type_metrics.json",
        ),
        "w",
    ) as f:
        json.dump(same_event_type_metrics, f, indent=4)


if __name__ == "__main__":
    main()
