"""
Module Name: model_comparison/same_event_type

Description:
    This module compares each model's performance by event type. It evaluates the models on the original test
    dataset, one file at a time. Artifacts include performance metrics and dual histograms for a select
    subset of files, and an accuracy plot across all files, to visualize models' performance across different
    event types.

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
from loss_functions import *
from model_evaluation.helpers import (
    MODEL_CONFIGS,
    calculate_all_metrics,
    create_performance_dict,
    load_all_models,
)
from plotting import *
from siamese import *
from utils import *


def process_model_predictions(
    model, X_test_scaled, y_scaler, model_name, y_true, performance_dict
):
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    pred_mean = np.mean(y_pred)
    pred_std = np.std(y_pred)

    performance_dict[model_name]["M_x(true)"].append(y_true[0])
    performance_dict[model_name]["mean"].append(pred_mean)
    performance_dict[model_name]["+1σ"].append(pred_mean + pred_std)
    performance_dict[model_name]["-1σ"].append(pred_mean - pred_std)
    performance_dict[model_name]["sample_count"].append(len(y_pred))

    return y_pred


def process_lorentz_predictions(X_test, y_true, performance_dict):
    particle_masses = np.zeros((X_test.shape[0], X_test.shape[1]))
    y_lorentz = vectorized_lorentz_addition(X_test, particle_masses)
    lorentz_mean = np.mean(y_lorentz)
    lorentz_std = np.std(y_lorentz)

    performance_dict["lorentz_addition"]["M_x(true)"].append(y_true[0])
    performance_dict["lorentz_addition"]["mean"].append(lorentz_mean)
    performance_dict["lorentz_addition"]["+1σ"].append(lorentz_mean + lorentz_std)
    performance_dict["lorentz_addition"]["-1σ"].append(lorentz_mean - lorentz_std)
    performance_dict["lorentz_addition"]["sample_count"].append(len(y_lorentz))

    return y_lorentz


def create_histograms_for_file(config, name, predictions, y_true):
    base_path = os.path.join(
        config["ROOT_DIR"],
        "model_evaluation",
        f"dual_histograms{config['DATASET_NAME']}",
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
            f"dual_histograms{config["DATASET_NAME"]}",
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
        y_true = test["y"]

        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)

        # Process all model predictions
        predictions = {}
        for model_name in MODEL_CONFIGS.keys():
            if model_name == "lorentz_addition":
                predictions[model_name] = process_lorentz_predictions(
                    X_test, y_true, model_performance_dict
                )
            else:
                predictions[model_name] = process_model_predictions(
                    models[model_name],
                    X_test_scaled,
                    y_scaler,
                    model_name,
                    y_true,
                    model_performance_dict,
                )

        # Log metrics and visualize selected event types
        if name in config["EVAL_DATA_FILES"]:
            metrics = calculate_all_metrics(y_true, predictions)
            same_event_type_metrics[name[5:-4]] = metrics
            create_histograms_for_file(config, name, predictions, y_true)

    # Aggregate data by M_x values to avoid multiple points per M_x
    aggregated_model_performance_dict = aggregate_model_performance_by_mx(
        model_performance_dict
    )

    compare_performance_all(
        model_performance_dict=aggregated_model_performance_dict,
        filename=os.path.join(
            config["ROOT_DIR"],
            "model_evaluation",
            f"accuracy_plots{config["DATASET_NAME"]}/standard_inputs.png",
        ),
        err_bar_h_spacing=config["ERR_BAR_H_SPACING"],
    )

    metric_dir = os.path.join(
        config["ROOT_DIR"], "model_evaluation", f"json{config["DATASET_NAME"]}"
    )
    os.makedirs(metric_dir, exist_ok=True)
    with open(os.path.join(metric_dir, "same_event_type_metrics.json"), "w") as f:
        json.dump(same_event_type_metrics, f, indent=4)


if __name__ == "__main__":
    main()
