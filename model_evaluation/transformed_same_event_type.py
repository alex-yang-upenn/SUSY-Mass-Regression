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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)

import json
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import config
from downstream_model import *
from graph_embeddings import GraphEmbeddings
from loss_functions import *
from plotting import *
from siamese import *
from transformation import create_transformed_dataset, ViewTransformedGenerator
from utils import *


def main():
    # Make directories
    os.makedirs(os.path.join(SCRIPT_DIR, "accuracy_plots"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "transformed_dual_histograms"), exist_ok=True)

    # Load in scalers and model
    scalers = []
    for i in range(3):
        with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))

    with open(os.path.join(config.PROCESSED_DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)
    
    gnn_baseline_model_path = os.path.join(ROOT_DIR, "gnn_baseline", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_baseline_model = tf.keras.models.load_model(gnn_baseline_model_path)

    gnn_transformed_model_path = os.path.join(ROOT_DIR, "gnn_transformed", f"model_{config.RUN_ID}", "best_model.keras")
    gnn_transformed_model = tf.keras.models.load_model(gnn_transformed_model_path)
    
    siamese_finetune_model_path = os.path.join(config.ROOT_DIR, "siamese", "model_3_finetune", "best_model.keras")
    siamese_finetune_model = tf.keras.models.load_model(
        siamese_finetune_model_path,
        custom_objects={
            "SimCLRNTXentLoss": SimCLRNTXentLoss,
            "GraphEmbeddings": GraphEmbeddings,
            "FinetunedNN": FinetunedNN,
        },
    )

    siamese_no_finetune_model_path = os.path.join(config.ROOT_DIR, "siamese", "model_3_no_finetune", "best_model.keras")
    siamese_no_finetune_model = tf.keras.models.load_model(
        siamese_no_finetune_model_path,
        custom_objects={
            "SimCLRNTXentLoss": SimCLRNTXentLoss,
            "GraphEmbeddings": GraphEmbeddings,
            "FinetunedNN": FinetunedNN,
        },
    )

    # Setup model performance dictionary
    model_performance_dict = {
        "gnn_baseline": {
            "M_x(true)": [],
            "mean": [],
            "+1σ": [],
            "-1σ": [],
        },
        "gnn_transformed": {
            "M_x(true)": [],
            "mean": [],
            "+1σ": [],
            "-1σ": [],
        },
        "siamese_finetune": {
            "M_x(true)": [],
            "mean": [],
            "+1σ": [],
            "-1σ": [],
        },
        "siamese_no_finetune": {
            "M_x(true)": [],
            "mean": [],
            "+1σ": [],
            "-1σ": [],
        },
        "lorentz_addition": {
            "M_x(true)": [],
            "mean": [],
            "+1σ": [],
            "-1σ": [],
        },
    }

    # Setup metrics tracker
    same_event_type_metrics = {}
    
    # Iterate across each file
    progress_bar = tqdm(os.listdir(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test")))
    for name in progress_bar:
        progress_bar.set_description(f"Processing file {name}")
        
        # Load in unscaled/untransposed data
        test = np.load(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test", name))
        X_test = test['X']
        y_test_flattened = test['y']

        # Scale and transpose
        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)
        y_test = y_test_flattened.reshape(-1, 1)

        # Apply augmentations
        test_batches = int(len(X_test_scaled) // config.BATCHSIZE)
        test_transformed = create_transformed_dataset(
            X_test_scaled, y_test, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES
        )
        test_transformed = test_transformed.take(test_batches)

        # Predictions given by gnn_baseline
        y_gnn_baseline_list = []
        y_true_list = []
        for features, labels in test_transformed:
            y_gnn_baseline_batch_scaled = gnn_baseline_model.predict(features, verbose=0)
            y_gnn_baseline_batch = y_scaler.inverse_transform(y_gnn_baseline_batch_scaled).flatten()
            y_gnn_baseline_list.append(y_gnn_baseline_batch)
            y_true_list.append(labels.numpy().flatten())
        y_gnn_baseline = np.concatenate(y_gnn_baseline_list)
        y_true = np.concatenate(y_true_list)
        
        gnn_baseline_mean = np.mean(y_gnn_baseline)
        gnn_baseline_std = np.std(y_gnn_baseline)
        model_performance_dict["gnn_baseline"]["M_x(true)"].append(y_true[0])
        model_performance_dict["gnn_baseline"]["mean"].append(gnn_baseline_mean)
        model_performance_dict["gnn_baseline"]["+1σ"].append(gnn_baseline_mean + gnn_baseline_std)
        model_performance_dict["gnn_baseline"]["-1σ"].append(gnn_baseline_mean - gnn_baseline_std)

        # Predictions given by gnn_transformed
        y_gnn_transformed_list = []
        for features, labels in test_transformed:
            y_gnn_transformed_batch_scaled = gnn_transformed_model.predict(features, verbose=0)
            y_gnn_transformed_batch = y_scaler.inverse_transform(y_gnn_transformed_batch_scaled).flatten()
            y_gnn_transformed_list.append(y_gnn_transformed_batch)
        y_gnn_transformed = np.concatenate(y_gnn_transformed_list)
        
        gnn_transformed_mean = np.mean(y_gnn_transformed)
        gnn_transformed_std = np.std(y_gnn_transformed)
        model_performance_dict["gnn_transformed"]["M_x(true)"].append(y_true[0])
        model_performance_dict["gnn_transformed"]["mean"].append(gnn_transformed_mean)
        model_performance_dict["gnn_transformed"]["+1σ"].append(gnn_transformed_mean + gnn_transformed_std)
        model_performance_dict["gnn_transformed"]["-1σ"].append(gnn_transformed_mean - gnn_transformed_std)

        # Prediction given by contrastive learning encoder + neural network (finetune)
        y_siamese_finetune_list = []
        for features, labels in test_transformed:
            y_siamese_finetune_batch_scaled = siamese_finetune_model.predict(features, verbose=0)
            y_siamese_finetune_batch = y_scaler.inverse_transform(y_siamese_finetune_batch_scaled).flatten()
            y_siamese_finetune_list.append(y_siamese_finetune_batch)
        y_siamese_finetune = np.concatenate(y_siamese_finetune_list)
        
        siamese_finetune_mean = np.mean(y_siamese_finetune)
        siamese_finetune_std = np.std(y_siamese_finetune)
        model_performance_dict["siamese_finetune"]["M_x(true)"].append(y_true[0])
        model_performance_dict["siamese_finetune"]["mean"].append(siamese_finetune_mean)
        model_performance_dict["siamese_finetune"]["+1σ"].append(siamese_finetune_mean + siamese_finetune_std)
        model_performance_dict["siamese_finetune"]["-1σ"].append(siamese_finetune_mean - siamese_finetune_std)

        # Prediction given by contrastive learning encoder + neural network (no finetune)
        y_siamese_no_finetune_list = []
        for features, labels in test_transformed:
            y_siamese_no_finetune_batch_scaled = siamese_no_finetune_model.predict(features, verbose=0)
            y_siamese_no_finetune_batch = y_scaler.inverse_transform(y_siamese_no_finetune_batch_scaled).flatten()
            y_siamese_no_finetune_list.append(y_siamese_no_finetune_batch)
        y_siamese_no_finetune = np.concatenate(y_siamese_no_finetune_list)
        
        siamese_no_finetune_mean = np.mean(y_siamese_no_finetune)
        siamese_no_finetune_std = np.std(y_siamese_no_finetune)
        model_performance_dict["siamese_no_finetune"]["M_x(true)"].append(y_true[0])
        model_performance_dict["siamese_no_finetune"]["mean"].append(siamese_no_finetune_mean)
        model_performance_dict["siamese_no_finetune"]["+1σ"].append(siamese_no_finetune_mean + siamese_no_finetune_std)
        model_performance_dict["siamese_no_finetune"]["-1σ"].append(siamese_no_finetune_mean - siamese_no_finetune_std)

        # Predictions given by lorentz addition
        X_orig = X_test.transpose(0, 2, 1)
        orig_test_transformed = create_transformed_dataset(
            X_orig, y_test, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES
        )
        orig_test_transformed = orig_test_transformed.take(test_batches)

        y_lorentz_list = []
        for features, labels in orig_test_transformed:
            particles = features.numpy().transpose(0, 2, 1)
            particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
            y_lorentz_batch = vectorized_lorentz_addition(particles, particle_masses)
            y_lorentz_list.append(y_lorentz_batch)
        y_lorentz = np.concatenate(y_lorentz_list).flatten()

        lorentz_mean = np.mean(y_lorentz)
        lorentz_std = np.std(y_lorentz)
        model_performance_dict["lorentz_addition"]["M_x(true)"].append(y_true[0])
        model_performance_dict["lorentz_addition"]["mean"].append(lorentz_mean)
        model_performance_dict["lorentz_addition"]["+1σ"].append(lorentz_mean + lorentz_std)
        model_performance_dict["lorentz_addition"]["-1σ"].append(lorentz_mean - lorentz_std)

        # Log metrics and visualize selected event types
        if name in config.EVAL_DATA_FILES:
            gnn_baseline_model_metrics = calculate_metrics(y_true, y_gnn_baseline, "gnn_baseline")
            gnn_transformed_model_metrics = calculate_metrics(y_true, y_gnn_transformed, "gnn_transformed")
            siamese_finetune_metrics = calculate_metrics(y_true, y_siamese_finetune, "siamese_finetune")
            siamese_no_finetune_metrics = calculate_metrics(y_true, y_siamese_no_finetune, "siamese_no_finetune")
            lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz_addition")

            metrics = {
                **gnn_baseline_model_metrics,
                **gnn_transformed_model_metrics,
                **siamese_finetune_metrics,
                **siamese_no_finetune_metrics,
                **lorentz_metrics
            }
            same_event_type_metrics[name[5:-4]] = metrics
            
            create_2var_histogram_with_marker(
                data1=y_gnn_baseline,
                data_label1="GNN Prediction",
                data2=y_lorentz,
                data_label2="Lorentz Addition Prediction",
                marker=y_true[0],
                marker_label="True Mass",
                title=f"Mass Regression for {name}",
                x_label="Mass (GeV / c^2)",
                filename=os.path.join(SCRIPT_DIR, f"transformed_dual_histograms/{name[5:-4]}.png")
            )

            create_2var_histogram_with_marker(
                data1=y_gnn_transformed,
                data_label1="GNN Transformed Prediction",
                data2=y_gnn_baseline,
                data_label2="GNN Baseline Prediction",
                marker=y_true[0],
                marker_label="True Mass",
                title=f"Mass Regression for {name}",
                x_label="Mass (GeV / c^2)",
                filename=os.path.join(SCRIPT_DIR, f"transformed_dual_histograms/{name[5:-4]}_gnn.png")
            )

            create_2var_histogram_with_marker(
                data1=y_siamese_finetune,
                data_label1="Siamese Finetune Prediction",
                data2=y_siamese_no_finetune,
                data_label2="Siamese No Finetune Prediction",
                marker=y_true[0],
                marker_label="True Mass",
                title=f"Mass Regression for {name}",
                x_label="Mass (GeV / c^2)",
                filename=os.path.join(SCRIPT_DIR, f"transformed_dual_histograms/{name[5:-4]}_finetuning.png")
            )
    
    compare_performance_all(
        model_performance_dict=model_performance_dict,
        filename=os.path.join(SCRIPT_DIR, "accuracy_plots/transformed_inputs.png")
    )

    # Create subset with only siamese_finetune, siamese_no_finetune, and gnn_baseline metrics
    subset_model_performance_dict = {
        "siamese_finetune": model_performance_dict["siamese_finetune"],
        "siamese_no_finetune": model_performance_dict["siamese_no_finetune"],
        "gnn_baseline": model_performance_dict["gnn_baseline"]
    }
    
    compare_performance_all(
        model_performance_dict=subset_model_performance_dict,
        filename=os.path.join(SCRIPT_DIR, "accuracy_plots/transformed_inputs_best.png")
    )

    with open(os.path.join(SCRIPT_DIR, "transformed_same_event_type_metrics.json"), 'w') as f:
        json.dump(same_event_type_metrics, f, indent=4)


if __name__ == "__main__":
    main()
