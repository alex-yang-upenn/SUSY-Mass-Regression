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
from tqdm import tqdm

import config
from graph_embeddings import GraphEmbeddings
from utils import *
from plotting import *


def main():
    # Make directories
    os.makedirs(os.path.join(SCRIPT_DIR, "accuracy_plots"), exist_ok=True)
    os.makedirs(os.path.join(SCRIPT_DIR, "dual_histograms"), exist_ok=True)

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
        y_true = test['y']

        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)

        # Predictions given by gnn_baseline
        y_gnn_baseline_scaled = gnn_baseline_model.predict(X_test_scaled, verbose=0)
        y_gnn_baseline = y_scaler.inverse_transform(y_gnn_baseline_scaled).flatten()
        gnn_baseline_mean = np.mean(y_gnn_baseline)
        gnn_baseline_std = np.std(y_gnn_baseline)
        model_performance_dict["gnn_baseline"]["M_x(true)"].append(y_true[0])
        model_performance_dict["gnn_baseline"]["mean"].append(gnn_baseline_mean)
        model_performance_dict["gnn_baseline"]["+1σ"].append(gnn_baseline_mean + gnn_baseline_std)
        model_performance_dict["gnn_baseline"]["-1σ"].append(gnn_baseline_mean - gnn_baseline_std)

        # Predictions given by gnn_transformed
        y_gnn_transformed_scaled = gnn_transformed_model.predict(X_test_scaled, verbose=0)
        y_gnn_transformed = y_scaler.inverse_transform(y_gnn_transformed_scaled).flatten()
        gnn_transformed_mean = np.mean(y_gnn_transformed)
        gnn_transformed_std = np.std(y_gnn_transformed)
        model_performance_dict["gnn_transformed"]["M_x(true)"].append(y_true[0])
        model_performance_dict["gnn_transformed"]["mean"].append(gnn_transformed_mean)
        model_performance_dict["gnn_transformed"]["+1σ"].append(gnn_transformed_mean + gnn_transformed_std)
        model_performance_dict["gnn_transformed"]["-1σ"].append(gnn_transformed_mean - gnn_transformed_std)

        # Predictions given by lorentz addition
        particle_masses = np.zeros((X_test.shape[0], X_test.shape[1]))
        y_lorentz = vectorized_lorentz_addition(X_test, particle_masses)
        lorentz_mean = np.mean(y_lorentz)
        lorentz_std = np.std(y_lorentz)
        model_performance_dict["lorentz_addition"]["M_x(true)"].append(y_true[0])
        model_performance_dict["lorentz_addition"]["mean"].append(lorentz_mean)
        model_performance_dict["lorentz_addition"]["+1σ"].append(lorentz_mean + lorentz_std)
        model_performance_dict["lorentz_addition"]["-1σ"].append(lorentz_mean - lorentz_std)

        # Log metrics and visualize selected event types
        if name in config.EVAL_DATA_FILES:
            gnn_baseline_metrics = calculate_metrics(y_true, y_gnn_baseline, "gnn_baseline")
            gnn_transformed_metrics = calculate_metrics(y_true, y_gnn_transformed, "gnn_transformed")
            lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz_addition")

            metrics = {
                **gnn_baseline_metrics,
                **gnn_transformed_metrics,
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
                filename=os.path.join(SCRIPT_DIR, f"dual_histograms/{name[5:-4]}.png")
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
                filename=os.path.join(SCRIPT_DIR, f"dual_histograms/{name[5:-4]}_gnn.png")
            )
    
    compare_performance_all(
        model_performance_dict=model_performance_dict,
        filename=os.path.join(SCRIPT_DIR, "accuracy_plots/standard_inputs.png")
    )

    with open(os.path.join(SCRIPT_DIR, "same_event_type_metrics.json"), 'w') as f:
        json.dump(same_event_type_metrics, f, indent=4)


if __name__ == "__main__":
    main()