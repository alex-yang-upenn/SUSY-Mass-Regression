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
from transformation import create_transformed_dataset, ViewTransformedGenerator


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
    
    # Setup model performance dictionary
    model_performance_dict = {
        "gnn_baseline": {
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
    for name in os.listdir(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test")):
        # Load in unscaled/untransposed data
        print(f"Processing file {name}")
        test = np.load(os.path.join(config.PROCESSED_DATA_DIRECTORY, "test", name))
        X_test = test['X']
        y_test_flattened = test['y']
        y_test = y_test_flattened.reshape(-1, 1)

        # Predictions given by gnn_baseline
        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)
        test_transformed = create_transformed_dataset(
            X_test_scaled, y_test, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES
        )
        test_batches = int(len(X_test_scaled) // config.BATCHSIZE)

        y_gnn_baseline_list = []
        y_true_list = []
        for features, labels in tqdm(test_transformed, desc="Processing batches", total=test_batches):
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

        # Predictions given by lorentz addition
        X_orig = X_test.transpose(0, 2, 1)
        orig_test_transformed = create_transformed_dataset(
            X_orig, y_test, batchsize=config.BATCHSIZE, n_features=config.N_FEATURES
        )

        y_lorentz_list = []
        for features, labels in tqdm(orig_test_transformed, desc="Processing batches", total=test_batches):
            particles = features.numpy().transpose(0, 2, 1)
            particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
            y = vectorized_lorentz_addition(particles, particle_masses)
            y_lorentz_list.append(y)
        y_lorentz = np.concatenate(y_lorentz_list).flatten()

        lorentz_mean = np.mean(y_lorentz)
        lorentz_std = np.std(y_lorentz)
        model_performance_dict["lorentz_addition"]["M_x(true)"].append(y_true[0])
        model_performance_dict["lorentz_addition"]["mean"].append(lorentz_mean)
        model_performance_dict["lorentz_addition"]["+1σ"].append(lorentz_mean + lorentz_std)
        model_performance_dict["lorentz_addition"]["-1σ"].append(lorentz_mean - lorentz_std)

        # Log metrics and visualize selected event types
        if name in config.EVAL_DATA_FILES:
            model_metrics = calculate_metrics(y_true, y_gnn_baseline, "gnn_baseline")
            lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz_addition")

            metrics = {
                **model_metrics,
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
    
    compare_performance_all(
        model_performance_dict=model_performance_dict,
        filename=os.path.join(SCRIPT_DIR, "accuracy_plots/transformed_inputs.png")
    )

    with open(os.path.join(SCRIPT_DIR, "transformed_same_event_type_metrics.json"), 'w') as f:
        json.dump(same_event_type_metrics, f, indent=4)


if __name__ == "__main__":
    main()
