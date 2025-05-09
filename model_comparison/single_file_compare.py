import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import *
from plotting import *

DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "processed_data")
SELECTED_FILES = [
    "test_qX_qWY_qqqlv_X200_Y60.npz",
    "test_qX_qWY_qqqlv_X250_Y80.npz",
    "test_qX_qWY_qqqlv_X300_Y100.npz",
    "test_qX_qWY_qqqlv_X350_Y130.npz",
    "test_qX_qWY_qqqlv_X400_Y160.npz",
]
MODEL_NAME = "best_model_1.keras"
OUTPUT_IMAGE_DIRECTORY = os.path.join(SCRIPT_DIR, "graphs")
OUTPUT_COMPARE_DIRECTORY = os.path.join(SCRIPT_DIR, "comparison_plots")

os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)


def main():
    single_file_metrics = {}

    scalers = []
    for i in range(3):
        with open(os.path.join(DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))
    
    model = tf.keras.models.load_model(os.path.join(ROOT_DIR, "gnn_baseline", "checkpoints", MODEL_NAME))
    
    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    for name in SELECTED_FILES:
        # Load in raw data
        test = np.load(os.path.join(DATA_DIRECTORY, "test", name))
        X_test = test['X']
        y_true = test['y']

        X_test_scaled = scale_data(X_test, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)

    
        y_model_scaled = model.predict(X_test_scaled, verbose=1)
        y_model = y_scaler.inverse_transform(y_model_scaled).flatten()

        # Vectorized lorentz addition
        particle_masses = np.zeros((X_test.shape[0], X_test.shape[1]))
        y_lorentz = vectorized_lorentz_addition(X_test, particle_masses)

        model_metrics = calculate_metrics(y_true, y_model, "best_model_1.keras")
        lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz")

        metrics = {
            **model_metrics,
            **lorentz_metrics
        }
        single_file_metrics[name] = metrics
        
        # Graph Double Histogram 
        create_2var_histogram_with_marker(
            data1=y_model,
            data_label1="GNN Prediction",
            data2=y_lorentz,
            data_label2="Lorentz Addition Prediction",
            marker=y_true[0],
            marker_label="True Mass",
            title=f"Mass Regression for {name}",
            x_label="Mass (GeV / c^2)",
            filename=f"{OUTPUT_IMAGE_DIRECTORY}/{name}.png"
        )
        
        model_performance_dict = {
            "GNN_original": {
                "mean": np.mean(y_model) / y_true[0],
                "+1σ": (np.mean(y_model) + np.std(y_model)) / y_true[0],
                "-1σ": (np.mean(y_model) - np.std(y_model)) / y_true[0],
            },
            "Lorentz_original": {
                "mean": np.mean(y_lorentz) / y_true[0],
                "+1σ": (np.mean(y_lorentz) + np.std(y_lorentz)) / y_true[0],
                "-1σ": (np.mean(y_lorentz) - np.std(y_lorentz)) / y_true[0],
            }
        }
        compare_performance_single(
            model_performance_dict=model_performance_dict,
            filename=f"{OUTPUT_COMPARE_DIRECTORY}/{name}.png",
        )

    with open(os.path.join(SCRIPT_DIR, "single_file_metrics.json"), 'w') as f:
        json.dump(single_file_metrics, f, indent=4)


if __name__ == "__main__":
    main()