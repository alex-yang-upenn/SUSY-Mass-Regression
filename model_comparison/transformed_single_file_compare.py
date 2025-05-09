import json
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from graph_embeddings import GraphEmbeddings
from transformation import ViewTransformedGenerator
from utils import *

DATA_DIRECTORY = os.path.join(os.path.dirname(SCRIPT_DIR), "processed_data")
SELECTED_FILES = [
    "test_qX_qWY_qqqlv_X200_Y60.npz",
    "test_qX_qWY_qqqlv_X250_Y80.npz",
    "test_qX_qWY_qqqlv_X300_Y100.npz",
    "test_qX_qWY_qqqlv_X350_Y130.npz",
    "test_qX_qWY_qqqlv_X400_Y160.npz",
]
MODEL_NAME = "best_model_1.keras"
OUTPUT_IMAGE_DIRECTORY = os.path.join(SCRIPT_DIR, "transformed_graphs")
N_FEATURES = 6
BATCHSIZE = 128

os.makedirs(OUTPUT_IMAGE_DIRECTORY, exist_ok=True)


def create_transformed_dataset(X, y):
    # Create TensorFlow datasets
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCHSIZE)

    view_transformed_generator = ViewTransformedGenerator(dataset)

    output_signature = (
        tf.TensorSpec(shape=(None, N_FEATURES, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )

    transformed_dataset = tf.data.Dataset.from_generator(
        view_transformed_generator.generate,
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    return transformed_dataset

def main():
    single_file_metrics = {}

    scalers = []
    for i in range(3):
        with open(os.path.join(DATA_DIRECTORY, f"x_scaler_{i}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))
    
    model = tf.keras.models.load_model(os.path.join(SCRIPT_DIR, "checkpoints", MODEL_NAME))
    
    with open(os.path.join(DATA_DIRECTORY, "y_scaler.pkl"), 'rb') as f:
        y_scaler = pickle.load(f)

    for name in SELECTED_FILES:
        # Load in data
        test = np.load(os.path.join(DATA_DIRECTORY, "test", name))
        X_orig = test['X']
        y_orig = test['y']

        # Transform data for Model
        X_test_scaled = scale_data(X_orig, scalers, [0, 1, 2])
        X_test_scaled = X_test_scaled.transpose(0, 2, 1)
        y_test = y_orig.reshape(-1, 1)
        
        test_batches = int(len(X_test_scaled) // BATCHSIZE)

        test_transformed = create_transformed_dataset(X_test_scaled, y_test)

        y_model_list = []
        y_true_list = []
        for features, labels in tqdm(test_transformed, desc="Processing batches", total=test_batches):
            y_model_batch_scaled = model.predict(features, verbose=0)
            
            y_model_batch = y_scaler.inverse_transform(y_model_batch_scaled)
            y_model_list.append(y_model_batch)
            y_true_list.append(labels.numpy())
        y_model = np.concatenate(y_model_list).flatten()
        y_true = np.concatenate(y_true_list).flatten()

        # Transform data for Vectorized lorentz addition
        X_orig_transformed = X_orig.transpose(0, 2, 1)
        test_orig_transformed = create_transformed_dataset(X_orig_transformed, y_test)

        y_lorentz_list = []
        y_lorentz_true_list = []
        for features, labels in tqdm(test_orig_transformed, desc="Processing batches", total=test_batches):
            particles = features.numpy().transpose(0, 2, 1)
            particle_masses = np.zeros((particles.shape[0], particles.shape[1]))
            y = vectorized_lorentz_addition(particles, particle_masses)
            y_lorentz_list.append(y)
            y_lorentz_true_list.append(labels.numpy())
        y_lorentz = np.concatenate(y_lorentz_list).flatten()
        y_lorentz_true = np.concatenate(y_lorentz_true_list).flatten()

        model_metrics = calculate_metrics(y_true, y_model, "best_model_1.keras")
        lorentz_metrics = calculate_metrics(y_true, y_lorentz, "lorentz")

        metrics = {
            **model_metrics,
            **lorentz_metrics
        }
        single_file_metrics[name] = metrics

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


    with open(os.path.join(SCRIPT_DIR, "transformed_single_file_metrics.json"), 'w') as f:
        json.dump(single_file_metrics, f, indent=4)


if __name__ == "__main__":
    main()